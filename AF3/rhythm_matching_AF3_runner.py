# rhythm_matching_AF3_runner.py
# AF3 — Rhythm Matching (stateless; ONE-audio-per-call using pre-combined files; trial-only prompts)

import os, re, gc, random, logging, warnings
from typing import List, Dict, Tuple
warnings.filterwarnings("ignore")

try:
    import torch
except Exception:
    torch = None

# =============================
# Constants / paths
# =============================
STIM_ROOT = "stimuli"

# Decoding params (match Instrument ID AF3 runner)
DECODE = dict(
    max_new_tokens=8192,
    temperature=1.0,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.0,
)

# Canonical answer strings and robust patterns
YES_CANON = 'Yes, these are the same exact drum set patterns.'
NO_CANON  = 'No, these are not the same exact drum set patterns.'

# Anywhere-in-text patterns (fallback)
YES_PAT_ANY = re.compile(r'(?i)\byes,\s*these\s+are\s+the\s+same\s+exact\s+drum\s+set\s+patterns\.')
NO_PAT_ANY  = re.compile(r'(?i)\bno,\s*these\s+are\s+not\s+the\s+same\s+exact\s+drum\s+set\s+patterns\.')

# Anchored "final-line-only" patterns (preferred)
YES_PAT_LINE = re.compile(r'(?im)^\s*"?yes,\s*these\s+are\s+the\s+same\s+exact\s+drum\s+set\s+patterns\."?\s*$')
NO_PAT_LINE  = re.compile(r'(?im)^\s*"?no,\s*these\s+are\s+not\s+the\s+same\s+exact\s+drum\s+set\s+patterns\."?\s*$')

# ===== Trial prompts (stateless; per-trial only) =====
TRIAL_PROMPT_SYS = """You are a participant in a psychological experiment on music perception.
In each question, you will be given a single audio file that contains two excerpts back-to-back with a voice saying "Here is the first excerpt" before the first excerpt is presented. After the first excerpt plays, the same voice then says "Here is the second excerpt" before the second excerpt is presented.

Your task is to decide whether the two music excerpts have the same exact drum set pattern — meaning the same exact rhythmic structure — or if they are different. All rhythms are at the same tempo.

Valid responses are exactly:
Yes, these are the same exact drum set patterns.
or
No, these are not the same exact drum set patterns.

Output exactly ONE of the two lines above. Do not add any other text, quotes, or commentary. Do not mention being a voice assistant.
"""

COT_PROMPT_TEMPLATE = """You are a participant in a psychological experiment on music perception.
In each question, you will be given a single audio file that contains two excerpts back-to-back with a voice saying "Here is the first excerpt" before the first excerpt is presented. After the first excerpt plays, the same voice then says "Here is the second excerpt" before the second excerpt is presented.

Analyze the two music excerpts to determine if they are the SAME exact drum set pattern.

Step 1: Establish the pulse and identify the smallest repeating cycle you hear for each excerpt (e.g., one bar or two bars).

Step 2: For each excerpt:
- Identify the overall feel of the rhythm (straight/duple vs. triplet/swing),
- Identify the meter, and
- Examine the positions of where the Kick, Snare, Hi-Hat/Cymbal, and Tom hits fall.

Step 3: Compare the two patterns. They are the same pattern if:
- The cycle lengths match,
- They have the same rhythmic feel,
- They have the same meter, and
- The positions for each drum voice match across the two excerpts (no added/removed/moved hits, and no voice swaps).

Notes:
- Tempo is the same across excerpts; focus on placement within the cycle.
- Don’t assume a specific meter; use the repeating structure you hear.

Step 4: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
Yes, these are the same exact drum set patterns.
OR
No, these are not the same exact drum set patterns.
Do not include quotes or extra text. Do not mention being a voice assistant.
"""

# =============================
# Logging
# =============================
def configure_logging(log_filename: str):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        try: h.close()
        finally: root.removeHandler(h)
    fh = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    root.addHandler(fh)

def make_log_filename(*, mode: str, group: str, seed: int) -> str:
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"rhythm_matching_AF3_CHAT_{mode}_{group}_seed{seed}.log"

# =============================
# Paths & parsing
# =============================
def _ppath(p: str) -> str:
    if os.path.isabs(p): return p
    if p.startswith(STIM_ROOT + os.sep) or p.startswith(STIM_ROOT + "/"):
        return p
    return os.path.join(STIM_ROOT, p)

def _anchored_find(text: str) -> str:
    """Prefer a single-line anchored final answer."""
    has_yes = bool(YES_PAT_LINE.search(text or ""))
    has_no  = bool(NO_PAT_LINE.search(text or ""))
    if has_yes and not has_no:
        return YES_CANON
    if has_no and not has_yes:
        return NO_CANON
    # if both anchored present, prefer the one that appears last
    if has_yes and has_no:
        last_yes = None
        last_no  = None
        for m in YES_PAT_LINE.finditer(text or ""):
            last_yes = m
        for m in NO_PAT_LINE.finditer(text or ""):
            last_no = m
        if last_yes and last_no:
            return YES_CANON if last_yes.end() > last_no.end() else NO_CANON
    return ""

def _anywhere_find(text: str) -> str:
    """Fallback: pick the LAST occurrence anywhere in text."""
    last_yes = None
    last_no  = None
    for m in YES_PAT_ANY.finditer(text or ""):
        last_yes = m
    for m in NO_PAT_ANY.finditer(text or ""):
        last_no = m
    if last_yes and last_no:
        return YES_CANON if last_yes.end() > last_no.end() else NO_CANON
    if last_yes:
        return YES_CANON
    if last_no:
        return NO_CANON
    return ""

def parse_final_decision(text: str) -> str:
    """Two-stage parsing: anchored line first, then anywhere fallback."""
    return _anchored_find(text) or _anywhere_find(text)

def _is_ambiguous_both_anywhere(text: str) -> bool:
    """True if both phrases appear somewhere, but NO anchored final line present."""
    both_any = bool(YES_PAT_ANY.search(text or "")) and bool(NO_PAT_ANY.search(text or ""))
    anchored = bool(YES_PAT_LINE.search(text or "")) or bool(NO_PAT_LINE.search(text or ""))
    return both_any and not anchored

def _beat_id_from_base(base_no_ext: str) -> str:
    m = re.match(r'(?i)^(Beat_\d+)', base_no_ext)
    return m.group(1) if m else base_no_ext

def pair_ids_from_combined(path: str) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(path))[0]
    if "__" in base:
        left, right = base.split("__", 1)
    else:
        left = right = base
    return _beat_id_from_base(left), _beat_id_from_base(right)

def stimuli_list_group(group: str) -> List[Dict[str, str]]:
    assert group in {"GroupA", "GroupB"}
    if group == "GroupA":
        names = [
            "stimuli/combined/Beat_1_140__Beat_1_140.wav",
            "stimuli/combined/Beat_1_140__Beat_2_140.wav",
            "stimuli/combined/Beat_2_140__Beat_2_140.wav",
            "stimuli/combined/Beat_2_140__Beat_3_140.wav",
            "stimuli/combined/Beat_3_140__Beat_3_140.wav",
            "stimuli/combined/Beat_3_140__Beat_4_140_34.wav",
            "stimuli/combined/Beat_4_140_34__Beat_4_140_34.wav",
            "stimuli/combined/Beat_4_140_34__Beat_5_140.wav",
            "stimuli/combined/Beat_5_140__Beat_5_140.wav",
            "stimuli/combined/Beat_5_140__Beat_6_140_34.wav",
        ]
    else:
        names = [
            "stimuli/combined/Beat_6_140_34__Beat_6_140_34.wav",
            "stimuli/combined/Beat_6_140_34__Beat_7_140.wav",
            "stimuli/combined/Beat_7_140__Beat_7_140.wav",
            "stimuli/combined/Beat_7_140__Beat_8_140.wav",
            "stimuli/combined/Beat_8_140__Beat_8_140.wav",
            "stimuli/combined/Beat_8_140__Beat_9_140_34.wav",
            "stimuli/combined/Beat_9_140_34__Beat_9_140_34.wav",
            "stimuli/combined/Beat_9_140_34__Beat_10_140.wav",
            "stimuli/combined/Beat_10_140__Beat_1_140.wav",
            "stimuli/combined/Beat_10_140__Beat_10_140.wav",
        ]
    return [{"file": _ppath(n)} for n in names]

# =============================
# AF3 bindings (llava)
# =============================
class AF3NotFound(Exception): pass

def _import_llava():
    tried = []
    try:
        import llava
        return llava
    except Exception as e:
        tried.append(("llava", e))
    try:
        import llava_next as llava
        return llava
    except Exception as e:
        tried.append(("llava_next", e))
    raise AF3NotFound("Could not import AF3 llava bindings. Tried: " + ", ".join(k for k,_ in tried))

_AF3_CACHE: Dict[str, object] = {"model": None, "llava": None}

def _load_af3_model():
    llava = _import_llava()
    model_name = os.environ.get("AF3_MODEL", "nvidia/audio-flamingo-3-chat")
    local_dir = os.environ.get("AF3_LOCAL_DIR", "").strip() or None
    try:
        model = llava.load(local_dir if local_dir else model_name, model_base=None, devices=[0])
    except Exception as e:
        raise AF3NotFound(f"Failed to load AF3 model: {e}")
    try:
        g = model.default_generation_config
        g.max_new_tokens = DECODE["max_new_tokens"]
        g.temperature = DECODE["temperature"]
        g.top_p = DECODE["top_p"]
        g.top_k = DECODE["top_k"]
        g.repetition_penalty = DECODE["repetition_penalty"]
        try:
            g.stop = ["\nUSER:", "\nASSISTANT:"]
        except Exception:
            pass
    except Exception:
        pass
    return model

def get_af3():
    if _AF3_CACHE["model"] is None:
        llava = _import_llava()
        model = _load_af3_model()
        _AF3_CACHE["model"] = model
        _AF3_CACHE["llava"] = llava
    return _AF3_CACHE["model"], _AF3_CACHE["llava"]

def af3_generate(model, parts) -> str:
    out = model.generate_content(parts)
    return (out.text if hasattr(out, "text") else str(out)).strip()

# =============================
# Trial helper (single call, ONE combined audio, trial-only prompt)
# =============================
def run_trial(model, llava, *, mode: str, trial_path: str, logger: logging.Logger) -> Tuple[str, str]:
    assert mode in {"SYSINST", "COT"}
    prompt = TRIAL_PROMPT_SYS if mode == "SYSINST" else COT_PROMPT_TEMPLATE

    # One-audio-per-call; audio FIRST, then the EXACT trial prompt text
    resp = af3_generate(model, [llava.Sound(trial_path), prompt.strip()])

    # If ambiguous (mentions both phrases) or failed to parse an anchored final line, force a stricter re-ask
    ambiguous = _is_ambiguous_both_anywhere(resp)
    parsed = parse_final_decision(resp)

    if ambiguous or not parsed:
        fallback = (
            """If the first and second drum set patterns are the same, write exactly: Yes, these are the same exact drum set patterns.
             If they are different, write exactly: No, these are not the same exact drum set patterns."""
        )
        resp = af3_generate(model, [llava.Sound(trial_path), fallback])
        parsed = parse_final_decision(resp)

        # One more ultra-minimal nudge if still not clean
        if _is_ambiguous_both_anywhere(resp) or not parsed:
            fallback2 = "Final answer (one line ONLY): Yes, these are the same exact drum set patterns. OR No, these are not the same exact drum set patterns."
            resp = af3_generate(model, [llava.Sound(trial_path), fallback2])

    history_text = f"USER: {prompt.strip()}\nASSISTANT: {resp}"
    return resp, history_text

# =============================
# Runner
# =============================
def run_once(*, mode: str, group: str, seed: int, log_filename: str) -> None:
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}

    configure_logging(log_filename)
    logging.info(f"=== RUN START (Rhythm Matching • AF3-CHAT • STATELESS • {mode} • {group}) ===")

    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    model, llava = get_af3()
    model_name = os.environ.get("AF3_MODEL", "nvidia/audio-flamingo-3-chat")
    logging.info(f"Config: model={model_name}, temp={DECODE['temperature']}, top_p={DECODE['top_p']}, "
                 f"top_k={DECODE['top_k']}, seed={seed}, group={group}, log={log_filename}")

    question_stims = stimuli_list_group(group)
    random.shuffle(question_stims)

    print(f"\n--- Rhythm Matching — STATELESS • {mode} • {group} | model={model_name} | seed={seed} ---\n")
    logging.info(f"\n--- Task: Rhythm Matching — STATELESS • {mode} • {group} ---\n")

    correct = 0
    total = len(question_stims)

    for idx, q in enumerate(question_stims, start=1):
        fpath = q["file"]

        logging.info(f"\n--- Question {idx} ---")
        logging.info(f"Stimulus: file={fpath}")

        if not os.path.exists(fpath):
            print(f"[WARN] Missing file: {fpath}")
            logging.error(f"Missing stimulus file: {fpath}")
            continue

        final_resp, text_trace = run_trial(
            model, llava, mode=mode, trial_path=fpath, logger=logging.getLogger()
        )

        print(f"\n--- Q{idx}: {os.path.basename(fpath)} ---")
        print("LLM Full Response:\n", final_resp)
        logging.info(f"[{mode}/{group}] Q{idx} - LLM Full Response:\n{final_resp}")

        parsed = parse_final_decision(final_resp)
        if not parsed:
            print("Evaluation: Failed. Could not parse the final answer phrase.")
            logging.error("Parse Error: missing/malformed final answer phrase.")
            continue

        logging.info(f"Parsed Final Answer: {parsed}")

        # Ground truth from combined filename halves (Beat_X vs Beat_Y)
        left_id, right_id = pair_ids_from_combined(fpath)
        expected = YES_CANON if left_id == right_id else NO_CANON
        logging.info(f"Derived IDs: left={left_id}, right={right_id}; Expected={expected}")

        if parsed == expected:
            correct += 1
            print("Evaluation: Correct!")
            logging.info("Evaluation: Correct")
        else:
            print(f"Evaluation: Incorrect. (expected={expected})")
            logging.info(f"Evaluation: Incorrect (expected={expected})")

    print(f"\nTotal Correct: {correct} out of {total}")
    logging.info(f"Total Correct: {correct} out of {total}")
    logging.info("=== RUN END ===\n")

# =============================
# Multi-run driver (12 total)
# =============================
if __name__ == "__main__":
    runs = []
    for mode in ["SYSINST", "COT"]:
        for group in ["GroupA"]:
            for s in (1, 2):
                runs.append(dict(
                    mode=mode,
                    group=group,
                    seed=s,
                    log_filename=make_log_filename(mode=mode, group=group, seed=s),
                ))
    for cfg in runs:
        try:
            run_once(**cfg)
        finally:
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
