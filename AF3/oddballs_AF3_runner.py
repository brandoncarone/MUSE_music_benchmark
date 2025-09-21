# oddballs_AF3_runner.py
# AF3 — Oddball Detection (stateless; one-audio-per-call; combined "Excerpt1__Excerpt2" files)

import os, re, gc, random, logging, warnings
from typing import List, Dict
warnings.filterwarnings("ignore")

try:
    import torch
except Exception:
    torch = None

# =============================
# Constants / paths
# =============================
STIM_ROOT = "stimuli"

DECODE = dict(
    max_new_tokens=8192,
    temperature=1.0,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.0,
)

# Canonical answer strings (EXACT)
YES_CANON = "Yes, these are the same exact melody."
NO_CANON  = "No, these are not the same exact melody."

# Robust patterns (case-insensitive) — prefer the LAST occurrence
YES_PAT = re.compile(r'(?i)\byes,\s*these\s+are\s+the\s+same\s+exact\s+melody\.')
NO_PAT  = re.compile(r'(?i)\bno,\s*these\s+are\s+not\s+the\s+same\s+exact\s+melody\.')

# =============================
# Trial Prompts (VERBATIM)
# =============================
TRIAL_PROMPT_SYS = """You are a participant in a psychological experiment on music perception.
In each question, you will be given a single audio file that contains two excerpts back-to-back with a voice saying "Here is the first excerpt" before the first excerpt is presented. After the first excerpt plays, the same voice then says "Here is the second excerpt" before the second excerpt is presented.

Your task is to decide whether the two audio examples are the same exact melody, or whether an "Oddball" is present. An “Oddball” in a musical or auditory experiment is simply a note or sound that doesn’t fit with what you’d expect based on what you’ve been hearing. Imagine you’re listening to a melody where all the notes line up nicely in the same key—then suddenly, one note is out of key. This unexpected note is the "oddball". If the note is present multiple times in the melody, then you will hear the oddball more than once.

Valid responses are:
“Yes, these are the same exact melody.” if they are exactly the same, or 
“No, these are not the same exact melody.” if you notice an oddball."""

COT_PROMPT_TEMPLATE = """You are a participant in a psychological experiment on music perception.
In each question, you will be given a single audio file that contains two excerpts back-to-back with a voice saying "Here is the first excerpt" before the first excerpt is presented. After the first excerpt plays, the same voice then says "Here is the second excerpt" before the second excerpt is presented.

Your task is to decide whether the two audio examples are the same exact melody, or whether an "Oddball" is present. An “Oddball” represents one or more unexpected notes that do not match the expected melody (e.g., out-of-key or altered notes). The oddball may occur more than once.

Step 1: For each audio, identify the monophonic note sequence (pitch over time). Ignore small timing differences and leading/trailing silence.

Step 2: Align the two sequences by their note order and compare pitch at each corresponding position (no transposition or octave equivalence).

Step 3: Decision rule:
- If all corresponding pitches match exactly, there is no oddball, and they are the same exact melody.
- If any pitch differs (e.g., out-of-key substitution or altered note), there is an oddball present, and they are not the same melody.

Step 4: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
Yes, these are the same exact melody.
OR
No, these are not the same exact melody."""

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
    """
    Two modes: SYSINST (plain), COT (reasoning)
    Two groups: GroupA / GroupB
    Example: oddballs_AF3_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"oddballs_AF3_CHAT_{mode}_{group}_seed{seed}.log"

# =============================
# Stimuli (combined; one file per trial)
# =============================
def _ppath(p: str) -> str:
    """Normalize: abs -> as-is; starts with 'stimuli/' -> as-is; else join with STIM_ROOT."""
    if os.path.isabs(p): return p
    if p.startswith(STIM_ROOT + os.sep) or p.startswith(STIM_ROOT + "/"):
        return p
    return os.path.join(STIM_ROOT, p)

def stimuli_list_group(group: str) -> List[Dict[str, str]]:
    """
    AF3 takes one audio per call, so we use pre-combined files of the form:
      '.../<left>__<right>.wav'
    where the audio contains:
      - Voice: "Here is the first excerpt" → Excerpt 1
      - Voice: "Here is the second excerpt" → Excerpt 2
    """
    assert group in {"GroupA", "GroupB"}
    if group == "GroupA":
        files = [
            "stimuli/combined/M1_EbMaj_90__M1_EbMaj_90.wav",
            "stimuli/combined/M1_EbMaj_90__M1_Odd_EbMaj_90.wav",
            "stimuli/combined/M2_Abm_155_3_4__M2_Abm_155_3_4.wav",
            "stimuli/combined/M2_Abm_155_3_4__M2_Odd_Abm_155_3_4.wav",
            "stimuli/combined/M3_DbMaj_78__M3_DbMaj_78.wav",
            "stimuli/combined/M8_FMaj_95_Piano__M8_Odd_FMaj_95_Piano.wav",
            "stimuli/combined/M9_Gm_200_3_4_Piano__M9_Gm_200_3_4_Piano.wav",
            "stimuli/combined/M9_Gm_200_3_4_Piano__M9_Odd_Gm_200_3_4_Piano.wav",
            "stimuli/combined/M10_Fm_165_3_4__M10_Fm_165_3_4.wav",
            "stimuli/combined/M10_Fm_165_3_4__M10_Odd_Fm_165_3_4.wav",
        ]
    else:
        files = [
            "stimuli/combined/M3_DbMaj_78__M3_Odd_DbMaj_78.wav",
            "stimuli/combined/M4_EMaj_130__M4_EMaj_130.wav",
            "stimuli/combined/M4_EMaj_130__M4_Odd_EMaj_130.wav",
            "stimuli/combined/M5_Dm_100__M5_Dm_100.wav",
            "stimuli/combined/M5_Dm_100__M5_Odd_Dm_100.wav",
            "stimuli/combined/M6_Cm_120_Piano__M6_Cm_120_Piano.wav",
            "stimuli/combined/M6_Cm_120_Piano__M6_Odd_Cm_120_Piano.wav",
            "stimuli/combined/M7_CMaj_140_Piano__M7_CMaj_140_Piano.wav",
            "stimuli/combined/M7_CMaj_140_Piano__M7_Odd_CMaj_140_Piano.wav",
            "stimuli/combined/M8_FMaj_95_Piano__M8_FMaj_95_Piano.wav",
        ]
    return [{"file": _ppath(p)} for p in files]

# =============================
# Parsing / evaluation helpers
# =============================
def parse_final_decision(text: str) -> str:
    """Return YES_CANON or NO_CANON, or '' if not found. Prefer the LAST occurrence."""
    last_yes = last_no = None
    for m in YES_PAT.finditer(text or ""): last_yes = m
    for m in NO_PAT.finditer(text or ""):  last_no  = m
    if last_yes and last_no:
        return YES_CANON if last_yes.end() > last_no.end() else NO_CANON
    if last_yes: return YES_CANON
    if last_no:  return NO_CANON
    return ""

def _normalize_pair_tokens_from_combined(path: str) -> tuple[str, str]:
    """
    Given '.../<LEFT>__<RIGHT>.wav', return (LEFT_base_no_ext, RIGHT_base_no_ext).
    LEFT usually has no '.wav' in the combined name; RIGHT ends with '.wav'.
    """
    base = os.path.basename(path)
    if "__" not in base:
        return base, base
    left, right = base.split("__", 1)
    left_noext = os.path.splitext(left)[0]
    right_noext = os.path.splitext(right)[0]
    return left_noext, right_noext

def expected_for_combined(path: str) -> str:
    """
    Ground truth rule for Oddball task:
      - SAME exact melody if LEFT token == RIGHT token (after stripping extensions)
      - Otherwise NOT the same exact melody (oddball present)
    """
    left, right = _normalize_pair_tokens_from_combined(path)
    return YES_CANON if left == right else NO_CANON

# =============================
# AF3 bindings (LLaVA-style API)
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
    # Apply decoding knobs
    try:
        g = model.default_generation_config
        g.max_new_tokens = DECODE["max_new_tokens"]
        g.temperature = DECODE["temperature"]
        g.top_p = DECODE["top_p"]
        g.top_k = DECODE["top_k"]
        g.repetition_penalty = DECODE["repetition_penalty"]
        try: g.stop = ["\nUSER:", "\nASSISTANT:"]
        except Exception: pass
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
# Trial helper (single call; one combined audio; trial-only prompt)
# =============================
def run_trial(model, llava, *, mode: str, trial_path: str) -> str:
    assert mode in {"SYSINST", "COT"}
    prompt = TRIAL_PROMPT_SYS if mode == "SYSINST" else COT_PROMPT_TEMPLATE

    resp = af3_generate(model, [llava.Sound(trial_path), prompt.strip()])

    # Tight fallback if parsing failed
    if not parse_final_decision(resp):
        fallback = (
            "This audio contains two excerpts with spoken markers. "
            "Answer with exactly ONE line:\n"
            "Yes, these are the same exact melody.\n"
            "OR\n"
            "No, these are not the same exact melody."
        )
        resp = af3_generate(model, [llava.Sound(trial_path), fallback])
    return resp

# =============================
# Runner
# =============================
def configure_cuda_for_repro(seed: int):
    if torch is not None:
        try: torch.manual_seed(seed)
        except Exception: pass
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def run_once(*, mode: str, group: str, seed: int, log_filename: str) -> None:
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}

    configure_logging(log_filename)
    logging.info(f"=== RUN START (Oddball Detection • AF3-CHAT • STATELESS • {mode} • {group}) ===")

    random.seed(seed)
    configure_cuda_for_repro(seed)

    model, llava = get_af3()
    model_name = os.environ.get("AF3_MODEL", "nvidia/audio-flamingo-3-chat")
    logging.info(f"Config: model={model_name}, temp={DECODE['temperature']}, top_p={DECODE['top_p']}, "
                 f"top_k={DECODE['top_k']}, seed={seed}, group={group}, log={log_filename}")

    question_stims = stimuli_list_group(group)
    random.shuffle(question_stims)

    print(f"\n--- Oddball Detection — STATELESS • {mode} • {group} | model={model_name} | seed={seed} ---\n")
    logging.info(f"\n--- Task: Oddball Detection — STATELESS • {mode} • {group} ---\n")

    correct = 0
    total = len(question_stims)

    for idx, q in enumerate(question_stims, start=1):
        fpath = q["file"]

        logging.info(f"\n--- Question {idx} ---")
        logging.info(f"Stimulus (combined): file={fpath}")

        if not os.path.exists(fpath):
            print(f"[WARN] Missing file: {fpath}")
            logging.error(f"Missing stimulus file: {fpath}")
            continue

        final_resp = run_trial(model, llava, mode=mode, trial_path=fpath)

        print(f"\n--- Q{idx}: {os.path.basename(fpath)} ---")
        print("LLM Full Response:\n", final_resp)
        logging.info(f"[{mode}/{group}] Q{idx} - LLM Full Response:\n{final_resp}")

        parsed = parse_final_decision(final_resp)
        if not parsed:
            print("Evaluation: Failed. Could not parse the final answer phrase.")
            logging.error("Parse Error: missing/malformed final answer phrase.")
            continue

        logging.info(f"Parsed Final Answer: {parsed}")

        expected = expected_for_combined(fpath)
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
        for group in ["GroupA", "GroupB"]:
            for s in (1, 2, 3):
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
