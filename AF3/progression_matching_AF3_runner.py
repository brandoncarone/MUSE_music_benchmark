# chord_progression_matching_AF3_runner.py
# AF3 — Chord Progression Matching (stateless; one-audio-per-call; combined "Left__Right" files)

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

DECODE = dict(
    max_new_tokens=8192,
    temperature=1.0,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.0,
)

# Canonical answer strings and robust patterns (accept optional A./B. labels)
A_CANON = "A. Yes, these are the same chord progression."
B_CANON = "B. No, these are not the same chord progression."

A_PAT = re.compile(r'(?i)\b(?:a\.\s*)?yes,\s*these\s+are\s+the\s+same\s+chord\s+progression\.')
B_PAT = re.compile(r'(?i)\b(?:b\.\s*)?no,\s*these\s+are\s+not\s+the\s+same\s+chord\s+progression\.')

# =============================
# ===== Trial prompts (stateless; per-trial only) =====
# =============================
TRIAL_PROMPT_SYS = """You are a participant in a psychological experiment on music perception.
In each question, you will be given a single audio file that contains two excerpts back-to-back with a voice saying "Here is the first excerpt" before the first excerpt is presented. After the first excerpt plays, the same voice then says "Here is the second excerpt" before the second excerpt is presented.

Your task is to decide if two excerpts follow the same underlying chord progression, even if they are played with different instruments or in different styles. Think of it like a "musical sentence" — the same sentence can be said by different people conveying the same meaning, but it may not always sound exactly the same.

Valid responses are:
"Yes, these are the same chord progression." or 
"No, these are not the same chord progression."
"""

COT_PROMPT_TEMPLATE = """You are a participant in a psychological experiment on music perception.
In each question, you will be given a single audio file that contains two excerpts back-to-back with a voice saying "Here is the first excerpt" before the first excerpt is presented. After the first excerpt plays, the same voice then says "Here is the second excerpt" before the second excerpt is presented.

Your task is to decide if two excerpts follow the same underlying chord progression, even if they are played with different instruments or in different styles.

Step 1: For each excerpt, segment by harmonic rhythm (where chords change) and abstract the progression as harmonic functions relative to its local key (e.g., I–vi–ii–V), ignoring inversions and extensions.

Step 2: Compare the two progressions across one cycle:
- Do they contain the same set of harmonic functions in the SAME ORDER?
- Differences in instrument, voicing, tempo, or chord durations are acceptable; ORDER must match.

Step 3: Decision rule:
- If the functional ORDER matches across the cycle, the two excerpts contain the same chord progression.
- If functions or their ORDER differ (insertions/deletions/substitutions that change function, or re-ordered segments), they are not the same.

Step 4: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
A. Yes, these are the same chord progression.
OR
B. No, these are not the same chord progression.
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
    """
    Two modes: SYSINST (plain), COT (reasoning)
    Two groups: GroupA / GroupB
    Example: chord_progression_matching_AF3_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"chord_progression_matching_AF3_CHAT_{mode}_{group}_seed{seed}.log"

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
      '.../<LEFT>__<RIGHT>.wav'
    where the audio contains:
      - Voice: "Here is the first excerpt" → Excerpt 1
      - Voice: "Here is the second excerpt" → Excerpt 2
    """
    assert group in {"GroupA", "GroupB"}
    if group == "GroupA":
        files = [
            "stimuli/combined/I-V-vi-IV_Dmaj_CrunchGuit_100__I-V-vi-IV_Dmaj_AcousticGuit_115.wav",
            "stimuli/combined/I-vi-VI-V_Fmaj_piano_172_3_4__I-vi-VI-V_Fmaj_piano_175_6_8.wav",
            "stimuli/combined/vi-IV-I-V_Gmaj_AcousticGuit_118__vi-IV-I-V_Gmaj_piano_135.wav",
            "stimuli/combined/I-IV-V_Emaj_piano_120__I-IV-V_Emaj_DistortedGuit_175.wav",
            "stimuli/combined/I-vi-ii-V_Cmaj_CleanGuitar_80__I-vi-ii-V_Cmaj_piano_125.wav",
            "stimuli/combined/I-IV-V_Emaj_DistortedGuit_175__I-V-vi-IV_Dmaj_CrunchGuit_100.wav",
            "stimuli/combined/vi-IV-I-V_Gmaj_piano_135__I-vi-VI-V_Fmaj_piano_172_3_4.wav",
            "stimuli/combined/I-V-vi-IV_Dmaj_AcousticGuit_115__I-vi-ii-V_Cmaj_piano_125.wav",
            "stimuli/combined/I-vi-VI-V_Fmaj_piano_175_6_8__vi-IV-I-V_Gmaj_AcousticGuit_118.wav",
            "stimuli/combined/I-IV-V_Emaj_piano_120__I-vi-ii-V_Cmaj_CleanGuitar_80.wav",
        ]
    else:
        files = [
            "stimuli/combined/I-vi-IV-V_Fmaj_CrunchGuit_140__I-vi-IV-V_Fmaj_CrunchEffectsGuit_160.wav",
            "stimuli/combined/I-IV-V_Emaj_piano_150__I-IV-V_Emaj_piano_120.wav",
            "stimuli/combined/I-V-vi-IV_Dmaj_piano_145__I-V-vi-IV_Dmaj_piano_115.wav",
            "stimuli/combined/I-vi-ii-V_Cmaj_CleanWahGuitar_120__I-vi-ii-V_Cmaj_piano_165.wav",
            "stimuli/combined/vi-IV-I-V_Gmaj_piano_165__vi-IV-I-V_Gmaj_CrunchGuit_150.wav",
            "stimuli/combined/I-vi-ii-V_Cmaj_CleanWahGuitar_120__I-vi-IV-V_Fmaj_CrunchGuit_140.wav",
            "stimuli/combined/I-IV-V_Emaj_piano_150__I-V-vi-IV_Dmaj_piano_115.wav",
            "stimuli/combined/I-vi-IV-V_Fmaj_CrunchEffectsGuit_160__vi-IV-I-V_Gmaj_piano_165.wav",
            "stimuli/combined/I-V-vi-IV_Dmaj_piano_145__vi-IV-I-V_Gmaj_CrunchGuit_150.wav",
            "stimuli/combined/I-IV-V_Emaj_piano_120__I-vi-ii-V_Cmaj_piano_165.wav",
        ]
    return [{"file": _ppath(p)} for p in files]

# =============================
# Parsing / evaluation helpers
# =============================
def parse_final_decision(text: str) -> str:
    """Return A_CANON or B_CANON, or '' if not found. Prefer the LAST occurrence."""
    last_a = last_b = None
    for m in A_PAT.finditer(text or ""): last_a = m
    for m in B_PAT.finditer(text or ""): last_b = m
    if last_a and last_b:
        return A_CANON if last_a.end() > last_b.end() else B_CANON
    if last_a: return A_CANON
    if last_b: return B_CANON
    return ""

_ROMAN_TOKEN = re.compile(r'^(?i)(i|ii|iii|iv|v|vi|vii)$')

def _normalize_progression_token(tok: str) -> str:
    t = tok.strip().replace('–', '-').replace('—', '-')
    t = t.strip('-').upper()
    return t

def extract_progression_from_name(name: str) -> str:
    """
    From a name like 'I-V-vi-IV_Dmaj_CrunchGuit_100' return normalized 'I-V-VI-IV'.
    Only the hyphen-separated roman section before the first underscore is considered.
    """
    base = os.path.basename(name)
    head = base.split('_', 1)[0]  # 'I-V-vi-IV'
    tokens = [_normalize_progression_token(t) for t in head.split('-') if t]
    filtered = [t for t in tokens if _ROMAN_TOKEN.match(t)]
    return "-".join(filtered)

def _split_combined_tokens(path: str) -> Tuple[str, str]:
    """
    Given '.../<LEFT>__<RIGHT>.wav', return (LEFT_base_no_ext, RIGHT_base_no_ext).
    """
    base = os.path.basename(path)
    if "__" not in base:
        stem = os.path.splitext(base)[0]
        return stem, stem
    left, right = base.split("__", 1)
    return os.path.splitext(left)[0], os.path.splitext(right)[0]

def expected_for_combined(path: str) -> str:
    """
    Ground truth via roman-numeral progression extracted from LEFT and RIGHT tokens.
    """
    left_stem, right_stem = _split_combined_tokens(path)
    prog_left  = extract_progression_from_name(left_stem)
    prog_right = extract_progression_from_name(right_stem)
    return A_CANON if prog_left and prog_right and (prog_left == prog_right) else B_CANON

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
            "Answer with exactly ONE of the following lines:\n"
            "A. Yes, these are the same chord progression.\n"
            "OR\n"
            "B. No, these are not the same chord progression."
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
    logging.info(f"=== RUN START (Chord Progression Matching • AF3-CHAT • STATELESS • {mode} • {group}) ===")

    random.seed(seed)
    configure_cuda_for_repro(seed)

    model, llava = get_af3()
    model_name = os.environ.get("AF3_MODEL", "nvidia/audio-flamingo-3-chat")
    logging.info(f"Config: model={model_name}, temp={DECODE['temperature']}, top_p={DECODE['top_p']}, "
                 f"top_k={DECODE['top_k']}, seed={seed}, group={group}, log={log_filename}")

    question_stims = stimuli_list_group(group)
    random.shuffle(question_stims)

    print(f"\n--- Chord Progression Matching — STATELESS • {mode} • {group} | model={model_name} | seed={seed} ---\n")
    logging.info(f"\n--- Task: Chord Progression Matching — STATELESS • {mode} • {group} ---\n")

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
            left_stem, right_stem = _split_combined_tokens(fpath)
            pl = extract_progression_from_name(left_stem)
            pr = extract_progression_from_name(right_stem)
            print(f"Evaluation: Incorrect. (expected={expected}, left={pl}, right={pr})")
            logging.info(f"Evaluation: Incorrect (expected={expected}, left={pl}, right={pr})")

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
            for s in (2, 3):
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
