# key_modulation_AF3_runner.py
# AF3 — Key Modulation Detection (stateless; one-audio-per-call; trial-only prompts)

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

# Decoding params (align with other AF3 runners)
DECODE = dict(
    max_new_tokens=8192,
    temperature=1.0,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.0,
)

# Canonical answer strings
A_CANON = "A. Yes, a key modulation occurs."
B_CANON = "B. No, the key stays the same."

# Robust patterns (accept exact lines, with/without trailing period; case-insensitive; per-line anchors)
A_PAT = re.compile(r'(?im)^\s*A\.\s*Yes,\s*a\s*key\s*modulation\s*occurs\.?\s*$', re.UNICODE)
B_PAT = re.compile(r'(?im)^\s*B\.\s*No,\s*the\s*key\s*stays\s*the\s*same\.?\s*$', re.UNICODE)
# Also accept natural-language variants for "No" without the "B." prefix
B_ALT = re.compile(r'(?im)^\s*(?:B\.\s*)?No,?\s*(?:a\s*)?key\s*modulation\s*(?:does\s*not|doesn\'t)\s*occur\.?\s*$', re.UNICODE)

# ===== Trial prompts (stateless; per-trial only) =====
TRIAL_PROMPT_SYS = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to.

Your task is to decide whether a "key change" (or modulation) occurs in the excerpt. Think of a song's key as its 
"home base." A modulation is a dramatic shift to a new home base, which can feel like a "lift" or change in the song's 
“home base.”

Valid responses are:
"A. Yes, a key modulation occurs."
"B. No, the key stays the same."

Please provide no additional commentary beyond one of the two lines above.
"""

COT_PROMPT_TEMPLATE = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to.

Analyze the music excerpt and decide whether a KEY MODULATION occurs.

Step 1: Identify the initial tonal center (key) from the opening harmony/melody.

Step 2: Scan the excerpt for a sustained shift to a NEW tonal center:
- Listen for cadential motion into a new key and a stable leading-tone/scale pattern supporting that key.
- Distinguish between a brief tonicization/borrowed chord (no modulation) and an established new key (modulation).

Step 3: Decision rule:
- If the music establishes and maintains a new tonal center, then a modulation occurs.
- If the music remains in the original key (only brief tonicizations/borrowed chords), no modulation occurs.

Step 4: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
A. Yes, a key modulation occurs.
OR
B. No, the key stays the same.
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
    Chat + per-trial system instructions (stateless), two modes:
      - SYSINST (plain)
      - COT     (reasoning)
    Two stimulus groups: GroupA / GroupB
    Example: key_modulation_AF3_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"key_modulation_AF3_CHAT_{mode}_{group}_seed{seed}.log"

# =============================
# Paths & parsing
# =============================
def _ppath(p: str) -> str:
    """Normalize: abs -> as-is; starts with 'stimuli/' -> as-is; else join with STIM_ROOT."""
    if os.path.isabs(p): return p
    if p.startswith(STIM_ROOT + os.sep) or p.startswith(STIM_ROOT + "/"):
        return p
    return os.path.join(STIM_ROOT, p)

def parse_final_decision(text: str) -> str:
    """Return A_CANON or B_CANON, or '' if not found. Prefer the LAST occurrence (across lines)."""
    last_a = last_b = last_b_alt = None
    for m in A_PAT.finditer(text or ""): last_a = m
    for m in B_PAT.finditer(text or ""): last_b = m
    for m in B_ALT.finditer(text or ""): last_b_alt = m
    candidates = []
    if last_a: candidates.append((last_a.end(), "A"))
    if last_b: candidates.append((last_b.end(), "B"))
    if last_b_alt: candidates.append((last_b_alt.end(), "B"))
    if not candidates: return ""
    _, label = max(candidates, key=lambda x: x[0])
    return A_CANON if label == "A" else B_CANON

def modulation_from_filename(path: str) -> str:
    """
    Ground truth from filename:
    - if 'mod' or 'Mod' present → A. Yes, a key modulation occurs.
    - else → B. No, the key stays the same.
    """
    name = os.path.basename(path)
    if ("mod" in name) or ("Mod" in name):
        return A_CANON
    return B_CANON

def stimuli_files_group(group: str) -> List[Dict[str, str]]:
    """Single audio per trial; mirror the Gemini/Qwen sets you provided."""
    assert group in {"GroupA", "GroupB"}
    if group == "GroupA":
        files = [
            "stimuli/Intermediate/I-vi-ii-V_Bmaj_mod_Csmin_CleanWahGuitar_120.wav",
            "stimuli/Intermediate/vi-IV-I-V_Fsmaj_mod_Dsmin_AcousticGuit_relativeminor_118.wav",
            "stimuli/Intermediate/I-V-vi-IV_Dbmaj_mod_Abmaj_commonchord_CrunchGuit_100.wav",
            "stimuli/Intermediate/I-vi-VI-V_Amaj_mod_Fsmin_piano_160_3_4.wav",
            "stimuli/Intermediate/vi-IV-I-V_Fsmaj_mod_Csmaj_piano_135.wav",
            "stimuli/Intermediate/I-vi-ii-V_Cmaj_CleanGuitar_80.wav",
            "stimuli/Intermediate/vi-IV-I-V_Gmaj_CrunchGuit_150.wav",
            "stimuli/Intermediate/I-vi-VI-V_Fmaj_piano_172_3_4.wav",
            "stimuli/Intermediate/I-V-vi-IV_Dmaj_piano_145.wav",
            "stimuli/Intermediate/I-IV-V_Emaj_piano_150.wav",
        ]
    else:
        files = [
            "stimuli/Intermediate/I-IV-V_Abmaj_Mod_Fmin_pivot_relative_minor_175.wav",
            "stimuli/Intermediate/I-vi_IV-V_Amaj_mod_Bbmaj_CrunchEffectsGuit_140.wav",
            "stimuli/Intermediate/I-IV-V_Abmaj_Mod_Bbmaj_piano_132.wav",
            "stimuli/Intermediate/I-V-vi-IV_Dbmaj_mod_Bbmin_piano_115.wav",
            "stimuli/Intermediate/I-vi-ii-V_Bmaj_mod_Fsmaj_128.wav",
            "stimuli/Intermediate/I-vi-IV-V_Fmaj_CrunchGuit_140.wav",
            "stimuli/Intermediate/I-V-vi-IV_Dmaj_AcousticGuit_115.wav",
            "stimuli/Intermediate/I-IV-V_Emaj_CleanGuit_132.wav",
            "stimuli/Intermediate/I-vi-ii-V_Cmaj_piano_125.wav",
            "stimuli/Intermediate/vi-IV-I-V_Gmaj_piano_165.wav",
        ]
    return [{"file": _ppath(p)} for p in files]

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
    # Apply decoding knobs
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
# Trial helper (single call; one audio; trial-only prompt)
# =============================
def run_trial(model, llava, *, mode: str, trial_path: str, logger: logging.Logger) -> Tuple[str, str]:
    assert mode in {"SYSINST", "COT"}
    prompt = TRIAL_PROMPT_SYS if mode == "SYSINST" else COT_PROMPT_TEMPLATE

    # One-audio-per-call; audio FIRST, then the EXACT trial prompt text
    resp = af3_generate(model, [llava.Sound(trial_path), prompt.strip()])

    # Tight fallback if parsing failed (same decoding params)
    if not parse_final_decision(resp):
        fallback = (
            "Does a key modulation occur in this excerpt? Answer with exactly ONE line:\n"
            "A. Yes, a key modulation occurs.\n"
            "OR\n"
            "B. No, the key stays the same."
        )
        resp = af3_generate(model, [llava.Sound(trial_path), fallback])

    history_text = f"USER: {prompt.strip()}\nASSISTANT: {resp}"
    return resp, history_text

# =============================
# Runner
# =============================
def configure_cuda_for_repro(seed: int):
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def run_once(*, mode: str, group: str, seed: int, log_filename: str) -> None:
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}

    configure_logging(log_filename)
    logging.info(f"=== RUN START (Key Modulation • AF3-CHAT • STATELESS • {mode} • {group}) ===")

    random.seed(seed)
    configure_cuda_for_repro(seed)

    model, llava = get_af3()
    model_name = os.environ.get("AF3_MODEL", "nvidia/audio-flamingo-3-chat")
    logging.info(f"Config: model={model_name}, temp={DECODE['temperature']}, top_p={DECODE['top_p']}, "
                 f"top_k={DECODE['top_k']}, seed={seed}, group={group}, log={log_filename}")

    question_stims = stimuli_files_group(group)
    random.shuffle(question_stims)

    print(f"\n--- Key Modulation Detection — STATELESS • {mode} • {group} | model={model_name} | seed={seed} ---\n")
    logging.info(f"\n--- Task: Key Modulation Detection — STATELESS • {mode} • {group} ---\n")

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

        final_resp, _ = run_trial(model, llava, mode=mode, trial_path=fpath, logger=logging.getLogger())

        print(f"\n--- Q{idx}: {os.path.basename(fpath)} ---")
        print("LLM Full Response:\n", final_resp)
        logging.info(f"[{mode}/{group}] Q{idx} - LLM Full Response:\n{final_resp}")

        parsed = parse_final_decision(final_resp)
        if not parsed:
            print("Evaluation: Failed. Could not parse the final answer phrase.")
            logging.error("Parse Error: missing/malformed final answer phrase.")
            continue

        logging.info(f"Parsed Final Answer: {parsed}")

        expected = modulation_from_filename(fpath)
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
    #for mode in ["SYSINST", "COT"]:
    for mode in ["COT"]:
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
