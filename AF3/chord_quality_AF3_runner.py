# chord_quality_AF3_runner.py
# AF3 — Chord Quality Identification (stateless; one-audio-per-call; trial-only prompts)

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

# Canonical answer strings and robust patterns
A_CANON = "A. Major"
B_CANON = "B. Minor"

A_PAT = re.compile(r'(?i)\bA\.\s*Major\b')
B_PAT = re.compile(r'(?i)\bB\.\s*Minor\b')

# ===== Trial prompts (stateless; per-trial only) =====
TRIAL_PROMPT_SYS = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to.

Your task is to decide if a chord is Major or Minor. First, you will hear the chord itself, and then you will hear the
individual notes of the chord played one at a time. You can think of the differences between Major and Minor in terms of
mood. For those with a Western enculturation:
- Major chords generally sound bright, happy, or triumphant.
- Minor chords often sound more somber, sad, or mysterious.

Valid responses are:
"A. Major"
"B. Minor"

Please provide no additional commentary beyond one of the two lines above.
"""

COT_PROMPT_TEMPLATE = """Analyze the music excerpt and identify the chord QUALITY (MAJOR vs MINOR).

Step 1: Listen to the sustained chord and the arpeggiated notes; focus on pitch classes (ignore inversion/voicing/register).

Step 2: Determine whether the set of chord tones contains:
- If there is a MAJOR third present, it is a Major chord
- If there is a MINOR third present, it is a Minor chord.

Step 3: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
A. Major
OR
B. Minor
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
    Example: chord_quality_AF3_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"chord_quality_AF3_CHAT_{mode}_{group}_seed{seed}.log"

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
    """Return A_CANON or B_CANON, or '' if not found. Prefer the LAST occurrence."""
    last_a = None
    last_b = None
    for m in A_PAT.finditer(text or ""):
        last_a = m
    for m in B_PAT.finditer(text or ""):
        last_b = m
    if last_a and last_b:
        return A_CANON if last_a.end() > last_b.end() else B_CANON
    if last_a:
        return A_CANON
    if last_b:
        return B_CANON
    return ""

def chord_quality_from_filename(path: str) -> str:
    """Heuristic ground truth from filename."""
    name = os.path.basename(path).lower()
    if "major" in name:
        return A_CANON
    if "minor" in name:
        return B_CANON
    return ""  # unknown

def stimuli_files_group(group: str) -> List[Dict[str, str]]:
    """Single audio per trial; mirror Gemini/Qwen sets."""
    assert group in {"GroupA", "GroupB"}
    if group == "GroupA":
        files = [
            "stimuli/Amajor_Guitar_120.wav",
            "stimuli/Gbmajor_Piano_120.wav",
            "stimuli/Bmajor_Guitar_120.wav",
            "stimuli/Dbmajor_Piano_120.wav",
            "stimuli/Cmajor_Guitar_120.wav",
            "stimuli/Abminor_Piano_120.wav",
            "stimuli/Dminor_Guitar_120.wav",
            "stimuli/Gbminor_Piano_120.wav",
            "stimuli/Cminor_Guitar_120.wav",
            "stimuli/Ebminor_Piano_120.wav",
        ]
    else:
        files = [
            "stimuli/Bbmajor_Piano_120.wav",
            "stimuli/Gmajor_Guitar_120.wav",
            "stimuli/Fmajor_Piano_120.wav",
            "stimuli/Dmajor_Guitar_120.wav",
            "stimuli/Ebmajor_Piano_120.wav",
            "stimuli/Gminor_Guitar_120.wav",
            "stimuli/Dbminor_Piano_120.wav",
            "stimuli/Bminor_Guitar_120.wav",
            "stimuli/Fminor_Piano_120.wav",
            "stimuli/Eminor_Guitar_120.wav",
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
            "Is this chord Major or Minor? Answer with exactly ONE line:\n"
            "A. Major\n"
            "OR\n"
            "B. Minor"
        )
        resp = af3_generate(model, [llava.Sound(trial_path), fallback])

    history_text = f"USER: {prompt.strip()}\nASSISTANT: {resp}"
    return resp, history_text

# =============================
# Runner
# =============================
def run_once(*, mode: str, group: str, seed: int, log_filename: str) -> None:
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}

    configure_logging(log_filename)
    logging.info(f"=== RUN START (Chord Quality • AF3-CHAT • STATELESS • {mode} • {group}) ===")

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

    question_stims = stimuli_files_group(group)
    random.shuffle(question_stims)

    print(f"\n--- Chord Quality Identification — STATELESS • {mode} • {group} | model={model_name} | seed={seed} ---\n")
    logging.info(f"\n--- Task: Chord Quality Identification — STATELESS • {mode} • {group} ---\n")

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

        expected = chord_quality_from_filename(fpath)
        if expected == "":
            print("Evaluation: Unknown ground truth for this filename.")
            logging.warning("Evaluation: Unknown ground truth (filename does not contain 'major'/'minor').")
        else:
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
