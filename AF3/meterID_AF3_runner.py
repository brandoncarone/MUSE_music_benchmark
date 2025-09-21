# meter_identification_AF3_runner.py
# AF3 — Meter Identification (stateless; one-audio-per-call; trial-only prompts)

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

# Canonical answer strings (EXACT)
A_CANON = "A. Groups of 3"
B_CANON = "B. Groups of 4"
C_CANON = "C. Groups of 5"

# Robust patterns (case-insensitive)
A_PAT = re.compile(r"(?i)\bA\.\s*Groups\s+of\s+3\b")
B_PAT = re.compile(r"(?i)\bB\.\s*Groups\s+of\s+4\b")
C_PAT = re.compile(r"(?i)\bC\.\s*Groups\s+of\s+5\b")

# ===== Trial prompts (stateless; per-trial only) =====
TRIAL_PROMPT_SYS = """You are a participant in a psychological experiment on music perception.

Your task is to identify the meter of a musical excerpt, or how you would count it in repeating groups. Almost all music 
has a basic, repeating pulse. Meter is how we group those pulses.
Counting in 4s (ONE-two-three-four, ONE-two-three-four) is the most common in pop and rock music.
Counting in 3s (ONE-two-three, ONE-two-three) is the feel of a waltz.
Counting in 5s is less common and feels like a longer, more unusual cycle (ONE-two-three-four-five, ONE-two-three-four-five).
Try to feel where the strongest pulse is and how many beats pass before it repeats.

Valid responses are:
"A. Groups of 3"
"B. Groups of 4"
"C. Groups of 5"

Reply with exactly ONE of the three lines above (and nothing else).
"""

COT_PROMPT_TEMPLATE = """You are a participant in a psychological experiment on music perception.

Your task is to identify the METER — how the steady pulse is grouped into repeating cycles.

Step 1: Find the steady pulse and count along to establish the beat.

Step 2: Locate the strongest recurring downbeat/accent and count how many beats occur before that accent pattern repeats.
- Choose the smallest repeating cycle that explains the accents (3, 4, or 5).

Step 3: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
A. Groups of 3
OR
B. Groups of 4
OR
C. Groups of 5
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
    Example: meter_identification_AF3_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"meter_identification_AF3_CHAT_{mode}_{group}_seed{seed}.log"

# =============================
# Paths & stimuli
# =============================
def _ppath(p: str) -> str:
    """Normalize: abs -> as-is; starts with 'stimuli/' -> as-is; else join with STIM_ROOT."""
    if os.path.isabs(p): return p
    if p.startswith(STIM_ROOT + os.sep) or p.startswith(STIM_ROOT + "/"):
        return p
    return os.path.join(STIM_ROOT, p)

def stimuli_list_group(group: str) -> List[Dict[str, str]]:
    assert group in {"GroupA", "GroupB"}
    if group == "GroupA":
        files = [
            "stimuli/Intermediate/Circles_3.wav",
            "stimuli/Intermediate/Piano_3.wav",
            "stimuli/Intermediate/I-vi-VI-V_Fmaj_piano_172_3_4.wav",
            "stimuli/Intermediate/vi-IV-I-V_Gmaj_AcousticGuit_118.wav",
            "stimuli/Intermediate/Rosewood_4.wav",
            "stimuli/Intermediate/SunKing_4.wav",
            "stimuli/Intermediate/opbeat_4.wav",
            "stimuli/Intermediate/off_4.wav",
            "stimuli/Intermediate/Five_solo_5.wav",
            "stimuli/Intermediate/GII_5.wav",
        ]
    else:
        files = [
            "stimuli/Intermediate/50s_3.wav",
            "stimuli/Intermediate/Circles_solo_3.wav",
            "stimuli/Intermediate/Scene_3.wav",
            "stimuli/Intermediate/I-vi-VI-V_Fmaj_piano_172_3_4.wav",
            "stimuli/Intermediate/ComeOn_4.wav",
            "stimuli/Intermediate/DoDoDoDoDo_4.wav",
            "stimuli/Intermediate/Flow_4.wav",
            "stimuli/Intermediate/Harm_4.wav",
            "stimuli/Intermediate/Dance_5.wav",
            "stimuli/Intermediate/Falling_5.wav",
        ]
    return [{"file": _ppath(p)} for p in files]

# Gold labels (by basename)
METER_GOLD: Dict[str, str] = {
    "Circles_3.wav":                         A_CANON,
    "Piano_3.wav":                           A_CANON,
    "I-vi-VI-V_Fmaj_piano_172_3_4.wav":      A_CANON,
    "vi-IV-I-V_Gmaj_AcousticGuit_118.wav":   A_CANON,
    "Rosewood_4.wav":                        B_CANON,
    "SunKing_4.wav":                         B_CANON,
    "opbeat_4.wav":                          B_CANON,
    "off_4.wav":                             B_CANON,
    "Five_solo_5.wav":                       C_CANON,
    "GII_5.wav":                             C_CANON,
    "50s_3.wav":                             A_CANON,
    "Circles_solo_3.wav":                    A_CANON,
    "Scene_3.wav":                           A_CANON,
    "ComeOn_4.wav":                          B_CANON,
    "DoDoDoDoDo_4.wav":                      B_CANON,
    "Flow_4.wav":                            B_CANON,
    "Harm_4.wav":                            B_CANON,
    "Dance_5.wav":                           C_CANON,
    "Falling_5.wav":                         C_CANON,
}

def expected_for_meter(path: str) -> str:
    return METER_GOLD.get(os.path.basename(path), "")

# =============================
# Parsing
# =============================
def parse_final_decision(text: str) -> str:
    """Return A_CANON / B_CANON / C_CANON, or '' if none; prefer LAST occurrence."""
    last_a = last_b = last_c = None
    for m in A_PAT.finditer(text or ""): last_a = m
    for m in B_PAT.finditer(text or ""): last_b = m
    for m in C_PAT.finditer(text or ""): last_c = m

    best = None
    for m, canon in [(last_a, A_CANON), (last_b, B_CANON), (last_c, C_CANON)]:
        if m and (best is None or m.end() > best[0].end()):
            best = (m, canon)
    return best[1] if best else ""

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
def run_trial(model, llava, *, mode: str, trial_path: str) -> str:
    assert mode in {"SYSINST", "COT"}
    prompt = TRIAL_PROMPT_SYS if mode == "SYSINST" else COT_PROMPT_TEMPLATE

    # One-audio-per-call; audio FIRST, then the EXACT trial prompt text
    resp = af3_generate(model, [llava.Sound(trial_path), prompt.strip()])

    # Tight fallback if parsing failed
    if not parse_final_decision(resp):
        fallback = (
            "Identify the meter. Answer with exactly ONE line:\n"
            "A. Groups of 3\n"
            "OR\n"
            "B. Groups of 4\n"
            "OR\n"
            "C. Groups of 5"
        )
        resp = af3_generate(model, [llava.Sound(trial_path), fallback])
    return resp

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
    logging.info(f"=== RUN START (Meter Identification • AF3-CHAT • STATELESS • {mode} • {group}) ===")

    random.seed(seed)
    configure_cuda_for_repro(seed)

    model, llava = get_af3()
    model_name = os.environ.get("AF3_MODEL", "nvidia/audio-flamingo-3-chat")
    logging.info(f"Config: model={model_name}, temp={DECODE['temperature']}, top_p={DECODE['top_p']}, "
                 f"top_k={DECODE['top_k']}, seed={seed}, group={group}, log={log_filename}")

    question_stims = stimuli_list_group(group)
    random.shuffle(question_stims)

    print(f"\n--- Meter Identification — STATELESS • {mode} • {group} | model={model_name} | seed={seed} ---\n")
    logging.info(f"\n--- Task: Meter Identification — STATELESS • {mode} • {group} ---\n")

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

        expected = expected_for_meter(fpath)
        if not expected:
            print("Evaluation: Unknown stimulus label for expected answer.")
            logging.error("Missing gold label for stimulus basename.")
            continue

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
