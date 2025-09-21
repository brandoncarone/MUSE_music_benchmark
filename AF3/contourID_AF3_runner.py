# contourID_AF3_runner.py
# AF3 — Contour Identification (stateless; one-audio-per-call; trial-only prompts)

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
A_CANON = 'A. Arch (ascending and then descending)'
B_CANON = 'B. Inverted Arch (descending and then ascending)'
C_CANON = 'C. Ascending (pitch raises over time)'
D_CANON = 'D. Descending (pitch falls over time)'

# Robust patterns (prefer LAST occurrence anywhere)
A_PAT = re.compile(r'(?i)\bA\.\s*Arch\s*\(ascending\s+and\s+then\s+descending\)')
B_PAT = re.compile(r'(?i)\bB\.\s*Inverted\s+Arch\s*\(descending\s+and\s+then\s+ascending\)')
C_PAT = re.compile(r'(?i)\bC\.\s*Ascending\s*\(pitch\s+raises\s+over\s+time\)')
D_PAT = re.compile(r'(?i)\bD\.\s*Descending\s*\(pitch\s+falls\s+over\s+time\)')

# ===== Trial prompts (stateless; per-trial only) =====
TRIAL_PROMPT_SYS = """You are a participant in a psychological experiment on music perception. 
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to. 

Your task is to determine the overall contour—the overall pattern or "shape" of the notes you hear. An Arch shape means that first the notes ascend to a higher pitch, and then they descend to a lower pitch. Thus, an inverted arch is the opposite, whereby the notes first drop in pitch and then rise up again. Ascending simply means that the pitches you hear go from lower to higher, and descending means they go from higher to lower.
Valid responses are:
"A. Arch (ascending and then descending)"
"B. Inverted Arch (descending and then ascending)"
"C. Ascending (pitch raises over time)"
"D. Descending (pitch falls over time)"

Please provide no additional commentary beyond the short answers previously mentioned. 
"""

COT_PROMPT_TEMPLATE = """Analyze the music excerpt and identify which CONTOUR best describes the melody.

Step 1: Attend to the single melodic line (no accompaniment). Determine whether the pitch overall rises, falls, rises then falls (single apex), or falls then rises (single trough).

Step 2: Decide which category best fits the global trend:
A. Arch — rises to a peak then falls.
B. Inverted Arch — falls to a trough then rises.
C. Ascending — overall upward trend start to end.
D. Descending — overall downward trend start to end.

Step 3: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
A. Arch (ascending and then descending)
OR
B. Inverted Arch (descending and then ascending)
OR
C. Ascending (pitch raises over time)
OR
D. Descending (pitch falls over time)
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
    Example: contourID_AF3_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"contourID_AF3_CHAT_{mode}_{group}_seed{seed}.log"

# =============================
# Paths & parsing
# =============================
def _ppath(p: str) -> str:
    """Normalize: abs -> as-is; starts with 'stimuli/' -> as-is; else join with STIM_ROOT."""
    if os.path.isabs(p): return p
    if p.startswith(STIM_ROOT + os.sep) or p.startswith(STIM_ROOT + "/"):
        return p
    return os.path.join(STIM_ROOT, p)

def parse_final_choice(text: str) -> str:
    """Return canonical A/B/C/D string, or '' if not found. Prefer LAST occurrence."""
    if not text:
        return ""
    last = None
    for m in A_PAT.finditer(text): last = ("A", m.end())
    for m in B_PAT.finditer(text):
        if last is None or m.end() > last[1]: last = ("B", m.end())
    for m in C_PAT.finditer(text):
        if last is None or m.end() > last[1]: last = ("C", m.end())
    for m in D_PAT.finditer(text):
        if last is None or m.end() > last[1]: last = ("D", m.end())
    if not last:
        return ""
    return {"A": A_CANON, "B": B_CANON, "C": C_CANON, "D": D_CANON}[last[0]]

def contour_label_from_filename(path: str) -> str:
    """Infer expected label from filename (InvArch before Arch)."""
    name = os.path.basename(path).lower()
    if "invarch" in name:
        return B_CANON
    if "arch" in name:
        return A_CANON
    if "asc" in name:
        return C_CANON
    if "desc" in name:
        return D_CANON
    return "Unknown"

def stimuli_list_group(group: str) -> List[Dict[str, str]]:
    """Single audio per trial; mirror Gemini/Qwen sets."""
    assert group in {"GroupA", "GroupB"}
    if group == "GroupA":
        names = [
            "stimuli/CMaj_Scale_Arch.wav",
            "stimuli/Ebmaj_Scale_Arch_Piano.wav",
            "stimuli/BbMaj_Scale_Arch_Bass.wav",
            "stimuli/Bmaj_Scale_InvArch_Piano.wav",
            "stimuli/Dbmaj_Scale_InvArch_Bass.wav",
            "stimuli/Gm_Arp_Asc.wav",
            "stimuli/AMaj_Arp_Asc_Piano.wav",
            "stimuli/Abmin_Arp_Asc_Bass.wav",
            "stimuli/DMaj_Arp_Desc.wav",
            "stimuli/EMaj_Arp_Desc_Bass.wav",
        ]
    else:
        names = [
            "stimuli/Cmin_Scale_Arch.wav",
            "stimuli/Ebmin_Scale_Arch_Piano.wav",
            "stimuli/FMaj_Scale_InvArch.wav",
            "stimuli/Bmin_Scale_InvArch_Piano.wav",
            "stimuli/Dbmin_Scale_InvArch_Bass.wav",
            "stimuli/GMaj_Arp_Asc.wav",
            "stimuli/AbMaj_Arp_Asc_Bass.wav",
            "stimuli/Dm_Arp_Desc.wav",
            "stimuli/Gbmin_Arp_Desc_Piano.wav",
            "stimuli/Emin_Arp_Desc_Bass.wav",
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
    if not parse_final_choice(resp):
        fallback = (
            "Which contour best describes the shape of the melody? Answer with exactly ONE of the multiple choice responses:\n"
            "A. Arch (ascending and then descending)\n"
            "B. Inverted Arch (descending and then ascending)\n"
            "C. Ascending (pitch raises over time)\n"
            "D. Descending (pitch falls over time)"
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
    logging.info(f"=== RUN START (Contour ID • AF3-CHAT • STATELESS • {mode} • {group}) ===")

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

    print(f"\n--- Contour Identification — STATELESS • {mode} • {group} | model={model_name} | seed={seed} ---\n")
    logging.info(f"\n--- Task: Contour Identification — STATELESS • {mode} • {group} ---\n")

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
        # logging.info(f"[{mode}/{group}] Q{idx} - Trace:\n{text_trace}")

        parsed = parse_final_choice(final_resp)
        if not parsed:
            print("Evaluation: Failed. Could not parse a valid final choice.")
            logging.error("Parse Error: missing/malformed final choice line.")
            continue

        logging.info(f"Parsed Final Answer: {parsed}")

        expected = contour_label_from_filename(fpath)
        if expected == "Unknown":
            print("Evaluation: Unknown ground truth for this filename.")
            logging.warning("Evaluation: Unknown ground truth (filename lacks Arch/InvArch/Asc/Desc keyword).")
            # You can choose to skip scoring unknowns; here we skip incrementing correct.
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
