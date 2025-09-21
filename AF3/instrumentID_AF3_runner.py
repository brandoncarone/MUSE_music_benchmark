# instrumentID_AF3_runner.py
# AF3 — Instrument Identification (stateless; one-audio-per-call; trial-only prompts; <music-X> tags)

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

# Decoding params (unchanged per your request)
DECODE = dict(
    max_new_tokens=8192,
    temperature=1.0,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.0,
)

# Canonical labels + robust parser
A_CANON = 'A. Piano'
B_CANON = 'B. Guitar'
C_CANON = 'C. Bass'
D_CANON = 'D. Drums'

A_PAT = re.compile(r'(?i)\bA\.\s*Piano\b|\bpiano\b')
B_PAT = re.compile(r'(?i)\bB\.\s*Guitar\b|\bguitar\b')
C_PAT = re.compile(r'(?i)\bC\.\s*Bass\b|\bbass\b')
D_PAT = re.compile(r'(?i)\bD\.\s*Drums\b|\bdrum(?:s)?\b')

# ===== Trial prompts (exact, as provided) =====
TRIAL_PROMPT_SYS = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to. 

Your task is to listen to different music excerpts and identify the musical instrument being played.
Valid responses are:
"A. Piano"
"B. Guitar"
"C. Bass"
"D. Drums"

Please provide no additional commentary beyond the short answers previously mentioned.
"""

COT_PROMPT_TEMPLATE = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to.

Analyze the music excerpt and identify which INSTRUMENT is being played.

Definitions and constraints:
- Possible instruments (and response options) are limited to:
  A. Piano
  B. Guitar
  C. Bass
  D. Drums
- Focus on overall timbre, envelope, register, and articulation.
- Ignore reverb, effects, and recording quality.

Step 1: Focus on timbre, envelope (attack/decay/sustain), register, and articulation characteristics.

Step 2: Compare what you hear to the four categories:
A. Piano — percussive key strike with harmonic resonance/decay.
B. Guitar — plucked/strummed strings; fretted articulation; mid-to-high register presence.
C. Bass — low-register plucked string; strong fundamental; longer sustain in low range.
D. Drums — percussive hits without sustained pitched notes; kit elements (kick/snare/hi-hat/cymbals).

Step 3: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
A. Piano
OR
B. Guitar
OR
C. Bass
OR
D. Drums
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
    return f"instrumentID_AF3_CHAT_{mode}_{group}_seed{seed}.log"

# =============================
# Paths & parsing
# =============================
def _ppath(p: str) -> str:
    if os.path.isabs(p): return p
    if p.startswith(STIM_ROOT + os.sep) or p.startswith(STIM_ROOT + "/"):
        return p
    return os.path.join(STIM_ROOT, p)

def parse_final_choice(text: str) -> str:
    if not text: return ""
    hits = []
    for lab, pat in (('A', A_PAT), ('B', B_PAT), ('C', C_PAT), ('D', D_PAT)):
        for m in pat.finditer(text):
            hits.append((m.end(), lab))
    if not hits: return ""
    hits.sort(key=lambda x: x[0])
    last_lab = hits[-1][1]
    return {"A": A_CANON, "B": B_CANON, "C": C_CANON, "D": D_CANON}[last_lab]

def instrument_label_from_filename(path: str) -> str:
    name = os.path.basename(path).lower()
    if "beat" in name:   return D_CANON
    if "piano" in name:  return A_CANON
    if "bass" in name:   return C_CANON
    return B_CANON  # default Guitar

def stimuli_list_group(group: str) -> List[Dict[str, str]]:
    assert group in {"GroupA", "GroupB"}
    if group == "GroupA":
        names = [
            "stimuli/Abmajor_Piano_120.wav",
            "stimuli/Bmaj_Scale_InvArch_Piano.wav",
            "stimuli/Ebmin_Scale_Arch_Piano.wav",
            "stimuli/DMaj_Arp_Desc.wav",
            "stimuli/Eminor_Guitar_120.wav",
            "stimuli/Abmin_Arp_Asc_Bass.wav",
            "stimuli/Dbmin_Scale_InvArch_Bass.wav",
            "stimuli/Emin_Arp_Desc_Bass.wav",
            "stimuli/Beat_2_140.wav",
            "stimuli/Beat_4_140_34.wav",
        ]
    else:
        names = [
            "stimuli/Bbminor_Piano_120.wav",
            "stimuli/Gbmin_Arp_Desc_Piano.wav",
            "stimuli/Bmajor_Guitar_120.wav",
            "stimuli/FMaj_Scale_InvArch.wav",
            "stimuli/GMaj_Arp_Asc.wav",
            "stimuli/BbMaj_Scale_Arch_Bass.wav",
            "stimuli/EMaj_Arp_Desc_Bass.wav",
            "stimuli/Beat_1_140.wav",
            "stimuli/Beat_3_140.wav",
            "stimuli/Beat_5_140.wav",
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

# ---- CACHE MUST BE DEFINED BEFORE get_af3() ----
_AF3_CACHE: Dict[str, object] = {"model": None, "llava": None}

def _load_af3_model():
    llava = _import_llava()
    model_name = os.environ.get("AF3_MODEL", "nvidia/audio-flamingo-3-chat")
    local_dir = os.environ.get("AF3_LOCAL_DIR", "").strip() or None
    try:
        model = llava.load(local_dir if local_dir else model_name, model_base=None, devices=[0])
    except Exception as e:
        raise AF3NotFound(f"Failed to load AF3 model: {e}")
    # Set your decoding params
    try:
        g = model.default_generation_config
        g.max_new_tokens = DECODE["max_new_tokens"]
        g.temperature = DECODE["temperature"]
        g.top_p = DECODE["top_p"]
        g.top_k = DECODE["top_k"]
        g.repetition_penalty = DECODE["repetition_penalty"]
        # Add gentle stop sequences to cap a single assistant turn (safe no-op if unsupported)
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
# Trial helper (single call, trial-only prompt, anchored)
# =============================
def run_trial(model, llava, *, mode: str, trial_path: str, logger: logging.Logger) -> Tuple[str, str]:
    assert mode in {"SYSINST", "COT"}
    prompt = TRIAL_PROMPT_SYS if mode == "SYSINST" else COT_PROMPT_TEMPLATE

    # One-audio-per-call; audio FIRST, then the EXACT trial prompt text
    resp = af3_generate(model, [llava.Sound(trial_path), prompt.strip()])

    # Optional: one tight fallback if parsing failed (same decoding params)
    if not parse_final_choice(resp):
        fallback = "Which instrument is this? Answer with exactly one of:\nA. Piano\nB. Guitar\nC. Bass\nD. Drums"
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
    logging.info(f"=== RUN START (Instrument ID • AF3-CHAT • STATELESS • {mode} • {group}) ===")

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

    print(f"\n--- Instrument Identification — STATELESS • {mode} • {group} | model={model_name} | seed={seed} ---\n")
    logging.info(f"\n--- Task: Instrument Identification — STATELESS • {mode} • {group} ---\n")

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
        expected = instrument_label_from_filename(fpath)
        if not parsed:
            print("Evaluation: Failed. Could not parse a valid final choice.")
            logging.error("Parse Error: missing/malformed final choice line.")
        else:
            logging.info(f"Parsed Final Answer: {parsed}")
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
