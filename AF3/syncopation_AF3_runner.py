# syncopation_detection_AF3_runner.py
# AF3 — Syncopation Detection (stateless; one-audio-per-call; combined "Excerpt1__Excerpt2" files)

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
A_CANON = "A. The rhythm in Excerpt 1 is more syncopated."
B_CANON = "B. The rhythm in Excerpt 2 is more syncopated."

# Robust patterns (case-insensitive; allow optional leading "A."/"B.")
A_PAT = re.compile(r'(?i)\b(?:a\.\s*)?the\s+rhythm\s+in\s+excerpt\s*1\s+is\s+more\s+syncopated\.')
B_PAT = re.compile(r'(?i)\b(?:b\.\s*)?the\s+rhythm\s+in\s+excerpt\s*2\s+is\s+more\s+syncopated\.')

# ===== Trial prompts (stateless; per-trial only) =====
TRIAL_PROMPT_SYS = """You are a participant in a psychological experiment on music perception.
In each question, you will be given a single audio file that contains two music excerpts back-to-back with a voice saying "Here is the first excerpt" before the first excerpt is presented. After the first excerpt plays, the same voice then says "Here is the second excerpt" before the second excerpt is presented.

Your task is to decide which excerpt is MORE SYNCOPATED (more off-beat kick/snare accents, displacements, ties across strong beats).
Think of syncopation as rhythmic surprise: a simple rhythm is steady and predictable (like a metronome: ONE-two-three-four), while a syncopated rhythm emphasizes the "off-beats" — the unexpected moments in between the main pulse — making it feel more complex or groovy.

Reply with exactly ONE of the two lines (and nothing else):
"A. The rhythm in Excerpt 1 is more syncopated."
OR
"B. The rhythm in Excerpt 2 is more syncopated.
"""

COT_PROMPT_TEMPLATE = """You are a participant in a psychological experiment on music perception.
In each question, you will be given a single audio file that contains two music excerpts back-to-back with a voice saying "Here is the first excerpt" before the first excerpt is presented. After the first excerpt plays, the same voice then says "Here is the second excerpt" before the second excerpt is presented.

Your task is to decide which drum set rhythm is MORE SYNCOPATED.


Step 1: Establish the pulse and smallest repeating cycle for each excerpt. 

Step 2: For each excerpt, note kick/snare placements relative to the beat grid: 
- Count/describe off-beat accents and displaced hits. 
- Note any strong-beat omissions with off-beat substitutions. 
- Treat hi-hat texture as neutral; focus on kick/snare. 

Step 3: Compare which excerpt exhibits more off-beat emphasis, displaced accents, or ties across beats — that excerpt is 
more syncopated. 

Step 4: Final Answer After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line): 
A. The rhythm in Excerpt 1 is more syncopated. 
OR 
B. The rhythm in Excerpt 2 is more syncopated.
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
    Example: syncopation_AF3_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"syncopation_AF3_CHAT_{mode}_{group}_seed{seed}.log"

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
    """
    AF3 takes one audio per call, so we use pre-combined files of the form:
      '<orig1>.wav__<orig2>.wav'
    where the audio contains Excerpt 1 followed by Excerpt 2.
    """
    assert group in {"GroupA", "GroupB"}
    if group == "GroupA":
        files = [
            "stimuli/combined/Sync1_A__NoSync_E.wav",
            "stimuli/combined/Sync2_A__NoSync_B.wav",
            "stimuli/combined/Sync2_B__Sync1_B.wav",
            "stimuli/combined/Sync3_E__Sync1_A.wav",
            "stimuli/combined/Sync3_B__Sync2_A.wav",
            "stimuli/combined/Sync2_B__Sync4_A.wav",
            "stimuli/combined/Sync3_E__Sync4_B.wav",
            "stimuli/combined/NoSync_B__Sync1_B.wav",
            "stimuli/combined/Sync1_A__Sync2_A.wav",
            "stimuli/combined/Sync3_B__Sync4_A.wav",
        ]
    else:
        files = [
            "stimuli/combined/Sync1_C__NoSync_C.wav",
            "stimuli/combined/Sync2_C__NoSync_D.wav",
            "stimuli/combined/Sync2_D__Sync1_D.wav",
            "stimuli/combined/Sync3_C__Sync1_C.wav",
            "stimuli/combined/Sync3_D__Sync2_C.wav",
            "stimuli/combined/Sync2_D__Sync4_C.wav",
            "stimuli/combined/Sync3_C__Sync4_D.wav",
            "stimuli/combined/NoSync_D__Sync1_D.wav",
            "stimuli/combined/Sync1_C__Sync2_C.wav",
            "stimuli/combined/Sync3_D__Sync4_C.wav",
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

def _sync_rank_from_name(name: str) -> int:
    """
    Rank syncopation by name:
      NoSync -> 0
      Sync1  -> 1
      Sync2  -> 2
      Sync3  -> 3
      Sync4  -> 4
    """
    s = name.lower()
    if "nosync" in s:
        return 0
    m = re.search(r"sync(\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return 0

def expected_for_combined(path: str) -> str:
    """
    Determine expected label from combined filename "<left>__<right>".
    Higher sync rank is more syncopated; any Sync# > NoSync.
    """
    base = os.path.basename(path)
    if "__" not in base:
        return ""
    left, right = base.split("__", 1)
    r1 = _sync_rank_from_name(left)
    r2 = _sync_rank_from_name(right)
    if r1 == r2:
        return ""  # tie/ambiguous; skip scoring
    return A_CANON if r1 > r2 else B_CANON

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
# Trial helper (single call; one combined audio; trial-only prompt)
# =============================
def run_trial(model, llava, *, mode: str, trial_path: str) -> str:
    assert mode in {"SYSINST", "COT"}
    prompt = TRIAL_PROMPT_SYS if mode == "SYSINST" else COT_PROMPT_TEMPLATE

    # One-audio-per-call; combined file contains Excerpt 1 then Excerpt 2
    resp = af3_generate(model, [llava.Sound(trial_path), prompt.strip()])

    # Tight fallback if parsing failed
    if not parse_final_decision(resp):
        fallback = (
            "This audio contains Excerpt 1 followed by Excerpt 2. "
            "Answer with exactly ONE line:\n"
            "A. The rhythm in Excerpt 1 is more syncopated.\n"
            "OR\n"
            "B. The rhythm in Excerpt 2 is more syncopated."
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
    logging.info(f"=== RUN START (Syncopation Detection • AF3-CHAT • STATELESS • {mode} • {group}) ===")

    random.seed(seed)
    configure_cuda_for_repro(seed)

    model, llava = get_af3()
    model_name = os.environ.get("AF3_MODEL", "nvidia/audio-flamingo-3-chat")
    logging.info(f"Config: model={model_name}, temp={DECODE['temperature']}, top_p={DECODE['top_p']}, "
                 f"top_k={DECODE['top_k']}, seed={seed}, group={group}, log={log_filename}")

    question_stims = stimuli_list_group(group)
    random.shuffle(question_stims)

    print(f"\n--- Syncopation Detection — STATELESS • {mode} • {group} | model={model_name} | seed={seed} ---\n")
    logging.info(f"\n--- Task: Syncopation Detection — STATELESS • {mode} • {group} ---\n")

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
        if not expected:
            print("Evaluation: Skipped (tie or unparseable filename).")
            logging.warning("Expected label unavailable (tie or bad name).")
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
