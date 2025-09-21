# syncopation_detection_Gemini_CHAT_SysInst_master.py
import os
import re
import gc
import random
import logging
import warnings
from typing import List, Dict, Any, Tuple

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# --- Google Gemini SDK (use only for calling & feeding text/audio) ---
from google import genai
from google.genai import types

# =============================
# Constants / paths
# =============================
STIM_ROOT = "/stimuli"
MAX_NEW_TOKENS = 8192

# Canonical answer strings and robust patterns
A_CANON = "A. The rhythm in Excerpt 1 is more syncopated."
B_CANON = "B. The rhythm in Excerpt 2 is more syncopated."

A_PAT = re.compile(r'(?i)\b(?:a\.\s*)?the\s+rhythm\s+in\s+excerpt\s*1\s+is\s+more\s+syncopated\.')
B_PAT = re.compile(r'(?i)\b(?:b\.\s*)?the\s+rhythm\s+in\s+excerpt\s*2\s+is\s+more\s+syncopated\.')

# =============================
# System instructions (VERBATIM)
# =============================
SYSINSTR_PLAIN = """You are a participant in a psychological experiment on music perception. 
In each question, you will be given:
    1. A brief instruction about the specific listening task.
    2. Two audio examples to listen to.
Syncopation Detection: Your task is to listen to two drum set rhythms and decide which is more syncopated. 
Think of syncopation as rhythmic surprise: a simple rhythm is steady and predictable (like a metronome: ONE-two-three-four), 
while a syncopated rhythm emphasizes the "off-beats" — the unexpected moments in between the main pulse — making it feel more complex or groovy.

Valid responses are:
"A. The rhythm in Excerpt 1 is more syncopated." or 
"B. The rhythm in Excerpt 2 is more syncopated."

Before you begin the task, I will provide you with examples of an excerpt that is not syncopated at all, 
as well as an excerpt that is highly syncopated so that you better understand the task. 
After examining the examples, please respond with "Yes, I understand." if you understand the task or 
"No, I don't understand." if you don't understand the task.

Please provide no additional commentary beyond the short answers previously mentioned.
"""

SYSINSTR_COT = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. Two audio examples to listen to.

Your task is to decide which drum set rhythm is MORE SYNCOPATED.

Definitions and constraints:
- Think of syncopation as emphasis on OFF-BEATS or unexpected placements relative to the main pulse.
- A rhythm is “more syncopated” when kick/snare accents more often land between the main beats, displace or tie across strong beats, or omit strong-beat hits in favor of off-beat hits.
- Focus primarily on kick and snare placement; treat hi-hat ostinatos as neutral texture.

Valid responses are exactly:
"A. The rhythm in Excerpt 1 is more syncopated."
"B. The rhythm in Excerpt 2 is more syncopated."

Before you begin the task, I will provide you with two examples so you better understand the task. After examining the examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't understand the task.

After your step-by-step reasoning, end with exactly one line:
A. The rhythm in Excerpt 1 is more syncopated.
OR
B. The rhythm in Excerpt 2 is more syncopated."""

# --- COT per-trial prompt (VERBATIM) ---
COT_PROMPT_TEMPLATE = """Analyze the two drum set excerpts and decide which is MORE SYNCOPATED.

Step 1: Establish the pulse and smallest repeating cycle for each excerpt.

Step 2: For each excerpt, note kick/snare placements relative to the beat grid:
- Count/describe off-beat accents and displaced hits.
- Note any strong-beat omissions with off-beat substitutions.
- Treat hi-hat texture as neutral; focus on kick/snare.

Step 3: Compare which excerpt exhibits more off-beat emphasis, displaced accents, or ties across beats — that excerpt is more syncopated.

Step 4: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
A. The rhythm in Excerpt 1 is more syncopated.
OR
B. The rhythm in Excerpt 2 is more syncopated.
"""

# =============================
# Logging utilities
# =============================
def configure_logging(log_filename: str):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        try:
            h.close()
        finally:
            root.removeHandler(h)
    fh = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    root.addHandler(fh)

def _model_tag_for(model_name: str) -> str:
    name = model_name.lower()
    if "2.5" in name and "pro" in name:
        return "G25Pro"
    if "2.5" in name and "flash" in name:
        return "G25Flash"
    return "Gemini"

def make_log_filename(*, model_name: str, mode: str, group: str, seed: int) -> str:
    """
    Chat + System Instructions runner, two modes:
      - SYSINST (plain)
      - COT     (reasoning)
    Two stimulus groups: GroupA / GroupB
    Example: syncopation_G25Pro_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    tag = _model_tag_for(model_name)
    return f"syncopation_{tag}_CHAT_{mode}_{group}_seed{seed}.log"

# =============================
# Data helpers
# =============================
def _ppath(p: str) -> str:
    """Normalize a given path to absolute under STIM_ROOT."""
    if os.path.isabs(p):
        return p
    if p.startswith("stimuli/") or p.startswith("stimuli" + os.sep):
        rel = p.split("/", 1)[1] if "/" in p else p.split(os.sep, 1)[1]
        return os.path.join(STIM_ROOT, rel)
    return os.path.join(STIM_ROOT, p)

def _p(name: str) -> str:
    return os.path.join(STIM_ROOT, name)

def stimuli_pairs_group(group: str) -> List[Dict[str, str]]:
    assert group in {"GroupA", "GroupB"}
    if group == "GroupA":
        pairs = [
            ("stimuli/Intermediate/Sync1_A.wav", "stimuli/Intermediate/NoSync_E.wav"),
            ("stimuli/Intermediate/Sync2_A.wav", "stimuli/Intermediate/NoSync_B.wav"),
            ("stimuli/Intermediate/Sync2_B.wav", "stimuli/Intermediate/Sync1_B.wav"),
            ("stimuli/Intermediate/Sync3_E.wav", "stimuli/Intermediate/Sync1_A.wav"),
            ("stimuli/Intermediate/Sync3_B.wav", "stimuli/Intermediate/Sync2_A.wav"),
            ("stimuli/Intermediate/Sync2_B.wav", "stimuli/Intermediate/Sync4_A.wav"),
            ("stimuli/Intermediate/Sync3_E.wav", "stimuli/Intermediate/Sync4_B.wav"),
            ("stimuli/Intermediate/NoSync_B.wav", "stimuli/Intermediate/Sync1_B.wav"),
            ("stimuli/Intermediate/Sync1_A.wav", "stimuli/Intermediate/Sync2_A.wav"),
            ("stimuli/Intermediate/Sync3_B.wav", "stimuli/Intermediate/Sync4_A.wav"),
        ]
    else:
        pairs = [
            ("stimuli/Intermediate/Sync1_C.wav", "stimuli/Intermediate/NoSync_C.wav"),
            ("stimuli/Intermediate/Sync2_C.wav", "stimuli/Intermediate/NoSync_D.wav"),
            ("stimuli/Intermediate/Sync2_D.wav", "stimuli/Intermediate/Sync1_D.wav"),
            ("stimuli/Intermediate/Sync3_C.wav", "stimuli/Intermediate/Sync1_C.wav"),
            ("stimuli/Intermediate/Sync3_D.wav", "stimuli/Intermediate/Sync2_C.wav"),
            ("stimuli/Intermediate/Sync2_D.wav", "stimuli/Intermediate/Sync4_C.wav"),
            ("stimuli/Intermediate/Sync3_C.wav", "stimuli/Intermediate/Sync4_D.wav"),
            ("stimuli/Intermediate/NoSync_D.wav", "stimuli/Intermediate/Sync1_D.wav"),
            ("stimuli/Intermediate/Sync1_C.wav", "stimuli/Intermediate/Sync2_C.wav"),
            ("stimuli/Intermediate/Sync3_D.wav", "stimuli/Intermediate/Sync4_C.wav"),
        ]
    return [{"file1": _ppath(a), "file2": _ppath(b)} for a, b in pairs]

# =============================
# Example audio for few-shot
# =============================
EX_NOSYNC_A = _p("Intermediate/NoSync_A.wav")
EX_SYNC3_A  = _p("Intermediate/Sync3_A.wav")
EX_SYNC1_C  = _p("Intermediate/Sync1_C.wav")
EX_SYNC2_C  = _p("Intermediate/Sync2_C.wav")
EX_SYNC1_A  = _p("Intermediate/Sync1_A.wav")
EX_SYNC2_A  = _p("Intermediate/Sync2_A.wav")

# =============================
# Parsing / evaluation helpers
# =============================
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

def syncopation_rank(path: str) -> int:
    """
    Assign a rank for syncopation based on filename.
    - 'NoSync'  -> 0
    - 'Sync1'   -> 1
    - 'Sync2'   -> 2
    - 'Sync3'   -> 3
    - 'Sync4'   -> 4
    """
    base = os.path.basename(path).lower()
    if "nosync" in base:
        return 0
    m = re.search(r"sync(\d+)", base)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return 0  # fallback

# =============================
# Gemini compatibility shims
# =============================
def PartText(s: str):
    if hasattr(types.Part, "from_text"):
        return types.Part.from_text(text=s)
    if hasattr(types, "Part"):
        try:
            return types.Part(text=s)
        except TypeError:
            pass
    return {"text": s}

def PartBytes(data: bytes, mime_type: str):
    if hasattr(types.Part, "from_bytes"):
        return types.Part.from_bytes(data=data, mime_type=mime_type)
    if hasattr(types, "Blob"):
        return types.Part(inline_data=types.Blob(mime_type=mime_type, data=data))
    return {"inline_data": {"mime_type": mime_type, "data": data}}

def read_audio_bytes_and_mime(path: str) -> tuple[bytes, str]:
    with open(path, "rb") as f:
        return f.read(), "audio/wav"

def gemini_decoding_config(*, temperature: float, seed: int, max_tokens: int, system_instruction: str):
    return types.GenerateContentConfig(
        temperature=1.0, top_p=0.95, top_k=40,
        max_output_tokens=max_tokens, response_mime_type="text/plain",
        seed=seed, system_instruction=system_instruction,
    )

# =============================
# Few-shot builders + confirmation
# =============================
def _example_nosync_user(wav: bytes, mt: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            PartText("Example 1: This excerpt is not syncopated. Listen carefully:"),
            PartBytes(wav, mt),
        ]
    )

def _example_sync_user(wav: bytes, mt: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            PartText("Example 2: This excerpt is highly syncopated. Listen carefully:"),
            PartBytes(wav, mt),
        ]
    )

def build_fewshot_messages_COT(log_filename: str) -> List[types.Content]:
    """
    Returns two in-context examples (assistant messages included), chosen by Group:
      - If "GroupA" in log name → use Sync2_C vs Sync1_C (A) and NoSync_A vs Sync3_A (B)
      - If "GroupB" in log name → use Sync2_A vs Sync1_A (A) and NoSync_A vs Sync3_A (B)

    Each example ends with the strict final line:
      A. The rhythm in Excerpt 1 is more syncopated.
      B. The rhythm in Excerpt 2 is more syncopated.
    """
    group_a = "GroupA" in os.path.basename(log_filename)

    # Common assets
    wav_nosync_A, mt_nsA  = read_audio_bytes_and_mime(EX_NOSYNC_A)
    wav_sync3_A,  mt_sy3A = read_audio_bytes_and_mime(EX_SYNC3_A)

    if group_a:
        # Group A: Sync2_C vs Sync1_C (A-case), NoSync_A vs Sync3_A (B-case)
        wav_sync1_X, mt_sy1X = read_audio_bytes_and_mime(EX_SYNC1_C)
        wav_sync2_X, mt_sy2X = read_audio_bytes_and_mime(EX_SYNC2_C)
    else:
        # Group B: Sync2_A vs Sync1_A (A-case), NoSync_A vs Sync3_A (B-case)
        wav_sync1_X, mt_sy1X = read_audio_bytes_and_mime(EX_SYNC1_A)
        wav_sync2_X, mt_sy2X = read_audio_bytes_and_mime(EX_SYNC2_A)

    # === Example 1 — Excerpt 1 more syncopated (Sync vs Sync) → "A." ===
    ex1_user = types.Content(
        role="user",
        parts=[
            PartText(COT_PROMPT_TEMPLATE),
            PartText("Here is Excerpt 1:"),
            PartBytes(wav_sync2_X, mt_sy2X),
            PartText("Here is Excerpt 2:"),
            PartBytes(wav_sync1_X, mt_sy1X),
        ],
    )
    ex1_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Both excerpts share a steady pulse.\n"
            "Step 2: Excerpt 1 places many of the kick/snare hits on strong beats, but there are 4 instances where the kick/snare hits land on off-beats instead of strong beats."
            "Excerpt 2 places most of the kick/snare hits on strong beats, but there are 2 instances where the kick/snare hits land on off-beats instead of strong beats.\n"
            "Step 3: Excerpt 1 emphasizes more off-beats than Excerpt 2. Thus, Excerpt 1 is more syncopated.\n"
            "A. The rhythm in Excerpt 1 is more syncopated."
        )]
    )

    # === Example 2 — Excerpt 2 more syncopated (NoSync vs Sync) → "B." ===
    ex2_user = types.Content(
        role="user",
        parts=[
            PartText(COT_PROMPT_TEMPLATE),
            PartText("Here is Excerpt 1:"),
            PartBytes(wav_nosync_A, mt_nsA),
            PartText("Here is Excerpt 2:"),
            PartBytes(wav_sync3_A, mt_sy3A),
        ],
    )
    ex2_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Both excerpts share a steady pulse.\n"
            "Step 2: Excerpt 1 places all kick/snare hits on strong beats; the only off-beats present are the consistent eighth note hi-hat ostinatos."
            "Excerpt 2 places several of the kick/snare hits on strong beats, but there are 6 instances where the kick/snare hits land on off-beats instead of strong beats.\n"
            "Step 3: Excerpt 2 emphasizes more off-beats than Excerpt 1. Thus, Excerpt 2 is more syncopated.\n"
            "B. The rhythm in Excerpt 2 is more syncopated."
        )]
    )

    return [ex1_user, ex1_model, ex2_user, ex2_model]

def run_examples_and_confirm(
    *,
    client: genai.Client,
    model_name: str,
    sysinstr_text: str,
    group: str,
    log: logging.Logger,
) -> Tuple[Any, str]:
    """Create chat with examples + confirmation; returns (chat, confirmation_text)."""
    is_cot = (sysinstr_text == SYSINSTR_COT)

    if is_cot:
        history = build_fewshot_messages_COT(group)
    else:
        # SYSINST: one non-syncopated example, one highly syncopated example (single-audio turns)
        wav_nosync_A, mt_nsA = read_audio_bytes_and_mime(EX_NOSYNC_A)
        wav_sync3_A,  mt_sy3A = read_audio_bytes_and_mime(EX_SYNC3_A)
        history = [
            _example_nosync_user(wav_nosync_A, mt_nsA),
            _example_sync_user(wav_sync3_A,  mt_sy3A),
        ]

    cfg = gemini_decoding_config(
        temperature=1.0, seed=1, max_tokens=MAX_NEW_TOKENS, system_instruction=sysinstr_text
    )

    chat = client.chats.create(model=model_name, config=cfg, history=history)
    logging.info("Using one persistent chat; examples and all trials share history and audio context.")

    confirm_prompt = PartText(
        'After examining the examples above, please respond with "Yes, I understand." if you understand the task, '
        'or "No, I don\'t understand." if you do not.'
    )
    response = chat.send_message([confirm_prompt])
    completion = (getattr(response, "text", None) or "").strip()
    print("Confirmation response:", completion)
    log.info(f"Confirmation response: {completion}")
    return chat, completion

# =============================
# Single-run experiment
# =============================
def run_once(*, model_name: str, mode: str, group: str, seed: int, log_filename: str) -> None:
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}

    configure_logging(log_filename)
    logging.info(f"=== RUN START (Syncopation Detection • CHAT+SysInst • {mode} • {group}) ===")
    logging.info(f"Config: model={model_name}, temp=1.0, seed={seed}, group={group}, log={log_filename}")

    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    sysinstr_text = SYSINSTR_PLAIN if mode == "SYSINST" else SYSINSTR_COT

    chat, _confirm = run_examples_and_confirm(
        client=client,
        model_name=model_name,
        sysinstr_text=sysinstr_text,
        group=group,
        log=logging.getLogger(),
    )

    question_stims = stimuli_pairs_group(group)
    random.seed(seed)
    random.shuffle(question_stims)

    print(
        f"\n--- Task: Syncopation Detection — CHAT+SysInst {mode} • {group} | model={model_name} | temp=1.0 | seed={seed} ---\n"
    )
    logging.info(f"\n--- Task: Syncopation Detection — CHAT+SysInst {mode} • {group} ---\n")

    correct = 0
    total = len(question_stims)

    per_trial_cfg = gemini_decoding_config(
        temperature=1.0, seed=seed, max_tokens=MAX_NEW_TOKENS, system_instruction=sysinstr_text
    )
    try:
        chat.config = per_trial_cfg  # type: ignore
    except Exception:
        pass

    for idx, q in enumerate(question_stims, start=1):
        print(f"\n--- Question {idx} ---\n")
        logging.info(f"\n--- Question {idx} ---\n")

        f1 = q["file1"]
        f2 = q["file2"]
        logging.info(f"Stimuli: file1={f1}, file2={f2}")

        wav1, mt1 = read_audio_bytes_and_mime(f1)
        wav2, mt2 = read_audio_bytes_and_mime(f2)

        if mode == "SYSINST":
            user_parts = [
                PartText("Here is Excerpt 1."),
                PartBytes(wav1, mt1),
                PartText("Here is Excerpt 2."),
                PartBytes(wav2, mt2),
                PartText('Reply with exactly ONE of the following lines:\n'
                         'A. The rhythm in Excerpt 1 is more syncopated.\n'
                         'OR\n'
                         'B. The rhythm in Excerpt 2 is more syncopated.'),
            ]
        else:
            user_parts = [
                PartText(COT_PROMPT_TEMPLATE),
                PartText("Here is Excerpt 1:"),
                PartBytes(wav1, mt1),
                PartText("Here is Excerpt 2:"),
                PartBytes(wav2, mt2),
            ]

        response = chat.send_message(user_parts)
        completion = (getattr(response, "text", None) or "").strip()

        print("LLM Full Response:\n", completion)
        logging.info(f"[{mode}/{group}] Q{idx} - LLM Full Response:\n{completion}")

        # Parse decision
        model_answer = parse_final_decision(completion)
        if not model_answer:
            print("Evaluation: Failed. Could not parse the final answer phrase.")
            logging.error("Parse Error: missing/malformed final answer phrase.")
            continue

        logging.info(f"Parsed Final Answer: {model_answer}")

        # Ground truth: higher Sync# wins; any Sync# > NoSync
        r1 = syncopation_rank(f1)
        r2 = syncopation_rank(f2)
        if r1 == r2:
            expected = ""  # rare tie; skip scoring
            logging.warning(f"Ambiguous tie on syncopation rank (r1=r2={r1}); skipping scoring for Q{idx}.")
        else:
            expected = A_CANON if r1 > r2 else B_CANON

        if not expected:
            print("Evaluation: Skipped (tie).")
            continue

        if model_answer == expected:
            correct += 1
            print("Evaluation: Correct!")
            logging.info("Evaluation: Correct")
        else:
            print("Evaluation: Incorrect.")
            logging.info(f"Evaluation: Incorrect (expected={expected})")

    print(f"\nTotal Correct: {correct} out of {total}")
    logging.info(f"Total Correct: {correct} out of {total}")
    logging.info("=== RUN END ===\n")

# =============================
# Multi-run driver (24 total)
# =============================
if __name__ == "__main__":
    runs = []
    MODELS = ["gemini-2.5-pro", "gemini-2.5-flash"]
    for model in MODELS:
        for mode in ["SYSINST", "COT"]:
            for group in ["GroupA", "GroupB"]:
                for s in (1, 2, 3):
                    runs.append(dict(
                        model_name=model,
                        mode=mode,
                        group=group,
                        seed=s,
                        log_filename=make_log_filename(model_name=model, mode=mode, group=group, seed=s),
                    ))

    for cfg in runs:
        run_once(**cfg)
        gc.collect()
