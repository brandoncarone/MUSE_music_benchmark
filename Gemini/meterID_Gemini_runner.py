# meter_identification_Gemini_CHAT_SysInst_master.py
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
# Preserve canonical root; _ppath below will map "stimuli/..." to this absolute root.
STIM_ROOT = "/Users/bcarone/PycharmProjects/GeminiAPI/stimuli"
MAX_NEW_TOKENS = 8192

# Canonical answer strings and robust patterns (strict but case-insensitive)
A_CANON = "A. Groups of 3"
B_CANON = "B. Groups of 4"
C_CANON = "C. Groups of 5"

A_PAT = re.compile(r"(?i)\bA\.\s*Groups\s+of\s+3\b")
B_PAT = re.compile(r"(?i)\bB\.\s*Groups\s+of\s+4\b")
C_PAT = re.compile(r"(?i)\bC\.\s*Groups\s+of\s+5\b")

# =============================
# System instructions (VERBATIM)
# =============================
SYSINSTR_PLAIN = """You are a participant in a psychological experiment on music perception. 
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to. 

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


Before you begin the task, I will provide you with examples of excerpts that are counted in groups of 3, in groups of 4, 
and in groups of 5 so that you better understand the task. After examining the examples, please respond 
with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't understand the task."""

SYSINSTR_COT = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to.

Your task is to identify the METER — how the steady pulse is grouped into repeating cycles.

Definitions and constraints:
- Meter = number of beats in the repeating cycle (groups of 3, 4, or 5).
- Focus on the strongest recurring downbeat and count how many beats elapse before that accent pattern repeats.
- Ignore tempo (speed), instrumentation, dynamics, fills, and small timing jitter.
- Surface syncopation does not change the underlying cycle; choose the SMALLEST repeating grouping that explains the accents.

Valid responses are exactly:
"A. Groups of 3"
"B. Groups of 4"
"C. Groups of 5"

Before you begin the task, I will provide you with one example for each meter so you better understand the task. After 
examining the examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." 
if you don't understand the task.

After any reasoning, end with exactly one line:
A. Groups of 3
OR
B. Groups of 4
OR
C. Groups of 5"""

# --- COT prompt (VERBATIM) ---
COT_PROMPT_TEMPLATE = """Analyze the music excerpts and identify the underlying METER (groups of 3, 4, or 5).

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
# Logging utilities (keep identical style)
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
    Example: meter_identification_G25Pro_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    tag = _model_tag_for(model_name)
    return f"meter_identification_{tag}_CHAT_{mode}_{group}_seed{seed}.log"

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

def _file_id_no_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

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

# Gold labels (by basename) — EXACT strings you provided
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
    # appears in both groups:
    "ComeOn_4.wav":                          B_CANON,
    "DoDoDoDoDo_4.wav":                      B_CANON,
    "Flow_4.wav":                            B_CANON,
    "Harm_4.wav":                            B_CANON,
    "Dance_5.wav":                           C_CANON,
    "Falling_5.wav":                         C_CANON,
}

def expected_for_meter(path: str) -> str:
    base = os.path.basename(path)
    return METER_GOLD.get(base, "")

# =============================
# Parsing / evaluation helpers
# =============================
def parse_final_decision(text: str) -> str:
    """
    Return the canonical final answer string (A_CANON/B_CANON/C_CANON), or '' if not found.
    Prefer the LAST occurrence among A/B/C matches.
    """
    last_a = last_b = last_c = None
    for m in A_PAT.finditer(text or ""):
        last_a = m
    for m in B_PAT.finditer(text or ""):
        last_b = m
    for m in C_PAT.finditer(text or ""):
        last_c = m

    best = None
    for m, canon in [(last_a, A_CANON), (last_b, B_CANON), (last_c, C_CANON)]:
        if m and (best is None or m.end() > best[0].end()):
            best = (m, canon)
    return best[1] if best else ""

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
# Example audio paths (resolved with your _p helper under STIM_ROOT)
EX_3 = _p("Intermediate/Whammy_3.wav")
EX_4 = _p("Intermediate/SS_4.wav")
EX_5 = _p("Intermediate/Five_5.wav")

def build_fewshot_messages_COT() -> List[types.Content]:
    """
    Three in-context examples with brief CoT and strict final line (assistant messages included).
    Ex1: Groups of 3 -> ends with: A. Groups of 3
    Ex2: Groups of 4 -> ends with: B. Groups of 4
    Ex3: Groups of 5 -> ends with: C. Groups of 5
    """
    # Load audio bytes
    wav3, mt3 = read_audio_bytes_and_mime(EX_3)
    wav4, mt4 = read_audio_bytes_and_mime(EX_4)
    wav5, mt5 = read_audio_bytes_and_mime(EX_5)

    id3 = "Intermediate/Whammy_3"
    id4 = "Intermediate/SS_4"
    id5 = "Intermediate/Five_5"

    # Example 1 — Groups of 3 (A)
    ex1_user = types.Content(
        role="user",
        parts=[
            PartText(COT_PROMPT_TEMPLATE),
            PartBytes(wav3, mt3),
        ],
    )
    ex1_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Establish a steady beat.\n"
            "Step 2: A strong downbeat recurs every three beats (ONE-two-three | ONE-two-three...).\n"
            "A. Groups of 3"
        )]
    )

    # Example 2 — Groups of 4 (B)
    ex2_user = types.Content(
        role="user",
        parts=[
            PartText(COT_PROMPT_TEMPLATE),
            PartBytes(wav4, mt4),
        ],
    )
    ex2_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Find the pulse.\n"
            "Step 2: The accent pattern repeats every four beats (ONE-two-three-four | ONE-two-three-four).\n"
            "B. Groups of 4"
        )]
    )

    # Example 3 — Groups of 5 (C)
    ex3_user = types.Content(
        role="user",
        parts=[
            PartText(COT_PROMPT_TEMPLATE),
            PartBytes(wav5, mt5),
        ],
    )
    ex3_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Lock to the beat.\n"
            "Step 2: The cycle resolves every five beats (ONE-two-three-four-five | ONE-two-three-four-five).\n"
            "C. Groups of 5"
        )]
    )

    return [ex1_user, ex1_model, ex2_user, ex2_model, ex3_user, ex3_model]

def _example_meter_user(label_text: str, wav: bytes, mt: str) -> types.Content:
    # Minimal plain example message for SYSINST
    return types.Content(
        role="user",
        parts=[PartText(label_text), PartBytes(wav, mt)]
    )

def run_examples_and_confirm(
    *,
    client: genai.Client,
    model_name: str,
    sysinstr_text: str,
    log: logging.Logger,
) -> Tuple[Any, str]:
    """
    Create a chat with the appropriate system instruction and few-shot history,
    then ask for the confirmation reply. Returns (chat, confirmation_text).
    """
    is_cot = (sysinstr_text == SYSINSTR_COT)

    if is_cot:
        history = build_fewshot_messages_COT()
    else:
        # SYSINST: three user-only example turns with audio, no assistant interjections
        wav3, mt3 = read_audio_bytes_and_mime(EX_3)
        wav4, mt4 = read_audio_bytes_and_mime(EX_4)
        wav5, mt5 = read_audio_bytes_and_mime(EX_5)
        history = [
            _example_meter_user("Example: This excerpt is counted in groups of 3. Listen carefully:", wav3, mt3),
            _example_meter_user("Example: This excerpt is counted in groups of 4. Listen carefully:", wav4, mt4),
            _example_meter_user("Example: This excerpt is counted in groups of 5. Listen carefully:", wav5, mt5),
        ]

    cfg = gemini_decoding_config(
        temperature=1.0,
        seed=1,  # per-run seed is applied below; this initial value is harmless
        max_tokens=MAX_NEW_TOKENS,
        system_instruction=sysinstr_text,
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
    logging.info(f"=== RUN START (Meter Identification • CHAT+SysInst • {mode} • {group}) ===")
    logging.info(f"Config: model={model_name}, temp=1.0, seed={seed}, group={group}, log={log_filename}")

    # Env & client
    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # System instructions (verbatim selection)
    sysinstr_text = SYSINSTR_PLAIN if mode == "SYSINST" else SYSINSTR_COT

    # Create chat with examples + confirmation in history
    chat, _confirm = run_examples_and_confirm(
        client=client,
        model_name=model_name,
        sysinstr_text=sysinstr_text,
        log=logging.getLogger(),
    )

    # Fixed stimuli list for the selected group; shuffle reproducibly by seed
    question_stims = stimuli_list_group(group)
    random.seed(seed)
    random.shuffle(question_stims)

    print(
        f"\n--- Task: Meter Identification — CHAT+SysInst {mode} • {group} | model={model_name} | temp=1.0 | seed={seed} ---\n"
    )
    logging.info(f"\n--- Task: Meter Identification — CHAT+SysInst {mode} • {group} ---\n")

    correct = 0
    total = len(question_stims)

    # Per-trial decoding (recreate config to stamp the seed for reproducibility logging)
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

        f1 = q["file"]
        logging.info(f"Stimulus: file={f1}")

        # Load audio (original bytes, no trimming, no downmix)
        wav1, mt1 = read_audio_bytes_and_mime(f1)

        # Build trial prompt (text + audio)
        if mode == "SYSINST":
            user_parts = [
                PartText("Listen to the following excerpt and identify the meter."),
                PartBytes(wav1, mt1),
                PartText("""Reply with exactly ONE of the following lines:
A. Groups of 3
OR
B. Groups of 4
OR
C. Groups of 5
"""),
            ]
        else:
            user_parts = [
                PartText(COT_PROMPT_TEMPLATE),
                PartBytes(wav1, mt1),
            ]

        # Send the message on the persistent chat
        response = chat.send_message(user_parts)
        completion = (getattr(response, "text", None) or "").strip()

        print("LLM Full Response:\n", completion)
        logging.info(f"[{mode}/{group}] Q{idx} - LLM Full Response:\n{completion}")

        # Parse decision (prefer last occurrence)
        model_answer = parse_final_decision(completion)
        if not model_answer:
            print("Evaluation: Failed. Could not parse the final answer phrase.")
            logging.error("Parse Error: missing/malformed final answer phrase.")
            continue

        logging.info(f"Parsed Final Answer: {model_answer}")

        # Ground truth via explicit mapping
        expected = expected_for_meter(f1)
        if not expected:
            print("Evaluation: Unknown stimulus label for expected answer.")
            logging.error("Missing gold label for stimulus basename.")
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
