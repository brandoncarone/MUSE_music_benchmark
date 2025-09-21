# oddball_detection_Gemini_CHAT_SysInst_master.py
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
STIM_ROOT = "/stimuli"
MAX_NEW_TOKENS = 8192

# Canonical answer strings and robust patterns
YES_CANON = "Yes, these are the same exact melody."
NO_CANON  = "No, these are not the same exact melody."

YES_PAT = re.compile(r'(?i)\byes,\s*these\s+are\s+the\s+same\s+exact\s+melody\.')
NO_PAT  = re.compile(r'(?i)\bno,\s*these\s+are\s+not\s+the\s+same\s+exact\s+melody\.')

# =============================
# System instructions (VERBATIM from your new text blocks)
# =============================
SYSINSTR_PLAIN = """You are a participant in a psychological experiment on music perception. In each question, you will be given:
    1. A brief instruction about the specific listening task.
    2. Two audio examples to listen to.

Your task is to decide whether the two audio examples are the same exact melody, or whether an "Oddball" is present. 
An “Oddball” in a musical or auditory experiment is simply a note or sound that doesn’t fit with what you’d expect based 
on what you’ve been hearing. Imagine you’re listening to a melody where all the notes line up nicely in the same key—then 
suddenly, one note is out of key. This unexpected note is the "oddball". If the note is present multiple times in the melody, 
then you will hear the oddball more than once.

Valid responses are:
“Yes, these are the same exact melody.” if they are exactly the same, or 
“No, these are not the same exact melody.” if you notice an oddball.

Before you begin the task, I will provide you with examples of two excerpts representing the same exact melody, as well as
examples of two excerpts where one contains oddballs so that you better understand the task. After examining the 
examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't 
understand the task.

Please provide no additional commentary beyond the short answers previously mentioned.
"""

SYSINSTR_COT = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. Two audio examples to listen to.

Your task is to decide whether the two audio examples are the same exact melody, or whether an ODDBALL is present in one of them.

Definitions and constraints:
- “Oddball” = one or more unexpected notes that do not match the expected melody (e.g., out-of-key or altered notes). The oddball may occur more than once.
- Judge exact pitch content at corresponding positions (no transposition or octave equivalence). Small timing deviations or leading/trailing silence can be ignored.
- If every note at each aligned position matches exactly, there is NO oddball (same exact melody).
- If any aligned note differs (pitch substitution/out-of-key change), an oddball is present and the melodies are NOT the same.

Valid responses are:
"Yes, these are the same exact melody."
or
"No, these are not the same exact melody."

Before you begin the task, I will provide you with examples of two excerpts representing the same exact melody, as well as
examples of two excerpts where one contains oddballs so that you better understand the task. After examining the 
examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't 
understand the task.

After any reasoning, end with exactly one line:
Yes, these are the same exact melody.
OR
No, these are not the same exact melody."""

# --- COT per-trial prompt (VERBATIM) ---
COT_PROMPT_TEMPLATE = """Analyze the two music excerpts and decide if they are the SAME exact melody or if an ODDBALL is present.

Step 1: For each audio, identify the monophonic note sequence (pitch over time). Ignore small timing differences and leading/trailing silence.

Step 2: Align the two sequences by their note order and compare pitch at each corresponding position (no transposition or octave equivalence).

Step 3: Decision rule:
- If all corresponding pitches match exactly, there is no oddball, and they are the same exact melody.
- If any pitch differs (e.g., out-of-key substitution or altered note), there is an oddball present, and they are not the same melody.

Step 4: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
Yes, these are the same exact melody.
OR
No, these are not the same exact melody.
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
    Example: oddballs_G25Pro_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    tag = _model_tag_for(model_name)
    return f"oddballs_{tag}_CHAT_{mode}_{group}_seed{seed}.log"

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

def stimuli_pairs_group(group: str) -> List[Dict[str, str]]:
    """
    Return list of dicts: {"file1": abs_path, "file2": abs_path}
    """
    assert group in {"GroupA", "GroupB"}
    if group == "GroupA":
        pairs = [
            ("stimuli/M1_EbMaj_90.wav",            "stimuli/M1_EbMaj_90.wav"),
            ("stimuli/M1_EbMaj_90.wav",            "stimuli/M1_Odd_EbMaj_90.wav"),
            ("stimuli/M2_Abm_155_3_4.wav",         "stimuli/M2_Abm_155_3_4.wav"),
            ("stimuli/M2_Abm_155_3_4.wav",         "stimuli/M2_Odd_Abm_155_3_4.wav"),
            ("stimuli/M3_DbMaj_78.wav",            "stimuli/M3_DbMaj_78.wav"),
            ("stimuli/M8_FMaj_95_Piano.wav",       "stimuli/M8_Odd_FMaj_95_Piano.wav"),
            ("stimuli/M9_Gm_200_3_4_Piano.wav",    "stimuli/M9_Gm_200_3_4_Piano.wav"),
            ("stimuli/M9_Gm_200_3_4_Piano.wav",    "stimuli/M9_Odd_Gm_200_3_4_Piano.wav"),
            ("stimuli/M10_Fm_165_3_4.wav",         "stimuli/M10_Fm_165_3_4.wav"),
            ("stimuli/M10_Fm_165_3_4.wav",         "stimuli/M10_Odd_Fm_165_3_4.wav"),
        ]
    else:
        pairs = [
            ("stimuli/M3_DbMaj_78.wav",            "stimuli/M3_Odd_DbMaj_78.wav"),
            ("stimuli/M4_EMaj_130.wav",            "stimuli/M4_EMaj_130.wav"),
            ("stimuli/M4_EMaj_130.wav",            "stimuli/M4_Odd_EMaj_130.wav"),
            ("stimuli/M5_Dm_100.wav",              "stimuli/M5_Dm_100.wav"),
            ("stimuli/M5_Dm_100.wav",              "stimuli/M5_Odd_Dm_100.wav"),
            ("stimuli/M6_Cm_120_Piano.wav",        "stimuli/M6_Cm_120_Piano.wav"),
            ("stimuli/M6_Cm_120_Piano.wav",        "stimuli/M6_Odd_Cm_120_Piano.wav"),
            ("stimuli/M7_CMaj_140_Piano.wav",      "stimuli/M7_CMaj_140_Piano.wav"),
            ("stimuli/M7_CMaj_140_Piano.wav",      "stimuli/M7_Odd_CMaj_140_Piano.wav"),
            ("stimuli/M8_FMaj_95_Piano.wav",       "stimuli/M8_FMaj_95_Piano.wav"),
        ]
    return [{"file1": _ppath(a), "file2": _ppath(b)} for a, b in pairs]

# Few-shot example file paths
EX_SAME = _p("M11_CMaj_180_Piano.wav")
EX_ODD  = _p("M11_Odd_CMaj_180_Piano.wav")

# =============================
# Parsing / evaluation helpers
# =============================
def parse_final_decision(text: str) -> str:
    """Return YES_CANON or NO_CANON, or '' if not found. Prefer the LAST occurrence."""
    last_yes = None
    last_no = None
    for m in YES_PAT.finditer(text or ""):
        last_yes = m
    for m in NO_PAT.finditer(text or ""):
        last_no = m
    if last_yes and last_no:
        return YES_CANON if last_yes.end() > last_no.end() else NO_CANON
    if last_yes:
        return YES_CANON
    if last_no:
        return NO_CANON
    return ""

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
    # We assume stimuli are WAV; if you ever add MP3/FLAC, branch on extension.
    with open(path, "rb") as f:
        return f.read(), "audio/wav"

def gemini_decoding_config(*, temperature: float, seed: int, max_tokens: int, system_instruction: str):
    # Temperature fixed at 1.0 for this benchmark.
    return types.GenerateContentConfig(
        temperature=1.0, top_p=0.95, top_k=40,
        max_output_tokens=max_tokens, response_mime_type="text/plain",
        seed=seed, system_instruction=system_instruction,
    )

# =============================
# Few-shot builders + confirmation
# =============================
def _example_same_user(wav_a: bytes, mt_a: str, wav_b: bytes, mt_b: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            PartText("Example 1: The following two audio examples are the same exact melody. Listen carefully:"),
            PartText("Audio example number 1:"),
            PartBytes(wav_a, mt_a),
            PartText("Audio example number 2:"),
            PartBytes(wav_b, mt_b),
        ]
    )

def _example_odd_user(wav_a: bytes, mt_a: str, wav_b: bytes, mt_b: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            PartText("Example 2: The following two audio examples are different; an oddball is present in one excerpt (it may occur more than once). Listen carefully:"),
            PartText("Audio example number 1:"),
            PartBytes(wav_a, mt_a),
            PartText("Audio example number 2:"),
            PartBytes(wav_b, mt_b),
        ]
    )

def build_fewshot_messages_COT() -> List[types.Content]:
    """
    Two in-context examples with brief CoT and strict final line (assistant messages included).
    Ex1: SAME → ends with: Yes, these are the same exact melody.
    Ex2: ODDBALL present → ends with: No, these are not the same exact melody.
    """
    wav_same1, mt1 = read_audio_bytes_and_mime(EX_SAME)
    wav_same2, mt2 = read_audio_bytes_and_mime(EX_SAME)
    wav_odd,   mt3 = read_audio_bytes_and_mime(EX_ODD)

    # Example 1 — SAME exact melody
    ex1_user = types.Content(
        role="user",
        parts=[PartText(COT_PROMPT_TEMPLATE),
               PartBytes(wav_same1, mt1),
               PartBytes(wav_same2, mt2)]
    )
    ex1_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Extract monophonic note sequences for both excerpts.\n"
            "Step 2: Align note-by-note; pitches match at every position.\n"
            "Step 3: No substitutions or out-of-key notes detected, thus, no oddball.\n"
            "Yes, these are the same exact melody."
        )]
    )

    # Example 2 — ODDBALL present
    ex2_user = types.Content(
        role="user",
        parts=[PartText(COT_PROMPT_TEMPLATE),
               PartBytes(wav_same1, mt1),
               PartBytes(wav_odd,   mt3)]
    )
    ex2_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Extract monophonic sequences.\n"
            "Step 2: Alignment reveals positions where the second excerpt’s pitch deviates (out-of-key substitutions).\n"
            "Step 3: Oddball(s) present, thus, the melodies are not the same.\n"
            "No, these are not the same exact melody."
        )]
    )

    return [ex1_user, ex1_model, ex2_user, ex2_model]

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
        # SYSINST: two user-only example turns with audio, no assistant interjections
        wav_same1, mt1 = read_audio_bytes_and_mime(EX_SAME)
        wav_same2, mt2 = read_audio_bytes_and_mime(EX_SAME)
        wav_odd,   mt3 = read_audio_bytes_and_mime(EX_ODD)
        history = [
            _example_same_user(wav_same1, mt1, wav_same2, mt2),
            _example_odd_user(wav_same1, mt1, wav_odd,   mt3),
        ]

    # Decoding config (temperature fixed at 1.0 for this benchmark)
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
    logging.info(f"=== RUN START (Oddball Detection • CHAT+SysInst • {mode} • {group}) ===")
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

    # Fixed stimuli for the selected group, then shuffle per seed
    question_stims = stimuli_pairs_group(group)
    random.seed(seed)  # ensures reproducible shuffle per run/seed
    random.shuffle(question_stims)

    print(
        f"\n--- Task: Oddball Detection — CHAT+SysInst {mode} • {group} | model={model_name} | temp=1.0 | seed={seed} ---\n"
    )
    logging.info(f"\n--- Task: Oddball Detection — CHAT+SysInst {mode} • {group} ---\n")

    correct = 0
    total = len(question_stims)

    # Per-trial decoding (recreate config to stamp the seed for reproducibility logging)
    per_trial_cfg = gemini_decoding_config(
        temperature=1.0, seed=seed, max_tokens=MAX_NEW_TOKENS, system_instruction=sysinstr_text
    )
    # Update the live chat config if supported (safe to call; else ignored)
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

        # Load audio (original bytes, no trimming, no downmix)
        wav1, mt1 = read_audio_bytes_and_mime(f1)
        wav2, mt2 = read_audio_bytes_and_mime(f2)

        # Build trial prompt (text + 2 audio parts)
        if mode == "SYSINST":
            user_parts = [
                PartText("Here is the first excerpt."),
                PartBytes(wav1, mt1),
                PartText("Here is the second excerpt."),
                PartBytes(wav2, mt2),
                PartText('Reply with exactly ONE of the following lines:\n'
                         'Yes, these are the same exact melody.\n'
                         'OR\n'
                         'No, these are not the same exact melody.'),
            ]
        else:
            user_parts = [
                PartText(COT_PROMPT_TEMPLATE),
                PartBytes(wav1, mt1),
                PartBytes(wav2, mt2),
            ]

        # Send the message on the persistent chat (history maintained automatically)
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

        # Ground truth: identical filenames => SAME; otherwise NOT SAME
        base1 = os.path.basename(f1)
        base2 = os.path.basename(f2)
        expected = YES_CANON if base1 == base2 else NO_CANON

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
