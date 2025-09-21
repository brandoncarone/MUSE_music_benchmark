# rhythm_matching_Gemini_CHAT_SysInst_master.py
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

# Canonical answer strings and robust patterns (VERBATIM from Qwen runner)
YES_CANON = 'Yes, these are the same exact drum set patterns.'
NO_CANON = 'No, these are not the same exact drum set patterns.'

YES_PAT = re.compile(r'(?i)\byes,\s*these\s+are\s+the\s+same\s+exact\s+drum\s+set\s+patterns\.')
NO_PAT = re.compile(r'(?i)\bno,\s*these\s+are\s+not\s+the\s+same\s+exact\s+drum\s+set\s+patterns\.')

# =============================
# System instructions
# =============================
SYSINSTR_PLAIN = """You are a participant in a psychological experiment on music perception. 
 In each question, you will be given:
    1. A brief instruction about the specific listening task.
    2. Two audio examples to listen to. 
    
Your task is to decide whether the two music excerpts have the same exact drum set pattern – meaning the same exact rhythmic structure – 
or if they are different. All rhythms will be played at the same tempo / speed. 
Valid responses are exactly:
"Yes, these are the same exact drum set patterns." or 
"No, these are not the same exact drum set patterns."

Before you begin the task, I will provide you with examples of two excerpts representing the same rhythmic pattern, as well as
examples of two excerpts representing different rhythmic patterns so that you better understand the task. After examining the 
examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't 
understand the task.

Please provide no additional commentary beyond the short answers previously mentioned."""

SYSINSTR_COT = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. Two audio examples to listen to.

Your task is to decide whether the two music excerpts have the same exact drum set pattern – meaning the same exact rhythmic structure – 
or if they are different. All rhythms will be played at the same tempo / speed. 

Definitions and constraints:
- Pattern equivalence: both excerpts share the same repeating cycle length and the same onset positions for the main kit voices (e.g., Kick, Snare, Hi-Hat/Cymbal, and Toms). Instrument identity across those voices matters (a kick-onset is not interchangeable with a snare-onset).
- Cycle & meter: choose the smallest repeating cycle you hear (one bar or two bars). Do not assume a specific time signature; align by pulse and repeating structure.
- Ignore dynamics (loudness), micro-ornaments, and mix; focus on which drums hit where in time. If any hits are added/removed or moved to different positions (or to a different drum voice), the patterns are different.

Valid responses are exactly:
"Yes, these are the same exact drum set patterns." or 
"No, these are not the same exact drum set patterns."

Before you begin the task, I will provide you with examples of two excerpts representing the same rhythmic pattern, as well as
examples of two excerpts representing different rhythmic patterns so that you better understand the task. After examining the 
examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't 
understand the task.

After any reasoning, end with exactly one line:
Yes, these are the same exact drum set patterns.
OR
No, these are not the same exact drum set patterns."""

# --- COT prompt (VERBATIM) ---
COT_PROMPT_TEMPLATE = """Analyze the two music excerpts to determine if they are the SAME exact drum set pattern.

Step 1: Establish the pulse and identify the smallest repeating cycle you hear for each excerpt (e.g., one bar or two bars).

Step 2: For each excerpt:
- Identify the overall feel of the rhythm (straight/duple vs. triplet/swing),
- Identify the meter, and
- Examine the positions of where the Kick, Snare, Hi-Hat/Cymbal, and Tom hits fall

Step 3: Compare the two patterns. They are the same pattern if:
- The cycle lengths match,
- They have the same rhythmic feel,
- They have the same meter, and
- The positions for each drum voice match across the two excerpts (no added/removed/moved hits, and no voice swaps).

Notes:
Tempo is the same across excerpts; focus on placement within the cycle.
Don’t assume a specific meter; use the repeating structure you hear.

Step 4: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
Yes, these are the same exact drum set patterns.
OR
No, these are not the same exact drum set patterns.
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
    Example: rhythm_matching_G25Pro_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    tag = _model_tag_for(model_name)
    return f"rhythm_matching_{tag}_CHAT_{mode}_{group}_seed{seed}.log"

# =============================
# Data helpers
# =============================
def _ppath(p: str) -> str:
    """Normalize a given path to absolute under STIM_ROOT."""
    if os.path.isabs(p):
        return p
    # If it starts with 'stimuli/', drop that prefix and join to STIM_ROOT
    if p.startswith("stimuli/") or p.startswith("stimuli" + os.sep):
        # keep everything after the first '/' (or os.sep)
        rel = p.split("/", 1)[1] if "/" in p else p.split(os.sep, 1)[1]
        return os.path.join(STIM_ROOT, rel)
    return os.path.join(STIM_ROOT, p)

def _p(name: str) -> str:
    return os.path.join(STIM_ROOT, name)

def _file_id_no_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def stimuli_pairs_group(group: str) -> List[Dict[str, str]]:
    assert group in {"GroupA", "GroupB"}
    if group == "GroupA":
        pairs = [
            ("stimuli/Beat_1_140.wav",       "stimuli/Beat_1_140.wav"),
            ("stimuli/Beat_1_140.wav",       "stimuli/Beat_2_140.wav"),
            ("stimuli/Beat_2_140.wav",       "stimuli/Beat_2_140.wav"),
            ("stimuli/Beat_2_140.wav",       "stimuli/Beat_3_140.wav"),
            ("stimuli/Beat_3_140.wav",       "stimuli/Beat_3_140.wav"),
            ("stimuli/Beat_3_140.wav",       "stimuli/Beat_4_140_34.wav"),
            ("stimuli/Beat_4_140_34.wav",   "stimuli/Beat_4_140_34.wav"),
            ("stimuli/Beat_4_140_34.wav",   "stimuli/Beat_5_140.wav"),
            ("stimuli/Beat_5_140.wav",       "stimuli/Beat_5_140.wav"),
            ("stimuli/Beat_5_140.wav",       "stimuli/Beat_6_140_34.wav"),
        ]
    else:
        pairs = [
            ("stimuli/Beat_6_140_34.wav",   "stimuli/Beat_6_140_34.wav"),
            ("stimuli/Beat_6_140_34.wav",   "stimuli/Beat_7_140.wav"),
            ("stimuli/Beat_7_140.wav",       "stimuli/Beat_7_140.wav"),
            ("stimuli/Beat_7_140.wav",       "stimuli/Beat_8_140.wav"),
            ("stimuli/Beat_8_140.wav",       "stimuli/Beat_8_140.wav"),
            ("stimuli/Beat_8_140.wav",       "stimuli/Beat_9_140_34.wav"),
            ("stimuli/Beat_9_140_34.wav",   "stimuli/Beat_9_140_34.wav"),
            ("stimuli/Beat_9_140_34.wav",   "stimuli/Beat_10_140.wav"),
            ("stimuli/Beat_10_140.wav",      "stimuli/Beat_10_140.wav"),
            ("stimuli/Beat_10_140.wav",      "stimuli/Beat_1_140.wav"),
        ]
    return [{"file1": _ppath(a), "file2": _ppath(b)} for a, b in pairs]

# Few-shot example file paths (VERBATIM IDs)
EX_SAME_A = _p("Beat_11_140.wav")
EX_SAME_B = _p("Beat_11_140.wav")
EX_DIFF_C = _p("Beat_12_140.wav")

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

def rhythm_id(fname: str) -> str:
    base = os.path.basename(fname)
    m = re.match(r"(?i)^(Beat_\d+)_", base)
    return m.group(1) if m else base

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
            PartText("Example 1: The following two audio excerpts represent the same exact drum set pattern."),
            PartBytes(wav_a, mt_a),
            PartText("Audio example 2:"),
            PartBytes(wav_b, mt_b),
        ]
    )

def _example_diff_user(wav_b: bytes, mt_b: str, wav_c: bytes, mt_c: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            PartText("Example 2: The following two audio excerpts represent different drum set patterns."),
            PartBytes(wav_b, mt_b),
            PartText("Audio example 2:"),
            PartBytes(wav_c, mt_c),
        ]
    )

def build_fewshot_messages_COT() -> List[types.Content]:
    """
    Two in-context examples with brief CoT and strict final line (assistant messages included).
    Ex1: SAME → ends with: Yes, these are the same exact drum set patterns.
    Ex2: DIFFERENT → ends with: No, these are not the same exact drum set patterns.
    """
    # Load audio bytes (no trimming / no downmix)
    wav_same1, mt1 = read_audio_bytes_and_mime(EX_SAME_A)  # Beat_11_140
    wav_same2, mt2 = read_audio_bytes_and_mime(EX_SAME_B)  # Beat_11_140
    wav_diff2, mt3 = read_audio_bytes_and_mime(EX_DIFF_C)  # Beat_12_140

    # Example 1 — SAME pattern
    ex1_user = types.Content(
        role="user",
        parts=[
            PartText(COT_PROMPT_TEMPLATE),
            PartBytes(wav_same1, mt1),
            PartBytes(wav_same2, mt2),
        ],
    )
    ex1_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Both excerpts establish a clear pulse and form a two-bar phrase.\n"
            "Step 2:\n"
            "Excerpt 1:\n"
            "Feel: triplet.\n"
            "Meter: inferred 4/4.\n"
            "Position analysis: Analyzed the Kick, Snare, Hi-Hat/Cymbal, and the Toms hits.\n"
            "Excerpt 2:\n"
            "Feel: triplet.\n"
            "Meter: inferred 4/4.\n"
            "Position analysis: Analyzed the Kick, Snare, Hi-Hat/Cymbal, and the Toms hits.\n"
            "Step 3: Cycle length, feel, and meter match; onset positions per voice are aligned in each of the excerpts.\n"
            "Yes, these are the same exact drum set patterns."
        )]
    )

    # Example 2 — DIFFERENT pattern
    ex2_user = types.Content(
        role="user",
        parts=[
            PartText(COT_PROMPT_TEMPLATE),
            PartBytes(wav_same2, mt2),   # Beat_11_140
            PartBytes(wav_diff2, mt3),   # Beat_12_140
        ],
    )
    ex2_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Both excerpts establish a clear pulse and form a two-bar phrase.\n"
            "Step 2:\n"
            "Excerpt 1:\n"
            "Feel: triplet.\n"
            "Meter: inferred 4/4.\n"
            "Position analysis: Analyzed the Kick, Snare, Hi-Hat/Cymbal, and the Toms hits.\n"
            "Excerpt 2:\n"
            "Feel: straight.\n"
            "Meter: inferred 4/4.\n"
            "Position analysis: Analyzed the Kick, Snare, Hi-Hat/Cymbal, and the Toms hits.\n"
            "Step 3: Cycle lengths and meters match, but the feel differs and the onset positions per voice are not aligned (added/moved hits and tom presence mismatch).\n"
            "No, these are not the same exact drum set patterns."
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
        wav_same1, mt1 = read_audio_bytes_and_mime(EX_SAME_A)
        wav_same2, mt2 = read_audio_bytes_and_mime(EX_SAME_B)
        wav_diff2, mt3 = read_audio_bytes_and_mime(EX_SAME_A)
        wav_diff3, mt4 = read_audio_bytes_and_mime(EX_DIFF_C)
        history = [
            _example_same_user(wav_same1, mt1, wav_same2, mt2),
            _example_diff_user(wav_diff2, mt3, wav_diff3, mt4),
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
    logging.info(f"=== RUN START (Rhythm Matching • CHAT+SysInst • {mode} • {group}) ===")
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

    # Fixed, ordered stimuli for the selected group
    question_stims = stimuli_pairs_group(group)
    random.seed(seed)  # ensures reproducible shuffle per run/seed
    random.shuffle(question_stims)

    print(
        f"\n--- Task: Rhythm Matching — CHAT+SysInst {mode} • {group} | model={model_name} | temp=1.0 | seed={seed} ---\n"
    )
    logging.info(f"\n--- Task: Rhythm Matching — CHAT+SysInst {mode} • {group} ---\n")

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
                PartText("""Reply with exactly ONE of the following lines:
"Yes, these are the same exact drum set patterns."
OR
"No, these are not the same exact drum set patterns."
"""),
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

        # Ground truth via robust rhythm id ('Beat_')
        rid1 = rhythm_id(f1)
        rid2 = rhythm_id(f2)
        expected = YES_CANON if rid1 == rid2 else NO_CANON

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
