# chord_quality_Gemini_CHAT_SysInst_master.py
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
A_CANON = "A. Major"
B_CANON = "B. Minor"

A_PAT = re.compile(r'(?i)\bA\.\s*Major\b')
B_PAT = re.compile(r'(?i)\bB\.\s*Minor\b')

# =============================
# System instructions (VERBATIM)
# =============================
SYSINSTR_PLAIN = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to.

Your task is to decide if a chord is Major or Minor. First, you will hear the chord itself, and then you will hear the 
individual notes of the chord played one at a time. You can think of the differences between Major and Minor in terms of 
mood. For those with a Western enculturation:
Major chords generally sound bright, happy, or triumphant.
Minor chords often sound more somber, sad, or mysterious.

Valid responses are:
"A. Major"
"B. Minor"

Before you begin the task, I will provide you with examples of major and minor chords so that you better understand the task.
After examining the examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't 
understand the task."""

SYSINSTR_COT = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to.

Your task is to decide whether the chord is MAJOR or MINOR. You will first hear the chord, then the individual notes arpeggiated.

Definitions and constraints:
- Judge the chord’s QUALITY (major vs. minor triad), not the key or root name.
- Ignore inversion (bass note), voicing/register, added doublings, dynamics, reverb/effects, and recording quality.
- Assume that the lowest note is the root
- Diagnostic cue: the interval between two chord tones forming the “third.”
  • Major chord: contains a MAJOR third within the set of chord tones.
  • Minor chord: contains a MINOR third within the set of chord tones.

Valid responses are exactly:
"A. Major"
"B. Minor"

Before you begin the task, I will provide you with one example of a major chord and one example of a minor chord so you 
better understand the task. After examining the examples, please respond with "Yes, I understand." if you understand 
the task or "No, I don't understand." if you don't understand the task.

After any reasoning, end with exactly one line:
A. Major
OR
B. Minor"""

# --- COT prompt (VERBATIM) ---
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
    Example: chord_quality_G25Pro_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    tag = _model_tag_for(model_name)
    return f"chord_quality_{tag}_CHAT_{mode}_{group}_seed{seed}_run2.log"

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

def stimuli_files_group(group: str) -> List[str]:
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
    return [_ppath(p) for p in files]

# Few-shot example file paths (VERBATIM IDs)
EX_MAJOR = _p("Abmajor_Piano_120.wav")   # Major example
EX_MINOR = _p("Aminor_Guitar_120.wav")    # Minor example

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

def chord_quality_from_filename(path: str) -> str:
    """
    Heuristic ground truth from filename:
    - if 'major' in name → A. Major
    - if 'minor' in name → B. Minor
    """
    name = os.path.basename(path).lower()
    if "major" in name:
        return A_CANON
    if "minor" in name:
        return B_CANON
    return ""  # unknown

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
# (EXACT, VERBATIM few-shot content you provided)
# =============================
def build_fewshot_messages_COT() -> List[types.Content]:
    """
    Two in-context examples with brief CoT and strict final line (assistant messages included).
    Ex1: Major  -> ends with: A. Major
    Ex2: Minor  -> ends with: B. Minor
    """
    # Load audio bytes
    wav_maj, mt_maj = read_audio_bytes_and_mime(EX_MAJOR)
    wav_min, mt_min = read_audio_bytes_and_mime(EX_MINOR)

    id_maj = "Abmajor_Piano_120"
    id_min = "Aminor_Guitar_120"

    # Example 1 — Major (A)
    ex1_user = types.Content(
        role="user",
        parts=[
            PartText(COT_PROMPT_TEMPLATE.format(id1=id_maj)),
            PartBytes(wav_maj, mt_maj),
        ],
    )
    ex1_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Hear the block chord then the arpeggiation; treat inversion/voicing as irrelevant.\n"
            "Step 2: Among the chord tones there is a MAJOR third span, consistent with a major triad.\n"
            "A. Major"
        )]
    )

    # Example 2 — Minor (B)
    ex2_user = types.Content(
        role="user",
        parts=[
            PartText(COT_PROMPT_TEMPLATE.format(id1=id_min)),
            PartBytes(wav_min, mt_min),
        ],
    )
    ex2_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Listen to the chord and then the arpeggiation; ignore register/voicing.\n"
            "Step 2: The chord tones include a MINOR third, indicating a minor triad.\n"
            "B. Minor"
        )]
    )

    return [ex1_user, ex1_model, ex2_user, ex2_model]

def _example_major_user(wav: bytes, mt: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            PartText("Example 1: The following audio example is a Major chord. Listen carefully:"),
            PartText("Audio example:"),
            PartBytes(wav, mt),
        ]
    )

def _example_minor_user(wav: bytes, mt: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            PartText("Example 2: The following audio example is a Minor chord. Listen carefully:"),
            PartText("Audio example:"),
            PartBytes(wav, mt),
        ]
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
        wav_maj, mt_maj = read_audio_bytes_and_mime(EX_MAJOR)
        wav_min, mt_min = read_audio_bytes_and_mime(EX_MINOR)
        history = [
            _example_major_user(wav_maj, mt_maj),
            _example_minor_user(wav_min, mt_min),
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
    logging.info(f"=== RUN START (Chord Quality • CHAT+SysInst • {mode} • {group}) ===")
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
    stim_files = stimuli_files_group(group)
    random.seed(seed)  # ensures reproducible shuffle per run/seed
    random.shuffle(stim_files)

    print(
        f"\n--- Task: Chord Quality (Major vs Minor) — CHAT+SysInst {mode} • {group} | model={model_name} | temp=1.0 | seed={seed} ---\n"
    )
    logging.info(f"\n--- Task: Chord Quality (Major vs Minor) — CHAT+SysInst {mode} • {group} ---\n")

    correct = 0
    total = len(stim_files)

    # Per-trial decoding (recreate config to stamp the seed for reproducibility logging)
    per_trial_cfg = gemini_decoding_config(
        temperature=1.0, seed=seed, max_tokens=MAX_NEW_TOKENS, system_instruction=sysinstr_text
    )
    # Update the live chat config if supported (safe to call; else ignored)
    try:
        chat.config = per_trial_cfg  # type: ignore
    except Exception:
        pass

    for idx, f in enumerate(stim_files, start=1):
        print(f"\n--- Question {idx} ---\n")
        logging.info(f"\n--- Question {idx} ---\n")
        logging.info(f"Stimulus: file={f}")

        # Load audio (original bytes, no trimming, no downmix)
        wav, mt = read_audio_bytes_and_mime(f)

        # Build trial prompt (text + 1 audio part)
        if mode == "SYSINST":
            user_parts = [
                PartText("Here is the audio excerpt."),
                PartBytes(wav, mt),
                PartText('Reply with exactly ONE of the following lines:\n'
                         'A. Major\n'
                         'OR\n'
                         'B. Minor'),
            ]
        else:
            user_parts = [
                PartText(COT_PROMPT_TEMPLATE),
                PartBytes(wav, mt),
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

        # Ground truth via filename heuristic (contains 'major' or 'minor')
        expected = chord_quality_from_filename(f)
        logging.info(f"Expected Final Answer: {expected}")

        if expected and (model_answer == expected):
            correct += 1
            print("Evaluation: Correct!")
            logging.info("Evaluation: Correct")
        elif expected and (model_answer in (A_CANON, B_CANON)):
            print("Evaluation: Incorrect.")
            logging.info(f"Evaluation: Incorrect (expected={expected})")
        else:
            print("Evaluation: Unexpected/Unknown stimulus label.")
            logging.info(f"Evaluation: Unknown mapping for file={f}")

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
