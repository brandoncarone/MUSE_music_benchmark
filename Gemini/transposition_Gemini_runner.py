# transposition_Gemini_CHAT_SysInst_master.py
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

# Canonical answer strings and robust patterns (VERBATIM from Qwen runner)
YES_CANON = 'Yes, these are the same melody.'
NO_CANON = 'No, these are not the same melody.'

YES_PAT = re.compile(r'(?i)\byes,\s*these\s+are\s+the\s+same\s+melody\.')
NO_PAT = re.compile(r'(?i)\bno,\s*these\s+are\s+not\s+the\s+same\s+melody\.')

# =============================
# System instructions (VERBATIM)
# =============================
SYSINSTR_PLAIN = """You are a participant in a psychological experiment on music perception. In each question, you will be given:
1. A brief instruction about the specific listening task.
2. Two audio examples to listen to.
Your task is to decide whether the two music excerpts represent the same melody, regardless of the musical key that they are played in. In other words, even if one sounds higher or lower than the other, they still might represent the same melody.

Valid responses are:
"Yes, these are the same melody." or
"No, these are not the same melody."

Before you begin the task, I will provide you with examples of two excerpts representing the same melody, as well as two excerpts representing different melodies so that you better understand the task. After examining the examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't understand the task.

Please provide no additional commentary beyond the short answers previously mentioned. """

SYSINSTR_COT = """You are a participant in a psychological experiment on music perception. In each question, you will be given:
1. A brief instruction about the specific listening task.
2. Two audio examples to listen to.

Your task is to decide whether the two music excerpts represent the same melody, regardless of the musical key that they are played in. In other words, even if one sounds higher or lower than the other, they still might represent the same melody.

Definitions and constraints:
- Transposition equivalence: the two melodies have the same number of notes and the same sequence of pitch INTERVALS between successive notes (including 0 for repeated notes).
- Ignore absolute key/register, starting pitch, and tempo. Small timing variations are acceptable. If the rhythmic patterns are drastically different (e.g., note insertions/deletions or re-ordered phrases), they are most likely NOT the same melody.
- Treat repeated notes as separate events and include 0 in the interval sequence when a note repeats.
- If there are leading/trailing silences, ignore them.

Valid responses:
"Yes, these are the same melody." or
"No, these are not the same melody."

Before you begin the task, I will provide you with examples of two excerpts representing the same melody, as well as two excerpts representing different melodies so that you better understand the task. After examining the examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't understand the task.

After any reasoning, end with exactly one line:
Yes, these are the same melody.
OR
No, these are not the same melody."""

# --- COT prompt (VERBATIM) ---
COT_PROMPT_TEMPLATE = """Analyze the two audio files ('{id1}' and '{id2}') to determine if they are the SAME melody up to TRANSPOSITION.

Step 1: For each audio, identify the sequence of pitched notes (monophonic). Treat repeated notes as repeated events.

Step 2: Compute the interval sequence in semitones for each melody:
Δp[i] = pitch[i+1] − pitch[i]  (include 0 when the next note repeats)

Step 3: Decide transposition equivalence
They are considered the same melody under transposition if:
- They have the SAME number of notes, AND
- Their interval sequences (Δp) match element-by-element.

Notes:
- Ignore absolute key/register and tempo differences.
- Small timing deviations are fine; large rhythmic re-organization suggests different melodies.

Step 4: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
Yes, these are the same melody.
OR
No, these are not the same melody.
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
    Example: transposition_G25Pro_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    tag = _model_tag_for(model_name)
    return f"transposition_{tag}_CHAT_{mode}_{group}_seed{seed}.log"

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
    """Return the 10-pair list for GroupA or GroupB (order preserved)."""
    assert group in {"GroupA", "GroupB"}
    if group == "GroupA":
        pairs = [
            ("stimuli/M1_EbMaj_90.wav", "stimuli/M1_GMaj_90.wav"),
            ("stimuli/M1_EbMaj_90.wav", "stimuli/M2_Abm_155_3_4.wav"),
            ("stimuli/M2_Abm_155_3_4.wav", "stimuli/M2_Fm_155_3_4.wav"),
            ("stimuli/M2_Fm_155_3_4.wav", "stimuli/M3_DbMaj_78.wav"),
            ("stimuli/M3_DbMaj_78.wav", "stimuli/M3_GbMaj_78.wav"),
            ("stimuli/M8_FMaj_95_Piano.wav", "stimuli/M9_Em_200_3_4_Piano.wav"),
            ("stimuli/M9_Em_200_3_4_Piano.wav", "stimuli/M9_Gm_200_3_4_Piano.wav"),
            ("stimuli/M9_Gm_200_3_4_Piano.wav", "stimuli/M10_Dbm_165_3_4.wav"),
            ("stimuli/M10_Dbm_165_3_4.wav", "stimuli/M10_Fm_165_3_4.wav"),
            ("stimuli/M10_Fm_165_3_4.wav", "stimuli/M6_Bbm_120_Piano.wav"),
        ]
    else:
        pairs = [
            ("stimuli/M3_GbMaj_78.wav", "stimuli/M4_AMaj_130.wav"),
            ("stimuli/M4_AMaj_130.wav", "stimuli/M4_EMaj_130.wav"),
            ("stimuli/M4_EMaj_130.wav", "stimuli/M5_Bm_100.wav"),
            ("stimuli/M5_Bm_100.wav", "stimuli/M5_Dm_100.wav"),
            ("stimuli/M5_Dm_100.wav", "stimuli/M1_GMaj_90.wav"),
            ("stimuli/M6_Cm_120_Piano.wav", "stimuli/M6_Bbm_120_Piano.wav"),
            ("stimuli/M6_Cm_120_Piano.wav", "stimuli/M7_CMaj_140_Piano.wav"),
            ("stimuli/M7_CMaj_140_Piano.wav", "stimuli/M7_DbMaj_140_Piano.wav"),
            ("stimuli/M7_DbMaj_140_Piano.wav", "stimuli/M8_AbMaj_95_Piano.wav"),
            ("stimuli/M8_AbMaj_95_Piano.wav", "stimuli/M8_FMaj_95_Piano.wav"),
        ]
    return [{"file1": _ppath(a), "file2": _ppath(b)} for a, b in pairs]

# Few-shot example file paths (VERBATIM IDs)
EX_SAME_A = _p("M11_CMaj_180_Piano.wav")
EX_SAME_B = _p("M11_EMaj_180_Piano.wav")
EX_DIFF_B = _p("M11_EMaj_180_Piano.wav")
EX_DIFF_C = _p("M12_FMaj_155_Piano.wav")

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

def melody_id(fname: str) -> str:
    base = os.path.basename(fname)
    m = re.match(r"^(M\d+)_", base)
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
            PartText("Example 1: The following two audio excerpts represent the same melody, played in different keys."),
            PartBytes(wav_a, mt_a),
            PartText("Audio example 2:"),
            PartBytes(wav_b, mt_b),
        ]
    )

def _example_diff_user(wav_b: bytes, mt_b: str, wav_c: bytes, mt_c: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            PartText("Example 2: The following two audio excerpts represent different melodies."),
            PartBytes(wav_b, mt_b),
            PartText("Audio example 2:"),
            PartBytes(wav_c, mt_c),
        ]
    )

def build_fewshot_messages_COT() -> List[types.Content]:
    """
    Two in-context examples with brief CoT and strict final line (assistant messages included).
    Ex1: SAME → ends with Yes, these are the same melody.
    Ex2: DIFFERENT → ends with No, these are not the same melody.
    """
    wav_same1, mt1 = read_audio_bytes_and_mime(EX_SAME_A)
    wav_same2, mt2 = read_audio_bytes_and_mime(EX_SAME_B)
    wav_diff2, mt3 = read_audio_bytes_and_mime(EX_DIFF_C)  # different melody

    id_same1 = "M11_CMaj_180_Piano"
    id_same2 = "M11_EMaj_180_Piano"
    id_diff2 = "M12_FMaj_155_Piano"

    ex1_user = types.Content(
        role="user",
        parts=[
            PartText(COT_PROMPT_TEMPLATE.format(id1=id_same1, id2=id_same2)),
            PartBytes(wav_same1, mt1),
            PartBytes(wav_same2, mt2),
        ]
    )
    ex1_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Identify notes for both clips; both sequences have the same length.\n"
            "Step 2: The successive semitone intervals match element-by-element (including 0 for repeats).\n"
            "Step 3: Therefore they are the same melody up to transposition.\n"
            "Yes, these are the same melody."
        )]
    )

    ex2_user = types.Content(
        role="user",
        parts=[
            PartText(COT_PROMPT_TEMPLATE.format(id1=id_same2, id2=id_diff2)),
            PartBytes(wav_same2, mt2),
            PartBytes(wav_diff2, mt3),
        ]
    )
    ex2_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Identify notes; sequences differ in structure.\n"
            "Step 2: Interval sequences do not align; there are insertions/deletions.\n"
            "Step 3: Thus not the same melody up to transposition.\n"
            "No, these are not the same melody."
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
        wav_diff2, mt3 = read_audio_bytes_and_mime(EX_DIFF_B)
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
    logging.info(f"=== RUN START (Transposition • CHAT+SysInst • {mode} • {group}) ===")
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
        f"\n--- Task: Melody Matching (Transposition) — CHAT+SysInst {mode} • {group} | model={model_name} | temp=1.0 | seed={seed} ---\n"
    )
    logging.info(f"\n--- Task: Melody Matching (Transposition) — CHAT+SysInst {mode} • {group} ---\n")

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
"Yes, these are the same melody."
OR
"No, these are not the same melody."
"""),
            ]
        else:
            id1 = _file_id_no_ext(f1)
            id2 = _file_id_no_ext(f2)
            user_parts = [
                PartText(COT_PROMPT_TEMPLATE.format(id1=id1, id2=id2)),
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

        # Ground truth via robust melody id ('M#')
        mid1 = melody_id(f1)
        mid2 = melody_id(f2)
        expected = YES_CANON if mid1 == mid2 else NO_CANON

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
