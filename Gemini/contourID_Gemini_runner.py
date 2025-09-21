# contourID_Gemini_CHAT_SysInst_master.py
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

# Canonical answer strings (EXACT, including capitalization / punctuation)
A_CANON = 'A. Arch (ascending and then descending)'
B_CANON = 'B. Inverted Arch (descending and then ascending)'
C_CANON = 'C. Ascending (pitch raises over time)'
D_CANON = 'D. Descending (pitch falls over time)'

# Robust patterns to find the final choice anywhere in the completion (prefer LAST occurrence)
A_PAT = re.compile(r'(?i)\bA\.\s*Arch\s*\(ascending\s+and\s+then\s+descending\)')
B_PAT = re.compile(r'(?i)\bB\.\s*Inverted\s+Arch\s*\(descending\s+and\s+then\s+ascending\)')
C_PAT = re.compile(r'(?i)\bC\.\s*Ascending\s*\(pitch\s+raises\s+over\s+time\)')
D_PAT = re.compile(r'(?i)\bD\.\s*Descending\s*\(pitch\s+falls\s+over\s+time\)')

# =============================
# System instructions (VERBATIM from your new text blocks)
# =============================
SYSINSTR_PLAIN = """You are a participant in a psychological experiment on music perception. 
 In each question, you will be given:
    1. A brief instruction about the specific listening task.
    2. One audio example to listen to. 

Your task is to determine the overall contour—the overall pattern or "shape" of the notes you hear. An Arch shape means that first the notes ascend to a higher pitch, and then they descend to a lower pitch. Thus, an inverted arch is the opposite, whereby the notes first drop in pitch and then rise up again. Ascending simply means that the pitches you hear go from lower to higher, and descending means they go from higher to lower.
Valid responses are:
"A. Arch (ascending and then descending)"
"B. Inverted Arch (descending and then ascending)"
"C. Ascending (pitch raises over time)"
"D. Descending (pitch falls over time)"

Before you begin the task, I will provide you with four examples of excerpts representing each of the possible multiple choice responses 
so that you better understand the task. After examining the examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't 
understand the task.

Please provide no additional commentary beyond the short answers previously mentioned. 
"""

SYSINSTR_COT = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to.

Your task is to determine the overall contour — the overall pattern or "shape" of the notes you hear. 
An Arch shape means that first the notes ascend to a higher pitch, and then they descend to a lower pitch. 
Thus, an inverted arch is the opposite, whereby the notes first drop in pitch and then rise up again. 
Ascending simply means that the pitches you hear go from lower to higher, and descending means they go from higher to lower.

Definitions and constraints:
- Ignore dynamics, timbre, articulation, and small embellishments; judge the global trend.
- Use coarse pitch motion, not exact notes
- Category definitions:
A. Arch: generally rises to a peak and then falls.
B. Inverted Arch: generally falls to a low point and then rises.
C. Ascending: generally rises overall from start to end.
D. Descending: generally falls overall from start to end.

Valid responses are exactly:
"A. Arch (ascending and then descending)"
"B. Inverted Arch (descending and then ascending)"
"C. Ascending (pitch raises over time)"
"D. Descending (pitch falls over time)"

Before you begin the task, I will provide you with four examples of excerpts representing each of the possible multiple choice responses 
so that you better understand the task. After examining the examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't 
understand the task.

After any reasoning, end with exactly one line:
A. Arch (ascending and then descending)
OR
B. Inverted Arch (descending and then ascending)
OR
C. Ascending (pitch raises over time)
OR
D. Descending (pitch falls over time)"""

# --- COT per-trial prompt (VERBATIM) ---
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
    Example: contourID_G25Pro_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    tag = _model_tag_for(model_name)
    return f"contourID_{tag}_CHAT_{mode}_{group}_seed{seed}.log"

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

# =============================
# Stimuli lists (single excerpt per trial)
# =============================
def stimuli_list_group(group: str) -> List[Dict[str, str]]:
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

# Few-shot example file paths (VERBATIM IDs)
EX_ARCH     = _p("Bbmin_Scale_Arch_Bass.wav")
EX_INVARCH  = _p("Fm_Scale_InvArch.wav")
EX_ASC      = _p("Amin_Arp_Asc_Piano.wav")
EX_DESC     = _p("Gbmin_Arp_Desc_Piano.wav")

# =============================
# Parsing / evaluation helpers
# =============================
def parse_final_choice(text: str) -> str:
    """Return the canonical A/B/C/D string, or '' if not found. Prefer the LAST occurrence."""
    if not text:
        return ""
    last = None
    for m in A_PAT.finditer(text):
        last = ("A", m.end())
    for m in B_PAT.finditer(text):
        if last is None or m.end() > last[1]:
            last = ("B", m.end())
    for m in C_PAT.finditer(text):
        if last is None or m.end() > last[1]:
            last = ("C", m.end())
    for m in D_PAT.finditer(text):
        if last is None or m.end() > last[1]:
            last = ("D", m.end())
    if not last:
        return ""
    label = last[0]
    return { "A": A_CANON, "B": B_CANON, "C": C_CANON, "D": D_CANON }[label]

def contour_label_from_filename(path: str) -> str:
    """Infer expected label from filename (case-insensitive); check InvArch before Arch."""
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
def build_fewshot_messages_COT() -> List[types.Content]:
    """
    Four in-context examples with brief CoT and strict final line (assistant messages included).
    Ex1: A. Arch
    Ex2: B. Inverted Arch
    Ex3: C. Ascending
    Ex4: D. Descending
    """
    # Load audio bytes (no trimming / no downmix)
    wav_arch,   mt_arch   = read_audio_bytes_and_mime(EX_ARCH)
    wav_inv,    mt_inv    = read_audio_bytes_and_mime(EX_INVARCH)
    wav_asc,    mt_asc    = read_audio_bytes_and_mime(EX_ASC)
    wav_desc,   mt_desc   = read_audio_bytes_and_mime(EX_DESC)

    # Example 1 — Arch (A)
    ex1_user = types.Content(
        role="user",
        parts=[PartText(COT_PROMPT_TEMPLATE), PartBytes(wav_arch, mt_arch)]
    )
    ex1_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Single melodic line rises to a clear apex around the midpoint, then falls.\n"
            "Step 2: This global trend is rise-then-fall.\n"
            "A. Arch (ascending and then descending)"
        )]
    )

    # Example 2 — Inverted Arch (B)
    ex2_user = types.Content(
        role="user",
        parts=[PartText(COT_PROMPT_TEMPLATE), PartBytes(wav_inv, mt_inv)]
    )
    ex2_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Single melodic line descends early to a low point, then rises toward the end.\n"
            "Step 2: This global trend is fall-then-rise.\n"
            "B. Inverted Arch (descending and then ascending)"
        )]
    )

    # Example 3 — Ascending (C)
    ex3_user = types.Content(
        role="user",
        parts=[PartText(COT_PROMPT_TEMPLATE), PartBytes(wav_asc, mt_asc)]
    )
    ex3_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Single melodic line with a clear overall upward movement from start to finish.\n"
            "Step 2: This matches an overall ascending contour.\n"
            "C. Ascending (pitch raises over time)"
        )]
    )

    # Example 4 — Descending (D)
    ex4_user = types.Content(
        role="user",
        parts=[PartText(COT_PROMPT_TEMPLATE), PartBytes(wav_desc, mt_desc)]
    )
    ex4_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Single melodic line moves downward overall from beginning to end.\n"
            "Step 2: This matches a descending contour.\n"
            "D. Descending (pitch falls over time)"
        )]
    )

    return [ex1_user, ex1_model, ex2_user, ex2_model, ex3_user, ex3_model, ex4_user, ex4_model]

def _example_user_plain(text: str, wav: bytes, mt: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            PartText(text),
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
        # SYSINST: four user-only example turns with audio, no assistant interjections
        wav_arch, mt_arch   = read_audio_bytes_and_mime(EX_ARCH)
        wav_inv,  mt_inv    = read_audio_bytes_and_mime(EX_INVARCH)
        wav_asc,  mt_asc    = read_audio_bytes_and_mime(EX_ASC)
        wav_desc, mt_desc   = read_audio_bytes_and_mime(EX_DESC)
        history = [
            _example_user_plain(
                "Example 1: The following audio example represents the contour 'A. Arch (ascending and then descending)'. Listen carefully:",
                wav_arch, mt_arch),
            _example_user_plain(
                "Example 2: The following audio example represents the contour 'B. Inverted Arch (descending and then ascending)'. Listen carefully:",
                wav_inv, mt_inv),
            _example_user_plain(
                "Example 3: The following audio example represents the contour 'C. Ascending (pitch raises over time)'. Listen carefully:",
                wav_asc, mt_asc),
            _example_user_plain(
                "Example 4: The following audio example represents the contour 'D. Descending (pitch falls over time)'. Listen carefully:",
                wav_desc, mt_desc),
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
    logging.info(f"=== RUN START (Contour ID • CHAT+SysInst • {mode} • {group}) ===")
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

    # Fixed, ordered stimuli for the selected group, then shuffle per seed
    question_stims = stimuli_list_group(group)
    random.seed(seed)
    random.shuffle(question_stims)

    print(
        f"\n--- Task: Contour Identification — CHAT+SysInst {mode} • {group} | model={model_name} | temp=1.0 | seed={seed} ---\n"
    )
    logging.info(f"\n--- Task: Contour Identification — CHAT+SysInst {mode} • {group} ---\n")

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

        f = q["file"]
        logging.info(f"Stimulus: file={f}")

        # Load audio (original bytes, no trimming, no downmix)
        wav, mt = read_audio_bytes_and_mime(f)

        # Build trial prompt
        if mode == "SYSINST":
            user_parts = [
                PartText("You are a participant in a psychological experiment on music perception. Please decide which option best describes the overall shape of the scale in the audio example:"),
                PartText('A. Arch (ascending and then descending)\nB. Inverted Arch (descending and then ascending)\nC. Ascending (pitch raises over time)\nD. Descending (pitch falls over time)'),
                PartText("This is the audio example. Listen carefully now:"),
                PartBytes(wav, mt),
                PartText('Now choose which option best represents the overall contour of the audio example. Answer exactly one of the four strings (and nothing else):\n'
                         '"A. Arch (ascending and then descending)"\n"B. Inverted Arch (descending and then ascending)"\n"C. Ascending (pitch raises over time)"\n"D. Descending (pitch falls over time)"'),
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

        # Parse decision (prefer last occurrence among A/B/C/D)
        model_answer = parse_final_choice(completion)
        if not model_answer:
            print("Evaluation: Failed. Could not parse a valid final choice.")
            logging.error("Parse Error: missing/malformed final choice line.")
            continue

        logging.info(f"Parsed Final Answer: {model_answer}")

        # Ground truth via filename keywords
        expected = contour_label_from_filename(f)

        if model_answer == expected:
            correct += 1
            print("Evaluation: Correct!")
            logging.info("Evaluation: Correct")
        else:
            if expected == "Unknown":
                print("Evaluation: Unknown ground truth for this filename.")
                logging.warning("Evaluation: Unknown ground truth (filename does not contain Arch/InvArch/Asc/Desc).")
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
