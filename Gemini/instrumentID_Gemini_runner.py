# instrument_id_Gemini_CHAT_SysInst_master.py
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

# Canonical final answer strings (exact)
A_CANON = 'A. Piano'
B_CANON = 'B. Guitar'
C_CANON = 'C. Bass'
D_CANON = 'D. Drums'

# Robust patterns to detect answers (prefer LAST occurrence)
A_PAT = re.compile(r'(?i)\bA\.\s*Piano\b')
B_PAT = re.compile(r'(?i)\bB\.\s*Guitar\b')
C_PAT = re.compile(r'(?i)\bC\.\s*Bass\b')
D_PAT = re.compile(r'(?i)\bD\.\s*Drums\b')

# =============================
# System instructions (VERBATIM from your new text blocks)
# =============================
SYSINSTR_PLAIN = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to. 

Your task is to listen to different music excerpts and identify the musical instrument being played.
Valid responses are:
"A. Piano"
"B. Guitar"
"C. Bass"
"D. Drums"

Before beginning the task, I will provide you with audio examples of each of the instruments, including piano, guitar, bass, and drums,
so that you better understand the task. After examining the examples, please respond with "Yes, I understand." if you 
understand the task or "No, I don't understand." if you don't understand the task.

Please provide no additional commentary beyond the short answers previously mentioned.
"""

SYSINSTR_COT = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to.

Your task is to listen to the excerpt and identify the musical instrument being played.

Definitions and constraints:
- Possible instruments (and response options) are limited to:
  A. Piano
  B. Guitar
  C. Bass
  D. Drums
- Focus on overall timbre, envelope, register, and articulation.
- Ignore reverb, effects, and recording quality.

Valid responses are exactly:
"A. Piano"
"B. Guitar"
"C. Bass"
"D. Drums"

Before beginning the task, I will provide you with audio examples of each of the instruments, including piano, guitar, bass, and drums, so that you better understand the task. After examining the 
examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't understand the task.

After any reasoning, end with exactly one line:
A. Piano
OR
B. Guitar
OR
C. Bass
OR
D. Drums"""

# --- COT per-trial prompt (VERBATIM) ---
COT_PROMPT_TEMPLATE = """Analyze the music excerpt and identify which INSTRUMENT is being played.

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
    Example: instrumentID_G25Pro_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    tag = _model_tag_for(model_name)
    return f"instrumentID_{tag}_CHAT_{mode}_{group}_seed{seed}.log"

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
# Parsing / evaluation helpers
# =============================
def parse_final_choice(text: str) -> str:
    """Return canonical A/B/C/D string, or '' if not found. Prefer the LAST occurrence."""
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
    return {"A": A_CANON, "B": B_CANON, "C": C_CANON, "D": D_CANON}[last[0]]

def instrument_label_from_filename(path: str) -> str:
    """
    Ground truth mapping per your rule:
      - 'Beat'  -> D. Drums
      - 'Piano' -> A. Piano
      - 'Bass'  -> C. Bass
      - 'Guitar' OR no instrument token -> B. Guitar
    (case-insensitive)
    """
    name = os.path.basename(path).lower()
    if "beat" in name:
        return D_CANON
    if "piano" in name:
        return A_CANON
    if "bass" in name:
        return C_CANON
    # Default/fallback == Guitar
    return B_CANON

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
# Example audio paths (resolved with _p)
EX_GUITAR = _p("Emajor_Guitar_120.wav")
EX_PIANO  = _p("GbMaj_Arp_Desc_Piano.wav")
EX_BASS   = _p("AbMaj_Arp_Asc_Bass.wav")
EX_DRUMS  = _p("Beat_11_140.wav")

def build_fewshot_messages_COT() -> List[types.Content]:
    """
    Four in-context examples with brief CoT and strict final line (assistant messages included).
    Ex1: Guitar  -> ends with B. Guitar
    Ex2: Piano   -> ends with A. Piano
    Ex3: Bass    -> ends with C. Bass
    Ex4: Drums   -> ends with D. Drums
    """
    wav_gtr, mt_gtr = read_audio_bytes_and_mime(EX_GUITAR)
    wav_pno, mt_pno = read_audio_bytes_and_mime(EX_PIANO)
    wav_bas, mt_bas = read_audio_bytes_and_mime(EX_BASS)
    wav_drm, mt_drm = read_audio_bytes_and_mime(EX_DRUMS)

    # Example 1 — Piano (A)
    ex1_user = types.Content(
        role="user",
        parts=[PartText(COT_PROMPT_TEMPLATE), PartBytes(wav_pno, mt_pno)]
    )
    ex1_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Percussive key-strike onset with rich harmonic resonance/decay and clear note separations.\n"
            "Step 2: Sustained resonant body aligns with piano timbre.\n"
            "A. Piano"
        )]
    )

    # Example 2 — Guitar (B)
    ex2_user = types.Content(
        role="user",
        parts=[PartText(COT_PROMPT_TEMPLATE), PartBytes(wav_gtr, mt_gtr)]
    )
    ex2_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Bright plucked-string timbre with fretted articulation; clear pick attack; mid register.\n"
            "Step 2: Matches a guitar rather than piano (hammered keys), bass (lower register), or drums (non-pitched percussive hits).\n"
            "B. Guitar"
        )]
    )

    # Example 3 — Bass (C)
    ex3_user = types.Content(
        role="user",
        parts=[PartText(COT_PROMPT_TEMPLATE), PartBytes(wav_bas, mt_bas)]
    )
    ex3_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Low-register plucked string with strong fundamental and longer sustain; subdued upper harmonics.\n"
            "Step 2: This profile fits an electric/acoustic bass rather than guitar or piano.\n"
            "C. Bass"
        )]
    )

    # Example 4 — Drums (D)
    ex4_user = types.Content(
        role="user",
        parts=[PartText(COT_PROMPT_TEMPLATE), PartBytes(wav_drm, mt_drm)]
    )
    ex4_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Broadband percussive hits (kick/snare/cymbal) with no sustained pitched notes.\n"
            "Step 2: Consistent with a drum kit rather than pitched string/keyboard instruments.\n"
            "D. Drums"
        )]
    )

    return [ex1_user, ex1_model, ex2_user, ex2_model, ex3_user, ex3_model, ex4_user, ex4_model]

def _example_user_plain(label: str, wav: bytes, mt: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            PartText(f"Example: The following audio example features {label}. Listen carefully:"),
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
        wav_gtr, mt_gtr = read_audio_bytes_and_mime(EX_GUITAR)
        wav_pno, mt_pno = read_audio_bytes_and_mime(EX_PIANO)
        wav_bas, mt_bas = read_audio_bytes_and_mime(EX_BASS)
        wav_drm, mt_drm = read_audio_bytes_and_mime(EX_DRUMS)
        history = [
            _example_user_plain("a piano", wav_pno, mt_pno),
            _example_user_plain("a guitar", wav_gtr, mt_gtr),
            _example_user_plain("a bass",   wav_bas, mt_bas),
            _example_user_plain("drums",    wav_drm, mt_drm),
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
    logging.info(f"=== RUN START (Instrument ID • CHAT+SysInst • {mode} • {group}) ===")
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
    random.seed(seed)  # ensures reproducible shuffle per run/seed
    random.shuffle(question_stims)

    print(
        f"\n--- Task: Instrument Identification — CHAT+SysInst {mode} • {group} | model={model_name} | temp=1.0 | seed={seed} ---\n"
    )
    logging.info(f"\n--- Task: Instrument Identification — CHAT+SysInst {mode} • {group} ---\n")

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
                PartText("You are a participant in a psychological experiment on music perception. "
                         "Your task is to listen to an audio example and identify the musical instrument being played."),
                PartText('Valid responses:\n"A. Piano"\n"B. Guitar"\n"C. Bass"\n"D. Drums"'),
                PartText("Listen carefully to the following audio:"),
                PartBytes(wav, mt),
                PartText('Now answer by stating exactly one of the four strings (and nothing else):\n'
                         '"A. Piano"\n"B. Guitar"\n"C. Bass"\n"D. Drums"'),
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

        # Ground truth from filename tokens
        expected = instrument_label_from_filename(f)

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
