# key_modulation_Gemini_CHAT_SysInst_master.py
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
A_CANON = "A. Yes, a key modulation occurs."
B_CANON = "B. No, the key stays the same."

# Accept the canonical lines (with or without trailing period), case-insensitive, per-line.
A_PAT = re.compile(r'(?im)^\s*A\.\s*Yes,\s*a\s*key\s*modulation\s*occurs\.?\s*$', re.UNICODE)
B_PAT = re.compile(r'(?im)^\s*B\.\s*No,\s*the\s*key\s*stays\s*the\s*same\.?\s*$', re.UNICODE)
B_ALT = re.compile(r'(?im)^\s*(?:B\.\s*)?No,?\s*(?:a\s*)?key\s*modulation\s*(?:does\s*not|doesn\'t)\s*occur\.?\s*$', re.UNICODE)


# =============================
# System instructions (VERBATIM)
# =============================
SYSINSTR_PLAIN = """You are a participant in a psychological experiment on music perception. 
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to. 

Your task is to decide if a "key change" (or modulation) occurs in a musical excerpt. Think of a song's key as its 
"home base." A modulation is a dramatic shift to a new home base, which can feel like a "lift" or change in the song's 
“home base.”

Valid responses are:
"A. Yes, a key modulation occurs."
"B. No, the key stays the same."

Before you begin the task, I will provide you with examples of an excerpt containing a key modulation, as well as an 
excerpt with no key modulation so that you better understand the task. After examining the examples, please respond 
with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't understand the task."""

SYSINSTR_COT = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to.

Your task is to decide whether a KEY MODULATION (key change) occurs within the excerpt.

Definitions and constraints:
- Treat a key as the stable “home base” or tonal center. A modulation is a shift to a NEW stable tonal center.
- Evidence for modulation can include: a clear cadence into the new key, a new leading tone consistent with the new key, and sustained harmony supporting the new center.
- Brief chromatic chords or short tonicizations that quickly return to the original key DO NOT count as a modulation.
- Ignore instrumentation, voicing/inversions, register, tempo, and mix. Small timing differences do not matter.

Valid responses are exactly:
"A. Yes, a key modulation occurs."
"B. No, the key stays the same."

Before you begin the task, I will provide you with one example that contains a key modulation and one example that remains in 
the same key so you better understand the task. After examining the examples, please respond with "Yes, I understand." 
if you understand the task or "No, I don't understand." if you don't understand the task.

After any reasoning, end with exactly one line:
A. Yes, a key modulation occurs.
OR
B. No, the key stays the same."""

# --- COT prompt (VERBATIM) ---
COT_PROMPT_TEMPLATE = """Analyze the music excerpt and decide whether a KEY MODULATION occurs.

Step 1: Identify the initial tonal center (key) from the opening harmony/melody.

Step 2: Scan the excerpt for a sustained shift to a NEW tonal center:
- Listen for cadential motion into a new key and a stable leading-tone/scale pattern supporting that key.
- Distinguish between a brief tonicization/borrowed chord (no modulation) and an established new key (modulation).

Step 3: Decision rule:
- If the music establishes and maintains a new tonal center, then a modulation occurs.
- If the music remains in the original key (only brief tonicizations/borrowed chords), no modulation occurs.

Step 4: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
A. Yes, a key modulation occurs.
OR
B. No, the key stays the same.
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
    Example: key_modulation_G25Pro_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    tag = _model_tag_for(model_name)
    return f"key_modulation_{tag}_CHAT_{mode}_{group}_seed{seed}.log"

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
            "stimuli/Intermediate/I-vi-ii-V_Bmaj_mod_Csmin_CleanWahGuitar_120.wav",
            "stimuli/Intermediate/vi-IV-I-V_Fsmaj_mod_Dsmin_AcousticGuit_relativeminor_118.wav",
            "stimuli/Intermediate/I-V-vi-IV_Dbmaj_mod_Abmaj_commonchord_CrunchGuit_100.wav",
            "stimuli/Intermediate/I-vi-VI-V_Amaj_mod_Fsmin_piano_160_3_4.wav",
            "stimuli/Intermediate/vi-IV-I-V_Fsmaj_mod_Csmaj_piano_135.wav",
            "stimuli/Intermediate/I-vi-ii-V_Cmaj_CleanGuitar_80.wav",
            "stimuli/Intermediate/vi-IV-I-V_Gmaj_CrunchGuit_150.wav",
            "stimuli/Intermediate/I-vi-VI-V_Fmaj_piano_172_3_4.wav",
            "stimuli/Intermediate/I-V-vi-IV_Dmaj_piano_145.wav",
            "stimuli/Intermediate/I-IV-V_Emaj_piano_150.wav",
        ]
    else:
        files = [
            "stimuli/Intermediate/I-IV-V_Abmaj_Mod_Fmin_pivot_relative_minor_175.wav",
            "stimuli/Intermediate/I-vi_IV-V_Amaj_mod_Bbmaj_CrunchEffectsGuit_140.wav",
            "stimuli/Intermediate/I-IV-V_Abmaj_Mod_Bbmaj_piano_132.wav",
            "stimuli/Intermediate/I-V-vi-IV_Dbmaj_mod_Bbmin_piano_115.wav",
            "stimuli/Intermediate/I-vi-ii-V_Bmaj_mod_Fsmaj_128.wav",
            "stimuli/Intermediate/I-vi-IV-V_Fmaj_CrunchGuit_140.wav",
            "stimuli/Intermediate/I-V-vi-IV_Dmaj_AcousticGuit_115.wav",
            "stimuli/Intermediate/I-IV-V_Emaj_CleanGuit_132.wav",
            "stimuli/Intermediate/I-vi-ii-V_Cmaj_piano_125.wav",
            "stimuli/Intermediate/vi-IV-I-V_Gmaj_piano_165.wav",
        ]
    return [_ppath(p) for p in files]

# =============================
# Parsing / evaluation helpers
# =============================
def parse_final_decision(text: str) -> str:
    last_a = last_b = last_b_alt = None
    for m in A_PAT.finditer(text or ""): last_a = m
    for m in B_PAT.finditer(text or ""): last_b = m
    for m in B_ALT.finditer(text or ""): last_b_alt = m

    # Prefer true canonical matches if both appear; otherwise allow the alt-B.
    candidates = []
    if last_a: candidates.append((last_a.end(), "A"))
    if last_b: candidates.append((last_b.end(), "B"))
    if last_b_alt: candidates.append((last_b_alt.end(), "B"))
    if not candidates: return ""

    _, label = max(candidates, key=lambda x: x[0])
    return A_CANON if label == "A" else B_CANON


def modulation_from_filename(path: str) -> str:
    """
    Ground truth from filename:
    - if 'mod' or 'Mod' present → A. Yes, a key modulation occurs.
    - else → B. No, the key stays the same.
    """
    name = os.path.basename(path)
    if ("mod" in name) or ("Mod" in name):
        return A_CANON
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
# ---- Example audio paths (resolved with your _p helper under STIM_ROOT) ----
# Group A (your current examples)
EX_A_MOD   = _p("Intermediate/I-IV-V_Abmaj_Mod_Bbmaj_piano_132.wav")       # contains a modulation
EX_A_NOMOD = _p("Intermediate/I-V-vi-IV_Dmaj_AcousticGuit_115.wav")        # stays in one key

# Group B (new examples)  # NOTE: if your file is actually .wav, change .wavv -> .wav
EX_B_MOD   = _p("Intermediate/vi-IV-I-V_Fsmaj_mod_Dsmin_relativemin_150.wav")  # contains a modulation
EX_B_NOMOD = _p("Intermediate/I-vi-VI-V_Fmaj_piano_172_3_4.wav")                # stays in one key

# =============================
# Few-shot builders + confirmation (VERBATIM content)
# =============================
def build_fewshot_messages_COT(log_filename: str) -> List[types.Content]:
    """
    Two in-context examples (assistant messages included), chosen by Group from the log name:
      - If "GroupA" in log name:
          Example 1: MODULATION present  -> ends with A. Yes, a key modulation occurs.
          Example 2: NO MODULATION       -> ends with B. No, the key stays the same.
      - Else (Group B):
          Example 1: MODULATION present  -> ends with A. Yes, a key modulation occurs.
          Example 2: NO MODULATION       -> ends with B. No, the key stays the same.

    Final lines are EXACT for reliable parsing.
    """
    group_a = "GroupA" in os.path.basename(log_filename)

    if group_a:
        wav_mod,   mt_mod   = read_audio_bytes_and_mime(EX_A_MOD)
        wav_nomod, mt_nomod = read_audio_bytes_and_mime(EX_A_NOMOD)
        id_mod   = "I-IV-V_Abmaj_Mod_Bbmaj_piano_132"
        id_nomod = "I-V-vi-IV_Dmaj_AcousticGuit_115"
    else:
        wav_mod,   mt_mod   = read_audio_bytes_and_mime(EX_B_MOD)
        wav_nomod, mt_nomod = read_audio_bytes_and_mime(EX_B_NOMOD)
        id_mod   = "vi-IV-I-V_Fsmaj_mod_Dsmin_relativemin_150"   # base name without extension
        id_nomod = "I-vi-VI-V_Fmaj_piano_172_3_4"

    # === Example 1 — MODULATION present → final "A." ===
    ex1_user = types.Content(
        role="user",
        parts=[
            PartText(COT_PROMPT_TEMPLATE.format(id1=id_mod)),
            PartBytes(wav_mod, mt_mod),
        ],
    )
    ex1_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Identify the initial tonal center.\n"
            "Step 2: In the middle of the excerpt, the harmony and leading-tone patterns establish a NEW stable center that is sustained.\n"
            "Step 3: A new tonal center is clearly established. Thus, a modulation occurs.\n"
            "A. Yes, a key modulation occurs."
        )]
    )

    # === Example 2 — NO MODULATION → final "B." ===
    ex2_user = types.Content(
        role="user",
        parts=[
            PartText(COT_PROMPT_TEMPLATE.format(id1=id_nomod)),
            PartBytes(wav_nomod, mt_nomod),
        ],
    )
    ex2_model = types.Content(
        role="model",
        parts=[PartText(
            "Step 1: Establish the opening key as the tonal center.\n"
            "Step 2: Harmony/melody continue to support the same center throughout the excerpt.\n"
            "Step 3: No sustained new center. Thus, no modulation occurs.\n"
            "B. No, the key stays the same."
        )]
    )

    return [ex1_user, ex1_model, ex2_user, ex2_model]

def _example_mod_user(wav: bytes, mt: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            PartText("Example 1: This audio example contains a key modulation. Listen carefully:"),
            PartText("Audio example:"),
            PartBytes(wav, mt),
        ]
    )

def _example_nomod_user(wav: bytes, mt: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            PartText("Example 2: This audio example contains no key modulation. Listen carefully:"),
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
    log_filename: str,
) -> Tuple[Any, str]:
    """
    Create a chat with the appropriate system instruction and few-shot history,
    then ask for the confirmation reply. Returns (chat, confirmation_text).
    """
    is_cot = (sysinstr_text == SYSINSTR_COT)
    group_a = "GroupA" in os.path.basename(log_filename)

    if is_cot:
        history = build_fewshot_messages_COT(log_filename)
    else:
        if group_a:
            wav_mod, mt_mod = read_audio_bytes_and_mime(EX_A_MOD)
            wav_nom, mt_nom = read_audio_bytes_and_mime(EX_A_NOMOD)
        else:
            wav_mod, mt_mod = read_audio_bytes_and_mime(EX_B_MOD)
            wav_nom, mt_nom = read_audio_bytes_and_mime(EX_B_NOMOD)
        history = [
            _example_mod_user(wav_mod, mt_mod),
            _example_nomod_user(wav_nom, mt_nom),
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
    logging.info(f"=== RUN START (Key Modulation • CHAT+SysInst • {mode} • {group}) ===")
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
        log_filename=log_filename,
    )

    # Fixed, ordered stimuli for the selected group
    stim_files = stimuli_files_group(group)
    random.seed(seed)  # ensures reproducible shuffle per run/seed
    random.shuffle(stim_files)

    print(
        f"\n--- Task: Key Modulation Detection — CHAT+SysInst {mode} • {group} | model={model_name} | temp=1.0 | seed={seed} ---\n"
    )
    logging.info(f"\n--- Task: Key Modulation Detection — CHAT+SysInst {mode} • {group} ---\n")

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
                         'A. Yes, a key modulation occurs.\n'
                         'OR\n'
                         'B. No, the key stays the same.'),
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

        # Ground truth from filename (presence of 'mod' / 'Mod')
        expected = modulation_from_filename(f)
        logging.info(f"Expected Final Answer: {expected}")

        if model_answer == expected:
            correct += 1
            print("Evaluation: Correct!")
            logging.info("Evaluation: Correct")
        elif model_answer in (A_CANON, B_CANON):
            print("Evaluation: Incorrect.")
            logging.info(f"Evaluation: Incorrect (expected={expected})")
        else:
            print("Evaluation: Unexpected response.")
            logging.info(f"Evaluation: Unexpected (parsed={model_answer})")

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
