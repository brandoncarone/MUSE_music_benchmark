# chord_progression_matching_Gemini_CHAT_SysInst_master.py
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

# Canonical answer strings and robust patterns (accept both A/B format and bare Yes/No)
A_CANON = "A. Yes, these are the same chord progression."
B_CANON = "B. No, these are not the same chord progression."

# Accept optional leading "A." / "B." and be robust to whitespace/case
A_PAT = re.compile(r'(?i)\b(?:a\.\s*)?yes,\s*these\s+are\s+the\s+same\s+chord\s+progression\.')
B_PAT = re.compile(r'(?i)\b(?:b\.\s*)?no,\s*these\s+are\s+not\s+the\s+same\s+chord\s+progression\.')

# =====================
# System instructions
# =====================
SYSINSTR_PLAIN = """You are a participant in a psychological experiment on music perception. 
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. Two audio examples to listen to.

Your task is to decide if two excerpts follow the same underlying chord progression, 
even if they are played with different instruments or in different styles. Think of it like a "musical sentence" — 
the same sentence can be said by different people conveying the same meaning, but it may not always sound exactly the same.
Valid responses are:
"Yes, these are the same chord progression." or 
"No, these are not the same chord progression."

Before you begin the task, I will provide you with examples of two excerpts representing the same chord progression, 
as well as two excerpts representing different chord progressions so that you better understand the task. 
After examining the examples, please respond with "Yes, I understand." if you understand the task or 
"No, I don't understand." if you don't understand the task.

Please provide no additional commentary beyond the short answers previously mentioned.
"""

SYSINSTR_COT = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. Two audio examples to listen to.

Your task is to decide whether the two excerpts follow the SAME underlying CHORD PROGRESSION, even if they differ in instrumentation, voicing, register, tempo, or playing style.

Definitions and constraints:
- Treat a chord progression as the ORDER of harmonic functions (e.g., I–vi–ii–V), independent of absolute key.
- Ignore instrumentation, voicings/inversions (e.g., I6, V7), added extensions (7, 9, sus), octave/register, and performance style.
- Small timing differences and different harmonic rhythms (durations) are acceptable as long as the ORDER of functions across one cycle is the same.
- If the functions or their ORDER differ (extra/missing chords, substitutions that change function, or re-ordered segments), they are NOT the same progression.

Valid responses are:
"A. Yes, these are the same chord progression."
"B. No, these are not the same chord progression."

Before you begin the task, I will provide you with examples of two excerpts representing the same chord progression, 
as well as two excerpts representing different chord progressions so that you better understand the task. 
After examining the examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." 
if you don't understand the task.

After any reasoning, end with exactly one line:
A. Yes, these are the same chord progression.
OR
B. No, these are not the same chord progression."""

# --- COT per-trial prompt ---
COT_PROMPT_TEMPLATE = """Analyze the two music excerpts and decide if they share the SAME underlying CHORD PROGRESSION.

Step 1: For each excerpt, segment by harmonic rhythm (where chords change) and abstract the progression as harmonic functions relative to its local key (e.g., I–vi–ii–V), ignoring inversions and extensions.

Step 2: Compare the two progressions across one cycle:
- Do they contain the same set of harmonic functions in the SAME ORDER?
- Differences in instrument, voicing, tempo, or chord durations are acceptable; ORDER must match.

Step 3: Decision rule:
- If the functional ORDER matches across the cycle, the two excerpts contain the same chord progression.
- If functions or their ORDER differ (insertions/deletions/substitutions that change function, or re-ordered segments), they are not the same.

Step 4: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
A. Yes, these are the same chord progression.
OR
B. No, these are not the same chord progression.
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
    Example: chord_progression_matching_G25Pro_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    tag = _model_tag_for(model_name)
    return f"chord_progression_matching_{tag}_CHAT_{mode}_{group}_seed{seed}.log"

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
            ("stimuli/Intermediate/I-V-vi-IV_Dmaj_CrunchGuit_100.wav", "stimuli/Intermediate/I-V-vi-IV_Dmaj_AcousticGuit_115.wav"),
            ("stimuli/Intermediate/I-vi-VI-V_Fmaj_piano_172_3_4.wav", "stimuli/Intermediate/I-vi-VI-V_Fmaj_piano_175_6_8.wav"),
            ("stimuli/Intermediate/vi-IV-I-V_Gmaj_AcousticGuit_118.wav", "stimuli/Intermediate/vi-IV-I-V_Gmaj_piano_135.wav"),
            ("stimuli/Intermediate/I-IV-V_Emaj_piano_120.wav", "stimuli/Intermediate/I-IV-V_Emaj_DistortedGuit_175.wav"),
            ("stimuli/Intermediate/I-vi-ii-V_Cmaj_CleanGuitar_80.wav", "stimuli/Intermediate/I-vi-ii-V_Cmaj_piano_125.wav"),
            ("stimuli/Intermediate/I-IV-V_Emaj_DistortedGuit_175.wav", "stimuli/Intermediate/I-V-vi-IV_Dmaj_CrunchGuit_100.wav"),
            ("stimuli/Intermediate/vi-IV-I-V_Gmaj_piano_135.wav", "stimuli/Intermediate/I-vi-VI-V_Fmaj_piano_172_3_4.wav"),
            ("stimuli/Intermediate/I-V-vi-IV_Dmaj_AcousticGuit_115.wav", "stimuli/Intermediate/I-vi-ii-V_Cmaj_piano_125.wav"),
            ("stimuli/Intermediate/I-vi-VI-V_Fmaj_piano_175_6_8.wav", "stimuli/Intermediate/vi-IV-I-V_Gmaj_AcousticGuit_118.wav"),
            ("stimuli/Intermediate/I-IV-V_Emaj_piano_120.wav", "stimuli/Intermediate/I-vi-ii-V_Cmaj_CleanGuitar_80.wav"),
        ]
    else:
        pairs = [
            ("stimuli/Intermediate/I-vi-IV-V_Fmaj_CrunchGuit_140.wav", "stimuli/Intermediate/I-vi-IV-V_Fmaj_CrunchEffectsGuit_160.wav"),
            ("stimuli/Intermediate/I-IV-V_Emaj_piano_150.wav", "stimuli/Intermediate/I-IV-V_Emaj_piano_120.wav"),
            ("stimuli/Intermediate/I-V-vi-IV_Dmaj_piano_145.wav", "stimuli/Intermediate/I-V-vi-IV_Dmaj_piano_115.wav"),
            ("stimuli/Intermediate/I-vi-ii-V_Cmaj_CleanWahGuitar_120.wav", "stimuli/Intermediate/I-vi-ii-V_Cmaj_piano_165.wav"),
            ("stimuli/Intermediate/vi-IV-I-V_Gmaj_piano_165.wav", "stimuli/Intermediate/vi-IV-I-V_Gmaj_CrunchGuit_150.wav"),
            ("stimuli/Intermediate/I-vi-ii-V_Cmaj_CleanWahGuitar_120.wav", "stimuli/Intermediate/I-vi-IV-V_Fmaj_CrunchGuit_140.wav"),
            ("stimuli/Intermediate/I-IV-V_Emaj_piano_150.wav", "stimuli/Intermediate/I-V-vi-IV_Dmaj_piano_115.wav"),
            ("stimuli/Intermediate/I-vi-IV-V_Fmaj_CrunchEffectsGuit_160.wav", "stimuli/Intermediate/vi-IV-I-V_Gmaj_piano_165.wav"),
            ("stimuli/Intermediate/I-V-vi-IV_Dmaj_piano_145.wav", "stimuli/Intermediate/vi-IV-I-V_Gmaj_CrunchGuit_150.wav"),
            ("stimuli/Intermediate/I-IV-V_Emaj_piano_120.wav", "stimuli/Intermediate/I-vi-ii-V_Cmaj_piano_165.wav"),
        ]
    return [{"file1": _ppath(a), "file2": _ppath(b)} for a, b in pairs]

# ---- Example audio paths (resolved with your _p helper under STIM_ROOT) ----
# Group A
EX_A_SAME_1 = _p("Intermediate/vi-IV-I-V_Gmaj_CrunchGuit_150.wav")
EX_A_SAME_2 = _p("Intermediate/vi-IV-I-V_Gmaj_piano_165.wav")
EX_A_DIFF_1 = _p("Intermediate/vi-IV-I-V_Gmaj_CrunchGuit_150.wav")
EX_A_DIFF_2 = _p("Intermediate/I-IV-V_Emaj_CleanGuit_132.wav")

# Group B
EX_B_SAME_1 = _p("Intermediate/I-vi-ii-V_Cmaj_CleanGuitar_80.wav")
EX_B_SAME_2 = _p("Intermediate/I-vi-ii-V_Cmaj_piano_125.wav")
EX_B_DIFF_1 = _p("Intermediate/I-vi-VI-V_Fmaj_piano_175_6_8.wav")
EX_B_DIFF_2 = _p("Intermediate/vi-IV-I-V_Gmaj_AcousticGuit_118.wav")

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

_ROMAN_TOKEN = re.compile(r'^(?i)(i|ii|iii|iv|v|vi|vii)$')
def _normalize_progression_token(tok: str) -> str:
    t = tok.strip().replace('–', '-').replace('—', '-')
    t = t.strip('-').upper()
    return t

def extract_progression_from_path(path: str) -> str:
    """
    From a filename like 'I-V-vi-IV_Dmaj_CrunchGuit_100.wav' return a normalized
    progression string 'I-V-VI-IV'. Only the hyphen-separated roman section before
    the first underscore is considered.
    """
    base = os.path.basename(path)
    head = base.split('_', 1)[0]  # e.g., 'I-V-vi-IV'
    tokens = [ _normalize_progression_token(t) for t in head.split('-') if t ]
    # Keep only plausible roman tokens (I..VII)
    filtered = [t for t in tokens if _ROMAN_TOKEN.match(t)]
    return "-".join(filtered)

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
def _example_same_user(wav1: bytes, mt1: str, wav2: bytes, mt2: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            PartText("Example 1: The following two audio excerpts represent the same chord progression."),
            PartText("Excerpt 1:"),
            PartBytes(wav1, mt1),
            PartText("Excerpt 2:"),
            PartBytes(wav2, mt2),
        ]
    )

def _example_diff_user(wav1: bytes, mt1: str, wav2: bytes, mt2: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            PartText("Example 2: The following two audio excerpts represent different chord progressions."),
            PartText("Excerpt 1:"),
            PartBytes(wav1, mt1),
            PartText("Excerpt 2:"),
            PartBytes(wav2, mt2),
        ]
    )

def build_fewshot_messages_COT(log_filename: str) -> List[types.Content]:
    """
    Returns two in-context examples (assistant messages included), chosen by Group from the log name:
      - If "GroupA" in log name:
          Example 1: SAME progression (vi–IV–I–V in different instruments/tempi)  -> ends with A.
          Example 2: DIFFERENT progressions (vi–IV–I–V vs I–IV–V)                  -> ends with B.
      - If "GroupB" in log name:
          Example 1: SAME progression (I–vi–ii–V in different instruments/tempi)   -> ends with A.
          Example 2: DIFFERENT progressions (I–vi–VI–V vs vi–IV–I–V)               -> ends with B.

    Final line strings are EXACT for reliable parsing/eval.
    """
    group_a = "GroupA" in os.path.basename(log_filename)

    if group_a:
        # Load bytes for Group A examples
        wav_same1, mt_same1 = read_audio_bytes_and_mime(EX_A_SAME_1)
        wav_same2, mt_same2 = read_audio_bytes_and_mime(EX_A_SAME_2)
        wav_diff1, mt_diff1 = read_audio_bytes_and_mime(EX_A_DIFF_1)
        wav_diff2, mt_diff2 = read_audio_bytes_and_mime(EX_A_DIFF_2)

        id_same1 = "vi-IV-I-V_Gmaj_CrunchGuit_150"
        id_same2 = "vi-IV-I-V_Gmaj_piano_165"
        id_diff1 = "vi-IV-I-V_Gmaj_CrunchGuit_150"
        id_diff2 = "I-IV-V_Emaj_CleanGuit_132"

        # === Example 1 — SAME chord progression → final "A." ===
        ex1_user = types.Content(
            role="user",
            parts=[
                PartText(COT_PROMPT_TEMPLATE.format(id1=id_same1, id2=id_same2)),
                PartText("Excerpt 1:"),
                PartBytes(wav_same1, mt_same1),
                PartText("Excerpt 2:"),
                PartBytes(wav_same2, mt_same2),
            ],
        )
        ex1_model = types.Content(
            role="model",
            parts=[PartText(
                "Step 1: Segment both excerpts by harmonic rhythm; abstract functions ignoring inversions/extensions.\n"
                "Excerpt 1: The first excerpt follows a vi-IV-I-V in the key of G major.\n"
                "Excerpt 2: The second excerpt follows a vi-IV-I-V in the key of G major.\n"
                "Step 2: Both outline the same functional ORDER across the cycle despite different instruments/tempi.\n"
                "Step 3: ORDER matches (e.g., vi–IV–I–V). Thus, they are the same progression.\n"
                "A. Yes, these are the same chord progression."
            )]
        )

        # === Example 2 — DIFFERENT chord progressions → final "B." ===
        ex2_user = types.Content(
            role="user",
            parts=[
                PartText(COT_PROMPT_TEMPLATE.format(id1=id_diff1, id2=id_diff2)),
                PartText("Excerpt 1:"),
                PartBytes(wav_diff1, mt_diff1),
                PartText("Excerpt 2:"),
                PartBytes(wav_diff2, mt_diff2),
            ],
        )
        ex2_model = types.Content(
            role="model",
            parts=[PartText(
                "Step 1: Segment and abstract each to functional ORDER.\n"
                "Excerpt 1: The first excerpt follows a vi-IV-I-V progression in the key of G major.\n"
                "Excerpt 2: The second excerpt follows a I-IV-V progression in the key of E major.\n"
                "Step 2: The sequences differ in functions and order (e.g., vi–IV–I–V vs I–IV–V).\n"
                "Step 3: ORDER does not match. Thus, these are not the same progression.\n"
                "B. No, these are not the same chord progression."
            )]
        )

        return [ex1_user, ex1_model, ex2_user, ex2_model]

    else:
        # Load bytes for Group B examples
        wav_same1, mt_same1 = read_audio_bytes_and_mime(EX_B_SAME_1)
        wav_same2, mt_same2 = read_audio_bytes_and_mime(EX_B_SAME_2)
        wav_diff1, mt_diff1 = read_audio_bytes_and_mime(EX_B_DIFF_1)
        wav_diff2, mt_diff2 = read_audio_bytes_and_mime(EX_B_DIFF_2)

        id_same1 = "I-vi-ii-V_Cmaj_CleanGuitar_80"
        id_same2 = "I-vi-ii-V_Cmaj_piano_125"
        id_diff1 = "I-vi-VI-V_Fmaj_piano_175_6_8"
        id_diff2 = "vi-IV-I-V_Gmaj_AcousticGuit_118"

        # === Example 1 — SAME chord progression → final "A." ===
        ex1_user = types.Content(
            role="user",
            parts=[
                PartText(COT_PROMPT_TEMPLATE.format(id1=id_same1, id2=id_same2)),
                PartText("Excerpt 1:"),
                PartBytes(wav_same1, mt_same1),
                PartText("Excerpt 2:"),
                PartBytes(wav_same2, mt_same2),
            ],
        )
        ex1_model = types.Content(
            role="model",
            parts=[PartText(
                "Step 1: Segment both excerpts by harmonic rhythm; abstract functions ignoring inversions/extensions.\n"
                "Excerpt 1: The first excerpt follows a I-vi-ii-V progression in the key of C major.\n"
                "Excerpt 2: The second excerpt follows a I-vi-ii-V progression in the key of C major.\n"
                "Step 2: Both outline the same functional ORDER across the cycle despite different instruments/tempi.\n"
                "Step 3: ORDER matches (e.g., I–vi–ii–V). Thus, they are the same progression.\n"
                "A. Yes, these are the same chord progression."
            )]
        )

        # === Example 2 — DIFFERENT chord progressions → final "B." ===
        ex2_user = types.Content(
            role="user",
            parts=[
                PartText(COT_PROMPT_TEMPLATE.format(id1=id_diff1, id2=id_diff2)),
                PartText("Excerpt 1:"),
                PartBytes(wav_diff1, mt_diff1),
                PartText("Excerpt 2:"),
                PartBytes(wav_diff2, mt_diff2),
            ],
        )
        ex2_model = types.Content(
            role="model",
            parts=[PartText(
                "Step 1: Segment and abstract each to functional ORDER.\n"
                "Excerpt 1: The first excerpt follows a I–vi–VI–V progression in the key of F major.\n"
                "Excerpt 2: The second excerpt follows a vi-IV-I-V progression in the key of G major.\n"
                "Step 2: The sequences differ in functions and/or order (e.g., I–vi–VI–V vs vi-IV-I-V).\n"
                "Step 3: ORDER does not match. Thus, these are not the same progression.\n"
                "B. No, these are not the same chord progression."
            )]
        )

        return [ex1_user, ex1_model, ex2_user, ex2_model]

def run_examples_and_confirm(
    *,
    client: genai.Client,
    model_name: str,
    sysinstr_text: str,
    log_filename: str,
    group: str,
    log: logging.Logger,
) -> Tuple[Any, str]:
    """
    Create a chat with the appropriate system instruction and few-shot history,
    then ask for the confirmation reply. Returns (chat, confirmation_text).
    """
    is_cot = (sysinstr_text == SYSINSTR_COT)

    if is_cot:
        history = build_fewshot_messages_COT(log_filename)
    else:
        # SYSINST mode: show one SAME-pair and one DIFFERENT-pair as user-only turns (no assistant messages)
        if group == "GroupA":
            wav_same1, mt_same1 = read_audio_bytes_and_mime(EX_A_SAME_1)
            wav_same2, mt_same2 = read_audio_bytes_and_mime(EX_A_SAME_2)
            wav_diff1, mt_diff1 = read_audio_bytes_and_mime(EX_A_DIFF_1)
            wav_diff2, mt_diff2 = read_audio_bytes_and_mime(EX_A_DIFF_2)
        else:
            wav_same1, mt_same1 = read_audio_bytes_and_mime(EX_B_SAME_1)
            wav_same2, mt_same2 = read_audio_bytes_and_mime(EX_B_SAME_2)
            wav_diff1, mt_diff1 = read_audio_bytes_and_mime(EX_B_DIFF_1)
            wav_diff2, mt_diff2 = read_audio_bytes_and_mime(EX_B_DIFF_2)

        history = [
            _example_same_user(wav_same1, mt_same1, wav_same2, mt_same2),
            _example_diff_user(wav_diff1, mt_diff1, wav_diff2, mt_diff2),
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
    logging.info(f"=== RUN START (Chord Progression Matching • CHAT+SysInst • {mode} • {group}) ===")
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
        log_filename=log_filename,
        group=group,
        log=logging.getLogger(),
    )

    # Stimuli for the selected group (shuffle per seed)
    question_stims = stimuli_pairs_group(group)
    random.seed(seed)
    random.shuffle(question_stims)

    print(
        f"\n--- Task: Chord Progression Matching — CHAT+SysInst {mode} • {group} | model={model_name} | temp=1.0 | seed={seed} ---\n"
    )
    logging.info(f"\n--- Task: Chord Progression Matching — CHAT+SysInst {mode} • {group} ---\n")

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

        f1 = q["file1"]
        f2 = q["file2"]
        logging.info(f"Stimuli: file1={f1}, file2={f2}")

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
                         'Yes, these are the same chord progression.\n'
                         'OR\n'
                         'No, these are not the same chord progression.'),
            ]
        else:
            user_parts = [
                PartText(COT_PROMPT_TEMPLATE),
                PartText("Excerpt 1:"),
                PartBytes(wav1, mt1),
                PartText("Excerpt 2:"),
                PartBytes(wav2, mt2),
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

        # Ground truth via roman-numeral progression extracted from filenames
        prog1 = extract_progression_from_path(f1)
        prog2 = extract_progression_from_path(f2)
        expected = A_CANON if prog1 and prog2 and (prog1 == prog2) else B_CANON

        if model_answer == expected:
            correct += 1
            print("Evaluation: Correct!")
            logging.info("Evaluation: Correct")
        else:
            print("Evaluation: Incorrect.")
            logging.info(f"Evaluation: Incorrect (expected={expected}, prog1={prog1}, prog2={prog2})")

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
