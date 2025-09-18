# transposition_Qwen2.5-Omni_CHAT_SysInst_master.py
import os
import re
import random
import gc
import logging
from typing import List, Dict, Any, Tuple

import warnings

warnings.filterwarnings("ignore")

# ===== Runtime knobs (match your other Qwen runners) =====
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import librosa
import torch
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    GenerationConfig,
)

# =============================
# Constants / paths
# =============================
MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
STIM_ROOT = "stimuli"  # relative root per your HPC layout
MAX_NEW_TOKENS = 8192

YES_CANON = 'Yes, these are the same melody.'
NO_CANON = 'No, these are not the same melody.'

# Robust patterns to find final answer anywhere in the completion (choose last occurrence)
YES_PAT = re.compile(r'(?i)\byes,\s*these\s+are\s+the\s+same\s+melody\.')
NO_PAT = re.compile(r'(?i)\bno,\s*these\s+are\s+not\s+the\s+same\s+melody\.')

# =============================
# System instructions (two modes)
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

# --- COT prompt (looser, no strict template) ---
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

# ANALYSIS_TEMPLATE = """Return your analysis in this exact format, followed by the Final Answer line:
#
# [Pitch sequence A]: <space-separated notes or Hz bins, e.g., C4 D4 E4 G4 C5>
# [Pitch sequence B]: <...>
# [Interval sequence A (semitones)]: [d1,d2,...]
# [Interval sequence B (semitones)]: [d1,d2,...]
# [Same number of notes?]: <yes/no>
# [Interval sequences match?]: <yes/no>
# [Comments ≤ 25 words]: <very brief justification>
#
# Final Answer: "<one of the two exact lines>"
# """
#
# Ex_Same_Inst = """Return your analysis in this exact format, followed by the Final Answer line:
#
# [Pitch sequence A]: B4, C5, B4, G4, A4, G4, E4, F4, E4
# [Pitch sequence B]: C#5, D5, C#5, A4, B4, A4, F#4, G4, F#4
# [Interval sequence A (semitones)]: [+1, -1, -4, +2, -2, -3, +1, -1]
# [Interval sequence B (semitones)]: [+1, -1, -4, +2, -2, -3, +1, -1]
# [Same number of notes?]: yes
# [Interval sequences match?]: yes
# [Comments ≤ 25 words]: Identical interval pattern across 8 steps; B is a transposed version of A (up a whole tone).
#
# Final Answer: "Yes, these are the same melody."
# """
#
# Ex_Diff_Inst = """Return your analysis in this exact format, followed by the Final Answer line:
#
# [Pitch sequence A]: C#5, D5, C#5, A4, B4, A4, F#4, G4, F#4
# [Pitch sequence B]: F3, A4, F4, E4, C4, B3, C4, F3
# [Interval sequence A (semitones)]: [+1, -1, -4, +2, -2, -3, +1, -1]
# [Interval sequence B (semitones)]: [+16, -4, -1, -4, -1, +1, -7]
# [Same number of notes?]: no
# [Interval sequences match?]: no
# [Comments ≤ 25 words]: Different note counts; B has a big +16 leap and final -7 not in A. Not transposition-equivalent.
#
# Final Answer: "No, these are not the same melody."
# """


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


def make_log_filename(*, mode: str, group: str, seed: int) -> str:
    """
    Chat + System Instructions runner, two modes:
      - SYSINST (plain)
      - COT     (reasoning)
    Two stimulus groups: GroupA / GroupB
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"transposition_Qwen2.5-Omni_CHAT_{mode}_{group}_seed{seed}.log"


# =============================
# Decoding setup
# =============================
def set_generation_config(model, *, min_new_tokens: int | None = None):
    cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        do_sample=True,
        use_cache=True,
    )
    if min_new_tokens is not None:
        cfg.min_new_tokens = min_new_tokens
    model.generation_config = cfg


# =============================
# Data helpers
# =============================
def _ppath(p: str) -> str:
    """Normalize a given path:
       - absolute path: returned as-is
       - path starting with 'stimuli/': returned as-is
       - otherwise, join with STIM_ROOT
    """
    if os.path.isabs(p):
        return p
    if p.startswith(STIM_ROOT + os.sep) or p.startswith(STIM_ROOT + "/"):
        return p
    return os.path.join(STIM_ROOT, p)


def _p(name: str) -> str:
    """Join a stimulus filename to the canonical root."""
    return os.path.join(STIM_ROOT, name)


def load_audio_array(path: str, sr: int):
    y, _ = librosa.load(path, sr=sr, mono=True)
    # remove leading/trailing silence to reduce audio tokens
    y, _ = librosa.effects.trim(y, top_db=35)  # 30–40 dB is typical; 35 is a good middle
    return y


def _file_id_no_ext(p: str) -> str:
    """e.g., '/path/M11_CMaj_180_Piano.wav' -> 'M11_CMaj_180_Piano'"""
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
    else:  # GroupB
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


# Example audio (few-shot)
EX_SAME_A = _p("M11_CMaj_180_Piano.wav")
EX_SAME_B = _p("M11_EMaj_180_Piano.wav")
EX_DIFF_B = _p("M11_EMaj_180_Piano.wav")
EX_DIFF_C = _p("M12_FMaj_155_Piano.wav")


# =============================
# Utility: parse final decision
# =============================
def parse_final_decision(text: str) -> str:
    """Return YES_CANON or NO_CANON, or '' if not found. Prefer the LAST occurrence."""
    last_yes = None
    last_no = None
    for m in YES_PAT.finditer(text):
        last_yes = m
    for m in NO_PAT.finditer(text):
        last_no = m
    if last_yes and last_no:
        return YES_CANON if last_yes.end() > last_no.end() else NO_CANON
    if last_yes:
        return YES_CANON
    if last_no:
        return NO_CANON
    return ""


def melody_id(fname: str) -> str:
    """Extract 'M#' id robustly (supports M1..M10..)."""
    base = os.path.basename(fname)
    m = re.match(r"^(M\d+)_", base)
    return m.group(1) if m else base


def _count_audio_msgs(messages: List[Dict[str, Any]]) -> int:
    n = 0
    for m in messages:
        for c in m.get("content", []):
            if isinstance(c, dict) and c.get("type") == "audio":
                n += 1
    return n


# =============================
# Qwen generation helper
# =============================
def qwen_generate_text(*, model, processor, messages, audio) -> str:
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=text,
        audio=audio if len(audio) > 0 else None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            v = v.to(model.device)
            if v.is_floating_point():
                v = v.to(model.dtype)
            inputs[k] = v

    with torch.inference_mode():
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
            out_ids = model.generate(**inputs, return_audio=False)
        new_tokens = out_ids[:, inputs["input_ids"].size(1):]
        completion = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
            0].strip()

    del inputs, out_ids, new_tokens
    torch.cuda.empty_cache()
    return completion


# =============================
# Build example sequence (single pre-task turn)
# =============================

def build_fewshot_messages_COT(sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """
    Two in-context examples with brief CoT and strict 'Final Answer:' line.
    Ex1: SAME melody → Final Answer: Yes, these are the same melody.
    Ex2: DIFFERENT melody → Final Answer: No, these are not the same melody.
    """
    s = STIM_ROOT
    a_same1 = load_audio_array(f"{s}/M11_CMaj_180_Piano.wav", sr)
    a_same2 = load_audio_array(f"{s}/M11_EMaj_180_Piano.wav", sr)
    a_diff2 = load_audio_array(f"{s}/M12_FMaj_155_Piano.wav", sr)

    id_same1 = "M11_CMaj_180_Piano"
    id_same2 = "M11_EMaj_180_Piano"
    id_diff2 = "M12_FMaj_155_Piano"

    ex1_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE.format(id1=id_same1, id2=id_same2)},
            {"type": "audio", "audio": a_same1, "sampling_rate": sr},
            {"type": "audio", "audio": a_same2, "sampling_rate": sr},
        ],
    }
    ex1_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text":
            "Step 1: Identify notes for both clips; both sequences have the same length.\n"
            "Step 2: The successive semitone intervals match element-by-element (including 0 for repeats).\n"
            "Step 3: Therefore they are the same melody up to transposition.\n"
            "Yes, these are the same melody."
        }]
    }

    ex2_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE.format(id1=id_same2, id2=id_diff2)},
            {"type": "audio", "audio": a_same2, "sampling_rate": sr},
            {"type": "audio", "audio": a_diff2, "sampling_rate": sr},
        ],
    }
    ex2_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text":
            "Step 1: Identify notes; sequences differ in structure.\n"
            "Step 2: Interval sequences do not align; there are insertions/deletions.\n"
            "Step 3: Thus not the same melody up to transposition.\n"
            "No, these are not the same melody."
        }]
    }

    messages = [ex1_user, ex1_assistant, ex2_user, ex2_assistant]
    audio_list = [a_same1, a_same2, a_same2, a_diff2]  # must match the order of audio placeholders
    return messages, audio_list


def run_examples_and_confirm(
        *,
        model: Qwen2_5OmniForConditionalGeneration,
        processor: Qwen2_5OmniProcessor,
        sysinstr_text: str,
        sr: int,
        log: logging.Logger,
) -> Tuple[List[Dict[str, Any]], str, List[Any]]:
    """Returns (history_with_examples_and_confirm, confirmation_string, example_audio_list)."""

    # System message
    history: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": sysinstr_text}]}
    ]
    is_cot = (sysinstr_text == SYSINSTR_COT)

    if is_cot:
        fewshot_msgs, fewshot_audio = build_fewshot_messages_COT(sr)
        history_with_examples = history + fewshot_msgs
        cumulative_audios = list(fewshot_audio)


        confirm_user = {
            "role": "user",
            "content": [
                {"type": "text", "text": 'After examining the examples above, please respond with "Yes, I understand." if you understand the task, or "No, I don\'t understand." if you do not.'}
            ],
        }

        completion = qwen_generate_text(
            model=model,
            processor=processor,
            messages=history_with_examples + [confirm_user],
            audio=cumulative_audios,  # exactly the few-shot audios; confirm has no audio
        )
        print("Confirmation response:", completion)
        log.info(f"Confirmation response: {completion}")

        history_with_examples_and_confirm = history_with_examples + [
            {"role": "assistant", "content": [{"type": "text", "text": completion.strip()}]},
        ]
        return history_with_examples_and_confirm, completion.strip(), cumulative_audios

    # Load example audios
    a_same_1 = load_audio_array(EX_SAME_A, sr)
    a_same_2 = load_audio_array(EX_SAME_B, sr)
    a_diff_2 = load_audio_array(EX_DIFF_B, sr)
    a_diff_3 = load_audio_array(EX_DIFF_C, sr)

    # Log + print
    log.info("--- Example 1: Same Melody ---")
    log.info(f"Example 1 - Stimuli: {EX_SAME_A}, {EX_SAME_B}")
    log.info("--- Example 2: Different Melody ---")
    log.info(f"Example 2 - Stimuli: {EX_DIFF_B}, {EX_DIFF_C}")
    print("\n--- Example 1: Same Melody ---")
    print("--- Example 2: Different Melody ---")


    ex1_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Example 1: The following two audio excerpts represent the same melody, played in different keys."},
            {"type": "audio", "audio": a_same_1, "sampling_rate": sr},
            {"type": "text", "text": "Audio example 2:"},
            {"type": "audio", "audio": a_same_2, "sampling_rate": sr},
        ],
    }

    ex2_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Example 2: The following two audio excerpts represent different melodies."},
            {"type": "audio", "audio": a_diff_2, "sampling_rate": sr},
            {"type": "text", "text": "Audio example 2:"},
            {"type": "audio", "audio": a_diff_3, "sampling_rate": sr},
        ],
    }

    # Build the few-shot history:
    # - SYSINST: just the two example user turns (audio only)
    # - COT: add assistant turns that SHOW the correct worked format (your Ex_* strings)
    history_with_examples: List[Dict[str, Any]] = [*history, ex1_user]

    history_with_examples = history + [ex1_user, ex2_user]
    cumulative_audios = [a_same_1, a_same_2, a_diff_2, a_diff_3]

    # Confirmation prompt (text)
    confirm_user = {
        "role": "user",
        "content": [
            {"type": "text",
             "text": 'After examining the examples above, please respond with "Yes, I understand." if you understand the task, or "No, I don\'t understand." if you do not.'}
        ],
    }

    # IMPORTANT: For this one call, supply all 4 example audios in order
    completion = qwen_generate_text(
        model=model,
        processor=processor,
        messages=history_with_examples + [confirm_user],
        audio=[a_same_1, a_same_2, a_diff_2, a_diff_3],
    )
    print("Confirmation response:", completion)
    log.info(f"Confirmation response: {completion}")

    history_with_examples_and_confirm = history_with_examples + [
        {"role": "assistant", "content": [{"type": "text", "text": completion.strip()}]},
    ]
    # Return history that INCLUDES the example turns (with audio) + the audio list in exact order
    return history_with_examples_and_confirm, completion.strip(), [a_same_1, a_same_2, a_diff_2, a_diff_3]

# =============================
# One full run (SYSINST / COT, GroupA / GroupB)
# =============================
def run_once(*, mode: str, group: str, seed: int, log_filename: str) -> None:
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    configure_logging(log_filename)
    logging.info(f"=== RUN START (Transposition • CHAT+SysInst • {mode} • {group}) ===")
    logging.info(f"Config: model={MODEL_ID}, temp=1.0, seed={seed}, group={group}, log={log_filename}")

    # Reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    LOCAL_QWEN_DIR = "/scratch/bc3189/hf_cache/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00"

    # Processor + Model
    processor = Qwen2_5OmniProcessor.from_pretrained(LOCAL_QWEN_DIR, local_files_only=True)

    gpu_count = torch.cuda.device_count()
    if gpu_count >= 2:
        # Cap lower to force spreading across all GPUs
        max_mem = {i: "22GiB" for i in range(gpu_count)}
        max_mem["cpu"] = "120GiB"
        device_map = "balanced"
    else:
        max_mem = None
        device_map = "auto"

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        LOCAL_QWEN_DIR,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        attn_implementation="sdpa",
        enable_audio_output=False,
        max_memory=max_mem,
    ).eval()

    # Helpful: confirm sharding in your logs
    try:
        logging.info(f"hf_device_map keys: {list(model.hf_device_map.items())[:6]} ...")
    except Exception:
        pass

    set_generation_config(model, min_new_tokens=80 if mode == "COT" else None)
    sr = processor.feature_extractor.sampling_rate

    # System instructions
    sysinstr_text = SYSINSTR_PLAIN if mode == "SYSINST" else SYSINSTR_COT

    # === Examples + confirmation (few-shot, keep audio in history) ===
    history, confirm, cumulative_audios = run_examples_and_confirm(
        model=model, processor=processor, sysinstr_text=sysinstr_text, sr=sr, log=logging.getLogger()
    )

    # Stimuli pairs: fixed order per group, no shuffle
    question_stims = stimuli_pairs_group(group)
    random.seed(seed)               # ensures reproducible shuffle per run/seed
    random.shuffle(question_stims)

    print(
        f"\n--- Task: Melody Matching (Transposition) — CHAT+SysInst {mode} • {group} | model={MODEL_ID} | temp=1.0 | seed={seed} ---\n")
    logging.info(f"\n--- Task: Melody Matching (Transposition) — CHAT+SysInst {mode} • {group} ---\n")

    correct = 0
    total = len(question_stims)

    for idx, q in enumerate(question_stims, start=1):
        print(f"\n--- Question {idx} ---\n")
        logging.info(f"\n--- Question {idx} ---\n")

        f1 = q["file1"]
        f2 = q["file2"]
        logging.info(f"Stimuli: file1={f1}, file2={f2}")

        # Load audio arrays for this trial
        a1 = load_audio_array(f1, sr)
        a2 = load_audio_array(f2, sr)

        # Build trial prompt (text only; audio passed separately)
        if mode == "SYSINST":
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the first excerpt."},
                    {"type": "audio", "audio": a1, "sampling_rate": sr},
                    {"type": "text", "text": "Here is the second excerpt."},
                    {"type": "audio", "audio": a2, "sampling_rate": sr},
                    {"type": "text", "text": """Reply with exactly ONE of the following lines:
"Yes, these are the same melody."
OR
"No, these are not the same melody."
"""},
                ],
            }
        else:
            # COT trial prompt uses the general template and only the two audios
            id1 = _file_id_no_ext(f1)
            id2 = _file_id_no_ext(f2)
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": COT_PROMPT_TEMPLATE.format(id1=id1, id2=id2)},
                    {"type": "audio", "audio": a1, "sampling_rate": sr},
                    {"type": "audio", "audio": a2, "sampling_rate": sr},
                ],
            }

        # Build messages and audio for THIS call (full history including all prior audio)
        messages = history + [user_turn]
        audio_for_call = list(cumulative_audios) + [a1, a2]

        # Safety check to ensure audio alignment
        assert _count_audio_msgs(messages) == len(audio_for_call), \
            f"Audio count mismatch: messages expect {_count_audio_msgs(messages)}, got {len(audio_for_call)}"

        completion = qwen_generate_text(
            model=model,
            processor=processor,
            messages=messages,
            audio=audio_for_call,
        )

        print("LLM Full Response:\n", completion)
        logging.info(f"[{mode}/{group}] Q{idx} - LLM Full Response:\n{completion}")

        # Parse decision
        model_answer = parse_final_decision(completion)
        if not model_answer:
            print("Evaluation: Failed. Could not parse the final answer phrase.")
            logging.error("Parse Error: missing/malformed final answer phrase.")
            # Append turn (with audio) to history and grow the cumulative audio list
            history.append(user_turn)
            history.append({"role": "assistant", "content": [{"type": "text", "text": completion}]})
            cumulative_audios.extend([a1, a2])
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

        # Append turn (with audio) to history and grow the cumulative audio list
        history.append(user_turn)
        history.append({"role": "assistant", "content": [{"type": "text", "text": completion}]})
        cumulative_audios.extend([a1, a2])

    print(f"\nTotal Correct: {correct} out of {total}")
    logging.info(f"Total Correct: {correct} out of {total}")
    logging.info("=== RUN END ===\n")


# =============================
# Multi-run driver (12 total)
# =============================
if __name__ == "__main__":
    runs = []
    for mode in ["SYSINST", "COT"]:
        for group in ["GroupA", "GroupB"]:
            for s in (1, 2, 3):
                runs.append(dict(
                    mode=mode,
                    group=group,
                    seed=s,
                    log_filename=make_log_filename(mode=mode, group=group, seed=s),
                ))

    for cfg in runs:
        run_once(**cfg)
        gc.collect()
        torch.cuda.empty_cache()
