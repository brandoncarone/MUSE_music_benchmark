# meter_identification_Qwen2.5-Omni_CHAT_SysInst_master.py
import os
import re
import gc
import random
import logging
from collections import deque
from itertools import chain
from typing import List, Dict, Any, Tuple

KEEP_LAST_K = 4

import warnings
warnings.filterwarnings("ignore")

# ===== Runtime knobs =====
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

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
STIM_ROOT = "stimuli"
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
    Example: meter_identification_Qwen2.5-Omni_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"meter_identification_Qwen2.5-Omni_CHAT_{mode}_{group}_seed{seed}.log"

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

def load_audio_array(path: str, sr: int):
    y, _ = librosa.load(path, sr=sr, mono=True)
    # Efficiency: trim leading/trailing silence to reduce token load
    y, _ = librosa.effects.trim(y, top_db=35)
    return y

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

def _count_audio_msgs(messages: List[Dict[str, Any]]) -> int:
    n = 0
    for m in messages:
        for c in m.get("content", []):
            if isinstance(c, dict) and c.get("type") == "audio":
                n += 1
    return n

# =============================
# Qwen text generation
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
        completion = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    del inputs, out_ids, new_tokens
    torch.cuda.empty_cache()
    return completion

# =============================
# Few-shot builders + confirmation
# =============================
# Example audio paths (resolved under STIM_ROOT)
EX_3 = _p("Intermediate/Whammy_3.wav")
EX_4 = _p("Intermediate/SS_4.wav")
EX_5 = _p("Intermediate/Five_5.wav")

def build_fewshot_messages_SYSINST(sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    a3 = load_audio_array(EX_3, sr)
    a4 = load_audio_array(EX_4, sr)
    a5 = load_audio_array(EX_5, sr)
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "Example: This excerpt is counted in groups of 3. Listen carefully:"},
            {"type": "audio", "audio": a3, "sampling_rate": sr},
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": "Example: This excerpt is counted in groups of 4. Listen carefully:"},
            {"type": "audio", "audio": a4, "sampling_rate": sr},
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": "Example: This excerpt is counted in groups of 5. Listen carefully:"},
            {"type": "audio", "audio": a5, "sampling_rate": sr},
        ]},
    ]
    return msgs, [a3, a4, a5]

def build_fewshot_messages_COT(sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    a3 = load_audio_array(EX_3, sr)
    a4 = load_audio_array(EX_4, sr)
    a5 = load_audio_array(EX_5, sr)

    ex1_user = {"role": "user", "content": [
        {"type": "text", "text": COT_PROMPT_TEMPLATE},
        {"type": "audio", "audio": a3, "sampling_rate": sr},
    ]}
    ex1_assistant = {"role": "assistant", "content": [{"type": "text", "text":
        "Step 1: Establish a steady beat.\n"
        "Step 2: A strong downbeat recurs every three beats (ONE-two-three | ONE-two-three...).\n"
        "A. Groups of 3"
    }]}

    ex2_user = {"role": "user", "content": [
        {"type": "text", "text": COT_PROMPT_TEMPLATE},
        {"type": "audio", "audio": a4, "sampling_rate": sr},
    ]}
    ex2_assistant = {"role": "assistant", "content": [{"type": "text", "text":
        "Step 1: Find the pulse.\n"
        "Step 2: The accent pattern repeats every four beats (ONE-two-three-four | ONE-two-three-four).\n"
        "B. Groups of 4"
    }]}

    ex3_user = {"role": "user", "content": [
        {"type": "text", "text": COT_PROMPT_TEMPLATE},
        {"type": "audio", "audio": a5, "sampling_rate": sr},
    ]}
    ex3_assistant = {"role": "assistant", "content": [{"type": "text", "text":
        "Step 1: Lock to the beat.\n"
        "Step 2: The cycle resolves every five beats (ONE-two-three-four-five | ONE-two-three-four-five).\n"
        "C. Groups of 5"
    }]}

    messages = [ex1_user, ex1_assistant, ex2_user, ex2_assistant, ex3_user, ex3_assistant]
    audio_list = [a3, a4, a5]
    return messages, audio_list

def run_examples_and_confirm(
    *,
    model: Qwen2_5OmniForConditionalGeneration,
    processor: Qwen2_5OmniProcessor,
    sysinstr_text: str,
    sr: int,
    log: logging.Logger,
) -> Tuple[List[Dict[str, Any]], str, List[Any]]:
    """Returns (history_with_examples_and_confirm, confirmation_string, audio_list)."""
    history: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": sysinstr_text}]}
    ]
    is_cot = (sysinstr_text == SYSINSTR_COT)

    if is_cot:
        fewshot_msgs, fewshot_audio = build_fewshot_messages_COT(sr)
    else:
        fewshot_msgs, fewshot_audio = build_fewshot_messages_SYSINST(sr)

    history_with_examples = history + fewshot_msgs
    cumulative_audios = list(fewshot_audio)

    confirm_user = {
        "role": "user",
        "content": [{"type": "text", "text":
            'After examining the examples above, please respond with "Yes, I understand." if you understand the task, or "No, I don\'t understand." if you do not.'}]
    }

    completion = qwen_generate_text(
        model=model,
        processor=processor,
        messages=history_with_examples + [confirm_user],
        audio=cumulative_audios,
    )
    print("Confirmation response:", completion)
    log.info(f"Confirmation response: {completion}")

    history_with_examples_and_confirm = history_with_examples + [
        {"role": "assistant", "content": [{"type": "text", "text": completion.strip()}]},
    ]
    return history_with_examples_and_confirm, completion.strip(), cumulative_audios

def _auto_max_memory(reserve_gib: int = 6) -> dict:
    mm = {}
    for i in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        mm[i] = f"{max(1, int(total_gib - reserve_gib))}GiB"
    mm["cpu"] = "120GiB"
    return mm

# =============================
# Single-run experiment
# =============================
def run_once(*, mode: str, group: str, seed: int, log_filename: str) -> None:
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}

    configure_logging(log_filename)
    logging.info(f"=== RUN START (Meter Identification • CHAT+SysInst • {mode} • {group}) ===")
    logging.info(f"Config: model={MODEL_ID}, temp=1.0, seed={seed}, group={group}, log={log_filename}")

    # Seeds
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
        max_mem = _auto_max_memory(reserve_gib=6)  # tune reserve as needed
        device_map = "balanced"
    else:
        max_mem = None
        device_map = "auto"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        LOCAL_QWEN_DIR,
        local_files_only=True,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation="sdpa",
        enable_audio_output=False,
        max_memory=max_mem,
    ).eval()

    # CoT needs a small headroom; SYSINST does not
    set_generation_config(model, min_new_tokens=48 if mode == "COT" else None)
    sr = processor.feature_extractor.sampling_rate

    # === Examples + confirmation ===
    base_history, _confirm, base_audios = run_examples_and_confirm(
        model=model,
        processor=processor,
        sysinstr_text=(SYSINSTR_PLAIN if mode == "SYSINST" else SYSINSTR_COT),
        sr=sr,
        log=logging.getLogger(),
    )

    # Rolling memory: only last K user/assistant pairs (with their audios)
    # Each deque element is: (msgs_list, audios_list)
    #   msgs_list = [user_turn, assistant_turn]
    #   audios_list = [a1]  # audio attached to that user_turn
    rolling_pairs = deque(maxlen=KEEP_LAST_K)

    # Stimuli (shuffle per seed)
    question_stims = stimuli_list_group(group)
    random.seed(seed)
    random.shuffle(question_stims)

    print(f"\n--- Task: Meter Identification — CHAT+SysInst {mode} • {group} | model={MODEL_ID} | temp=1.0 | seed={seed} ---\n")
    logging.info(f"\n--- Task: Meter Identification — CHAT+SysInst {mode} • {group} ---\n")

    correct = 0
    total = len(question_stims)

    for idx, q in enumerate(question_stims, start=1):
        print(f"\n--- Question {idx} ---\n")
        logging.info(f"\n--- Question {idx} ---\n")

        f1 = q["file"]
        logging.info(f"Stimulus: file={f1}")

        a1 = load_audio_array(f1, sr)

        if mode == "SYSINST":
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Listen to the following excerpt and identify the meter."},
                    {"type": "audio", "audio": a1, "sampling_rate": sr},
                    {"type": "text", "text":
                     "Reply with exactly ONE of the following lines:\n"
                     "A. Groups of 3\n"
                     "OR\n"
                     "B. Groups of 4\n"
                     "OR\n"
                     "C. Groups of 5\n"},
                ],
            }
        else:
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": COT_PROMPT_TEMPLATE},
                    {"type": "audio", "audio": a1, "sampling_rate": sr},
                ],
            }

        # Build messages for this call:
        #   base_history (system + examples + confirm)
        #   + flattened last K pairs
        #   + current user_turn
        past_msgs_flat = list(chain.from_iterable(msgs for msgs, _ in rolling_pairs))
        messages = base_history + past_msgs_flat + [user_turn]

        # Build aligned audio list:
        past_audios_flat = list(chain.from_iterable(auds for _, auds in rolling_pairs))
        audio_for_call = list(base_audios) + past_audios_flat + [a1]

        # Safety check
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

        # Build assistant message for the rolling window (regardless of parse success)
        assistant_turn = {"role": "assistant", "content": [{"type": "text", "text": completion}]}
        # Update rolling memory to include THIS pair for future turns
        rolling_pairs.append(([user_turn, assistant_turn], [a1]))

        # Parse decision (prefer last occurrence)
        model_answer = parse_final_decision(completion)
        if not model_answer:
            print("Evaluation: Failed. Could not parse the final answer phrase.")
            logging.error("Parse Error: missing/malformed final answer phrase.")
            # continue to next trial; rolling_pairs already updated
            continue

        logging.info(f"Parsed Final Answer: {model_answer}")

        # Ground truth via explicit mapping
        expected = expected_for_meter(f1)
        if not expected:
            print("Evaluation: Unknown stimulus label for expected answer.")
            logging.error("Missing gold label for stimulus basename.")
        else:
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
