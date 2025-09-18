# rhythm_matching_Qwen2.5-Omni_CHAT_SysInst_master.py
import os
import re
import gc
import random
import logging
from typing import List, Dict, Any, Tuple

import warnings
warnings.filterwarnings("ignore")

# ===== Runtime knobs (align with other Qwen runners) =====
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
STIM_ROOT = "stimuli"   # relative canonical root
MAX_NEW_TOKENS = 8192

# Canonical answer strings and robust patterns (VERBATIM)
YES_CANON = 'Yes, these are the same exact drum set patterns.'
NO_CANON = 'No, these are not the same exact drum set patterns.'

YES_PAT = re.compile(r'(?i)\byes,\s*these\s+are\s+the\s+same\s+exact\s+drum\s+set\s+patterns\.')
NO_PAT = re.compile(r'(?i)\bno,\s*these\s+are\s+not\s+the\s+same\s+exact\s+drum\s+set\s+patterns\.')

# =============================
# System instructions (VERBATIM)
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
# Logging utilities (identical style)
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
    Example: rhythm_matching_Qwen2.5-Omni_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"rhythm_matching_Qwen2.5-Omni_CHAT_{mode}_{group}_seed{seed}.log"

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
    """Normalize: abs -> as-is; starts with 'stimuli/' -> as-is; else join with STIM_ROOT."""
    if os.path.isabs(p):
        return p
    if p.startswith(STIM_ROOT + os.sep) or p.startswith(STIM_ROOT + "/"):
        return p
    return os.path.join(STIM_ROOT, p)

def _p(name: str) -> str:
    return os.path.join(STIM_ROOT, name)

def load_audio_array(path: str, sr: int):
    y, _ = librosa.load(path, sr=sr, mono=True)
    # QWEN TOKEN EFFICIENCY: trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=35)
    return y

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
            ("stimuli/Beat_4_140_34.wav",    "stimuli/Beat_4_140_34.wav"),
            ("stimuli/Beat_4_140_34.wav",    "stimuli/Beat_5_140.wav"),
            ("stimuli/Beat_5_140.wav",       "stimuli/Beat_5_140.wav"),
            ("stimuli/Beat_5_140.wav",       "stimuli/Beat_6_140_34.wav"),
        ]
    else:
        pairs = [
            ("stimuli/Beat_6_140_34.wav",    "stimuli/Beat_6_140_34.wav"),
            ("stimuli/Beat_6_140_34.wav",    "stimuli/Beat_7_140.wav"),
            ("stimuli/Beat_7_140.wav",       "stimuli/Beat_7_140.wav"),
            ("stimuli/Beat_7_140.wav",       "stimuli/Beat_8_140.wav"),
            ("stimuli/Beat_8_140.wav",       "stimuli/Beat_8_140.wav"),
            ("stimuli/Beat_8_140.wav",       "stimuli/Beat_9_140_34.wav"),
            ("stimuli/Beat_9_140_34.wav",    "stimuli/Beat_9_140_34.wav"),
            ("stimuli/Beat_9_140_34.wav",    "stimuli/Beat_10_140.wav"),
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
        completion = processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    del inputs, out_ids, new_tokens
    torch.cuda.empty_cache()
    return completion

# =============================
# Few-shot builders + confirmation
# =============================
def _example_same_SYSINST(sr: int):
    a = load_audio_array(EX_SAME_A, sr)
    b = load_audio_array(EX_SAME_B, sr)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "Example 1: The following two audio excerpts represent the same exact drum set pattern."},
            {"type": "audio", "audio": a, "sampling_rate": sr},
            {"type": "text", "text": "Audio example 2:"},
            {"type": "audio", "audio": b, "sampling_rate": sr},
        ],
    }, [a, b]

def _example_diff_SYSINST(sr: int):
    b = load_audio_array(EX_SAME_A, sr)  # as in Gemini: first clip of diff uses same Beat_11
    c = load_audio_array(EX_DIFF_C, sr)  # second clip uses Beat_12
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "Example 2: The following two audio excerpts represent different drum set patterns."},
            {"type": "audio", "audio": b, "sampling_rate": sr},
            {"type": "text", "text": "Audio example 2:"},
            {"type": "audio", "audio": c, "sampling_rate": sr},
        ],
    }, [b, c]

def build_fewshot_messages_SYSINST(sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    ex1, aud1 = _example_same_SYSINST(sr)
    ex2, aud2 = _example_diff_SYSINST(sr)
    return [ex1, ex2], aud1 + aud2

def build_fewshot_messages_COT(sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    a_same1 = load_audio_array(EX_SAME_A, sr)  # Beat_11_140
    a_same2 = load_audio_array(EX_SAME_B, sr)  # Beat_11_140
    a_diff  = load_audio_array(EX_DIFF_C, sr)  # Beat_12_140

    # Example 1 — SAME pattern
    ex1_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "audio", "audio": a_same1, "sampling_rate": sr},
            {"type": "audio", "audio": a_same2, "sampling_rate": sr},
        ],
    }
    ex1_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text":
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
        }]
    }

    # Example 2 — DIFFERENT pattern (Beat_11 vs Beat_12)
    ex2_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "audio", "audio": a_same2, "sampling_rate": sr},
            {"type": "audio", "audio": a_diff,  "sampling_rate": sr},
        ],
    }
    ex2_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text":
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
        }]
    }

    messages = [ex1_user, ex1_assistant, ex2_user, ex2_assistant]
    # Audio order must match user-turn audio placeholders:
    audio_list = [a_same1, a_same2, a_same2, a_diff]
    return messages, audio_list

def run_examples_and_confirm(
    *,
    model: Qwen2_5OmniForConditionalGeneration,
    processor: Qwen2_5OmniProcessor,
    sysinstr_text: str,
    sr: int,
    log: logging.Logger,
) -> Tuple[List[Dict[str, Any]], str, List[Any]]:
    """Create chat history with examples, then ask for confirmation."""
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
        "content": [
            {"type": "text", "text":
             'After examining the examples above, please respond with "Yes, I understand." if you understand the task, or "No, I don\'t understand." if you do not.'}
        ],
    }

    completion = qwen_generate_text(
        model=model,
        processor=processor,
        messages=history_with_examples + [confirm_user],
        audio=cumulative_audios,  # confirm adds no audio
    )
    print("Confirmation response:", completion)
    log.info(f"Confirmation response: {completion}")

    history_with_examples_and_confirm = history_with_examples + [
        {"role": "assistant", "content": [{"type": "text", "text": completion.strip()}]},
    ]
    return history_with_examples_and_confirm, completion.strip(), cumulative_audios

# =============================
# Single-run experiment
# =============================
def run_once(*, mode: str, group: str, seed: int, log_filename: str) -> None:
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}

    configure_logging(log_filename)
    logging.info(f"=== RUN START (Rhythm Matching • CHAT+SysInst • {mode} • {group}) ===")
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

    # CoT needs headroom to reason; SYSINST does not
    set_generation_config(model, min_new_tokens=80 if mode == "COT" else None)
    sr = processor.feature_extractor.sampling_rate

    # === Examples + confirmation ===
    history, _confirm, cumulative_audios = run_examples_and_confirm(
        model=model, processor=processor, sysinstr_text=(SYSINSTR_PLAIN if mode == "SYSINST" else SYSINSTR_COT),
        sr=sr, log=logging.getLogger()
    )

    # Fixed stimuli for the selected group, then shuffle per seed (mirror Gemini)
    question_stims = stimuli_pairs_group(group)
    random.seed(seed)
    random.shuffle(question_stims)

    print(f"\n--- Task: Rhythm Matching — CHAT+SysInst {mode} • {group} | model={MODEL_ID} | temp=1.0 | seed={seed} ---\n")
    logging.info(f"\n--- Task: Rhythm Matching — CHAT+SysInst {mode} • {group} ---\n")

    correct = 0
    total = len(question_stims)

    for idx, q in enumerate(question_stims, start=1):
        print(f"\n--- Question {idx} ---\n")
        logging.info(f"\n--- Question {idx} ---\n")

        f1 = q["file1"]
        f2 = q["file2"]
        logging.info(f"Stimuli: file1={f1}, file2={f2}")

        # Load trimmed audio arrays
        a1 = load_audio_array(f1, sr)
        a2 = load_audio_array(f2, sr)

        # Build trial prompt
        if mode == "SYSINST":
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the first excerpt."},
                    {"type": "audio", "audio": a1, "sampling_rate": sr},
                    {"type": "text", "text": "Here is the second excerpt."},
                    {"type": "audio", "audio": a2, "sampling_rate": sr},
                    {"type": "text", "text": """Reply with exactly ONE of the following lines:
"Yes, these are the same exact drum set patterns."
OR
"No, these are not the same exact drum set patterns."
"""},
                ],
            }
        else:
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": COT_PROMPT_TEMPLATE},
                    {"type": "audio", "audio": a1, "sampling_rate": sr},
                    {"type": "audio", "audio": a2, "sampling_rate": sr},
                ],
            }

        # Build messages + audio for THIS call (full prior history included)
        messages = history + [user_turn]
        audio_for_call = list(cumulative_audios) + [a1, a2]

        # Safety: audio placeholders must match
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
            # Maintain strict history regardless
            history.append(user_turn)
            history.append({"role": "assistant", "content": [{"type": "text", "text": completion}]})
            cumulative_audios.extend([a1, a2])
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

        # Append turn + completion; grow audio list
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
