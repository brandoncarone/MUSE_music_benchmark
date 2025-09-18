# oddballs_Qwen2.5-Omni_CHAT_SysInst_master.py
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

# Canonical answer strings and robust patterns
YES_CANON = "Yes, these are the same exact melody."
NO_CANON  = "No, these are not the same exact melody."

YES_PAT = re.compile(r'(?i)\byes,\s*these\s+are\s+the\s+same\s+exact\s+melody\.')
NO_PAT  = re.compile(r'(?i)\bno,\s*these\s+are\s+not\s+the\s+same\s+exact\s+melody\.')

# =============================
# System instructions (VERBATIM)
# =============================
SYSINSTR_PLAIN = """You are a participant in a psychological experiment on music perception. In each question, you will be given:
    1. A brief instruction about the specific listening task.
    2. Two audio examples to listen to.

Your task is to decide whether the two audio examples are the same exact melody, or whether an "Oddball" is present. 
An “Oddball” in a musical or auditory experiment is simply a note or sound that doesn’t fit with what you’d expect based 
on what you’ve been hearing. Imagine you’re listening to a melody where all the notes line up nicely in the same key—then 
suddenly, one note is out of key. This unexpected note is the "oddball". If the note is present multiple times in the melody, 
then you will hear the oddball more than once.

Valid responses are:
“Yes, these are the same exact melody.” if they are exactly the same, or 
“No, these are not the same exact melody.” if you notice an oddball.

Before you begin the task, I will provide you with examples of two excerpts representing the same exact melody, as well as
examples of two excerpts where one contains oddballs so that you better understand the task. After examining the 
examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't 
understand the task.

Please provide no additional commentary beyond the short answers previously mentioned.
"""

SYSINSTR_COT = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. Two audio examples to listen to.

Your task is to decide whether the two audio examples are the same exact melody, or whether an ODDBALL is present in one of them.

Definitions and constraints:
- “Oddball” = one or more unexpected notes that do not match the expected melody (e.g., out-of-key or altered notes). The oddball may occur more than once.
- Judge exact pitch content at corresponding positions (no transposition or octave equivalence). Small timing deviations or leading/trailing silence can be ignored.
- If every note at each aligned position matches exactly, there is NO oddball (same exact melody).
- If any aligned note differs (pitch substitution/out-of-key change), an oddball is present and the melodies are NOT the same.

Valid responses are:
"Yes, these are the same exact melody."
or
"No, these are not the same exact melody."

Before you begin the task, I will provide you with examples of two excerpts representing the same exact melody, as well as
examples of two excerpts where one contains oddballs so that you better understand the task. After examining the 
examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't 
understand the task.

After any reasoning, end with exactly one line:
Yes, these are the same exact melody.
OR
No, these are not the same exact melody."""

# --- COT per-trial prompt (VERBATIM) ---
COT_PROMPT_TEMPLATE = """Analyze the two music excerpts and decide if they are the SAME exact melody or if an ODDBALL is present.

Step 1: For each audio, identify the monophonic note sequence (pitch over time). Ignore small timing differences and leading/trailing silence.

Step 2: Align the two sequences by their note order and compare pitch at each corresponding position (no transposition or octave equivalence).

Step 3: Decision rule:
- If all corresponding pitches match exactly, there is no oddball, and they are the same exact melody.
- If any pitch differs (e.g., out-of-key substitution or altered note), there is an oddball present, and they are not the same melody.

Step 4: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
Yes, these are the same exact melody.
OR
No, these are not the same exact melody.
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
    Example: oddballs_Qwen2.5-Omni_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"oddballs_Qwen2.5-Omni_CHAT_{mode}_{group}_seed{seed}.log"

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
    """
    Return list of dicts: {"file1": abs_path, "file2": abs_path}
    """
    assert group in {"GroupA", "GroupB"}
    if group == "GroupA":
        pairs = [
            ("stimuli/M1_EbMaj_90.wav",            "stimuli/M1_EbMaj_90.wav"),
            ("stimuli/M1_EbMaj_90.wav",            "stimuli/M1_Odd_EbMaj_90.wav"),
            ("stimuli/M2_Abm_155_3_4.wav",         "stimuli/M2_Abm_155_3_4.wav"),
            ("stimuli/M2_Abm_155_3_4.wav",         "stimuli/M2_Odd_Abm_155_3_4.wav"),
            ("stimuli/M3_DbMaj_78.wav",            "stimuli/M3_DbMaj_78.wav"),
            ("stimuli/M8_FMaj_95_Piano.wav",       "stimuli/M8_Odd_FMaj_95_Piano.wav"),
            ("stimuli/M9_Gm_200_3_4_Piano.wav",    "stimuli/M9_Gm_200_3_4_Piano.wav"),
            ("stimuli/M9_Gm_200_3_4_Piano.wav",    "stimuli/M9_Odd_Gm_200_3_4_Piano.wav"),
            ("stimuli/M10_Fm_165_3_4.wav",         "stimuli/M10_Fm_165_3_4.wav"),
            ("stimuli/M10_Fm_165_3_4.wav",         "stimuli/M10_Odd_Fm_165_3_4.wav"),
        ]
    else:
        pairs = [
            ("stimuli/M3_DbMaj_78.wav",            "stimuli/M3_Odd_DbMaj_78.wav"),
            ("stimuli/M4_EMaj_130.wav",            "stimuli/M4_EMaj_130.wav"),
            ("stimuli/M4_EMaj_130.wav",            "stimuli/M4_Odd_EMaj_130.wav"),
            ("stimuli/M5_Dm_100.wav",              "stimuli/M5_Dm_100.wav"),
            ("stimuli/M5_Dm_100.wav",              "stimuli/M5_Odd_Dm_100.wav"),
            ("stimuli/M6_Cm_120_Piano.wav",        "stimuli/M6_Cm_120_Piano.wav"),
            ("stimuli/M6_Cm_120_Piano.wav",        "stimuli/M6_Odd_Cm_120_Piano.wav"),
            ("stimuli/M7_CMaj_140_Piano.wav",      "stimuli/M7_CMaj_140_Piano.wav"),
            ("stimuli/M7_CMaj_140_Piano.wav",      "stimuli/M7_Odd_CMaj_140_Piano.wav"),
            ("stimuli/M8_FMaj_95_Piano.wav",       "stimuli/M8_FMaj_95_Piano.wav"),
        ]
    return [{"file1": _ppath(a), "file2": _ppath(b)} for a, b in pairs]

# Few-shot example file paths
EX_SAME = _p("M11_CMaj_180_Piano.wav")
EX_ODD  = _p("M11_Odd_CMaj_180_Piano.wav")

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
        completion = processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    del inputs, out_ids, new_tokens
    torch.cuda.empty_cache()
    return completion

# =============================
# Few-shot builders + confirmation
# =============================
def build_fewshot_messages_COT(sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """
    Two in-context examples with brief CoT and strict final line (assistant messages included).
    Ex1: SAME → ends with: Yes, these are the same exact melody.
    Ex2: ODDBALL present → ends with: No, these are not the same exact melody.
    """
    a_same1 = load_audio_array(EX_SAME, sr)
    a_same2 = load_audio_array(EX_SAME, sr)
    a_odd   = load_audio_array(EX_ODD,  sr)

    # Example 1 — SAME exact melody
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
            "Step 1: Extract monophonic note sequences for both excerpts.\n"
            "Step 2: Align note-by-note; pitches match at every position.\n"
            "Step 3: No substitutions or out-of-key notes detected, thus, no oddball.\n"
            "Yes, these are the same exact melody."
        }]
    }

    # Example 2 — ODDBALL present
    ex2_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "audio", "audio": a_same1, "sampling_rate": sr},
            {"type": "audio", "audio": a_odd,   "sampling_rate": sr},
        ],
    }
    ex2_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text":
            "Step 1: Extract monophonic sequences.\n"
            "Step 2: Alignment reveals positions where the second excerpt’s pitch deviates (out-of-key substitutions).\n"
            "Step 3: Oddball(s) present, thus, the melodies are not the same.\n"
            "No, these are not the same exact melody."
        }]
    }

    messages = [ex1_user, ex1_assistant, ex2_user, ex2_assistant]
    audio_list = [a_same1, a_same2, a_same1, a_odd]  # align with user-turn audio order
    return messages, audio_list

def build_fewshot_messages_SYSINST(sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    a_same1 = load_audio_array(EX_SAME, sr)
    a_same2 = load_audio_array(EX_SAME, sr)
    a_odd   = load_audio_array(EX_ODD,  sr)

    ex1_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Example 1: The following two audio examples are the same exact melody. Listen carefully:"},
            {"type": "text", "text": "Audio example number 1:"},
            {"type": "audio", "audio": a_same1, "sampling_rate": sr},
            {"type": "text", "text": "Audio example number 2:"},
            {"type": "audio", "audio": a_same2, "sampling_rate": sr},
        ],
    }

    ex2_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Example 2: The following two audio examples are different; an oddball is present in one excerpt (it may occur more than once). Listen carefully:"},
            {"type": "text", "text": "Audio example number 1:"},
            {"type": "audio", "audio": a_same1, "sampling_rate": sr},
            {"type": "text", "text": "Audio example number 2:"},
            {"type": "audio", "audio": a_odd,   "sampling_rate": sr},
        ],
    }

    messages = [ex1_user, ex2_user]
    audio_list = [a_same1, a_same2, a_same1, a_odd]
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
    logging.info(f"=== RUN START (Oddball Detection • CHAT+SysInst • {mode} • {group}) ===")
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

    # CoT needs headroom to reason
    set_generation_config(model, min_new_tokens=80 if mode == "COT" else None)
    sr = processor.feature_extractor.sampling_rate

    # === Examples + confirmation ===
    history, _confirm, cumulative_audios = run_examples_and_confirm(
        model=model, processor=processor, sysinstr_text=(SYSINSTR_PLAIN if mode == "SYSINST" else SYSINSTR_COT),
        sr=sr, log=logging.getLogger()
    )

    # Fixed stimuli, then shuffle per seed (mirror Gemini)
    question_stims = stimuli_pairs_group(group)
    random.seed(seed)
    random.shuffle(question_stims)

    print(f"\n--- Task: Oddball Detection — CHAT+SysInst {mode} • {group} | model={MODEL_ID} | temp=1.0 | seed={seed} ---\n")
    logging.info(f"\n--- Task: Oddball Detection — CHAT+SysInst {mode} • {group} ---\n")

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
                    {"type": "text", "text": 'Reply with exactly ONE of the following lines:\n'
                                              'Yes, these are the same exact melody.\n'
                                              'OR\n'
                                              'No, these are not the same exact melody.'},
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

        # Safety check: audio placeholders must match
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

        # Parse decision (prefer last occurrence)
        model_answer = parse_final_decision(completion)
        if not model_answer:
            print("Evaluation: Failed. Could not parse the final answer phrase.")
            logging.error("Parse Error: missing/malformed final answer phrase.")
            # Maintain strict chat history
            history.append(user_turn)
            history.append({"role": "assistant", "content": [{"type": "text", "text": completion}]})
            cumulative_audios.extend([a1, a2])
            continue

        logging.info(f"Parsed Final Answer: {model_answer}")

        # Ground truth: identical filenames => SAME; otherwise NOT SAME
        base1 = os.path.basename(f1)
        base2 = os.path.basename(f2)
        expected = YES_CANON if base1 == base2 else NO_CANON

        if model_answer == expected:
            correct += 1
            print("Evaluation: Correct!")
            logging.info("Evaluation: Correct")
        else:
            print("Evaluation: Incorrect.")
            logging.info(f"Evaluation: Incorrect (expected={expected})")

        # Append turn and completion; grow audio list
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
