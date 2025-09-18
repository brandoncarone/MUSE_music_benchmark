# chord_quality_Qwen2.5-Omni_CHAT_SysInst_master.py
import os
import re
import gc
import random
import logging
from typing import List, Dict, Any, Tuple

import warnings
warnings.filterwarnings("ignore")

# ===== Runtime knobs (consistent with other Qwen runners) =====
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
STIM_ROOT = "stimuli"  # your repo root; absolute paths still work
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

def make_log_filename(*, mode: str, group: str, seed: int) -> str:
    """
    Chat + System Instructions runner, two modes:
      - SYSINST (plain)
      - COT     (reasoning)
    Two stimulus groups: GroupA / GroupB
    Example: chord_quality_Qwen2.5-Omni_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"chord_quality_Qwen2.5-Omni_CHAT_{mode}_{group}_seed{seed}.log"

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
    """Normalize path: absolute → as-is; otherwise join with STIM_ROOT."""
    if os.path.isabs(p):
        return p
    if p.startswith(STIM_ROOT + os.sep) or p.startswith(STIM_ROOT + "/"):
        return p
    return os.path.join(STIM_ROOT, p)

def _p(name: str) -> str:
    return os.path.join(STIM_ROOT, name)

def load_audio_array(path: str, sr: int):
    y, _ = librosa.load(path, sr=sr, mono=True)
    # Efficiency: trim leading/trailing silence to save tokens
    y, _ = librosa.effects.trim(y, top_db=35)
    return y

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

# Few-shot example file paths
EX_MAJOR = _p("Abmajor_Piano_120.wav")   # Major example
EX_MINOR = _p("Aminor_Guitar_120.wav")   # Minor example

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
def build_fewshot_messages_SYSINST(sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    maj = load_audio_array(EX_MAJOR, sr)
    minc = load_audio_array(EX_MINOR, sr)
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "Example 1: The following audio example is a Major chord. Listen carefully:"},
            {"type": "text", "text": "Audio example:"},
            {"type": "audio", "audio": maj, "sampling_rate": sr},
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": "Example 2: The following audio example is a Minor chord. Listen carefully:"},
            {"type": "text", "text": "Audio example:"},
            {"type": "audio", "audio": minc, "sampling_rate": sr},
        ]},
    ]
    return msgs, [maj, minc]

def build_fewshot_messages_COT(sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    maj = load_audio_array(EX_MAJOR, sr)
    minc = load_audio_array(EX_MINOR, sr)

    ex1_user = {"role": "user", "content": [
        {"type": "text", "text": COT_PROMPT_TEMPLATE},
        {"type": "audio", "audio": maj, "sampling_rate": sr},
    ]}
    ex1_assistant = {"role": "assistant", "content": [{"type": "text", "text":
        "Step 1: Hear the block chord then the arpeggiation; treat inversion/voicing as irrelevant.\n"
        "Step 2: Among the chord tones there is a MAJOR third span, consistent with a major triad.\n"
        "A. Major"
    }]}

    ex2_user = {"role": "user", "content": [
        {"type": "text", "text": COT_PROMPT_TEMPLATE},
        {"type": "audio", "audio": minc, "sampling_rate": sr},
    ]}
    ex2_assistant = {"role": "assistant", "content": [{"type": "text", "text":
        "Step 1: Listen to the chord and then the arpeggiation; ignore register/voicing.\n"
        "Step 2: The chord tones include a MINOR third, indicating a minor triad.\n"
        "B. Minor"
    }]}

    messages = [ex1_user, ex1_assistant, ex2_user, ex2_assistant]
    audio_list = [maj, minc]
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

# =============================
# Single-run experiment
# =============================
def run_once(*, mode: str, group: str, seed: int, log_filename: str) -> None:
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}

    configure_logging(log_filename)
    logging.info(f"=== RUN START (Chord Quality • CHAT+SysInst • {mode} • {group}) ===")
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

    # CoT needs a small headroom; SYSINST does not
    set_generation_config(model, min_new_tokens=40 if mode == "COT" else None)
    sr = processor.feature_extractor.sampling_rate

    # === Examples + confirmation ===
    history, _confirm, cumulative_audios = run_examples_and_confirm(
        model=model,
        processor=processor,
        sysinstr_text=(SYSINSTR_PLAIN if mode == "SYSINST" else SYSINSTR_COT),
        sr=sr,
        log=logging.getLogger(),
    )

    # Stimuli (shuffle per seed)
    stim_files = stimuli_files_group(group)
    random.seed(seed)
    random.shuffle(stim_files)

    print(f"\n--- Task: Chord Quality (Major vs Minor) — CHAT+SysInst {mode} • {group} | model={MODEL_ID} | temp=1.0 | seed={seed} ---\n")
    logging.info(f"\n--- Task: Chord Quality (Major vs Minor) — CHAT+SysInst {mode} • {group} ---\n")

    correct = 0
    total = len(stim_files)

    for idx, f in enumerate(stim_files, start=1):
        print(f"\n--- Question {idx} ---\n")
        logging.info(f"\n--- Question {idx} ---\n")
        logging.info(f"Stimulus: file={f}")

        a = load_audio_array(f, sr)

        if mode == "SYSINST":
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the audio excerpt."},
                    {"type": "audio", "audio": a, "sampling_rate": sr},
                    {"type": "text", "text":
                     'Reply with exactly ONE of the following lines:\n'
                     'A. Major\n'
                     'OR\n'
                     'B. Minor'},
                ],
            }
        else:
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": COT_PROMPT_TEMPLATE},
                    {"type": "audio", "audio": a, "sampling_rate": sr},
                ],
            }

        messages = history + [user_turn]
        audio_for_call = list(cumulative_audios) + [a]
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

        model_answer = parse_final_decision(completion)
        if not model_answer:
            print("Evaluation: Failed. Could not parse the final answer phrase.")
            logging.error("Parse Error: missing/malformed final answer phrase.")
            # keep context growth anyway
            history.append(user_turn)
            history.append({"role": "assistant", "content": [{"type": "text", "text": completion}]})
            cumulative_audios.append(a)
            continue

        logging.info(f"Parsed Final Answer: {model_answer}")

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

        # Grow persistent chat context
        history.append(user_turn)
        history.append({"role": "assistant", "content": [{"type": "text", "text": completion}]})
        cumulative_audios.append(a)

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
