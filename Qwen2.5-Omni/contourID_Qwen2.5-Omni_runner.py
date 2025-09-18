# contourID_Qwen2.5-Omni_CHAT_SysInst_master.py
import os
import re
import gc
import random
import logging
from typing import List, Dict, Any, Tuple

import warnings
warnings.filterwarnings("ignore")

# ===== Runtime knobs (match other Qwen runners) =====
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
STIM_ROOT = "stimuli"   # relative canonical root in your HPC layout
MAX_NEW_TOKENS = 8192

# Canonical answer strings (EXACT)
A_CANON = 'A. Arch (ascending and then descending)'
B_CANON = 'B. Inverted Arch (descending and then ascending)'
C_CANON = 'C. Ascending (pitch raises over time)'
D_CANON = 'D. Descending (pitch falls over time)'

# Robust patterns (prefer LAST occurrence)
A_PAT = re.compile(r'(?i)\bA\.\s*Arch\s*\(ascending\s+and\s+then\s+descending\)')
B_PAT = re.compile(r'(?i)\bB\.\s*Inverted\s+Arch\s*\(descending\s+and\s+then\s+ascending\)')
C_PAT = re.compile(r'(?i)\bC\.\s*Ascending\s*\(pitch\s+raises\s+over\s+time\)')
D_PAT = re.compile(r'(?i)\bD\.\s*Descending\s*\(pitch\s+falls\s+over\s+time\)')

# =============================
# System instructions (VERBATIM)
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
    Example: contourID_Qwen2.5-Omni_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"contourID_Qwen2.5-Omni_CHAT_{mode}_{group}_seed{seed}.log"

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
    """Normalize path:
       - absolute: as-is
       - starts with 'stimuli/': as-is
       - else: join with STIM_ROOT
    """
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
    """Return canonical A/B/C/D string, or '' if not found. Prefer LAST occurrence."""
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

def contour_label_from_filename(path: str) -> str:
    """Infer expected label from filename (InvArch before Arch)."""
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
# Build few-shot sequences
# =============================
def build_fewshot_messages_COT(sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """
    Four in-context examples with brief CoT + strict final line.
    Ex1: A. Arch
    Ex2: B. Inverted Arch
    Ex3: C. Ascending
    Ex4: D. Descending
    """
    a_arch   = load_audio_array(EX_ARCH, sr)
    a_inv    = load_audio_array(EX_INVARCH, sr)
    a_asc    = load_audio_array(EX_ASC, sr)
    a_desc   = load_audio_array(EX_DESC, sr)

    ex1_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "audio", "audio": a_arch, "sampling_rate": sr},
        ],
    }
    ex1_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text":
            "Step 1: Single melodic line rises to a clear apex around the midpoint, then falls.\n"
            "Step 2: This global trend is rise-then-fall.\n"
            "A. Arch (ascending and then descending)"
        }]
    }

    ex2_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "audio", "audio": a_inv, "sampling_rate": sr},
        ],
    }
    ex2_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text":
            "Step 1: Single melodic line descends early to a low point, then rises toward the end.\n"
            "Step 2: This global trend is fall-then-rise.\n"
            "B. Inverted Arch (descending and then ascending)"
        }]
    }

    ex3_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "audio", "audio": a_asc, "sampling_rate": sr},
        ],
    }
    ex3_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text":
            "Step 1: Single melodic line with a clear overall upward movement from start to finish.\n"
            "Step 2: This matches an overall ascending contour.\n"
            "C. Ascending (pitch raises over time)"
        }]
    }

    ex4_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "audio", "audio": a_desc, "sampling_rate": sr},
        ],
    }
    ex4_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text":
            "Step 1: Single melodic line moves downward overall from beginning to end.\n"
            "Step 2: This matches a descending contour.\n"
            "D. Descending (pitch falls over time)"
        }]
    }

    messages = [ex1_user, ex1_assistant, ex2_user, ex2_assistant, ex3_user, ex3_assistant, ex4_user, ex4_assistant]
    audio_list = [a_arch, a_inv, a_asc, a_desc]  # matches audio placeholders in user turns
    return messages, audio_list

def build_fewshot_messages_SYSINST(sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    a_arch   = load_audio_array(EX_ARCH, sr)
    a_inv    = load_audio_array(EX_INVARCH, sr)
    a_asc    = load_audio_array(EX_ASC, sr)
    a_desc   = load_audio_array(EX_DESC, sr)

    ex1_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Example 1: The following audio example represents the contour 'A. Arch (ascending and then descending)'. Listen carefully:"},
            {"type": "text", "text": "Audio example:"},
            {"type": "audio", "audio": a_arch, "sampling_rate": sr},
        ],
    }
    ex2_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Example 2: The following audio example represents the contour 'B. Inverted Arch (descending and then ascending)'. Listen carefully:"},
            {"type": "text", "text": "Audio example:"},
            {"type": "audio", "audio": a_inv, "sampling_rate": sr},
        ],
    }
    ex3_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Example 3: The following audio example represents the contour 'C. Ascending (pitch raises over time)'. Listen carefully:"},
            {"type": "text", "text": "Audio example:"},
            {"type": "audio", "audio": a_asc, "sampling_rate": sr},
        ],
    }
    ex4_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Example 4: The following audio example represents the contour 'D. Descending (pitch falls over time)'. Listen carefully:"},
            {"type": "text", "text": "Audio example:"},
            {"type": "audio", "audio": a_desc, "sampling_rate": sr},
        ],
    }

    messages = [ex1_user, ex2_user, ex3_user, ex4_user]
    audio_list = [a_arch, a_inv, a_asc, a_desc]
    return messages, audio_list

def run_examples_and_confirm(
    *,
    model: Qwen2_5OmniForConditionalGeneration,
    processor: Qwen2_5OmniProcessor,
    sysinstr_text: str,
    sr: int,
    log: logging.Logger,
) -> Tuple[List[Dict[str, Any]], str, List[Any]]:
    """Return (history_with_examples_and_confirm, confirmation_text, cumulative_audio_list)."""
    # System message
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

    # Confirmation prompt
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
        audio=cumulative_audios,  # confirm has no new audio
    )
    print("Confirmation response:", completion)
    log.info(f"Confirmation response: {completion}")

    history_with_examples_and_confirm = history_with_examples + [
        {"role": "assistant", "content": [{"type": "text", "text": completion.strip()}]},
    ]
    return history_with_examples_and_confirm, completion.strip(), cumulative_audios

# =============================
# One full run (SYSINST / COT, GroupA / GroupB)
# =============================
def run_once(*, mode: str, group: str, seed: int, log_filename: str) -> None:
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    configure_logging(log_filename)
    logging.info(f"=== RUN START (Contour ID • CHAT+SysInst • {mode} • {group}) ===")
    logging.info(f"Config: model={MODEL_ID}, temp=1.0, seed={seed}, group={group}, log={log_filename}")

    # Reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Local snapshot path (adjust if needed)
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

    # Decoding knobs: ensure CoT produces some reasoning
    set_generation_config(model, min_new_tokens=80 if mode == "COT" else None)
    sr = processor.feature_extractor.sampling_rate

    # === Examples + confirmation ===
    history, _confirm, cumulative_audios = run_examples_and_confirm(
        model=model, processor=processor, sysinstr_text=(SYSINSTR_PLAIN if mode == "SYSINST" else SYSINSTR_COT),
        sr=sr, log=logging.getLogger()
    )

    # Stimuli (shuffle per seed to mirror Gemini)
    question_stims = stimuli_list_group(group)
    random.seed(seed)
    random.shuffle(question_stims)

    print(f"\n--- Task: Contour Identification — CHAT+SysInst {mode} • {group} | model={MODEL_ID} | temp=1.0 | seed={seed} ---\n")
    logging.info(f"\n--- Task: Contour Identification — CHAT+SysInst {mode} • {group} ---\n")

    correct = 0
    total = len(question_stims)

    for idx, q in enumerate(question_stims, start=1):
        print(f"\n--- Question {idx} ---\n")
        logging.info(f"\n--- Question {idx} ---\n")

        f = q["file"]
        logging.info(f"Stimulus: file={f}")

        # Load trimmed audio array for this trial
        a = load_audio_array(f, sr)

        if mode == "SYSINST":
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are a participant in a psychological experiment on music perception. Please decide which option best describes the overall shape of the scale in the audio example:"},
                    {"type": "text", "text": 'A. Arch (ascending and then descending)\nB. Inverted Arch (descending and then ascending)\nC. Ascending (pitch raises over time)\nD. Descending (pitch falls over time)'},
                    {"type": "text", "text": "This is the audio example. Listen carefully now:"},
                    {"type": "audio", "audio": a, "sampling_rate": sr},
                    {"type": "text", "text": 'Now choose which option best represents the overall contour of the audio example. Answer exactly one of the four strings (and nothing else):\n"A. Arch (ascending and then descending)"\n"B. Inverted Arch (descending and then ascending)"\n"C. Ascending (pitch raises over time)"\n"D. Descending (pitch falls over time)"'},
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

        # Build messages + audio for THIS call (include full prior history audio)
        messages = history + [user_turn]
        audio_for_call = list(cumulative_audios) + [a]

        # Safety check: audio alignment with placeholders
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
        model_answer = parse_final_choice(completion)
        if not model_answer:
            print("Evaluation: Failed. Could not parse a valid final choice.")
            logging.error("Parse Error: missing/malformed final choice line.")
            # still append turn to maintain strict history
            history.append(user_turn)
            history.append({"role": "assistant", "content": [{"type": "text", "text": completion}]})
            cumulative_audios.append(a)
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

        # Append turn and completion to persistent history; grow audio list
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
