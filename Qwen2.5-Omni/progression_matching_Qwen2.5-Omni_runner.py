# chord_progression_matching_Qwen2.5-Omni_CHAT_SysInst_master.py
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

# ===== Runtime knobs (align with other Qwen runners) =====
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

# Canonical answer strings and robust patterns (accept both A/B format and bare Yes/No)
A_CANON = "A. Yes, these are the same chord progression."
B_CANON = "B. No, these are not the same chord progression."

# Accept optional leading "A." / "B." and be robust to whitespace/case
A_PAT = re.compile(r'(?i)\b(?:a\.\s*)?yes,\s*these\s+are\s+the\s+same\s+chord\s+progression\.')
B_PAT = re.compile(r'(?i)\b(?:b\.\s*)?no,\s*these\s+are\s+not\s+the\s+same\s+chord\s+progression\.')

# =============================
# System instructions (VERBATIM)
# =============================
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

# --- COT per-trial prompt (VERBATIM) ---
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

def make_log_filename(*, mode: str, group: str, seed: int) -> str:
    """
    Chat + System Instructions runner, two modes:
      - SYSINST (plain)
      - COT     (reasoning)
    Two stimulus groups: GroupA / GroupB
    Example: chord_progression_matching_Qwen2.5-Omni_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"chord_progression_matching_Qwen2.5-Omni_CHAT_{mode}_{group}_seed{seed}.log"

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

# ---- Example audio paths ----
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

_ROMAN_TOKEN = re.compile(r'^(i|ii|iii|iv|v|vi|vii)$', re.I)

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
    tokens = [_normalize_progression_token(t) for t in head.split('-') if t]
    filtered = [t for t in tokens if _ROMAN_TOKEN.match(t)]
    return "-".join(filtered)

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
        #with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
        #    out_ids = model.generate(**inputs, return_audio=False)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            out_ids = model.generate(**inputs, return_audio=False)
        new_tokens = out_ids[:, inputs["input_ids"].size(1):]
        completion = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
            0
        ].strip()

    del inputs, out_ids, new_tokens
    torch.cuda.empty_cache()
    return completion

# =============================
# Few-shot builders + confirmation
# =============================
def build_fewshot_messages_SYSINST(group: str, sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    if group == "GroupA":
        a1 = load_audio_array(EX_A_SAME_1, sr)
        a2 = load_audio_array(EX_A_SAME_2, sr)
        b1 = load_audio_array(EX_A_DIFF_1, sr)
        b2 = load_audio_array(EX_A_DIFF_2, sr)
    else:
        a1 = load_audio_array(EX_B_SAME_1, sr)
        a2 = load_audio_array(EX_B_SAME_2, sr)
        b1 = load_audio_array(EX_B_DIFF_1, sr)
        b2 = load_audio_array(EX_B_DIFF_2, sr)

    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "Example 1: The following two audio excerpts represent the same chord progression."},
            {"type": "text", "text": "Excerpt 1:"},
            {"type": "audio", "audio": a1, "sampling_rate": sr},
            {"type": "text", "text": "Excerpt 2:"},
            {"type": "audio", "audio": a2, "sampling_rate": sr},
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": "Example 2: The following two audio excerpts represent different chord progressions."},
            {"type": "text", "text": "Excerpt 1:"},
            {"type": "audio", "audio": b1, "sampling_rate": sr},
            {"type": "text", "text": "Excerpt 2:"},
            {"type": "audio", "audio": b2, "sampling_rate": sr},
        ]},
    ]
    return msgs, [a1, a2, b1, b2]

def build_fewshot_messages_COT(group: str, sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    if group == "GroupA":
        same1 = load_audio_array(EX_A_SAME_1, sr)
        same2 = load_audio_array(EX_A_SAME_2, sr)
        diff1 = load_audio_array(EX_A_DIFF_1, sr)
        diff2 = load_audio_array(EX_A_DIFF_2, sr)

        ex1_user = {"role": "user", "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "text", "text": "Excerpt 1:"},
            {"type": "audio", "audio": same1, "sampling_rate": sr},
            {"type": "text", "text": "Excerpt 2:"},
            {"type": "audio", "audio": same2, "sampling_rate": sr},
        ]}
        ex1_assistant = {"role": "assistant", "content": [{"type": "text", "text":
            "Step 1: Segment both excerpts by harmonic rhythm; abstract functions ignoring inversions/extensions.\n"
            "Excerpt 1: The first excerpt follows a vi-IV-I-V in the key of G major.\n"
            "Excerpt 2: The second excerpt follows a vi-IV-I-V in the key of G major.\n"
            "Step 2: Both outline the same functional ORDER across the cycle despite different instruments/tempi.\n"
            "Step 3: ORDER matches (e.g., vi–IV–I–V). Thus, they are the same progression.\n"
            "A. Yes, these are the same chord progression."
        }]}
        ex2_user = {"role": "user", "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "text", "text": "Excerpt 1:"},
            {"type": "audio", "audio": diff1, "sampling_rate": sr},
            {"type": "text", "text": "Excerpt 2:"},
            {"type": "audio", "audio": diff2, "sampling_rate": sr},
        ]}
        ex2_assistant = {"role": "assistant", "content": [{"type": "text", "text":
            "Step 1: Segment and abstract each to functional ORDER.\n"
            "Excerpt 1: The first excerpt follows a vi-IV-I-V progression in the key of G major.\n"
            "Excerpt 2: The second excerpt follows a I-IV-V progression in the key of E major.\n"
            "Step 2: The sequences differ in functions and order (e.g., vi–IV–I–V vs I–IV–V).\n"
            "Step 3: ORDER does not match. Thus, these are not the same progression.\n"
            "B. No, these are not the same chord progression."
        }]}
    else:
        same1 = load_audio_array(EX_B_SAME_1, sr)
        same2 = load_audio_array(EX_B_SAME_2, sr)
        diff1 = load_audio_array(EX_B_DIFF_1, sr)
        diff2 = load_audio_array(EX_B_DIFF_2, sr)

        ex1_user = {"role": "user", "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "text", "text": "Excerpt 1:"},
            {"type": "audio", "audio": same1, "sampling_rate": sr},
            {"type": "text", "text": "Excerpt 2:"},
            {"type": "audio", "audio": same2, "sampling_rate": sr},
        ]}
        ex1_assistant = {"role": "assistant", "content": [{"type": "text", "text":
            "Step 1: Segment both excerpts by harmonic rhythm; abstract functions ignoring inversions/extensions.\n"
            "Excerpt 1: The first excerpt follows a I-vi-ii-V progression in the key of C major.\n"
            "Excerpt 2: The second excerpt follows a I-vi-ii-V progression in the key of C major.\n"
            "Step 2: Both outline the same functional ORDER across the cycle despite different instruments/tempi.\n"
            "Step 3: ORDER matches (e.g., I–vi–ii–V). Thus, they are the same progression.\n"
            "A. Yes, these are the same chord progression."
        }]}
        ex2_user = {"role": "user", "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "text", "text": "Excerpt 1:"},
            {"type": "audio", "audio": diff1, "sampling_rate": sr},
            {"type": "text", "text": "Excerpt 2:"},
            {"type": "audio", "audio": diff2, "sampling_rate": sr},
        ]}
        ex2_assistant = {"role": "assistant", "content": [{"type": "text", "text":
            "Step 1: Segment and abstract each to functional ORDER.\n"
            "Excerpt 1: The first excerpt follows a I–vi–VI–V progression in the key of F major.\n"
            "Excerpt 2: The second excerpt follows a vi-IV-I-V progression in the key of G major.\n"
            "Step 2: The sequences differ in functions and/or order (e.g., I–vi–VI–V vs vi-IV-I-V).\n"
            "Step 3: ORDER does not match. Thus, these are not the same progression.\n"
            "B. No, these are not the same chord progression."
        }]}

    messages = [ex1_user, ex1_assistant, ex2_user, ex2_assistant]
    audio_list = [same1, same2, diff1, diff2]
    return messages, audio_list

def run_examples_and_confirm(
    *,
    model: Qwen2_5OmniForConditionalGeneration,
    processor: Qwen2_5OmniProcessor,
    sysinstr_text: str,
    group: str,
    sr: int,
    log: logging.Logger,
) -> Tuple[List[Dict[str, Any]], str, List[Any]]:
    """Returns (history_with_examples_and_confirm, confirmation_string, audio_list)."""
    history: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": sysinstr_text}]}
    ]
    is_cot = (sysinstr_text == SYSINSTR_COT)

    if is_cot:
        fewshot_msgs, fewshot_audio = build_fewshot_messages_COT(group, sr)
    else:
        fewshot_msgs, fewshot_audio = build_fewshot_messages_SYSINST(group, sr)

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
    from collections import deque
    from itertools import chain

    KEEP_LAST_K = 4  # remember only the last K user+assistant pairs (and their audios)

    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}

    configure_logging(log_filename)
    logging.info(f"=== RUN START (Chord Progression Matching • CHAT+SysInst • {mode} • {group}) ===")
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

    # CoT needs headroom; SYSINST does not
    set_generation_config(model, min_new_tokens=80 if mode == "COT" else None)
    sr = processor.feature_extractor.sampling_rate

    # === Examples + confirmation (kept fixed for entire run) ===
    base_history, _confirm, base_audios = run_examples_and_confirm(
        model=model,
        processor=processor,
        sysinstr_text=(SYSINSTR_PLAIN if mode == "SYSINST" else SYSINSTR_COT),
        group=group,
        sr=sr,
        log=logging.getLogger(),
    )

    # Rolling memory: only last K user/assistant pairs (and their audios)
    # Each deque element is: (msgs_list, audios_list)
    #   msgs_list   = [user_turn, assistant_turn]
    #   audios_list = [a1, a2]
    rolling_pairs = deque(maxlen=KEEP_LAST_K)

    # Stimuli for the selected group (shuffle per seed)
    question_stims = stimuli_pairs_group(group)
    random.seed(seed)
    random.shuffle(question_stims)

    print(
        f"\n--- Task: Chord Progression Matching — CHAT+SysInst {mode} • {group} | model={MODEL_ID} | temp=1.0 | seed={seed} ---\n"
    )
    logging.info(f"\n--- Task: Chord Progression Matching — CHAT+SysInst {mode} • {group} ---\n")

    correct = 0
    total = len(question_stims)

    for idx, q in enumerate(question_stims, start=1):
        print(f"\n--- Question {idx} ---\n")
        logging.info(f"\n--- Question {idx} ---\n")

        f1 = q["file1"]
        f2 = q["file2"]
        logging.info(f"Stimuli: file1={f1}, file2={f2}")

        # Load audio for this trial
        a1 = load_audio_array(f1, sr)
        a2 = load_audio_array(f2, sr)

        # Build the user turn
        if mode == "SYSINST":
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the first excerpt."},
                    {"type": "audio", "audio": a1, "sampling_rate": sr},
                    {"type": "text", "text": "Here is the second excerpt."},
                    {"type": "audio", "audio": a2, "sampling_rate": sr},
                    {"type": "text", "text":
                     'Reply with exactly ONE of the following lines:\n'
                     'Yes, these are the same chord progression.\n'
                     'OR\n'
                     'No, these are not the same chord progression.'},
                ],
            }
        else:
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": COT_PROMPT_TEMPLATE},
                    {"type": "text", "text": "Excerpt 1:"},
                    {"type": "audio", "audio": a1, "sampling_rate": sr},
                    {"type": "text", "text": "Excerpt 2:"},
                    {"type": "audio", "audio": a2, "sampling_rate": sr},
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
        audio_for_call = list(base_audios) + past_audios_flat + [a1, a2]

        # Safety check to ensure audio alignment
        assert _count_audio_msgs(messages) == len(audio_for_call), \
            f"Audio count mismatch: messages expect {_count_audio_msgs(messages)}, got {len(audio_for_call)}"

        # Generate
        completion = qwen_generate_text(
            model=model,
            processor=processor,
            messages=messages,
            audio=audio_for_call,
        )

        print("LLM Full Response:\n", completion)
        logging.info(f"[{mode}/{group}] Q{idx} - LLM Full Response:\n{completion}")

        # Add this pair to rolling memory (regardless of parse success)
        assistant_turn = {"role": "assistant", "content": [{"type": "text", "text": completion}]}
        rolling_pairs.append(([user_turn, assistant_turn], [a1, a2]))

        # Parse final decision
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
