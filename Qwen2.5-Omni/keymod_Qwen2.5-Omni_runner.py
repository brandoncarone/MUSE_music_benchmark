# key_modulation_Qwen2.5-Omni_CHAT_SysInst_master.py
import os
import re
import gc
import random
import logging
from typing import List, Dict, Any, Tuple

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

def make_log_filename(*, mode: str, group: str, seed: int) -> str:
    """
    Chat + System Instructions runner, two modes:
      - SYSINST (plain)
      - COT     (reasoning)
    Two stimulus groups: GroupA / GroupB
    Example: key_modulation_Qwen2.5-Omni_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"key_modulation_Qwen2.5-Omni_CHAT_{mode}_{group}_seed{seed}.log"

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

def load_audio_array(path: str, sr: int):
    y, _ = librosa.load(path, sr=sr, mono=True)
    # Trim leading/trailing silence to reduce token load
    y, _ = librosa.effects.trim(y, top_db=35)
    return y

# =============================
# Parsing / evaluation helpers
# =============================
def parse_final_decision(text: str) -> str:
    last_a = last_b = last_b_alt = None
    for m in A_PAT.finditer(text or ""): last_a = m
    for m in B_PAT.finditer(text or ""): last_b = m
    for m in B_ALT.finditer(text or ""): last_b_alt = m

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

def _count_audio_msgs(messages: List[Dict[str, Any]]) -> int:
    n = 0
    for m in messages:
        for c in m.get("content", []):
            if isinstance(c, dict) and c.get("type") == "audio":
                n += 1
    return n

# =============================
# Qwen generation
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
        # after (A100-friendly)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            out_ids = model.generate(**inputs, return_audio=False)
        new_tokens = out_ids[:, inputs["input_ids"].size(1):]
        completion = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    del inputs, out_ids, new_tokens
    #torch.cuda.empty_cache()
    return completion

# =============================
# ---- Example audio paths (resolved with _p under STIM_ROOT) ----
# Group A
EX_A_MOD   = _p("Intermediate/I-IV-V_Abmaj_Mod_Bbmaj_piano_132.wav")       # contains a modulation
EX_A_NOMOD = _p("Intermediate/I-V-vi-IV_Dmaj_AcousticGuit_115.wav")        # stays in one key
# Group B
EX_B_MOD   = _p("Intermediate/vi-IV-I-V_Fsmaj_mod_Dsmin_relativemin_150.wav")  # contains a modulation
EX_B_NOMOD = _p("Intermediate/I-vi-VI-V_Fmaj_piano_172_3_4.wav")               # stays in one key

# =============================
# Few-shot builders + confirmation
# =============================
def build_fewshot_messages_SYSINST(sr: int, group: str) -> Tuple[List[Dict[str, Any]], List[Any]]:
    if group == "GroupA":
        a_mod  = load_audio_array(EX_A_MOD, sr)
        a_nom  = load_audio_array(EX_A_NOMOD, sr)
    else:
        a_mod  = load_audio_array(EX_B_MOD, sr)
        a_nom  = load_audio_array(EX_B_NOMOD, sr)

    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "Example 1: This audio example contains a key modulation. Listen carefully:"},
            {"type": "audio", "audio": a_mod, "sampling_rate": sr},
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": "Example 2: This audio example contains no key modulation. Listen carefully:"},
            {"type": "audio", "audio": a_nom, "sampling_rate": sr},
        ]},
    ]
    return msgs, [a_mod, a_nom]

def build_fewshot_messages_COT(sr: int, group: str) -> Tuple[List[Dict[str, Any]], List[Any]]:
    if group == "GroupA":
        a_mod  = load_audio_array(EX_A_MOD, sr)
        a_nom  = load_audio_array(EX_A_NOMOD, sr)
        id_mod, id_nom = "I-IV-V_Abmaj_Mod_Bbmaj_piano_132", "I-V-vi-IV_Dmaj_AcousticGuit_115"
    else:
        a_mod  = load_audio_array(EX_B_MOD, sr)
        a_nom  = load_audio_array(EX_B_NOMOD, sr)
        id_mod, id_nom = "vi-IV-I-V_Fsmaj_mod_Dsmin_relativemin_150", "I-vi-VI-V_Fmaj_piano_172_3_4"

    ex1_user = {"role": "user", "content": [
        {"type": "text", "text": COT_PROMPT_TEMPLATE},
        {"type": "audio", "audio": a_mod, "sampling_rate": sr},
    ]}
    ex1_assistant = {"role": "assistant", "content": [{"type": "text", "text":
        "Step 1: Identify the initial tonal center.\n"
        "Step 2: In the middle of the excerpt, the harmony and leading-tone patterns establish a NEW stable center that is sustained.\n"
        "Step 3: A new tonal center is clearly established. Thus, a modulation occurs.\n"
        "A. Yes, a key modulation occurs."
    }]}

    ex2_user = {"role": "user", "content": [
        {"type": "text", "text": COT_PROMPT_TEMPLATE},
        {"type": "audio", "audio": a_nom, "sampling_rate": sr},
    ]}
    ex2_assistant = {"role": "assistant", "content": [{"type": "text", "text":
        "Step 1: Establish the opening key as the tonal center.\n"
        "Step 2: Harmony/melody continue to support the same center throughout the excerpt.\n"
        "Step 3: No sustained new center. Thus, no modulation occurs.\n"
        "B. No, the key stays the same."
    }]}

    messages = [ex1_user, ex1_assistant, ex2_user, ex2_assistant]
    audio_list = [a_mod, a_nom]
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
    if sysinstr_text == SYSINSTR_COT:
        fewshot_msgs, fewshot_audio = build_fewshot_messages_COT(sr, group)
    else:
        fewshot_msgs, fewshot_audio = build_fewshot_messages_SYSINST(sr, group)

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
    logging.info(f"=== RUN START (Key Modulation • CHAT+SysInst • {mode} • {group}) ===")
    logging.info(f"Config: model={MODEL_ID}, temp=1.0, seed={seed}, group={group}, log={log_filename}")

    # Seeds
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Prefer a local snapshot (fast & offline); override via env QWEN2_5_OMNI_LOCAL_DIR if desired
    LOCAL_QWEN_DIR = os.environ.get(
        "QWEN2_5_OMNI_LOCAL_DIR",
        "/scratch/bc3189/hf_cache/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00"
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(LOCAL_QWEN_DIR, local_files_only=True)

    #gpu_count = torch.cuda.device_count()
    #if gpu_count >= 2:
    #    max_mem = {i: "22GiB" for i in range(gpu_count)}
    #    max_mem["cpu"] = "120GiB"
    #    device_map = "balanced"
    #else:
    #    max_mem = None
    #    device_map = "auto"
    
    gpu_count = torch.cuda.device_count()
    if gpu_count >= 2:
        max_mem = _auto_max_memory(reserve_gib=6)  # tune reserve as needed
        device_map = "balanced"
    else:
        max_mem = None
        device_map = "auto"


    #model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    #    LOCAL_QWEN_DIR,
    #    local_files_only=True,
    #    torch_dtype=torch.float16,
    #    device_map=device_map,
    #    attn_implementation="sdpa",
    #    enable_audio_output=False,
    #    max_memory=max_mem,
    #).eval()
    
    # after
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
    history, _confirm, cumulative_audios = run_examples_and_confirm(
        model=model,
        processor=processor,
        sysinstr_text=(SYSINSTR_PLAIN if mode == "SYSINST" else SYSINSTR_COT),
        group=group,
        sr=sr,
        log=logging.getLogger(),
    )

    # Stimuli (shuffle per seed)
    stim_files = stimuli_files_group(group)
    random.seed(seed)
    random.shuffle(stim_files)

    print(f"\n--- Task: Key Modulation Detection — CHAT+SysInst {mode} • {group} | model={MODEL_ID} | temp=1.0 | seed={seed} ---\n")
    logging.info(f"\n--- Task: Key Modulation Detection — CHAT+SysInst {mode} • {group} ---\n")

    correct = 0
    total = len(stim_files)

    for idx, f in enumerate(stim_files, start=1):
        print(f"\n--- Question {idx} ---\n")
        logging.info(f"\n--- Question {idx} ---\n")
        logging.info(f"Stimulus: file={f}")

        a1 = load_audio_array(f, sr)

        if mode == "SYSINST":
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the audio excerpt."},
                    {"type": "audio", "audio": a1, "sampling_rate": sr},
                    {"type": "text", "text":
                     "Reply with exactly ONE of the following lines:\n"
                     "A. Yes, a key modulation occurs.\n"
                     "OR\n"
                     "B. No, the key stays the same."},
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

        messages = history + [user_turn]
        audio_for_call = list(cumulative_audios) + [a1]
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

        # Parse
        model_answer = parse_final_decision(completion)
        if not model_answer:
            print("Evaluation: Failed. Could not parse the final answer phrase.")
            logging.error("Parse Error: missing/malformed final answer phrase.")
            # grow context anyway
            history.append(user_turn)
            history.append({"role": "assistant", "content": [{"type": "text", "text": completion}]})
            cumulative_audios.append(a1)
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

        # Grow persistent chat context
        history.append(user_turn)
        history.append({"role": "assistant", "content": [{"type": "text", "text": completion}]})
        cumulative_audios.append(a1)

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
