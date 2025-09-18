# instrumentID_Qwen2.5-Omni_CHAT_SysInst_master.py
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

# Canonical final answer strings (exact)
A_CANON = 'A. Piano'
B_CANON = 'B. Guitar'
C_CANON = 'C. Bass'
D_CANON = 'D. Drums'

# Robust patterns to detect answers (prefer LAST occurrence)
A_PAT = re.compile(r'(?i)\bA\.\s*Piano\b')
B_PAT = re.compile(r'(?i)\bB\.\s*Guitar\b')
C_PAT = re.compile(r'(?i)\bC\.\s*Bass\b')
D_PAT = re.compile(r'(?i)\bD\.\s*Drums\b')

# =============================
# System instructions (VERBATIM)
# =============================
SYSINSTR_PLAIN = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to. 

Your task is to listen to different music excerpts and identify the musical instrument being played.
Valid responses are:
"A. Piano"
"B. Guitar"
"C. Bass"
"D. Drums"

Before beginning the task, I will provide you with audio examples of each of the instruments, including piano, guitar, bass, and drums,
so that you better understand the task. After examining the examples, please respond with "Yes, I understand." if you 
understand the task or "No, I don't understand." if you don't understand the task.

Please provide no additional commentary beyond the short answers previously mentioned.
"""

SYSINSTR_COT = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to.

Your task is to listen to the excerpt and identify the musical instrument being played.

Definitions and constraints:
- Possible instruments (and response options) are limited to:
  A. Piano
  B. Guitar
  C. Bass
  D. Drums
- Focus on overall timbre, envelope, register, and articulation.
- Ignore reverb, effects, and recording quality.

Valid responses are exactly:
"A. Piano"
"B. Guitar"
"C. Bass"
"D. Drums"

Before beginning the task, I will provide you with audio examples of each of the instruments, including piano, guitar, bass, and drums, so that you better understand the task. After examining the 
examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't understand the task.

After any reasoning, end with exactly one line:
A. Piano
OR
B. Guitar
OR
C. Bass
OR
D. Drums"""

# --- COT per-trial prompt (VERBATIM) ---
COT_PROMPT_TEMPLATE = """Analyze the music excerpt and identify which INSTRUMENT is being played.

Step 1: Focus on timbre, envelope (attack/decay/sustain), register, and articulation characteristics.

Step 2: Compare what you hear to the four categories:
A. Piano — percussive key strike with harmonic resonance/decay.
B. Guitar — plucked/strummed strings; fretted articulation; mid-to-high register presence.
C. Bass — low-register plucked string; strong fundamental; longer sustain in low range.
D. Drums — percussive hits without sustained pitched notes; kit elements (kick/snare/hi-hat/cymbals).

Step 3: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
A. Piano
OR
B. Guitar
OR
C. Bass
OR
D. Drums
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
    Example: instrumentID_Qwen2.5-Omni_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"instrumentID_Qwen2.5-Omni_CHAT_{mode}_{group}_seed{seed}.log"

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

# =============================
# Stimuli lists (single excerpt per trial)
# =============================
def stimuli_list_group(group: str) -> List[Dict[str, str]]:
    assert group in {"GroupA", "GroupB"}
    if group == "GroupA":
        names = [
            "stimuli/Abmajor_Piano_120.wav",
            "stimuli/Bmaj_Scale_InvArch_Piano.wav",
            "stimuli/Ebmin_Scale_Arch_Piano.wav",
            "stimuli/DMaj_Arp_Desc.wav",
            "stimuli/Eminor_Guitar_120.wav",
            "stimuli/Abmin_Arp_Asc_Bass.wav",
            "stimuli/Dbmin_Scale_InvArch_Bass.wav",
            "stimuli/Emin_Arp_Desc_Bass.wav",
            "stimuli/Beat_2_140.wav",
            "stimuli/Beat_4_140_34.wav",
        ]
    else:
        names = [
            "stimuli/Bbminor_Piano_120.wav",
            "stimuli/Gbmin_Arp_Desc_Piano.wav",
            "stimuli/Bmajor_Guitar_120.wav",
            "stimuli/FMaj_Scale_InvArch.wav",
            "stimuli/GMaj_Arp_Asc.wav",
            "stimuli/BbMaj_Scale_Arch_Bass.wav",
            "stimuli/EMaj_Arp_Desc_Bass.wav",
            "stimuli/Beat_1_140.wav",
            "stimuli/Beat_3_140.wav",
            "stimuli/Beat_5_140.wav",
        ]
    return [{"file": _ppath(n)} for n in names]

# =============================
# Parsing / evaluation helpers
# =============================
def parse_final_choice(text: str) -> str:
    """Return canonical A/B/C/D string, or '' if not found. Prefer the LAST occurrence."""
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

def instrument_label_from_filename(path: str) -> str:
    """
    Ground truth mapping per your rule:
      - 'Beat'  -> D. Drums
      - 'Piano' -> A. Piano
      - 'Bass'  -> C. Bass
      - 'Guitar' OR no instrument token -> B. Guitar
    (case-insensitive)
    """
    name = os.path.basename(path).lower()
    if "beat" in name:
        return D_CANON
    if "piano" in name:
        return A_CANON
    if "bass" in name:
        return C_CANON
    # Default/fallback == Guitar
    return B_CANON

def _count_audio_msgs(messages: List[Dict[str, Any]]) -> int:
    n = 0
    for m in messages:
        for c in m.get("content", []):
            if isinstance(c, dict) and c.get("type") == "audio":
                n += 1
    return n

# =============================
# Few-shot examples (VERBATIM content & ordering)
# =============================
# Example audio paths
EX_GUITAR = _p("Emajor_Guitar_120.wav")
EX_PIANO  = _p("GbMaj_Arp_Desc_Piano.wav")
EX_BASS   = _p("AbMaj_Arp_Asc_Bass.wav")
EX_DRUMS  = _p("Beat_11_140.wav")

def build_fewshot_messages_COT(sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """
    Four in-context examples with brief CoT and strict final line (assistant messages included).
    NOTE: Ordering mirrors the Gemini code (Example 1 uses Piano audio, not Guitar).
    """
    a_gtr = load_audio_array(EX_GUITAR, sr)
    a_pno = load_audio_array(EX_PIANO, sr)
    a_bas = load_audio_array(EX_BASS, sr)
    a_drm = load_audio_array(EX_DRUMS, sr)

    # Example 1 — Piano (A)
    ex1_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "audio", "audio": a_pno, "sampling_rate": sr},
        ],
    }
    ex1_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text":
            "Step 1: Percussive key-strike onset with rich harmonic resonance/decay and clear note separations.\n"
            "Step 2: Sustained resonant body aligns with piano timbre.\n"
            "A. Piano"
        }]
    }

    # Example 2 — Guitar (B)
    ex2_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "audio", "audio": a_gtr, "sampling_rate": sr},
        ],
    }
    ex2_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text":
            "Step 1: Bright plucked-string timbre with fretted articulation; clear pick attack; mid register.\n"
            "Step 2: Matches a guitar rather than piano (hammered keys), bass (lower register), or drums (non-pitched percussive hits).\n"
            "B. Guitar"
        }]
    }

    # Example 3 — Bass (C)
    ex3_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "audio", "audio": a_bas, "sampling_rate": sr},
        ],
    }
    ex3_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text":
            "Step 1: Low-register plucked string with strong fundamental and longer sustain; subdued upper harmonics.\n"
            "Step 2: This profile fits an electric/acoustic bass rather than guitar or piano.\n"
            "C. Bass"
        }]
    }

    # Example 4 — Drums (D)
    ex4_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "audio", "audio": a_drm, "sampling_rate": sr},
        ],
    }
    ex4_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text":
            "Step 1: Broadband percussive hits (kick/snare/cymbal) with no sustained pitched notes.\n"
            "Step 2: Consistent with a drum kit rather than pitched string/keyboard instruments.\n"
            "D. Drums"
        }]
    }

    messages = [ex1_user, ex1_assistant, ex2_user, ex2_assistant, ex3_user, ex3_assistant, ex4_user, ex4_assistant]
    # Audio list must match the user-turn audio order: piano, guitar, bass, drums
    audio_list = [a_pno, a_gtr, a_bas, a_drm]
    return messages, audio_list

def build_fewshot_messages_SYSINST(sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    a_gtr = load_audio_array(EX_GUITAR, sr)
    a_pno = load_audio_array(EX_PIANO, sr)
    a_bas = load_audio_array(EX_BASS, sr)
    a_drm = load_audio_array(EX_DRUMS, sr)

    def _example_user_plain(label: str, audio_arr):
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Example: The following audio example features {label}. Listen carefully:"},
                {"type": "text", "text": "Audio example:"},
                {"type": "audio", "audio": audio_arr, "sampling_rate": sr},
            ],
        }

    # Mirror Gemini ordering: piano, guitar, bass, drums
    ex1_user = _example_user_plain("a piano", a_pno)
    ex2_user = _example_user_plain("a guitar", a_gtr)
    ex3_user = _example_user_plain("a bass",   a_bas)
    ex4_user = _example_user_plain("drums",    a_drm)

    messages = [ex1_user, ex2_user, ex3_user, ex4_user]
    audio_list = [a_pno, a_gtr, a_bas, a_drm]
    return messages, audio_list

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
# Few-shot + confirmation wrapper
# =============================
def run_examples_and_confirm(
    *,
    model: Qwen2_5OmniForConditionalGeneration,
    processor: Qwen2_5OmniProcessor,
    sysinstr_text: str,
    sr: int,
    log: logging.Logger,
) -> Tuple[List[Dict[str, Any]], str, List[Any]]:
    """Return (history_with_examples_and_confirm, confirmation_text, cumulative_audio_list)."""
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
    logging.info(f"=== RUN START (Instrument ID • CHAT+SysInst • {mode} • {group}) ===")
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

    # CoT needs some space to reason; SYSINST does not.
    set_generation_config(model, min_new_tokens=80 if mode == "COT" else None)
    sr = processor.feature_extractor.sampling_rate

    # === Examples + confirmation ===
    history, _confirm, cumulative_audios = run_examples_and_confirm(
        model=model, processor=processor, sysinstr_text=(SYSINSTR_PLAIN if mode == "SYSINST" else SYSINSTR_COT),
        sr=sr, log=logging.getLogger()
    )

    # Fixed stimuli, then shuffle per seed (to mirror Gemini)
    question_stims = stimuli_list_group(group)
    random.seed(seed)
    random.shuffle(question_stims)

    print(f"\n--- Task: Instrument Identification — CHAT+SysInst {mode} • {group} | model={MODEL_ID} | temp=1.0 | seed={seed} ---\n")
    logging.info(f"\n--- Task: Instrument Identification — CHAT+SysInst {mode} • {group} ---\n")

    correct = 0
    total = len(question_stims)

    for idx, q in enumerate(question_stims, start=1):
        print(f"\n--- Question {idx} ---\n")
        logging.info(f"\n--- Question {idx} ---\n")

        f = q["file"]
        logging.info(f"Stimulus: file={f}")

        # Load trimmed audio array
        a = load_audio_array(f, sr)

        if mode == "SYSINST":
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are a participant in a psychological experiment on music perception. Your task is to listen to an audio example and identify the musical instrument being played."},
                    {"type": "text", "text": 'Valid responses:\n"A. Piano"\n"B. Guitar"\n"C. Bass"\n"D. Drums"'},
                    {"type": "text", "text": "Listen carefully to the following audio:"},
                    {"type": "audio", "audio": a, "sampling_rate": sr},
                    {"type": "text", "text": 'Now answer by stating exactly one of the four strings (and nothing else):\n"A. Piano"\n"B. Guitar"\n"C. Bass"\n"D. Drums"'},
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

        # Build messages + audio for THIS call (preserve full prior audio history)
        messages = history + [user_turn]
        audio_for_call = list(cumulative_audios) + [a]

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
        model_answer = parse_final_choice(completion)
        if not model_answer:
            print("Evaluation: Failed. Could not parse a valid final choice.")
            logging.error("Parse Error: missing/malformed final choice line.")
            # Maintain strict chat history regardless
            history.append(user_turn)
            history.append({"role": "assistant", "content": [{"type": "text", "text": completion}]})
            cumulative_audios.append(a)
            continue

        logging.info(f"Parsed Final Answer: {model_answer}")

        # Ground truth via filename tokens
        expected = instrument_label_from_filename(f)

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
