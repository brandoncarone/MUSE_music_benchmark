# syncopation_detection_Qwen2.5-Omni_CHAT_SysInst_master.py
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
STIM_ROOT = "stimuli"
MAX_NEW_TOKENS = 8192

# Canonical answer strings and robust patterns (VERBATIM)
A_CANON = "A. The rhythm in Excerpt 1 is more syncopated."
B_CANON = "B. The rhythm in Excerpt 2 is more syncopated."

A_PAT = re.compile(r'(?i)\b(?:a\.\s*)?the\s+rhythm\s+in\s+excerpt\s*1\s+is\s+more\s+syncopated\.')
B_PAT = re.compile(r'(?i)\b(?:b\.\s*)?the\s+rhythm\s+in\s+excerpt\s*2\s+is\s+more\s+syncopated\.')

# =============================
# System instructions (VERBATIM)
# =============================
SYSINSTR_PLAIN = """You are a participant in a psychological experiment on music perception. 
In each question, you will be given:
    1. A brief instruction about the specific listening task.
    2. Two audio examples to listen to.
Syncopation Detection: Your task is to listen to two drum set rhythms and decide which is more syncopated. 
Think of syncopation as rhythmic surprise: a simple rhythm is steady and predictable (like a metronome: ONE-two-three-four), 
while a syncopated rhythm emphasizes the "off-beats" — the unexpected moments in between the main pulse — making it feel more complex or groovy.

Valid responses are:
"A. The rhythm in Excerpt 1 is more syncopated." or 
"B. The rhythm in Excerpt 2 is more syncopated."

Before you begin the task, I will provide you with examples of an excerpt that is not syncopated at all, 
as well as an excerpt that is highly syncopated so that you better understand the task. 
After examining the examples, please respond with "Yes, I understand." if you understand the task or 
"No, I don't understand." if you don't understand the task.

Please provide no additional commentary beyond the short answers previously mentioned.
"""

SYSINSTR_COT = """You are a participant in a psychological experiment on music perception.
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. Two audio examples to listen to.

Your task is to decide which drum set rhythm is MORE SYNCOPATED.

Definitions and constraints:
- Think of syncopation as emphasis on OFF-BEATS or unexpected placements relative to the main pulse.
- A rhythm is “more syncopated” when kick/snare accents more often land between the main beats, displace or tie across strong beats, or omit strong-beat hits in favor of off-beat hits.
- Focus primarily on kick and snare placement; treat hi-hat ostinatos as neutral texture.

Valid responses are exactly:
"A. The rhythm in Excerpt 1 is more syncopated."
"B. The rhythm in Excerpt 2 is more syncopated."

Before you begin the task, I will provide you with two examples so you better understand the task. After examining the examples, please respond with "Yes, I understand." if you understand the task or "No, I don't understand." if you don't understand the task.

After your step-by-step reasoning, end with exactly one line:
A. The rhythm in Excerpt 1 is more syncopated.
OR
B. The rhythm in Excerpt 2 is more syncopated."""

# --- COT per-trial prompt (VERBATIM) ---
COT_PROMPT_TEMPLATE = """Analyze the two drum set excerpts and decide which is MORE SYNCOPATED.

Step 1: Establish the pulse and smallest repeating cycle for each excerpt.

Step 2: For each excerpt, note kick/snare placements relative to the beat grid:
- Count/describe off-beat accents and displaced hits.
- Note any strong-beat omissions with off-beat substitutions.
- Treat hi-hat texture as neutral; focus on kick/snare.

Step 3: Compare which excerpt exhibits more off-beat emphasis, displaced accents, or ties across beats — that excerpt is more syncopated.

Step 4: Final Answer
After any reasoning, reply with exactly ONE of the following lines (and nothing else on that line):
A. The rhythm in Excerpt 1 is more syncopated.
OR
B. The rhythm in Excerpt 2 is more syncopated.
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
    Example: syncopation_Qwen2.5-Omni_CHAT_SYSINST_GroupA_seed1.log
    """
    assert mode in {"SYSINST", "COT"}
    assert group in {"GroupA", "GroupB"}
    return f"syncopation_Qwen2.5-Omni_CHAT_{mode}_{group}_seed{seed}.log"

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
    """Normalize path: abs → as-is; starts with 'stimuli/' → as-is; else join with STIM_ROOT."""
    if os.path.isabs(p):
        return p
    if p.startswith(STIM_ROOT + os.sep) or p.startswith(STIM_ROOT + "/"):
        return p
    return os.path.join(STIM_ROOT, p)

def _p(name: str) -> str:
    return os.path.join(STIM_ROOT, name)

def load_audio_array(path: str, sr: int):
    y, _ = librosa.load(path, sr=sr, mono=True)
    # Efficiency: trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=35)
    return y

def stimuli_pairs_group(group: str) -> List[Dict[str, str]]:
    assert group in {"GroupA", "GroupB"}
    if group == "GroupA":
        pairs = [
            ("stimuli/Intermediate/Sync1_A.wav", "stimuli/Intermediate/NoSync_E.wav"),
            ("stimuli/Intermediate/Sync2_A.wav", "stimuli/Intermediate/NoSync_B.wav"),
            ("stimuli/Intermediate/Sync2_B.wav", "stimuli/Intermediate/Sync1_B.wav"),
            ("stimuli/Intermediate/Sync3_E.wav", "stimuli/Intermediate/Sync1_A.wav"),
            ("stimuli/Intermediate/Sync3_B.wav", "stimuli/Intermediate/Sync2_A.wav"),
            ("stimuli/Intermediate/Sync2_B.wav", "stimuli/Intermediate/Sync4_A.wav"),
            ("stimuli/Intermediate/Sync3_E.wav", "stimuli/Intermediate/Sync4_B.wav"),
            ("stimuli/Intermediate/NoSync_B.wav", "stimuli/Intermediate/Sync1_B.wav"),
            ("stimuli/Intermediate/Sync1_A.wav", "stimuli/Intermediate/Sync2_A.wav"),
            ("stimuli/Intermediate/Sync3_B.wav", "stimuli/Intermediate/Sync4_A.wav"),
        ]
    else:
        pairs = [
            ("stimuli/Intermediate/Sync1_C.wav", "stimuli/Intermediate/NoSync_C.wav"),
            ("stimuli/Intermediate/Sync2_C.wav", "stimuli/Intermediate/NoSync_D.wav"),
            ("stimuli/Intermediate/Sync2_D.wav", "stimuli/Intermediate/Sync1_D.wav"),
            ("stimuli/Intermediate/Sync3_C.wav", "stimuli/Intermediate/Sync1_C.wav"),
            ("stimuli/Intermediate/Sync3_D.wav", "stimuli/Intermediate/Sync2_C.wav"),
            ("stimuli/Intermediate/Sync2_D.wav", "stimuli/Intermediate/Sync4_C.wav"),
            ("stimuli/Intermediate/Sync3_C.wav", "stimuli/Intermediate/Sync4_D.wav"),
            ("stimuli/Intermediate/NoSync_D.wav", "stimuli/Intermediate/Sync1_D.wav"),
            ("stimuli/Intermediate/Sync1_C.wav", "stimuli/Intermediate/Sync2_C.wav"),
            ("stimuli/Intermediate/Sync3_D.wav", "stimuli/Intermediate/Sync4_C.wav"),
        ]
    return [{"file1": _ppath(a), "file2": _ppath(b)} for a, b in pairs]

# Example audio for few-shot
EX_NOSYNC_A = _p("Intermediate/NoSync_A.wav")
EX_SYNC3_A  = _p("Intermediate/Sync3_A.wav")
EX_SYNC1_C  = _p("Intermediate/Sync1_C.wav")
EX_SYNC2_C  = _p("Intermediate/Sync2_C.wav")
EX_SYNC1_A  = _p("Intermediate/Sync1_A.wav")
EX_SYNC2_A  = _p("Intermediate/Sync2_A.wav")

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

def syncopation_rank(path: str) -> int:
    """
    Assign a rank for syncopation based on filename.
    - 'NoSync'  -> 0
    - 'Sync1'   -> 1
    - 'Sync2'   -> 2
    - 'Sync3'   -> 3
    - 'Sync4'   -> 4
    """
    base = os.path.basename(path).lower()
    if "nosync" in base:
        return 0
    m = re.search(r"sync(\d+)", base)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return 0

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
        completion = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
            0
        ].strip()

    del inputs, out_ids, new_tokens
    torch.cuda.empty_cache()
    return completion

# =============================
# Few-shot builders + confirmation
# =============================
def _example_nosync_user_SYSINST(sr: int):
    a = load_audio_array(EX_NOSYNC_A, sr)
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Example 1: This excerpt is not syncopated. Listen carefully:"},
            {"type": "audio", "audio": a, "sampling_rate": sr},
        ],
    }
    return msg, [a]

def _example_sync_user_SYSINST(sr: int):
    b = load_audio_array(EX_SYNC3_A, sr)
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Example 2: This excerpt is highly syncopated. Listen carefully:"},
            {"type": "audio", "audio": b, "sampling_rate": sr},
        ],
    }
    return msg, [b]

def build_fewshot_messages_SYSINST(sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    m1, a1 = _example_nosync_user_SYSINST(sr)
    m2, a2 = _example_sync_user_SYSINST(sr)
    return [m1, m2], a1 + a2

def build_fewshot_messages_COT(group: str, sr: int) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """
    Group-dependent COT examples exactly mirroring the Gemini script:
      - GroupA: Sync2_C vs Sync1_C  → A. (Excerpt 1 more syncopated)
      - GroupB: Sync2_A vs Sync1_A  → A. (Excerpt 1 more syncopated)
      - Both:  NoSync_A vs Sync3_A → B. (Excerpt 2 more syncopated)
    """
    # Common
    a_nosync_A = load_audio_array(EX_NOSYNC_A, sr)
    a_sync3_A  = load_audio_array(EX_SYNC3_A, sr)

    if group == "GroupA":
        a_sync1_X = load_audio_array(EX_SYNC1_C, sr)
        a_sync2_X = load_audio_array(EX_SYNC2_C, sr)
    else:
        a_sync1_X = load_audio_array(EX_SYNC1_A, sr)
        a_sync2_X = load_audio_array(EX_SYNC2_A, sr)

    # Example 1 — Excerpt 1 more syncopated (A)
    ex1_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "text", "text": "Here is Excerpt 1:"},
            {"type": "audio", "audio": a_sync2_X, "sampling_rate": sr},
            {"type": "text", "text": "Here is Excerpt 2:"},
            {"type": "audio", "audio": a_sync1_X, "sampling_rate": sr},
        ],
    }
    ex1_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text":
            "Step 1: Both excerpts share a steady pulse.\n"
            "Step 2: Excerpt 1 places many of the kick/snare hits on strong beats, but there are 4 instances where the kick/snare hits land on off-beats instead of strong beats."
            "Excerpt 2 places most of the kick/snare hits on strong beats, but there are 2 instances where the kick/snare hits land on off-beats instead of strong beats.\n"
            "Step 3: Excerpt 1 emphasizes more off-beats than Excerpt 2. Thus, Excerpt 1 is more syncopated.\n"
            "A. The rhythm in Excerpt 1 is more syncopated."
        }]
    }

    # Example 2 — Excerpt 2 more syncopated (B)
    ex2_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": COT_PROMPT_TEMPLATE},
            {"type": "text", "text": "Here is Excerpt 1:"},
            {"type": "audio", "audio": a_nosync_A, "sampling_rate": sr},
            {"type": "text", "text": "Here is Excerpt 2:"},
            {"type": "audio", "audio": a_sync3_A, "sampling_rate": sr},
        ],
    }
    ex2_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text":
            "Step 1: Both excerpts share a steady pulse.\n"
            "Step 2: Excerpt 1 places all kick/snare hits on strong beats; the only off-beats present are the consistent eighth note hi-hat ostinatos."
            "Excerpt 2 places several of the kick/snare hits on strong beats, but there are 6 instances where the kick/snare hits land on off-beats instead of strong beats.\n"
            "Step 3: Excerpt 2 emphasizes more off-beats than Excerpt 1. Thus, Excerpt 2 is more syncopated.\n"
            "B. The rhythm in Excerpt 2 is more syncopated."
        }]
    }

    messages = [ex1_user, ex1_assistant, ex2_user, ex2_assistant]
    audio_list = [a_sync2_X, a_sync1_X, a_nosync_A, a_sync3_A]
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
    logging.info(f"=== RUN START (Syncopation Detection • CHAT+SysInst • {mode} • {group}) ===")
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
        model=model,
        processor=processor,
        sysinstr_text=(SYSINSTR_PLAIN if mode == "SYSINST" else SYSINSTR_COT),
        group=group,
        sr=sr,
        log=logging.getLogger(),
    )

    # Fixed stimuli, then shuffle per seed (mirror Gemini)
    question_stims = stimuli_pairs_group(group)
    random.seed(seed)
    random.shuffle(question_stims)

    print(f"\n--- Task: Syncopation Detection — CHAT+SysInst {mode} • {group} | model={MODEL_ID} | temp=1.0 | seed={seed} ---\n")
    logging.info(f"\n--- Task: Syncopation Detection — CHAT+SysInst {mode} • {group} ---\n")

    correct = 0
    total = len(question_stims)

    for idx, q in enumerate(question_stims, start=1):
        print(f"\n--- Question {idx} ---\n")
        logging.info(f"\n--- Question {idx} ---\n")

        f1 = q["file1"]
        f2 = q["file2"]
        logging.info(f"Stimuli: file1={f1}, file2={f2}")

        a1 = load_audio_array(f1, sr)
        a2 = load_audio_array(f2, sr)

        if mode == "SYSINST":
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is Excerpt 1."},
                    {"type": "audio", "audio": a1, "sampling_rate": sr},
                    {"type": "text", "text": "Here is Excerpt 2."},
                    {"type": "audio", "audio": a2, "sampling_rate": sr},
                    {"type": "text", "text":
                     'Reply with exactly ONE of the following lines:\n'
                     'A. The rhythm in Excerpt 1 is more syncopated.\n'
                     'OR\n'
                     'B. The rhythm in Excerpt 2 is more syncopated.'},
                ],
            }
        else:
            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": COT_PROMPT_TEMPLATE},
                    {"type": "text", "text": "Here is Excerpt 1:"},
                    {"type": "audio", "audio": a1, "sampling_rate": sr},
                    {"type": "text", "text": "Here is Excerpt 2:"},
                    {"type": "audio", "audio": a2, "sampling_rate": sr},
                ],
            }

        messages = history + [user_turn]
        audio_for_call = list(cumulative_audios) + [a1, a2]
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
            # Still record turn & grow history/audio to preserve context
            history.append(user_turn)
            history.append({"role": "assistant", "content": [{"type": "text", "text": completion}]})
            cumulative_audios.extend([a1, a2])
            continue

        logging.info(f"Parsed Final Answer: {model_answer}")

        # Ground truth: higher Sync# is more syncopated; any Sync# > NoSync
        r1 = syncopation_rank(f1)
        r2 = syncopation_rank(f2)
        if r1 == r2:
            print("Evaluation: Skipped (tie).")
            logging.warning(f"Ambiguous tie on syncopation rank (r1=r2={r1}); skipping scoring for Q{idx}.")
        else:
            expected = A_CANON if r1 > r2 else B_CANON
            if model_answer == expected:
                correct += 1
                print("Evaluation: Correct!")
                logging.info("Evaluation: Correct")
            else:
                print("Evaluation: Incorrect.")
                logging.info(f"Evaluation: Incorrect (expected={expected})")

        # Append turn/history; grow audio list
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
    #for mode in ["SYSINST", "COT"]:
    #    for group in ["GroupA", "GroupB"]:
    for mode in ["COT"]:
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
