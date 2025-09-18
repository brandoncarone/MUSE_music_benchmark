#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys
from pathlib import Path
import argparse
from typing import Optional, Tuple
import pandas as pd

# ---------- Filename pattern ----------
FNAME_RE = re.compile(
    r"""^
    (?P<task>.+?)_
    (?P<model>[^_]+)_
    CHAT_
    (?P<mode>SYSINST|COT)_
    (?P<group>GroupA|GroupB)_
    seed(?P<seed>\d+)
    \.log$
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ---------- Canonical strings (must match runners) ----------
CANON = {
    "transposition": {
        "YES": "Yes, these are the same melody.",
        "NO":  "No, these are not the same melody.",
    },
    "oddballs": {
        "YES": "Yes, these are the same exact melody.",
        "NO":  "No, these are not the same exact melody.",
    },
    "rhythm_matching": {
        "YES": "Yes, these are the same exact drum set patterns.",
        "NO":  "No, these are not the same exact drum set patterns.",
    },
    "syncopation": {
        "A": 'A. The rhythm in Excerpt 1 is more syncopated.',
        "B": 'B. The rhythm in Excerpt 2 is more syncopated.',
    },
    "chord_progression_matching": {
        "A": "A. Yes, these are the same chord progression.",
        "B": "B. No, these are not the same chord progression.",
    },
    "key_modulation": {
        "A": "A. Yes, a key modulation occurs.",
        "B": "B. No, the key stays the same.",
    },
    "meter_identification": {
        "A": "A. Groups of 3",
        "B": "B. Groups of 4",
        "C": "C. Groups of 5",
    },
    "chord_quality": {
        "A": "A. Major",
        "B": "B. Minor",
    },
    "instrument_identification": {
        "A": "A. Piano",
        "B": "B. Guitar",
        "C": "C. Bass",
        "D": "D. Drums",
    },
    "contour_identification": {
        "A": "A. Arch (ascending and then descending)",
        "B": "B. Inverted Arch (descending and then ascending)",
        "C": "C. Ascending (pitch raises over time)",
        "D": "D. Descending (pitch falls over time)",
    },
}

# ---------- Task normalization ----------
def norm_task(raw: str) -> str:
    t = raw.lower()
    if "transposition" in t or "melody" in t:
        return "transposition"
    if "oddball" in t:
        return "oddballs"
    if "rhythm" in t:
        return "rhythm_matching"
    if "syncopation" in t:
        return "syncopation"
    if "progression" in t:
        return "chord_progression_matching"
    if "keymod" in t or "key_modulation" in t or "modulation" in t:
        return "key_modulation"
    if "meter" in t:
        return "meter_identification"
    if "chord_quality" in t:
        return "chord_quality"
    if "instrument" in t:
        return "instrument_identification"
    if "contour" in t:
        return "contour_identification"
    return raw

# ---------- Ground-truth derivation from filenames ----------
M_ID = re.compile(r"(?:^|[/\\])(M\d+)_", re.IGNORECASE)                # transposition
BEAT_ID = re.compile(r"(?:^|[/\\])(Beat_\d+)", re.IGNORECASE)          # rhythm
SYNC_LEVEL = re.compile(r"(?:^|[/\\])Sync\s*([1-4])[_\-A-Z]?", re.IGNORECASE)
QUESTION_RE = re.compile(r"--- Question \d+ ---")


def expected_transposition(f1: str, f2: str) -> str:
    m1 = M_ID.search(f1 or "")
    m2 = M_ID.search(f2 or "")
    same = (m1 and m2 and (m1.group(1).lower() == m2.group(1).lower()))
    return CANON["transposition"]["YES"] if same else CANON["transposition"]["NO"]

def expected_oddballs(f1: str, f2: str) -> str:
    same = (os.path.basename(f1 or "") == os.path.basename(f2 or ""))
    return CANON["oddballs"]["YES"] if same else CANON["oddballs"]["NO"]

def expected_rhythm_matching(f1: str, f2: str) -> str:
    b1 = os.path.basename(f1 or "")
    b2 = os.path.basename(f2 or "")
    if b1 == b2:
        return CANON["rhythm_matching"]["YES"]
    m1 = BEAT_ID.search(f1 or "")
    m2 = BEAT_ID.search(f2 or "")
    rid1 = m1.group(1) if m1 else None
    rid2 = m2.group(1) if m2 else None
    same = bool(rid1) and (rid1.lower() == (rid2 or "").lower())
    return CANON["rhythm_matching"]["YES"] if same else CANON["rhythm_matching"]["NO"]

def sync_level(path: str) -> int:
    name = os.path.basename(path or "")
    if "nosync" in name.lower():
        return 0
    m = SYNC_LEVEL.search(name)
    return int(m.group(1)) if m else 0

def expected_syncopation(f1: str, f2: str) -> str:
    l1, l2 = sync_level(f1), sync_level(f2)
    if l1 == l2:
        return CANON["syncopation"]["B"]  # ties shouldn't occur; deterministic fallback
    return CANON["syncopation"]["A"] if l1 > l2 else CANON["syncopation"]["B"]

def progression_token(path: str) -> str:
    return os.path.basename(path or "").split("_")[0].strip()

def expected_progression(f1: str, f2: str) -> str:
    same = progression_token(f1).lower() == progression_token(f2).lower()
    return CANON["chord_progression_matching"]["A"] if same else CANON["chord_progression_matching"]["B"]

def expected_key_modulation(f1: str) -> str:
    name = os.path.basename(f1 or "")
    return CANON["key_modulation"]["A"] if "mod" in name.lower() else CANON["key_modulation"]["B"]

METER_GOLD = {  # from your spec
    # Group A
    "Circles_3.wav": "A", "Piano_3.wav": "A", "I-vi-VI-V_Fmaj_piano_172_3_4.wav": "A", "vi-IV-I-V_Gmaj_AcousticGuit_118.wav": "A",
    "Rosewood_4.wav": "B", "SunKing_4.wav": "B", "opbeat_4.wav": "B", "off_4.wav": "B",
    "Five_solo_5.wav": "C", "GII_5.wav": "C",
    # Group B
    "50s_3.wav": "A", "Circles_solo_3.wav": "A", "Scene_3.wav": "A",
    "ComeOn_4.wav": "B", "DoDoDoDoDo_4.wav": "B", "Flow_4.wav": "B", "Harm_4.wav": "B",
    "Dance_5.wav": "C", "Falling_5.wav": "C",
}
def expected_meter(path: str):
    base = os.path.basename(path or "")
    label = METER_GOLD.get(base)
    if label == "A": return CANON["meter_identification"]["A"]
    if label == "B": return CANON["meter_identification"]["B"]
    if label == "C": return CANON["meter_identification"]["C"]
    m = re.search(r"_([345])\.wav$", base, re.IGNORECASE)  # best-effort fallback
    if m: return {"3": CANON["meter_identification"]["A"], "4": CANON["meter_identification"]["B"], "5": CANON["meter_identification"]["C"]}[m.group(1)]
    return None

def expected_chord_quality(path: str):
    n = os.path.basename(path or "").lower()
    if "major" in n: return CANON["chord_quality"]["A"]
    if "minor" in n: return CANON["chord_quality"]["B"]
    return None

def expected_instrument(path: str):
    n = os.path.basename(path or "").lower()
    if "piano" in n: return CANON["instrument_identification"]["A"]
    if "bass" in n: return CANON["instrument_identification"]["C"]
    if "beat" in n or "drum" in n: return CANON["instrument_identification"]["D"]
    return CANON["instrument_identification"]["B"]  # default guitar

def expected_contour(path: str):
    n = os.path.basename(path or "")
    if "InvArch" in n: return CANON["contour_identification"]["B"]
    if "Arch" in n:    return CANON["contour_identification"]["A"]
    if "Asc" in n:     return CANON["contour_identification"]["C"]
    if "Desc" in n:    return CANON["contour_identification"]["D"]
    return None

def expected_from_context(task: str, f1: Optional[str], f2: Optional[str]):
    t = norm_task(task)
    if t == "transposition":               return expected_transposition(f1 or "", f2 or "")
    if t == "oddballs":                    return expected_oddballs(f1 or "", f2 or "")
    if t == "rhythm_matching":             return expected_rhythm_matching(f1 or "", f2 or "")
    if t == "syncopation":                 return expected_syncopation(f1 or "", f2 or "")
    if t == "chord_progression_matching":  return expected_progression(f1 or "", f2 or "")
    if t == "key_modulation":              return expected_key_modulation(f1 or f2 or "")
    if t == "meter_identification":        return expected_meter(f1 or f2 or "")
    if t == "chord_quality":               return expected_chord_quality(f1 or f2 or "")
    if t == "instrument_identification":   return expected_instrument(f1 or f2 or "")
    if t == "contour_identification":      return expected_contour(f1 or f2 or "")
    return None

# ---------- Log parsing ----------
STIM_PAIR_RE   = re.compile(r"Stimuli:\s*file1=(.+?),\s*file2=(.+)$")
STIM_SINGLE_RE = re.compile(r"Stimulus:\s*file=(.+)$")
PARSED_RE      = re.compile(r"Parsed Final Answer:\s*(.+)$")

def parse_log_file(path: Path) -> dict:
    m = FNAME_RE.match(path.name)
    if not m:
        return {}
    task_raw = m.group("task")
    task = norm_task(task_raw)
    model = m.group("model")
    mode  = m.group("mode")
    group = m.group("group")
    seed  = int(m.group("seed"))

    n_scored = 0
    n_correct = 0

    # State for the current question block
    in_block = False
    has_scored = False
    curr_f1: Optional[str] = None
    curr_f2: Optional[str] = None
    curr_lines: list[str] = []

    def finalize_block():
        nonlocal n_scored, n_correct, in_block, has_scored, curr_f1, curr_f2, curr_lines
        # If we already scored (via 'Parsed Final Answer:'), do nothing
        if not in_block or has_scored:
            # reset block state
            in_block = False
            has_scored = False
            curr_f1 = None
            curr_f2 = None
            curr_lines = []
            return

        # Try tolerant fallback only if we have enough context to compute expected
        exp = expected_from_context(task, curr_f1, curr_f2)
        if exp is None:
            # reset and bail
            in_block = False
            has_scored = False
            curr_f1 = None
            curr_f2 = None
            curr_lines = []
            return

        block_text = "\n".join(curr_lines)
        model_ans = parse_from_llm_full_response(task, block_text)

        # If the model gave no answer (blank) or we still can't parse, SKIP (do not count wrong)
        if not model_ans:
            in_block = False
            has_scored = False
            curr_f1 = None
            curr_f2 = None
            curr_lines = []
            return

        # We parsed a model answer → score it
        n_scored += 1
        if model_ans == exp:
            n_correct += 1

        # reset block state
        in_block = False
        has_scored = False
        curr_f1 = None
        curr_f2 = None
        curr_lines = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # New question starts → finalize the previous one if needed, then start a new block
            if QUESTION_RE.search(line):
                finalize_block()
                in_block = True
                has_scored = False
                curr_f1 = None
                curr_f2 = None
                curr_lines = [line]
                continue

            # If we’re inside a question block, keep buffering lines
            if in_block:
                curr_lines.append(line)

            # Capture stimuli (pair or single) for expected-answer calculation
            mp = STIM_PAIR_RE.search(line)
            if mp:
                curr_f1, curr_f2 = mp.group(1).strip(), mp.group(2).strip()
                continue

            ms = STIM_SINGLE_RE.search(line)
            if ms:
                curr_f1, curr_f2 = ms.group(1).strip(), None
                continue

            # Normal case: use the logged "Parsed Final Answer:" when present
            ma = PARSED_RE.search(line)
            if ma and in_block and not has_scored:
                model_ans = ma.group(1).strip()
                exp = expected_from_context(task, curr_f1, curr_f2)
                if exp is not None and model_ans:
                    n_scored += 1
                    if model_ans == exp:
                        n_correct += 1
                    has_scored = True
                # (If exp is None, we can’t score; leave block open for potential fallback)

    # End of file → finalize the last open block (for fallback parsing)
    finalize_block()

    # n_items isn’t used downstream; keep the fields your summary expects
    return {
        "task": task, "model": model, "mode": mode, "group": group, "seed": seed,
        "n_items": n_scored,           # aligns with how you aggregate
        "n_scored": n_scored,
        "n_correct": n_correct,
    }


def _period_optional_regex(s: str) -> re.Pattern:
    """
    Return a case-insensitive regex that matches s, allowing a missing trailing period
    and optional straight/curly quotes around the whole answer.
    """
    s = s.strip()
    if s.endswith("."):
        core = re.escape(s[:-1])
        pat = rf'(?i)[\'"“”]?\s*{core}\.?\s*[\'"“”]?'
    else:
        pat = rf'(?i)[\'"“”]?\s*{re.escape(s)}\s*[\'"“”]?'
    return re.compile(pat)

def _syncopation_variants():
    # Optional leading "A." / "B." and optional trailing period
    a_core = r"The rhythm in Excerpt 1 is more syncopated"
    b_core = r"The rhythm in Excerpt 2 is more syncopated"
    a = re.compile(rf'(?i)(?:A\.\s*)?{a_core}\.?\s*')
    b = re.compile(rf'(?i)(?:B\.\s*)?{b_core}\.?\s*')
    return a, b

# Build tolerant patterns that map DIRECTLY to the canonical strings
FALLBACK_PATTERNS = {
    # Rhythm matching: allow missing trailing period
    "rhythm_matching": [
        (_period_optional_regex(CANON["rhythm_matching"]["YES"]), CANON["rhythm_matching"]["YES"]),
        (_period_optional_regex(CANON["rhythm_matching"]["NO"]),  CANON["rhythm_matching"]["NO"]),
    ],
    # Transposition: allow missing trailing period
    "transposition": [
        (_period_optional_regex(CANON["transposition"]["YES"]), CANON["transposition"]["YES"]),
        (_period_optional_regex(CANON["transposition"]["NO"]),  CANON["transposition"]["NO"]),
    ],
    # Syncopation: optional A./B. and optional trailing period
    "syncopation": [
        (_syncopation_variants()[0], CANON["syncopation"]["A"]),
        (_syncopation_variants()[1], CANON["syncopation"]["B"]),
    ],
}

def parse_from_llm_full_response(task: str, block_text: str) -> Optional[str]:
    """
    Fallback when 'Parsed Final Answer:' is missing.
    Extract the LLM Full Response block and try tolerant matching.
    Return the exact canonical string if matched; else None.
    """
    # Grab the whole LLM response payload for this question
    m = re.search(r'LLM Full Response:\s*\n(.*?)(?=\n--- Question \d+ ---|\Z)', block_text, flags=re.DOTALL)
    if not m:
        return None
    llm = m.group(1).strip()
    if not llm:
        return None  # true blank: skip, don't count wrong

    pairs = FALLBACK_PATTERNS.get(task, [])
    for pat, canonical in pairs:
        if pat.search(llm):
            return canonical
    return None



# ---------- Aggregation ----------
def summarize_logs(log_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(log_dir.glob("*.log")):
        row = parse_log_file(p)
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[
            "task","model","mode","groups_combined","seeds_combined",
            "items_scored","correct","accuracy_pct"
        ])
    df = pd.DataFrame(rows)
    g = df.groupby(["task","model","mode"], as_index=False).agg(
        items_scored=("n_scored","sum"),
        correct=("n_correct","sum"),
        groups_combined=("group","nunique"),
        seeds_combined=("seed","nunique"),
    )
    g["accuracy_pct"] = (100.0 * g["correct"] / g["items_scored"]).round(2)
    return g.sort_values(["task","model","mode"]).reset_index(drop=True)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Summarize per-task accuracies from log files.")
    ap.add_argument("log_dir", nargs="?", default=".", help="Folder containing *.log files")
    ap.add_argument("--out", default="log_accuracy_summary.csv", help="Path to write CSV summary")
    args = ap.parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve()
    if not log_dir.exists():
        print(f"ERROR: {log_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    df = summarize_logs(log_dir)
    if df.empty:
        print("No matching .log files found.")
        return

    out_path = Path(args.out).expanduser().resolve()
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved CSV → {out_path}")

if __name__ == "__main__":
    main()
