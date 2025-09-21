
# MUSE: Music Understanding in Systems and Experiments
**Benchmarking human-like music perception in multimodal LLMs**

> This repository accompanies our ICASSP submission and provides the exact task runners, stimuli, and logs used in the paper. We evaluate state-of-the-art multimodal models on controlled listening tasks designed to probe specific dimensions of music perception.

---

## What this project is about (paper-style overview)

MUSE is a controlled benchmark for **music perception** rather than music captioning or text-only theory exams. Each task presents **audio stimuli** and a tightly constrained **forced-choice** or **binary** decision. We match model instructions to those used for human participants and keep **stateful chat history** (text + audio) for models that support it. Tasks span pitch, rhythm, harmony, and timbre:

- **Chord progression matching** – Do two excerpts realize the same functional order (e.g., I–vi–ii–V) despite surface differences?  
- **Chord quality (Major/Minor)** – Identify the triad quality from sustained + arpeggiated presentations; diagnose via the third (M3 vs m3), ignoring inversion/voicing/register.  
- **Key modulation (Key Mod.)** – Detect whether the key changes inside the excerpt.  
- **Pitch-shift detection** – Decide if a uniform pitch shift is present between excerpts.  
- **Melody contour ID (Mel. Shape)** – Classify global contour: Arch / Inverted Arch / Ascending / Descending.  
- **Rhythm matching** – Decide if two drum patterns are exactly identical (same cycle length and kit voice positions).  
- **Syncopation** – Choose which excerpt is more syncopated (off‑beat accents, ties across strong beats).  
- **Meter ID** – Identify the meter from a single excerpt.  
- **Instrument ID** – Identify the instrument class (Piano, Guitar, Bass, Drums).  
- **Oddball detection** – Decide if the comparison melody contains out‑of‑key substitutions relative to a reference.

The benchmark isolates **core perceptual competencies** (structure, harmony, meter, contour, timbre) while preventing shortcuts. Each task uses **explicit answer strings**, **robust parsers**, and **automatic scoring** so results are reproducible and comparable to human baselines.

---

## Models and prompting strategies (Table A in paper)

We evaluate:
- **Gemini 2.5 Flash** and **Gemini 2.5 Pro** with **stateful chat** using either **System Instructions** (plain) or **Chain‑of‑Thought (CoT)** variants; audio + text history is maintained across trials.
- **Qwen2.5‑Omni** using a **CoT** strategy aligned to the same per‑trial prompts and canonical answer formats.
- **Audio Flamingo** in a **stateless** configuration, merging instruction and trial info per prompt.

**Placeholder (add your PNG):**  
`![Table A: Prompting methodology](figs/tableA.png)`

---

## Few-shot performance summary (Table B in paper)

We sweep the number of in‑context examples (shots) per task, reporting the **best** and **second‑best** shot counts per row. The **“Orig.”** column shows baseline scores from the original fixed‑shot setup (N=2 for most tasks; N=4 for Melody Shape ID; N=3 for Meter ID).

**Placeholder (add your PNG):**  
`![Table B: Few-shot results](figs/tableB.png)`

---

## Overall benchmark results (Figure 1 in paper)

Figure 1 summarizes performance across tasks, comparing **Gemini 2.5 Pro**, **Gemini 2.5 Flash**, **Qwen2.5‑Omni**, and **Audio Flamingo**, alongside **human** and **musician** baselines. Models are plotted with solid lines; humans with dashed/dotted lines.

**Placeholder (add your PNG):**  
`![Figure 1: Radar comparison](figs/fig1_muse_radar.png)`

---

## Repository structure

```
stimuli/                     # All audio stimuli (root preserved in runners)
runners/
  ├── gemini/
  │   ├── transposition_Gemini_runner.py
  │   ├── rhythm_matching_Gemini_runner.py
  │   ├── contourID_Gemini_runner.py
  │   ├── instrumentID_Gemini_runner.py
  │   ├── oddballs_Gemini_runner.py
  │   ├── progression_matching_Gemini_runner.py
  │   ├── syncopation_Gemini_runner.py
  │   ├── chord_quality_Gemini_runner.py
  │   ├── meterID_Gemini_runner.py
  │   └── keymod_Gemini_runner.py
  └── qwen/
      ├── rhythm_matching_Qwen2.5-Omni_runner.py
      ├── contourID_Qwen2.5-Omni_runner.py
      ├── transposition_Qwen2.5-Omni_runner.py
      ├── meterID_Qwen2.5-Omni_runner.py
      ├── chord_quality_Qwen2.5-Omni_runner.py
      ├── oddballs_Qwen2.5-Omni_runner.py
      ├── syncopation_Qwen2.5-Omni_runner.py
      ├── instrumentID_Qwen2.5-Omni_runner.py
      ├── keymod_Qwen2.5-Omni_runner.py
      └── progression_matching_Qwen2.5-Omni_runner.py
figs/
  ├── fig1_muse_radar.png            # (add)
  ├── tableA.png                     # (add)
  └── tableB.png                     # (add)
logs/                                # Auto-created per run
```

> **Stimulus root**: Runners assume `stimuli/` as the canonical root; keep this path intact.

---

## Reproducible runs (framework conventions)

All runners share the same **grid**, **logging**, **parsing**, and **evaluation** conventions:

- **Modes**: `SYSINST` (plain system instructions) vs `COT` (reasoning).  
- **Stateful chat**: Gemini/Qwen maintain **full audio + text history** across trials in a run.  
- **Stimulus groups**: `GroupA` / `GroupB` fixed splits; shuffling per `seed`.  
- **Canonical answers**: Exact strings + regex parsers; we score the **last** valid answer line.  
- **Logging**: one log per run with config, stimulus IDs/paths, full model response, parsed final answer & evaluation, and total accuracy.

**Log filename pattern (exact):**
```
{taskname}_{modelTag}_CHAT_{mode}_{group}_seed{seed}.log
```
Example:
```
rhythm_matching_G25Pro_CHAT_COT_GroupA_seed1.log
```

---

## Setup

1. **Install deps**
   ```bash
   pip install -r requirements.txt
   ```
2. **API keys**
   - Gemini: set `GEMINI_API_KEY`
3. **Stimuli**
   - Keep `stimuli/` as shipped; filenames encode ground truth for scoring.

---

## How to run (examples)

```bash
python runners/gemini/chord_quality_Gemini_runner.py
python runners/gemini/progression_matching_Gemini_runner.py
python runners/qwen/rhythm_matching_Qwen2.5-Omni_runner.py
```

Logs will appear in your working directory following the naming convention above.

---

## Task details (paper-mirrored summaries)

- **Chord Progression Matching** — same functional order despite surface differences.  
- **Chord Quality (Major/Minor)** — diagnose via the third; ignore inversion/voicing.  
- **Key Modulation** — detect a key change.  
- **Pitch-Shift Detection** — detect uniform pitch shift.  
- **Melody Contour ID** — Arch / Inverted Arch / Ascending / Descending.  
- **Rhythm Matching** — exact match across kit voices/positions.  
- **Syncopation** — which is more syncopated (off-beat accents, ties).  
- **Meter ID** — identify meter from excerpt.  
- **Instrument ID** — Piano, Guitar, Bass, Drums.  
- **Oddball Detection** — out-of-key substitutions vs. reference.

---

## Results (high-level, paper-aligned)

- **Overall (Fig. 1)**: Gemini 2.5 **Pro** generally leads; **Flash** is close; **Qwen2.5‑Omni** is competitive on several tasks but trails on structure‑heavy ones; **humans/musicians** set the ceiling.  
  `![Figure 1: Radar comparison](figs/fig1_muse_radar.png)`

- **Prompting & history (Table A)**: Stateful chat for Gemini/Qwen; Audio Flamingo stateless.  
  `![Table A: Prompting methodology](figs/tableA.png)`

- **Few-shot sensitivity (Table B)**: Task‑dependent gains; several tasks saturate at 0–1 shots, while structure tasks benefit from 3–8 shots.  
  `![Table B: Few-shot results](figs/tableB.png)`

---

## Citation
Please cite the ICASSP submission (preprint/DOI to come).

## License
Add license choice (e.g., MIT).
