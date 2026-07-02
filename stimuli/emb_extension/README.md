# EMB Extension Stimuli & Evaluator for MUSE

This directory contains the extended evaluation audio stimuli generated using the Extended Music Benchmarks (`emb`) repository. These files are part of the additive integration designed to benchmark LLM auditory perception on temporal structural shifts.

---

## 1. The New Tasks

Unlike the static, single-classification tasks in the baseline MUSE benchmark, these extended tests focus on **temporal localization**—checking if a model can perceive transitions that occur at specific points in time.

### A. Temporal Meter (Time Signature) Shifts
*   **Stimuli:** `meter_shift_early.wav`, `meter_shift_late.wav`, `meter_steady_34.wav`, `meter_steady_44.wav`
*   **Behavior:** The audio begins in a 4/4 meter and transitions to a 3/4 meter at either bar 5 (early) or bar 12 (late) in a 16-bar sequence. Steady tracks maintain the same meter throughout.
*   **Runner:** [Gemini/temporal_meter_Gemini_runner.py](../../Gemini/temporal_meter_Gemini_runner.py)

### B. Temporal Key Modulation
*   **Stimuli:** `key_shift_early.wav`, `key_shift_late.wav`, `key_steady_c.wav`, `key_steady_g.wav`
*   **Behavior:** The audio plays a chord progression starting in C Major and modulates to G Major at either bar 5 (early) or bar 12 (late). Steady tracks stay in a single key throughout.
*   **Runner:** [Gemini/temporal_key_Gemini_runner.py](../../Gemini/temporal_key_Gemini_runner.py)

---

## 2. How to Run the Tests

To run the evaluations, use the respective Gemini runner scripts located in the `Gemini/` directory of the `MUSE_music_benchmark` repository.

```bash
# Ensure your GOOGLE_API_KEY is configured in your .env or shell environment
# To run the meter shift benchmark:
python Gemini/temporal_meter_Gemini_runner.py

# To run the key modulation benchmark:
python Gemini/temporal_key_Gemini_runner.py
```

These scripts will upload the audio stimuli to the Gemini API, prompt the model with standard multiple-choice questions matching the formatting of other MUSE benchmarks, and verify the accuracy of the model's outputs.

---

## 3. Integration with EMB & Creation Method

### Source Repository of EMB
All stimuli are generated programmatically by the **Extended Music Benchmarks (EMB)** repository:
*   **Source URL:** [https://github.com/raniarokiahfiroozye/extended_music_benchmarks.git](https://github.com/raniarokiahfiroozye/extended_music_benchmarks.git)

### Where to Find the MIDI Files
The MIDI files containing the source musical notes, time signature tracks, and key mappings are stored in the `tests/` folder of the sibling `emb` repository:
*   [emb/tests/meter_shift_early.mid](../../../emb/tests/meter_shift_early.mid)
*   [emb/tests/meter_shift_late.mid](../../../emb/tests/meter_shift_late.mid)
*   [emb/tests/key_shift_early.mid](../../../emb/tests/key_shift_early.mid)
*   [emb/tests/key_shift_late.mid](../../../emb/tests/key_shift_late.mid)

### Synthesis Method (WAV Generator)
1.  **MIDI Generation:** The `generate_muse_extension.py` script in the `emb` repository programmatically builds MIDI tracks with specific velocity accentuations (for meter) or chord transformations (for key modulations).
2.  **Algorithmic Verification:** The source MIDI is verified using `emb`'s internal `TimeSignatureSolver` and `KeySolver` to ensure the mathematical validity of the transitions before rendering.
3.  **FluidSynth Rendering:** The MIDI is synthesized to high-quality audio using **FluidSynth** and the system's default **FluidR3_GM.sf2** SoundFont to produce realistic piano synth WAV files, which are then copied directly into this directory.
