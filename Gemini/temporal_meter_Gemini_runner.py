# temporal_meter_Gemini_runner.py
import os
import re
import gc
import random
import logging
import warnings
from typing import List, Dict, Any, Tuple

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# --- Google Gemini SDK ---
from google import genai
from google.genai import types

# =============================
# Constants / paths
# =============================
STIM_ROOT = "stimuli/emb_extension"
MAX_NEW_TOKENS = 8192

# Canonical answer strings and robust patterns
A_CANON = "A. Yes, it changes"
B_CANON = "B. No, it remains steady"

A_PAT = re.compile(r"(?i)\bA\.\s*Yes\b")
B_PAT = re.compile(r"(?i)\bB\.\s*No\b")

# =============================
# System instructions
# =============================
SYSINSTR_PLAIN = """You are a participant in a psychological experiment on music perception. 
In each question, you will be given:
1. A brief instruction about the specific listening task.
2. One audio example to listen to. 

Your task is to identify if the METER (time signature) of a musical excerpt CHANGES during the performance.
For example, it might start in groups of 4 (4/4) and shift to groups of 3 (3/4).
Or it might stay in the same meter for the entire duration.

Valid responses are:
"A. Yes, it changes"
"B. No, it remains steady"

Please respond with "Yes, I understand." if you understand the task."""

# =============================
# Helper Functions
# =============================

def get_stimuli() -> List[Dict[str, str]]:
    """Returns the list of stimuli for this task."""
    stimuli = [
        {"path": "meter_shift_early.wav", "label": A_CANON},
        {"path": "meter_shift_late.wav", "label": A_CANON},
        {"path": "meter_steady_44.wav", "label": B_CANON},
        {"path": "meter_steady_34.wav", "label": B_CANON},
    ]
    return stimuli

def run_task():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment.")
        return

    client = genai.Client(api_key=api_key)
    
    stimuli = get_stimuli()
    random.shuffle(stimuli)
    
    print(f"Starting Temporal Meter Task with {len(stimuli)} trials...")
    
    correct_count = 0
    
    # Simple stateful-like loop for this minimal extension
    for i, stim in enumerate(stimuli):
        full_path = os.path.join("MUSE_music_benchmark", STIM_ROOT, stim['path'])
        print(f"Trial {i+1}: Testing {stim['path']}...")
        
        # Upload file
        uploaded_file = client.files.upload(path=full_path)
        
        # Simple prompt
        prompt = f"Does the meter change in this audio excerpt? Respond with exactly '{A_CANON}' or '{B_CANON}'."
        
        # Generate response
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(file_uri=uploaded_file.uri, mime_type="audio/wav"),
                        types.Part.from_text(text=prompt)
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSINSTR_PLAIN,
                max_output_tokens=MAX_NEW_TOKENS,
            )
        )
        
        model_output = response.text.strip()
        print(f"Model response: {model_output}")
        
        # Evaluate
        if A_PAT.search(model_output) and stim['label'] == A_CANON:
            correct_count += 1
            print("✅ Correct")
        elif B_PAT.search(model_output) and stim['label'] == B_CANON:
            correct_count += 1
            print("✅ Correct")
        else:
            print(f"❌ Incorrect (Expected: {stim['label']})")
            
        # Clean up
        client.files.delete(name=uploaded_file.name)
        
    print(f"\nFinal Accuracy: {correct_count}/{len(stimuli)} ({correct_count/len(stimuli)*100:.2f}%)")

if __name__ == "__main__":
    run_task()
