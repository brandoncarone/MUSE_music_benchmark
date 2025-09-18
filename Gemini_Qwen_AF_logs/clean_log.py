# clean_log_file_corrected.py
import os
import re

def clean_log_file(log_file):
    """
    Removes lines containing specific, repetitive logging patterns from a log file.

    Args:
        log_file (str): The path to the log file.
    """

    # --- THE FIX IS HERE ---
    # The first pattern is now generalized to match any gemini model version.
    # The other patterns were already correct.
    patterns = [
        #Gemini
        r"HTTP Request: POST https://generativelanguage\.googleapis\.com/v1beta/models/gemini-.*?:generateContent",
        r"AFC is enabled with max remote calls: 10\.",
        r"AFC remote call \d+ is done\.",

        # Qwen (combined one-liner)
        r"System prompt modified, audio output may not work as expected\.\s*Audio output mode only works when using default system prompt.*"
    ]

    try:
        with open(log_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        cleaned_lines = []
        for line in lines:
            # Keep the line if it does *not* match any of the patterns
            keep_line = True
            for pattern in patterns:
                if re.search(pattern, line):
                    keep_line = False
                    break  # Stop checking other patterns if one matches
            if keep_line:
                cleaned_lines.append(line)

        with open(log_file, 'w', encoding='utf-8') as outfile:  # Overwrite the original file
            outfile.writelines(cleaned_lines)

        print(f"Log file '{log_file}' cleaned successfully.")

    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Find all log files in the current directory that end with .log or .txt
    log_files = [f for f in os.listdir('.') if f.endswith(('.log', '.txt'))]

    if not log_files:
        print("No log files (.log) found in the current directory.")
    else:
        for log_file in log_files:
            print(f"Cleaning log file: {log_file}")
            clean_log_file(log_file)
        print("All log files processed.")