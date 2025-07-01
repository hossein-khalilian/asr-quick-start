import os
import shutil
import subprocess

# --- Configuration ---
# Directory containing the original MP3 files.
INPUT_DIR = "audio_files"
# Directory where the resampled MP3 files will be saved.
OUTPUT_DIR = "resampled_audio_files"
# The target audio sampling rate in Hz.
TARGET_SR = 16000
# The audio codec to use for the output files.
# 'copy' can be used if you only want to change the container, not re-encode.
# For mp3, 'libmp3lame' is a high-quality choice.
AUDIO_CODEC = "libmp3lame"


def check_ffmpeg_installed():
    """
    Checks if ffmpeg is installed and available in the system's PATH.
    """
    if not shutil.which("ffmpeg"):
        print("---")
        print("Error: ffmpeg is not installed or not found in your system's PATH.")
        print("Please install ffmpeg to run this script.")
        print(
            "Installation instructions can be found at: https://ffmpeg.org/download.html"
        )
        print("---")
        return False
    return True


def resample_files_with_ffmpeg():
    """
    Finds all .mp3 files in the input directory, resamples them using ffmpeg,
    and saves them to the output directory.
    """
    # --- 1. Pre-run Checks ---
    if not check_ffmpeg_installed():
        return  # Exit if ffmpeg is not available

    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        # Create a dummy directory so the user knows where to put files.
        print(
            f"Creating '{INPUT_DIR}' for you. Please add your .mp3 files there and run again."
        )
        os.makedirs(INPUT_DIR)
        return

    # Ensure the output directory exists.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Resampled files will be saved in: '{OUTPUT_DIR}'")

    # --- 2. Find and Process Audio Files ---
    files_to_process = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".mp3")]

    if not files_to_process:
        print(f"No .mp3 files found in '{INPUT_DIR}'.")
        return

    print(f"\nFound {len(files_to_process)} .mp3 file(s) to process.")
    success_count = 0
    fail_count = 0

    for filename in files_to_process:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        print(f"\nProcessing '{filename}'...")

        # --- 3. Construct and Run FFmpeg Command ---
        # Command: ffmpeg -i <input_file> -ar <sampling_rate> -acodec <codec> <output_file>
        # -i: specifies the input file
        # -ar: sets the audio sampling rate
        # -acodec: sets the audio codec
        # -y: overwrites the output file if it already exists
        command = [
            "ffmpeg",
            "-i",
            input_path,
            "-ar",
            str(TARGET_SR),
            "-acodec",
            AUDIO_CODEC,
            "-y",
            output_path,
        ]

        try:
            # We use subprocess.run to execute the command.
            # capture_output=True and text=True will capture stdout/stderr.
            result = subprocess.run(
                command,
                check=True,  # This will raise an exception if ffmpeg returns an error
                capture_output=True,
                text=True,
                encoding="utf-8",  # Ensure correct encoding for output
            )
            print(f"Successfully resampled '{filename}'")
            success_count += 1

        except subprocess.CalledProcessError as e:
            # This block runs if ffmpeg returns a non-zero exit code (an error).
            print(f"--- FAILED to process '{filename}' ---")
            print(f"FFmpeg Error Output:\n{e.stderr}")
            fail_count += 1
        except FileNotFoundError:
            # This should not happen due to the check at the start, but is good practice.
            print("Error: 'ffmpeg' command not found. Please ensure it is installed.")
            return
        except Exception as e:
            print(f"An unexpected error occurred while processing '{filename}': {e}")
            fail_count += 1

    # --- 4. Final Report ---
    print("\n--- Processing Complete ---")
    print(f"Successfully converted: {success_count} file(s)")
    print(f"Failed to convert: {fail_count} file(s)")
    print(f"Resampled files are located in the '{OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    resample_files_with_ffmpeg()
