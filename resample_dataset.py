import os
import librosa
import numpy as np
from datasets import load_from_disk, Dataset
from functools import partial

# --- Configuration ---
# Define paths for input and output. This makes them easier to change.
# It's good practice to use os.path.expanduser to handle '~' correctly.
INPUT_DATASET_PATH = os.path.expanduser("~/.cache/huggingface/datasets/filimo/")
OUTPUT_DATASET_PATH = os.path.expanduser("~/.cache/huggingface/datasets/filimo_16000")
TARGET_SR = 16000
# Using a smaller number of processors or letting the library decide is often safer.
# If you have many cores, you can increase this, but 4 is a safe default.
NUM_PROC = 4 

def resample_audio(example, target_sr):
    """
    Resamples a single audio example to the target sampling rate.

    Args:
        example (dict): A dictionary containing the audio data from the dataset.
                        It must have an "audio" key with "array" and "sampling_rate".
        target_sr (int): The target sampling rate to resample to.

    Returns:
        dict: A dictionary with the resampled audio array and the new sampling rate.
    """
    try:
        audio_data = example["audio"]
        waveform = audio_data["array"]
        original_sr = audio_data["sampling_rate"]

        # Ensure the waveform is a NumPy array for processing
        if not isinstance(waveform, np.ndarray):
            waveform = np.array(waveform)

        # Only resample if the original sampling rate is different from the target.
        if original_sr != target_sr:
            # librosa.resample requires the input array to be float.
            # We check the dtype and convert if necessary.
            if waveform.dtype != np.float32:
                waveform = waveform.astype(np.float32)
            
            resampled_waveform = librosa.resample(y=waveform, orig_sr=original_sr, target_sr=target_sr)
            return {"audio": {"array": resampled_waveform, "sampling_rate": target_sr}}
        else:
            # If already at the target rate, just return the original data.
            return {"audio": {"array": waveform, "sampling_rate": original_sr}}
            
    except Exception as e:
        # If any error occurs during resampling, log it and return an empty audio array.
        # This prevents the whole mapping process from crashing due to one bad file.
        print(f"Error processing an audio file: {e}")
        return {"audio": {"array": np.array([], dtype=np.float32), "sampling_rate": target_sr}}


def main():
    """
    Main function to load, resample, and save the audio dataset.
    """
    # --- 1. Load Dataset ---
    print(f"Attempting to load dataset from: {INPUT_DATASET_PATH}")
    try:
        # Check if the directory exists before trying to load
        if not os.path.isdir(INPUT_DATASET_PATH):
            print(f"Error: Input directory not found at {INPUT_DATASET_PATH}")
            return # Exit the script if the data isn't there

        dataset = load_from_disk(INPUT_DATASET_PATH)
        print("Dataset loaded successfully.")
        print(f"Dataset features: {dataset.features}")

    except Exception as e:
        print(f"An unexpected error occurred while loading the dataset: {e}")
        return

    # --- 2. Resample Dataset ---
    print(f"\nStarting resampling process to {TARGET_SR} Hz...")
    try:
        # Use functools.partial to pass the target_sr argument to the map function
        resample_fn = partial(resample_audio, target_sr=TARGET_SR)
        
        resampled_dataset = dataset.map(
            resample_fn,
            num_proc=NUM_PROC
        )
        print("Resampling complete.")

    except Exception as e:
        print(f"An error occurred during the .map() operation: {e}")
        return

    # --- 3. Save Resampled Dataset ---
    print(f"\nSaving resampled dataset to: {OUTPUT_DATASET_PATH}")
    try:
        # Ensure the output directory exists before saving.
        os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
        resampled_dataset.save_to_disk(OUTPUT_DATASET_PATH)
        print("Dataset saved successfully!")

    except Exception as e:
        print(f"An error occurred while saving the dataset: {e}")
        return

if __name__ == "__main__":
    main()
