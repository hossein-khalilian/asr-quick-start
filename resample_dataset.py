import os

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

dataset = load_dataset("hsekhalilian/persian-youtube", num_proc=32)

output_dir = "/home/user/.cache/datasets/persian-youtube/audio_files"
os.makedirs(output_dir, exist_ok=True)

target_sr = 16000


def process(example):
    audio_array = example["audio"]["array"]
    original_sr = example["audio"]["sampling_rate"]
    resampled = librosa.resample(
        np.array(audio_array), orig_sr=original_sr, target_sr=target_sr
    )

    # Output file name
    filename = os.path.splitext(example["audio"]["path"])[0] + ".flac"
    output_path = os.path.join(output_dir, filename)

    # Save as FLAC
    sf.write(output_path, resampled, target_sr, format="FLAC")

    # Return updated record
    return {
        "audio": output_path,
        "sentence": example["sentence"],
        "normalized_transcription": example["normalized_transcription"],
    }


# Map over dataset
new_dataset = dataset.map(process, num_proc=16)
new_dataset = new_dataset.cast_column("audio", Audio(sampling_rate=target_sr))

new_dataset.save_to_disk(
    "/home/user/.cache/huggingface/datasets/persian-youtube-resampled"
)
