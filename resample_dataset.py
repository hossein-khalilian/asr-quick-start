import os

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

dataset_name = "filimo"

dataset = load_dataset(f"hsekhalilian/{dataset_name}", num_proc=32)

output_dir = f"/home/user/.cache/datasets/{dataset_name}/audio_files"
os.makedirs(output_dir, exist_ok=True)

target_sr = 16000


def process(example):
    audio_array = example["audio"]["array"]
    original_sr = example["audio"]["sampling_rate"]
    resampled = librosa.resample(
        np.array(audio_array), orig_sr=original_sr, target_sr=target_sr
    )

    filename = os.path.splitext(example["audio"]["path"])[0] + ".flac"
    output_path = os.path.join(output_dir, filename)
    sf.write(output_path, resampled, target_sr, format="FLAC")

    return {
        "audio": output_path,
        "text": example["text"],
        "normalized_transcription": example["normalized_transcription"],
    }


new_dataset = dataset.map(process, num_proc=32)
new_dataset = new_dataset.cast_column("audio", Audio(sampling_rate=target_sr))

new_dataset.save_to_disk(
    f"/home/user/.cache/huggingface/datasets/{dataset_name}-resampled",
    num_proc=32,
)
