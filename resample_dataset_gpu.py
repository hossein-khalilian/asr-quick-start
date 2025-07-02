import os

import soundfile as sf
import torch
import torchaudio
from datasets import Audio, load_dataset

dataset = load_dataset("hsekhalilian/filimo", num_proc=32)


output_dir = "/home/user/.cache/datasets/filimo/audio_files"
os.makedirs(output_dir, exist_ok=True)

target_sr = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resampler_dict = {}


def process(example):
    waveform = torch.tensor(example["audio"]["array"]).unsqueeze(0)  # Shape: [1, T]
    original_sr = example["audio"]["sampling_rate"]
    waveform = waveform.to(torch.float32).to(device)

    if not resampler_dict.get(str(original_sr)):
        resampler_dict[str(original_sr)] = torchaudio.transforms.Resample(
            orig_freq=original_sr, new_freq=target_sr
        ).to(device)

    if original_sr != target_sr:
        resampled = resampler_dict[str(original_sr)](waveform).squeeze(0).cpu().numpy()
    else:
        resampled = waveform.squeeze(0).cpu().numpy()

    filename = os.path.splitext(example["audio"]["path"])[0] + ".flac"
    output_path = os.path.join(output_dir, filename)
    sf.write(output_path, resampled, target_sr, format="FLAC")

    return {
        "audio": output_path,
        "text": example["text"],
        "normalized_transcription": example["normalized_transcription"],
    }


new_dataset = dataset.map(process)
new_dataset = new_dataset.cast_column("audio", Audio(sampling_rate=target_sr))

new_dataset.save_to_disk(
    "/home/user/.cache/huggingface/datasets/filimo-resampled",
    num_proc=16,
)
