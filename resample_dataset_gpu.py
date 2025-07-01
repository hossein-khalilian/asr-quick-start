import numpy as np
import torch
import torchaudio
from datasets import load_from_disk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_from_disk("/home/user/.cache/huggingface/datasets/filimo/")
target_sr = 16000


def pad_waveforms(waveforms):
    # Pad all waveforms in batch to max length
    max_len = max(w.shape[0] for w in waveforms)
    padded = torch.zeros(len(waveforms), max_len, dtype=torch.float32)
    for i, w in enumerate(waveforms):
        padded[i, : w.shape[0]] = w
    return padded


def resample_with_torchaudio_batch(batch):
    waveforms = []
    orig_srs = []

    # Convert all to torch tensors first
    for audio in batch["audio"]:
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        waveforms.append(waveform)
        orig_srs.append(audio["sampling_rate"])

    # Pad waveforms to same length and move to device
    padded_waveforms = pad_waveforms(waveforms).to(
        device
    )  # shape: (batch_size, max_len)
    batch_size = padded_waveforms.size(0)

    resampled_batch = []
    for i in range(batch_size):
        orig_sr = orig_srs[i]
        waveform = padded_waveforms[i].unsqueeze(0)  # shape: (1, max_len)

        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=target_sr
            ).to(device)
            waveform = resampler(waveform)
        resampled_batch.append(waveform.squeeze(0).cpu().numpy())

    # Return list of dicts with resampled waveforms and new sampling rate
    return {
        "audio": [
            {"array": resampled_waveform, "sampling_rate": target_sr}
            for resampled_waveform in resampled_batch
        ]
    }


dataset = dataset.map(
    resample_with_torchaudio_batch,
    batched=True,
    batch_size=16,
    num_proc=1,
)

dataset.save_to_disk("/home/user/.cache/huggingface/datasets/filimo_16000")
