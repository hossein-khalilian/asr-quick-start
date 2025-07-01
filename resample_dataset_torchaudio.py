import torch
import torchaudio
from datasets import Audio, load_from_disk

# Load dataset
dataset = load_from_disk("/home/user/.cache/huggingface/datasets/filimo/")

# Target sampling rate
target_sr = 16000

# Ensure audio decoding returns PyTorch tensors
dataset = dataset.cast_column("audio", Audio(sampling_rate=None))

# Resampler cache (to avoid re-creating it each time)
resamplers = {}


def resample_with_torchaudio(example):
    audio = example["audio"]
    waveform, orig_sr = audio["array"], audio["sampling_rate"]

    if orig_sr != target_sr:
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).to(torch.float32)
        else:
            waveform = waveform.to(torch.float32)

        if orig_sr not in resamplers:
            resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, target_sr)
            print(orig_sr)


        waveform = resamplers[orig_sr](waveform)
        waveform = waveform.squeeze(0)

    return {"audio": {"array": waveform.numpy(), "sampling_rate": target_sr}}


dataset = dataset.map(resample_with_torchaudio, num_proc=32)
dataset.save_to_disk("/home/user/.cache/huggingface/datasets/filimo_16000")
