import torch
from datasets import load_dataset
from transformers import pipeline
from jiwer import wer
from tqdm import tqdm
import soundfile as sf
import os

# --- CONFIGURATION ---
device = "cuda" if torch.cuda.is_available() else "cpu"
lang_code = "fa_ir"
num_samples = 20  # Adjust for larger evaluation
save_audio = False  # Set to True to save audio files
output_audio_dir = "fleurs_audio"

models = ["openai/whisper-small", "m3hrdadfi/wav2vec2-large-xlsr-persian"]


def main():
    print("🚀 Loading Persian FLEURS dataset...")
    dataset = load_dataset("google/fleurs", lang_code, split=f"test[:{num_samples}]")

    if save_audio:
        os.makedirs(output_audio_dir, exist_ok=True)

    for model_name in models:
        print(f"\n🔍 Evaluating model: {model_name}")
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=0 if device == "cuda" else -1,
        )

        references, hypotheses = [], []

        for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
            audio_array = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
            reference_text = sample["transcription"].strip().lower()

            try:
                result = asr(audio_array, chunk_length_s=30, return_timestamps=False)
                predicted_text = result["text"].strip().lower()

                references.append(reference_text)
                hypotheses.append(predicted_text)

                if save_audio:
                    out_path = os.path.join(output_audio_dir, f"sample_{idx}.wav")
                    sf.write(out_path, audio_array, sampling_rate)

            except Exception as e:
                print(f"❌ Error on sample {idx}: {e}")
                continue

        error = wer(references, hypotheses)
        print(f"✅ WER for {model_name} on FLEURS (Persian): {error:.3f}")


if __name__ == "__main__":
    main()
