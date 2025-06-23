import torch
from datasets import load_dataset
from transformers import pipeline
from jiwer import wer
from tqdm import tqdm

# Optional: to save or play audio
import soundfile as sf
from IPython.display import Audio, display

# --- CONFIG ---
device = "cuda" if torch.cuda.is_available() else "cpu"
lang_code = "fa_ir"
num_samples = 20  # use more for full evaluation
save_audio = False  # set to True to save files
play_audio = False  # set to True in notebooks to play files

models = ["openai/whisper-small", "m3hrdadfi/wav2vec2-large-xlsr-persian"]

# --- Load Persian FLEURS dataset ---
print("Loading dataset...")
dataset = load_dataset("google/fleurs", lang_code, split=f"test[:{num_samples}]")

# --- Evaluation loop ---
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
            # --- Run ASR ---
            result = asr(audio_array, chunk_length_s=30, return_timestamps=False)
            predicted_text = result["text"].strip().lower()

            # --- Collect results ---
            references.append(reference_text)
            hypotheses.append(predicted_text)

            # --- Optional: Save or play audio ---
            if save_audio:
                out_path = f"audio_{idx}.wav"
                sf.write(out_path, audio_array, sampling_rate)
            if play_audio:
                display(Audio(data=audio_array, rate=sampling_rate))

        except Exception as e:
            print(f"Error on sample {idx}: {e}")

    # --- Calculate and display WER ---
    error = wer(references, hypotheses)
    print(f"✅ WER for {model_name} on FLEURS (Persian): {error:.3f}")
