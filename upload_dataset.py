from datasets import load_from_disk

dataset_name = "filimo"

dataset = load_from_disk(
    f"/home/user/.cache/huggingface/datasets/{dataset_name}-resampled/"
)

dataset = dataset.rename_column("text", "sentence")

print(dataset)
print(dataset["train"][0])

all_text = " ".join(dataset["train"]["sentence"])
unique_chars = sorted(set(all_text))

print("Unique characters:", unique_chars)
print("Number of unique characters:", len(unique_chars))

all_text = " ".join(dataset["train"]["normalized_transcription"])
unique_chars = sorted(set(all_text))

print("Unique characters:", unique_chars)
print("Number of unique characters:", len(unique_chars))


dataset.push_to_hub(f"hsekhalilian/{dataset_name}", private=True)
