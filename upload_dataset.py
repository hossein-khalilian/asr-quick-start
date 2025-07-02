from datasets import load_from_disk

dataset = load_from_disk(
    "/home/user/.cache/huggingface/datasets/persian-youtube-resampled/"
)

dataset = dataset.remove_columns("file_name")

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


dataset.push_to_hub("hsekhalilian/persian-youtube", private=True)
