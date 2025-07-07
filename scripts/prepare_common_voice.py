import os
import pandas as pd
from datasets import Dataset, Audio


def process_common_voice_split(base_path, split, output_dir):
    tsv_path = os.path.join(base_path, f"{split}.tsv")
    clips_path = os.path.join(base_path, "clips")

    df = pd.read_csv(tsv_path, sep="\t", quoting=3, on_bad_lines="skip")
    df["audio"] = df["path"].apply(lambda x: os.path.join(clips_path, x))

    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio", Audio())

    output_path = os.path.join(output_dir, f"commonvoice-fa-{split}")
    dataset.save_to_disk(output_path)
    print(f"[INFO] Saved {split} split to {output_path}")


def main():
    home_dir = os.path.expanduser("~")
    base_path = os.path.join(
        home_dir, ".cache", "commonvoice", "cv-corpus-22.0-2025-06-20", "fa"
    )
    output_dir = os.path.join(
        home_dir, ".cache", "huggingface", "datasets", "common_voice"
    )
    os.makedirs(output_dir, exist_ok=True)

    # for split in ["train", "test", "dev"]:
    for split in [
        "dev",
        # "invalidated",
        # "other",
        # "reported",
        "test",
        "train",
        # "unvalidated_sentences",
        # "validated",
        # "validated_sentences.ts",
    ]:
        print(split)
        try:
            process_common_voice_split(base_path, split, output_dir)
        except:
            continue


if __name__ == "__main__":
    main()
