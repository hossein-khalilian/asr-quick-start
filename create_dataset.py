import os

import pandas as pd
from datasets import Audio, Dataset

csv_path = "~/temp/unvalidated.csv"

audio_dir = "/home/user/temp/audio_files"

df = pd.read_csv(csv_path, sep=",")


df["audio"] = df["file_name"].apply(lambda x: os.path.join(audio_dir, x))

dataset = Dataset.from_pandas(df)

dataset = dataset.cast_column("audio", Audio())

print(dataset[0]["sentence"])
print(dataset[0]["audio"]["array"].shape)

dataset.save_to_disk("/home/user/.cache/huggingface/datasets/persian_youtube")
