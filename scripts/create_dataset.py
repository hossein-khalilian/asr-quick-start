import os

import pandas as pd
from datasets import Audio, Dataset

csv_path = "~/.cache/filimo/files.csv"
df = pd.read_csv(csv_path, sep=",")

audio_dir = "/home/user/temp/resampled_audio"
df["path"] = df["path"].apply(lambda x: x.split("/")[-1].split(".mp3")[0] + ".wav")
df["path"] = df["path"].apply(lambda x: os.path.join(audio_dir, x))

print(df.head())
dataset = Dataset.from_pandas(df)

dataset = dataset.cast_column("path", Audio())
dataset = dataset.rename_column("path", "audio")

print(dataset[0]["text"])
print(dataset[0]["audio"]["array"].shape)
print(dataset[0])

dataset.save_to_disk("/home/user/.cache/huggingface/datasets/filimo")
