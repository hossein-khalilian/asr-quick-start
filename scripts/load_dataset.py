import os

from datasets import load_dataset

# dataset = load_dataset("SeyedAli/Persian-Speech-Dataset", trust_remote_code=True)

# dataset = load_dataset("Kamtera/ParsiGoo", trust_remote_code=True)

# dataset = load_dataset("PerSets/filimo-persian-asr", trust_remote_code=True)

# dataset = load_dataset("PerSets/youtube-persian-asr", trust_remote_code=True)

# dataset = load_dataset(
#     "pariajm/sharif_emotional_speech_dataset", trust_remote_code=True
# )

# dataset = load_dataset(
#     "MohammadGholizadeh/costumer-service-persian-1", trust_remote_code=True
# )

# dataset = load_dataset("espnet/floras", "multilingual")

# dataset = load_dataset("wuenlp/fleurs-belebele", "pes_Arab", trust_remote_code=True)

# My own datasets
# dataset = load_dataset("hsekhalilian/filimo", num_proc=32)

# dataset = load_dataset("hsekhalilian/persian-youtube", num_proc=32)

# dataset = load_dataset("hsekhalilian/fleurs", num_proc=32)

dataset = load_dataset("hsekhalilian/psrb", num_proc=32)

print(dataset)
