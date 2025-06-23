from datasets import load_dataset

lang_code = "fa_ir"
fleurs = load_dataset("google/fleurs", lang_code, split="test", trust_remote_code=True)
