from huggingface_hub import login
from datasets import load_from_disk
ds = load_from_disk(r"TRABAJO CREATIVO\transformer\dataset.hf")
ds.push_to_hub("tukx/processed_fake_news")