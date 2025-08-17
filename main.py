# train_tokenizer.py
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

# 1. Load your dataset (using TinyStories as an example)
dataset = load_dataset("roneneldan/TinyStories", split="train")

# A generator to yield text from the dataset
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

# 2. Instantiate and train the tokenizer
tokenizer = ByteLevelBPETokenizer()

tokenizer.train_from_iterator(
    batch_iterator(),
    vocab_size=10000, # Small vocabulary size for a nano model
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# 3. Save the tokenizer
tokenizer.save_model(".", "nanollm-tokenizer")
# This will create 'vocab.json' and 'merges.txt'
