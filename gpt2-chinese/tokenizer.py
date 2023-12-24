from glob import glob
from datasets import load_dataset
from tokenizers import (
    Tokenizer, models, pre_tokenizers, trainers
)
from transformers import GPT2TokenizerFast

files = glob('dataset/wiki-zh/*/*', recursive=True)
raw_dataset = load_dataset('json', data_files=files)

def get_training_corpus():
    dataset = raw_dataset['train']
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples['text']

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["<|endoftext|>"], min_frequency=10, show_progress=True)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)

wrapped_tokenizer.save_pretrained('wiki-zh-tokenizer')