from glob import glob
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

context_length = 512
tokenizer = AutoTokenizer.from_pretrained('wiki-zh-tokenizer')

files = glob('dataset/wiki-zh/*/*', recursive=True)
raw_dataset = load_dataset('json', data_files={'train': files[:-2], 'val': files[-2:]})

def tokenize(element):
    outputs = tokenizer(
        element['text'],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs['length'], outputs['input_ids']):
        if length == context_length:
            input_batch.append(input_ids)
    return {'input_ids': input_batch}


tokenized_datasets = raw_dataset.map(
    tokenize, batched=True, remove_columns=raw_dataset['train'].column_names
)

config = AutoConfig.from_pretrained(
    'gpt2',
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
args = TrainingArguments(
    output_dir='outputs',
    logging_dir='logs',
    logging_steps=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy='steps',
    eval_steps=500,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    weight_decay=0.1,
    warmup_steps=1000,
    lr_scheduler_type='cosine',
    learning_rate=5e-4,
    save_steps=500,
    fp16=True
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['val'],
)

trainer.train()
model.save_pretrained('gpt2-chinese')
