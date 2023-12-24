from transformers import GPT2LMHeadModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('wiki-zh-tokenizer')
model = GPT2LMHeadModel.from_pretrained('gpt2-chinese', pad_token_id=tokenizer.eos_token_id)

txt = ''
input_ids = tokenizer.encode(txt, return_tensors='pt')
beam_output = model.generate(
    input_ids, 
    max_length=200, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    early_stopping=True
)
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))