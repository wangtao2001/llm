## 预训练中文GPT2模型

### 数据

wiki百科中文数据，包含1043224个词条，<a href='https://github.com/brightmart/nlp_chinese_corpus'>数据来源</a>

### 训练Tokenizer

分词算法：BPE，`vocab_size`：30000，`min_frequency`：10，真实的GPT系列模型使用的是BBPE

### 训练 & 推理


### 参考

- <a href='https://huggingface.co/learn/nlp-course'>🤗 Hugging Face NLP Course</a>