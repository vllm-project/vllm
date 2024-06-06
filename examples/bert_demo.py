import torch
from transformers import BertTokenizer, BertModel

# 初始化 BERT tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 输入句子
sentence = "This is an example sentence."

# 对输入句子进行编码
inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=128)

# 获取模型输出
with torch.no_grad():
    outputs = model(**inputs)

# 提取句子向量（这里我们使用 [CLS] token 的向量作为句子向量）
sentence_vector = outputs.last_hidden_state[:, 0, :].squeeze()

print(sentence_vector)

print(sentence_vector.shape)
