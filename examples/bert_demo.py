import torch
from transformers import BertTokenizer, BertModel

# Init BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

print("Model: ")
print(model)

print("Parameter: ")
for name, weight in model.named_parameters():
    print(f"Name: {name}".ljust(60) + f"Weight: {weight.shape}".ljust(40) + f"dtype: {weight.dtype}".ljust(20))


# sentence = "This is an example sentence."

# inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=128)

# with torch.no_grad():
#     outputs = model(**inputs)

# # Get the sentence vector. Here use [CLS] token as the sentence vector.
# sentence_vector = outputs.last_hidden_state[:, 0, :].squeeze()

# print(sentence_vector)
# print(sentence_vector.shape)
