from transformers import AutoModel, AutoTokenizer, AutoConfig
import os
import shutil

model_name = "google/gemma-2b-it"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

local_dir = "./model"

model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)
