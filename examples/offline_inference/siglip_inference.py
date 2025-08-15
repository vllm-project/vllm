# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import requests
import torch
from PIL import Image

from vllm import LLM
from vllm.inputs import TextPrompt

MODEL_ID = "HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit"
TOKENIZER_ID = "google/siglip-base-patch16-224"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
TEXTS: list[str] = ["a photo of a cat", "a photo of a dog"]

llm = LLM(
    model=MODEL_ID,
    tokenizer=TOKENIZER_ID,
    trust_remote_code=True,
    dtype="half",
    gpu_memory_utilization=0.8,
)

image = Image.open(requests.get(IMAGE_URL, stream=True).raw).convert("RGB")

image_input = TextPrompt(prompt="", multi_modal_data={"image": image})
text_inputs = [TextPrompt(prompt=p) for p in TEXTS]


image_outputs = llm.encode([image_input])
image_embedding = image_outputs[0].outputs.data.squeeze()

text_outputs = llm.encode(text_inputs)

cat_text_embedding = text_outputs[0].outputs.data.squeeze()
dog_text_embedding = text_outputs[1].outputs.data.squeeze()

sim_cat = torch.nn.functional.cosine_similarity(
    image_embedding, cat_text_embedding, dim=0
)
sim_dog = torch.nn.functional.cosine_similarity(
    image_embedding, dog_text_embedding, dim=0
)

print(f"Similarity between image and '{TEXTS[0]}': {sim_cat.item():.4f}")
print(f"Similarity with '{TEXTS[1]}': {sim_dog.item():.4f}")

if sim_cat > sim_dog:
    print("\nSanity check PASSED: Model correctly identified the cat image.")
else:
    print("\nSanity check FAILED: Model could not distinguish between cat and dog.")
