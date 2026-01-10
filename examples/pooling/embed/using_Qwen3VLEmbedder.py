import torch
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
import numpy as np
model = Qwen3VLEmbedder(
    model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
    # flash_attention_2 for better acceleration and memory saving
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2"
)

image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"

inputs = [{
    "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."
}, {
    "image": image_url
}, {
    "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
    "image": image_url
}]

embeddings = model.process(inputs)


def print_embeddings(embeds):
    embeds = np.array(embeds)
    norm = np.linalg.norm(embeds)
    embeds = embeds / norm
    embeds = embeds.tolist()
    embeds_trimmed = (str(embeds[:4])[:-1] + ", ...]") if len(embeds) > 4 else embeds
    print(f"Embeddings: {embeds_trimmed} (size={len(embeds)})")

print("Text embedding output:")
print_embeddings(embeddings[0].tolist())

print("Image embedding output:")
print_embeddings(embeddings[1].tolist())

print("Image+Text embedding output:")
print_embeddings(embeddings[2].tolist())
