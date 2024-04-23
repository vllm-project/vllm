import requests
import torch
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig, AutoModel
from transformers.image_utils import load_image

# DEVICE = "cuda:0"

# # # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
# image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
# image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
# image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

# processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b",  do_image_splitting=False)
# model = AutoModelForVision2Seq.from_pretrained(
#     "HuggingFaceM4/idefics2-8b",
#     # torch_dtype=torch.float16,   
# ).to(DEVICE)

# print(processor)

# # Create inputs
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image"},
#             {"type": "text", "text": "What do we see in this image?"},
#         ]
#     },
#     {
#         "role": "assistant",
#         "content": [
#             {"type": "text", "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty."},
#         ]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "image"},
#             {"type": "text", "text": "And how about this image?"},
#         ]
#     },       
# ]
# prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
# inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt")
# inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

# for k in inputs.keys():
#     print(k)
#     print(inputs[k].shape)


# print(model)
# # Generate
# # generated_ids = model.generate(**inputs, max_new_tokens=500)
# # generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

# # print("RET", generated_texts)
# pixel_values = inputs['pixel_values']
# batch_size, num_images, num_channels, height, width = pixel_values.shape
# pixel_values = pixel_values.to(torch.float16)  # fp16 compatibility
# pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])
# print(pixel_values.shape)

# nb_values_per_image = pixel_values.shape[1:].numel()
# real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
# pixel_values = pixel_values[real_images_inds].contiguous()
# print(pixel_values.shape)

model = AutoModel.from_pretrained("bert-base-uncased")
print("ST")
# for name, param in model.named_parameters():
#     print("hello_world")
#     break