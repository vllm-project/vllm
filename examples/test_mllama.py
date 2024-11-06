import argparse
from io import BytesIO
from PIL import Image
import requests
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from vllm.attention.selector import (_Backend, global_force_attn_backend)

model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

QUESTION = "What is the content of each image?"
IMAGE_URLS = [
  "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
  "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
]
attn_backend_map = {
  'flash': _Backend.FLASH_ATTN,
  'xformer': _Backend.XFORMERS,
}

images = [fetch_image(url) for url in IMAGE_URLS]

def run_vllm(attn_backend):
  if attn_backend != 'auto':
    global_force_attn_backend(attn_backend_map[attn_backend])
  llm = LLM(
    model=model_name,
    enforce_eager=True,
    max_model_len=4096,
    max_num_seqs=16,
    limit_mm_per_prompt={"image": len(images)},
  )
  prompt = f"<|image|><|image|><|begin_of_text|>{QUESTION}"
  inputs = {
    "prompt": prompt,
    "multi_modal_data": {
      "image": images
    },
  }
  sampling_params = SamplingParams(temperature=0.0,
                   max_tokens=9,)
  outputs = llm.generate(inputs, sampling_params=sampling_params)
  for o in outputs:
    print(o.outputs[0].text)


def run_huggingface():
  prompt = f"<|image|><|image|><|begin_of_text|>{QUESTION}"

  model = MllamaForConditionalGeneration.from_pretrained(model_name, device_map="auto")
  processor = AutoProcessor.from_pretrained(model_name)
  inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)
  output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
  print(processor.decode(output[0]))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--backend', choices=['vllm', 'hf'])
  parser.add_argument('--attn', choices=['flash', 'xformer', 'auto'], default='auto')
  args = parser.parse_args()
  if args.backend == 'vllm':
    run_vllm(attn_backend=args.attn)
  else:
    run_huggingface()
