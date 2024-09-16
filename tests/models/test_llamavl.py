from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser

from functools import partial
from PIL import Image as PIL_Image


if __name__ == "__main__":
    model_size_map = {
        "llama-3.2-11b": "11B",
        "llama-3.2-90b": "90B",
    }
    parser = FlexibleArgumentParser(
            description='Demo on using vLLM for offline inference with '
            'vision language models')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="llama-3.2-11b",
                        choices=model_size_map.keys(),
                        help='Huggingface "model_type".')

    args = parser.parse_args()

    size = model_size_map[args.model_type]
    checkpoint_dir = "/data/zhang-chen/llama/checkpoints" # update checkpoint path here
    llm = LLM(model=f"{checkpoint_dir}/Meta-Llama-3.2-{size}-Vision-Early/",
              enforce_eager=True,
              limit_mm_per_prompt={"image": 2},
              max_num_seqs=16,
              tensor_parallel_size=1,
            #   load_format="dummy"
              )

    resource_dir = "/home/eecs/zhang-chen/venv/vllm-multimodal/lib/python3.10/site-packages/llama_models/scripts/resources/"
    # Input image and question
    with open(f"{resource_dir}/dog.jpg", "rb") as f:
        image = PIL_Image.open(f).convert("RGB")
    with open(f"{resource_dir}/pasta.jpeg", "rb") as f:
        image2 = PIL_Image.open(f).convert("RGB")

    inputs = [
        {
            "encoder_prompt":{
                "prompt": "",
                "multi_modal_data": {
                    "image": [image]
                }
            },
            "decoder_prompt": "<|image|><|begin_of_text|>If I had to write a haiku for this one",
        },
        {
            "encoder_prompt":{
                "prompt": "",
            },
            "decoder_prompt": "The color of the sky is blue but sometimes it can also be",
        },
    ]
    outputs = llm.generate(inputs, SamplingParams(temperature=0, top_p=0.9, max_tokens=512))
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("==================================")

