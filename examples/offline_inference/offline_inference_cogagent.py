"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
from dataclasses import dataclass
import os
import random
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.entrypoints.chat_utils import apply_hf_chat_template
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.attention.backends.registry import AttentionBackendEnum

ImageData: TypeAlias = Any
if TYPE_CHECKING:
    import torch
    import numpy as np
    from PIL import Image

    ImageData: TypeAlias = torch.Tensor | np.ndarray | Image.Image
    
# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.

@dataclass
class ChatTemplate:
    role: str
    content: list[str | ImageData]

    def asdict(self):
        assert isinstance(self.content, (list, tuple))
        output = {
            "role": self.role,
            "content": [
                {"type": "image" if not isinstance(content, str) else "text",
                 "image" if not isinstance(content, str) else "text": content}
                 for content in self.content
            ]
        }
        return output

# Cogagent
def load_model(name, **kwargs):
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    name = "/home/burto/base/programs/transcoder/SuperTranslator/checkpoints/cogagent_ckpt4500merged"
    llm = LLM(
        model=name,
        #tokenizer="lmsys/vicuna-7b-v1.5",
        #tokenizer_mode="slow",
        enforce_eager=True,
        dtype="bfloat16",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        max_model_len=2048,
        gpu_memory_utilization=0.95,
        max_num_seqs=4,
        mm_encoder_attn_backend=AttentionBackendEnum.TORCH_SDPA,
        **kwargs
    )
    
    return llm

def apply_template(llm: LLM, conversation):
    if isinstance(conversation, dict):
        conversation = [conversation]
    prompt = apply_hf_chat_template(
        llm.get_tokenizer(),
        chat_template=None,
        model_config=llm.model_config,
        tools=None, 
        conversation=conversation
    )
    return prompt

def get_multi_modal_input(
    llm,
    num_prompts,
    image_repeat_prob
) -> list[dict[Literal['prompt', 'multi_modal_data'], str | dict]]:
    
    # Input image and question
    image = ImageAsset("cherry_blossom") \
        .pil_image.convert("RGB")
    prompts = [
        # "What is the content of this image?",
        # "Describe this image.",
        # "How does this image feel?",
        "What is happening in this image?"
    ]

    # images = apply_image_repeat(
    #     image_repeat_prob,
    #     num_prompts,
    #     image
    # )
    images = [image] * num_prompts
    data = list()
    for image in images:
        prompt = random.choice(prompts)
        conversation = ChatTemplate("chat_old", [prompt, image]).asdict()
        template = apply_template(llm, conversation=conversation)
        data.append({
            "prompt": template, 
            "multi_modal_data": {"image": image}
        })

    return data

def apply_image_repeat(image_repeat_prob, num_prompts, image):
    """Repeats images with provided probability of "image_repeat_prob". 
    Used to simulate hit/miss for the MM preprocessor cache.
    """
    assert (image_repeat_prob <= 1.0 and image_repeat_prob >= 0)

    images = []
    cur_image = image
    for i in range(num_prompts):
        res = random.random()
        if res >= image_repeat_prob:
            # No repeat => Modify one pixel
            cur_image = cur_image.copy()
            new_val = (i // 256 // 256, i // 256, i % 256)
            cur_image.putpixel((0, 0), new_val)

        images.append(cur_image)

    return images


def main(args):
    assert args.num_prompts > 0

    model_name = args.model_type
    llm = load_model(model_name)
    
    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=64,
        skip_special_tokens=False,
        spaces_between_special_tokens=False
    )
    
    inputs = get_multi_modal_input(
        llm,
        args.num_prompts,
        args.image_repeat_prob
    )
    
    if args.time_generate:
        import time
        start_time = time.time()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        elapsed_time = time.time() - start_time
        print("-- generate time = {}".format(elapsed_time))

    else:
        outputs = llm.generate(inputs, sampling_params=sampling_params)

    for inp, out in zip(inputs, outputs):
        generated_text = out.outputs[0].text
        print("_" * 80)
        print(inp['prompt'])
        print(generated_text)
        print("_" * 80)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for text generation')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="cogagent",
                        choices=['cogagent'],
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=1,
                        help='Number of prompts to run.')
    
    parser.add_argument('--modality',
                        type=str,
                        default="image",
                        choices=['image'],
                        help='Modality of the input.')
    
    parser.add_argument(
        '--image-repeat-prob',
        type=float,
        default=1,
        help='Simulates the hit-ratio for multi-modal preprocessor cache'
        ' (if enabled)')

    parser.add_argument(
        '--disable_mm_preprocessor_cache',
        action='store_true',
        help='If True, enable caching of multi-modal preprocessor/mapper.')

    parser.add_argument(
        '--time-generate',
        action='store_true',
        help='If True, then print the total generate() call time')

    args = parser.parse_args()
    main(args)

