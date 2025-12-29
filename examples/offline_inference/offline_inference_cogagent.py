"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""

from dataclasses import dataclass
import os
import random
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode
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

class GenerationData(TypedDict):
    prompt: str
    multi_modal_data: dict[str, ImageData]
    multi_modal_uuids: dict[str, str]

# Cogagent
def run_cogagent(
    prompts: GenerationData | list[GenerationData],
    tokenizer: str = "lmsys/vicuna-7b-v1.5",
    **kwargs
) -> tuple[LLM, list[str]]:
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    name = "zai-org/cogagent-vqa-hf"
    llm = LLM(
        model=name,
        tokenizer=tokenizer,
        enforce_eager=True,
        dtype="bfloat16",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        max_model_len=2048,
        gpu_memory_utilization=0.71,
        max_num_seqs=4,
        mm_encoder_attn_backend=AttentionBackendEnum.TORCH_SDPA,
        **kwargs
    )
    
    if isinstance(prompts, dict):
        prompts = [prompts]

    new_prompts = list()
    for prompt in prompts:
        image_data = prompt["multi_modal_data"].get('image')
        content = [prompt["prompt"]]
        if image_data is not None:
            content.append(image_data)

        new_prompt = ChatTemplate(
            'chat_old',
            content=content,
        )

        new_prompt = apply_hf_chat_template(
            llm.get_tokenizer(),
            chat_template=None,
            conversation=new_prompt.asdict(),
            model_config=llm.model_config,
            tools=None,
        )
        new_prompt = GenerationData(
            prompt=new_prompt,
            multi_modal_data=prompt['multi_modal_data'],
            multi_modal_uuids=prompt["multi_modal_uuids"]
        )
        new_prompts.append(new_prompt)

    return llm, new_prompts

def get_multi_modal_input(args) -> tuple[list[GenerationData], list[GenerationData]]:
    
    # Input image and question (Fixed input for testing)
    
    image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    img_questions = [
        "What is the content of this image?",
        "Describe the content of this image in detail.",
        "What's in the image?",
        "Where is this image taken?",
    ]
    data, data_with_empty_media = maybe_apply_image_repeat(
        image_repeat_prob=args.image_repeat_prob,
        num_prompts=args.num_prompts,
        data=image,
        prompts=img_questions,
        modality=args.modality,
    )
    return data, data_with_empty_media

def maybe_apply_image_repeat(
    image_repeat_prob, num_prompts, data, prompts: list[str], modality
):
    """Repeats images with provided probability of "image_repeat_prob".
    Used to simulate hit/miss for the MM preprocessor cache.
    """
    assert image_repeat_prob <= 1.0 and image_repeat_prob >= 0
    no_yes = [0, 1]
    probs = [1.0 - image_repeat_prob, image_repeat_prob]

    inputs = []
    inputs_with_empty_media = []
    cur_image = data
    repeats = random.choices(no_yes, probs, k=num_prompts)
    for i, repeat in enumerate(repeats):
        if repeat == 0:
            # No repeat => Modify one pixel
            cur_image = cur_image.copy()
            new_val = (i // 256 // 256, i // 256, i % 256)
            cur_image.putpixel((0, 0), new_val)

        uuid = "uuid_{}".format(i)

        inputs.append(
            GenerationData(
                prompt=prompts[i % len(prompts)],
                multi_modal_data={modality: cur_image},
                multi_modal_uuids={modality: uuid},
            )
        )

        inputs_with_empty_media.append(
            GenerationData(
                prompt=prompts[i % len(prompts)],
                multi_modal_data={modality: None},
                multi_modal_uuids={modality: uuid},
            )
        )

    return inputs, inputs_with_empty_media


def main(args):
    assert args.num_prompts > 0
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=64,
        skip_special_tokens=False,
        spaces_between_special_tokens=False
    )

    kwargs = dict()
    inputs, inputs_no_data = get_multi_modal_input(args)
    if args.disable_mm_preprocessor_cache:
        kwargs["mm_processor_cache_gb"] = 0

    llm, inputs = run_cogagent(inputs, args.tokenizer, **kwargs)
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
    parser.add_argument(
        '--model-type',
        '-m',
        type=str,
        default="cogagent",
        choices=['cogagent'],
        help='Huggingface "model_type".'
    )
    parser.add_argument(
        '--tokenizer',
        '-t',
        type=str,
        default='lmsys/vicuna-7b-v1.5',
    )

    parser.add_argument(
        '--num-prompts',
        type=int,
        default=1,
        help='Number of prompts to run.'
    )
    
    parser.add_argument(
        '--modality',
        type=str,
        default="image",
        choices=['image'],
        help='Modality of the input.'
    )
    
    parser.add_argument(
        '--image-repeat-prob',
        type=float,
        default=1,
        help='Simulates the hit-ratio for multi-modal preprocessor cache'
             ' (if enabled)'
    )

    parser.add_argument(
        '--disable_mm_preprocessor_cache',
        action='store_true',
        help='If True, enable caching of multi-modal preprocessor/mapper.'
    )

    parser.add_argument(
        '--time-generate',
        action='store_true',
        help='If True, then print the total generate() call time')

    args = parser.parse_args()
    main(args)


