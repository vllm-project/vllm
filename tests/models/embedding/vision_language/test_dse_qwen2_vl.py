import os
from typing import List, Type

import pytest
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from ....conftest import IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner
from ....utils import large_gpu_test
from ..utils import check_embeddings_close

HF_TEXT_PROMPTS = [
    # T -> X
    (
        "Query: Find me an everyday image that matches the given caption: The label of the object is stop sign",  # noqa: E501,
        Image.new("RGB", (56, 56))),
    # T -> X
    ("Query: Retrieve an image of this caption: cherry blossom",
     Image.new("RGB", (56, 56))),
]

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "What is shown in this image?",
    "cherry_blossom":
    "What is shown in this image?"
})

MODELS = ["MrLight/dse-qwen2-2b-mrl-v1"]


class QwenVLEncoder:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        attn = "flash_attention_2" if self.device == "cuda" else None

        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.processor = AutoProcessor.from_pretrained(MODELS[0])
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODELS[0], attn_implementation=attn,
            torch_dtype=torch.bfloat16).to(self.device).eval()
        self.processor.tokenizer.padding_side = "left"
        self.model.padding_side = "left"
        self.base_embed_dim = 1536

    def _get_embedding(self, last_hidden_state: torch.Tensor,
                       dimension: int) -> torch.Tensor:
        reps = last_hidden_state[:, -1]
        reps = torch.nn.functional.normalize(reps[0, :dimension], p=2, dim=-1)
        return reps

    def embed(self, inp: dict, embed_dim: int = 1536) -> torch.Tensor:
        """
        inp: dict
            {
                "dtype": "image",
                "image": PIL.Image,
            }
            or 
            {
                "dtype": "text",
                "question": (str) the question to embed,
            }
        embed_dim: int
            Will slice embeddings like emb[:embed_dim]
        """
        if inp["dtype"] == "image":
            messages = [[{
                "role":
                "user",
                "content": [{
                    "type": "image",
                    "image": inp["image"]
                }, {
                    "type": "text",
                    "text": "What is shown in this image?"
                }]
            }]]
        else:
            messages = [[{
                "role":
                "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.new("RGB", (28, 28)),
                        "resized_height": 1,
                        "resized_width": 1
                    },  # need a dummy image here for an easier process.
                    {
                        "type": "text",
                        "text": f"{inp['question']}"
                    },
                ]
            }]]
        image_inputs, _ = process_vision_info(messages)

        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True) +
            "<|endoftext|>" for msg in messages
        ]
        inputs = self.processor(text=texts,
                                images=image_inputs,
                                padding="longest",
                                return_tensors="pt").to(self.device)
        inputs = self.model.prepare_inputs_for_generation(**inputs,
                                                          use_cache=False)

        with torch.no_grad():
            output = self.model(**inputs,
                                return_dict=True,
                                output_hidden_states=True)

        embeddings = self._get_embedding(output.hidden_states[-1], embed_dim)
        return embeddings


def _run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    input_texts: List[str],
    input_images: PromptImageInput,
    model: str,
    *,
    dtype: str,
) -> None:
    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).
    processor = AutoProcessor.from_pretrained(MODELS[0])
    with vllm_runner(model,
                     task="embedding",
                     dtype=dtype,
                     enforce_eager=True,
                     max_model_len=8192) as vllm_model:
        texts = [
            processor.apply_chat_template([{
                "role":
                "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.new("RGB", (28, 28)),
                        "resized_height": 1,
                        "resized_width": 1
                    },
                    {
                        "type": "text",
                        "text": text
                    },
                ]
            }],
                                          tokenize=False,
                                          add_generation_prompt=True) +
            "<|endoftext|>" for text in input_texts
        ]
        vllm_outputs = vllm_model.encode(texts, images=input_images)

    hf_model = QwenVLEncoder()
    hf_outputs = []
    for text, image in zip(input_texts, input_images):
        if text.startswith("Query:"):
            inp = {"dtype": "text", "question": text}
        else:
            inp = {"dtype": "image", "image": image}
        hf_outputs.append(hf_model.embed(inp).tolist())
    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_models_text(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    input_texts_images = [(text, image_placeholder)
                          for text, image_placeholder in HF_TEXT_PROMPTS]
    input_texts = [text for text, _ in input_texts_images]
    input_images = [image for _, image in input_texts_images]

    _run_test(
        hf_runner,
        vllm_runner,
        input_texts,
        input_images,  # type: ignore
        model,
        dtype=dtype,
    )


@large_gpu_test(min_gb=48)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_models_image(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    input_texts_images = [
        (text, asset.pil_image)
        for text, asset in zip(HF_IMAGE_PROMPTS, image_assets)
    ]
    input_texts = [text for text, _ in input_texts_images]
    input_images = [image for _, image in input_texts_images]

    _run_test(
        hf_runner,
        vllm_runner,
        input_texts,
        input_images,
        model,
        dtype=dtype,
    )
