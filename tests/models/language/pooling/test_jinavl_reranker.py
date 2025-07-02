# SPDX-License-Identifier: Apache-2.0
import logging
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from transformers import Qwen2VLForConditionalGeneration
from transformers.image_utils import load_image

logger = logging.getLogger(__name__)

model_name = "jinaai/jina-reranker-m0"

mm_processor_kwargs = {
    "min_pixels": 3136,
    "max_pixels": 602112,
}

limit_mm_per_prompt = {"image": 4}


def load_images(images, lazy_load: bool = True):
    # Disable PIL DecompositionBomb threshold for reading large images.
    pil_max_px = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None

    images_batch = []
    for image in images:
        if isinstance(image, Image.Image):
            images_batch.append(image)
        else:
            pil_image = load_image(image)
            if lazy_load:
                images_batch.append(pil_image)
            else:
                # avoid Too many open files error
                images_batch.append(pil_image.copy())
                pil_image.close()
    Image.MAX_IMAGE_PIXELS = pil_max_px

    return images_batch


def formatting_prompts_func(
    query: str,
    doc: str,
    query_type: str = 'text',
    doc_type: str = 'text',
    prefix_str: str = '',
) -> str:
    """
    Format prompts for different combinations of query and content types.

    Args:
        query: Query text or image path
        doc: Content text or image path
        query_type: Whether query is an image
        doc_type: Whether content is an image
        prefix_str: Optional prefix string to add
    """
    # Format query part
    if query_type == 'image':
        query_part = "**Query**:\n<|vision_start|><|image_pad|><|vision_end|>"
    else:
        query_part = f"**Query**:\n{query}"

    # Format content part
    if doc_type == 'image':
        doc_part = "**Document**:\n<|vision_start|><|image_pad|><|vision_end|>"
    else:
        doc_part = f"**Document**:\n{doc}"

    # Combine parts
    prompt = doc_part + '\n' + query_part

    # Add prefix if provided
    if prefix_str:
        prompt = prefix_str + '\n' + prompt

    return prompt


# refer to https://huggingface.co/jinaai/jina-reranker-m0/blob/main/modeling.py
class JinaVLForRanking(Qwen2VLForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)

        self.padding_side = "left"
        self.num_labels = 1  # config.num_labels
        self.LOGIT_BIAS = 2.65  # logit bias for sigmoid normalization

        # hack the lm_head to do nothing, since we only want the hidden states
        self.lm_head = nn.Identity()

        # copy the idea from `Qwen2ForRewardModel` to have a MLP layer to get the final score
        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.num_labels),
        )

        # Initialize weights and apply final processing
        self.post_init()

        self.score_token_id = 100

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        assert kwargs.pop(
            "labels", None) is None, "labels should not be passed to forward()"

        outputs = super().forward(
            *args,
            use_cache=False,
            output_hidden_states=True,
            **kwargs,
        )

        # get the hidden states of the last layer
        hidden_states = outputs.hidden_states[-1]

        # IMPORTANT: the padding token must be on the left side
        # get the hidden states of the last token and apply the linear layer
        pooled_logits = self.score(hidden_states[:, -1])

        return pooled_logits.squeeze(-1)

    @torch.no_grad()
    def compute_score(
        self,
        pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 8,
        max_length: int = 10240,
        max_query_length: int = 512,
        max_doc_length: Optional[int] = None,
        query_type: str = 'text',
        doc_type: str = 'text',
        normalize_scores: bool = True,
        show_progress: bool = False,
    ) -> List[float]:

        if not hasattr(self, "_processor"):
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.name_or_path,
                max_pixels=602112,
                min_pixels=3136,
                trust_remote_code=True)

        assert isinstance(pairs, list)

        if isinstance(pairs[0], str):
            pairs = [pairs]

        max_length = max_length or self.config.max_length

        if max_doc_length is None:
            max_doc_length = max(max_length - max_query_length,
                                 max_query_length)

        if max_doc_length < max_query_length:
            warnings.warn(
                f"max_doc_length={max_doc_length} should be greater than max_query_length={max_query_length}"
            )

        assert (
            max_doc_length + max_query_length <= max_length
        ), f"max_doc_length ({max_doc_length}) + max_query_length ({max_query_length}) should be less than max_length ({max_length})"

        max_length = max_length - 1

        all_scores = []

        device = next(self.parameters()).device

        batch_iter = range(0, len(pairs), batch_size)
        if show_progress:
            from tqdm import trange

            batch_iter = trange(0,
                                len(pairs),
                                batch_size,
                                desc="Computing scores")

        for start_index in batch_iter:
            mini_batch = pairs[start_index:start_index + batch_size]

            batch_inputs = []
            for q, d in mini_batch:
                # TEMP FIX: Truncate long documents
                if doc_type == 'text':
                    tokens = self._processor.tokenizer(
                        d, truncation=True, max_length=max_doc_length)
                    if len(tokens['input_ids']) >= max_doc_length:
                        d = self._processor.tokenizer.decode(
                            tokens['input_ids'])

                batch_inputs.append(
                    formatting_prompts_func(q,
                                            d,
                                            query_type=query_type,
                                            doc_type=doc_type))

            batch_images = None

            doc_images = []
            query_images = []
            if doc_type == 'image':
                doc_images = load_images([d for (q, d) in mini_batch])
            if query_type == 'image':
                query_images = load_images([q for (q, d) in mini_batch])

            if len(doc_images) == len(query_images) and len(doc_images) > 0:
                batch_images = [[d, q]
                                for q, d in zip(query_images, doc_images)]
            elif len(doc_images) > 0:
                batch_images = doc_images
            elif len(query_images) > 0:
                batch_images = query_images

            batch = self._processor(
                text=batch_inputs,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            # append the reward token to the input_ids and attention_mask
            batch_size = batch["input_ids"].size(0)
            batch["input_ids"] = torch.cat(
                [
                    batch["input_ids"],
                    torch.full((batch_size, 1),
                               self.score_token_id,
                               device=batch["input_ids"].device),
                ],
                dim=1,
            )
            batch["attention_mask"] = torch.cat(
                [
                    batch["attention_mask"],
                    torch.ones((batch_size, 1),
                               device=batch["attention_mask"].device),
                ],
                dim=1,
            )
            # move the batch to the correct device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            scores = self.forward(**batch).view(-1).cpu().float().numpy()

            # normalize scores to [0, 1] with sigmoid with a scale
            scores = 1.0 / (1.0 + np.exp(-(scores - self.LOGIT_BIAS)))

            all_scores.extend(scores.tolist())

        if len(all_scores) == 1:
            return all_scores[0]
        return all_scores


def vllm_reranker(model_name,
                  query,
                  documents,
                  query_type="text",
                  doc_type="text"):
    from vllm import LLM

    model = LLM(
        model=model_name,
        task="score",
        dtype="float16",
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )

    def create_image_param(url: str):
        return {"type": "image_url", "image_url": {"url": f"{url}"}}

    if query_type == "image":
        query = {"content": [create_image_param(url) for url in query]}

    if doc_type == "image":
        documents = {"content": [create_image_param(url) for url in documents]}

    print(query)
    print(documents)
    outputs = model.score(query, documents)

    return [output.outputs.score for output in outputs]


def hf_reranker(model_name,
                query,
                documents,
                query_type="text",
                doc_type="text"):

    checkpoint_to_hf_mapper = {
        "visual.": "model.visual.",
        "model.": "model.language_model.",
    }

    model = JinaVLForRanking.from_pretrained(
        model_name, key_mapping=checkpoint_to_hf_mapper).eval()

    data_pairs = [[query[0], d] for d in documents]

    scores = model.compute_score(data_pairs,
                                 max_length=2048,
                                 query_type=query_type,
                                 doc_type=doc_type)
    return scores


# Visual Documents Reranking
@pytest.mark.parametrize("model_name", [model_name])
def test_model_text_image(model_name):

    query = ["slm markdown"]
    documents = [
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png",
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png",
    ]

    hf_outputs = hf_reranker(model_name, query, documents, "text", "image")
    vllm_outputs = vllm_reranker(model_name, query, documents, "text", "image")

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


# Textual Documents Reranking
@pytest.mark.parametrize("model_name", [model_name])
def test_model_text_text(model_name):

    query = ["slm markdown"]
    documents = [
        "We present ReaderLM-v2, a compact 1.5 billion parameter language model designed for efficient web content extraction. Our model processes documents up to 512K tokens, transforming messy HTML into clean Markdown or JSON formats with high accuracy -- making it an ideal tool for grounding large language models. The models effectiveness results from two key innovations: (1) a three-stage data synthesis pipeline that generates high quality, diverse training data by iteratively drafting, refining, and critiquing web content extraction; and (2) a unified training framework combining continuous pre-training with multi-objective optimization. Intensive evaluation demonstrates that ReaderLM-v2 outperforms GPT-4o-2024-08-06 and other larger models by 15-20% on carefully curated benchmarks, particularly excelling at documents exceeding 100K tokens, while maintaining significantly lower computational requirements.",
        "数据提取么？为什么不用正则啊，你用正则不就全解决了么？",
    ]

    hf_outputs = hf_reranker(model_name, query, documents, "text", "text")
    vllm_outputs = vllm_reranker(model_name, query, documents, "text", "text")

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


# Image Querying for Textual Documents
@pytest.mark.parametrize("model_name", [model_name])
def test_model_image_text(model_name):

    query = [
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png"
    ]
    documents = [
        "We present ReaderLM-v2, a compact 1.5 billion parameter language model designed for efficient web content extraction. Our model processes documents up to 512K tokens, transforming messy HTML into clean Markdown or JSON formats with high accuracy -- making it an ideal tool for grounding large language models. The models effectiveness results from two key innovations: (1) a three-stage data synthesis pipeline that generates high quality, diverse training data by iteratively drafting, refining, and critiquing web content extraction; and (2) a unified training framework combining continuous pre-training with multi-objective optimization. Intensive evaluation demonstrates that ReaderLM-v2 outperforms GPT-4o-2024-08-06 and other larger models by 15-20% on carefully curated benchmarks, particularly excelling at documents exceeding 100K tokens, while maintaining significantly lower computational requirements.",
        "数据提取么？为什么不用正则啊，你用正则不就全解决了么？",
    ]

    hf_outputs = hf_reranker(model_name, query, documents, "image", "text")
    vllm_outputs = vllm_reranker(model_name, query, documents, "image", "text")

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


# Image Querying for Image Documents
@pytest.mark.parametrize("model_name", [model_name])
def test_model_image_image(model_name):

    query = [
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png"
    ]
    documents = [
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png",
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png",
    ]

    hf_outputs = hf_reranker(model_name, query, documents, "image", "image")
    vllm_outputs = vllm_reranker(model_name, query, documents, "image",
                                 "image")

    assert hf_outputs == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)
