# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, TypedDict, cast

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torchvision.transforms.v2.functional import InterpolationMode
from transformers import (
    AddedToken,
    AutoProcessor,
    BatchFeature,
    # CLIPImageProcessorFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    ProcessorMixin,
)

ImageData: TypeAlias = (
    list[Image.Image | np.ndarray | torch.Tensor]
    | np.ndarray
    | torch.Tensor
    | Image.Image
)


class ChatTemplate(TypedDict):
    type: str
    content: list[dict[str, Any]]


class SizedDict(TypedDict):
    size: int | tuple[int, int] | None
    max_size: int | None


def compose(
    dtype: torch.dtype,
    size: int | tuple[int, int] | None = None,
    max_size: int | None = None,
    crop_size: int | tuple[int, int] | None = None,
):
    assert size is not None or max_size is not None

    if crop_size is None:
        crop_size = max_size if max_size is not None else size

    compose = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.uint8, scale=True),
            transforms.Resize(
                size=size, max_size=max_size, interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToDtype(dtype=dtype, scale=True),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
            transforms.CenterCrop(crop_size),
            transforms.ToPureTensor(),
        ]
    )

    return compose


class CogAgentProcessor(ProcessorMixin):
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    image_processor_class = "AutoImageProcessor"
    attributes = ["tokenizer"]
    valid_kwargs = [
        "cross_image_size",
        "cross_crop_size",
        "crop_size",
        "dtype",
        "image_size",
        "patch_size",
        "aspect_ratio",
        "template_version",
        "image_token",
    ]

    def __init__(
        self,
        tokenizer: LlamaTokenizer | LlamaTokenizerFast = None,
        cross_image_size: int = 1120,
        cross_crop_size: int | tuple[int, int] | None = None,
        crop_size: int | tuple[int, int] | None = None,
        dtype: torch.dtype = torch.bfloat16,
        image_size: int = 224,
        patch_size: int = 14,
        aspect_ratio: str = "square",
        template_version: Literal["base", "chat", "chat_old", "vqa"] = "chat_old",
        image_token: str = "<EOI>",
    ):
        """
        tokenizer: LlamaTokenizer.

        template_version: str. base, chat, chat_old, or vqa. Default chat_old.
            Original version uses chat_old. chat uses different BOS EOS.
            vqa is short answer only. History is not supported with vqa.

        patch_size: int. Default is 14.
            patch size for both image encoder and cross encoder.

        image_size: int. Default is 224.
            image size for the image encoder (EVACLIP).

        cross_image_size: int. Default is 1120.
            cross image size for the cross encoder (EVALarge).
            cross_image_size should always be larger than the image_size.

        crop_size: int. Defaults to image_size.
            crops the image to a square equal to this value. If greater, pads the image.

        cross_crop_size: int. Defaults to cross_image_size.
            crops the cross encoder image to a square equal to this value.
            If greater pads the image.

        """

        self.patch_size = patch_size
        self.template_version = template_version
        self.tokenizer = tokenizer

        self.cross_image_size = self.get_size_dict(cross_image_size, aspect_ratio)
        self.image_size = self.get_size_dict(image_size, aspect_ratio)

        self.crop_size = crop_size
        self.cross_crop_size = cross_crop_size

        self.image_processor = compose(
            dtype=dtype,
            crop_size=self.crop_size,
            size=self.image_size["size"],
            max_size=self.image_size["max_size"],
        )
        self.cross_image_processor = compose(
            dtype=dtype,
            crop_size=self.cross_crop_size,
            size=self.cross_image_size["size"],
            max_size=self.cross_image_size["max_size"],
        )

        # add image token
        image_token_id = self.tokenizer.vocab.get(image_token)
        if image_token_id is None:
            special_tokens = [AddedToken(image_token, single_word=True, special=True)]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        self.image_token_id = self.tokenizer.vocab.get(image_token)
        self.image_token = image_token

        super().__init__(tokenizer)

    @staticmethod
    def get_size_dict(image_size: int, aspect_ratio: str = "square"):
        if aspect_ratio == "min":
            size = SizedDict(size=image_size, max_size=None)
        elif aspect_ratio == "max":
            size = SizedDict(size=None, max_size=image_size)
        elif aspect_ratio == "square":
            size = SizedDict(size=(image_size, image_size), max_size=None)
        else:
            raise NotImplementedError(
                f"Resize format {aspect_ratio} is not Implemented"
            )

        return size

    def default_template(
        self, text: str | ChatTemplate | list[str] | list[ChatTemplate]
    ) -> str:
        if isinstance(text, dict):
            text = [text]

        if isinstance(text, list):
            if isinstance(text[0], dict):
                text = cast(list[ChatTemplate], text)
                text = [
                    line["text"]
                    for lines in text
                    for line in lines["content"]
                    if line.get("type", "") == "text"
                ]

            text = cast(list[str], text)
            text = " ".join(text)

        if self.template_version == "vqa":
            text = "Question: " + text + " Short answer: "
        elif self.template_version == "chat_old":
            text = "Question: " + text + " Answer: "
        elif self.template_version == "chat":
            text = " [INST] " + text + " [/INST] "

        return text

    def applied_template(self, prompt: str) -> bool:
        """check the string prompt for the template based on version"""

        if prompt.startswith("Question:") or prompt.startswith(
            "Question:", self.num_image_tokens + 2
        ):
            end = "answer: "
            return prompt[-len(end) :].lower() == end

        return prompt.startswith(" [INST] ") and prompt.endswith(" [/INST] ")

    @property
    def num_image_tokens(self):
        crop_size = self.crop_size
        size = (0, 0)

        if crop_size is not None:
            if not isinstance(crop_size, Sequence):
                image_size = cast(int, crop_size)
                size = (image_size, image_size)
        else:
            size_param = (
                "max_size" if self.image_size["max_size"] is not None else "size"
            )
            image_size = self.image_size[size_param]
            if not isinstance(image_size, tuple):
                image_size = cast(int, image_size)
                size = (image_size, image_size)

        num_image_tokens = (size[0] / self.patch_size) * (size[1] / self.patch_size) + 2

        return int(round(num_image_tokens))

    def _process_token_ids(
        self,
        token_ids: list[int] | list[list[int]],
        has_image: Sequence[bool] = (True,),
        **kwargs,
    ) -> torch.Tensor:
        return_tensors = kwargs.get("return_tensors", "pt")
        bos_token: int = self.tokenizer.bos_token_id
        image_pretoken_sequence: list[int] = [
            self.tokenizer.vocab[self.image_token]
        ] * self.num_image_tokens

        if isinstance(token_ids[0], int):
            flat = cast(list[int], token_ids)
            token_ids = [flat]

        max_len = 0
        output_ids = []
        token_ids = cast(list[list[int]], token_ids)
        for with_image, tokens in zip(has_image, token_ids):
            num_image_tokens = sum(
                token == sequence_token
                for token, sequence_token in zip(tokens[1:], image_pretoken_sequence)
            )

            if with_image and num_image_tokens == 0:
                tokens = image_pretoken_sequence + tokens
            if tokens[0] != bos_token:
                tokens = [bos_token] + tokens
            if num_image_tokens != (self.num_image_tokens * int(with_image)):
                raise RuntimeError(
                    f"Wrong number of image tokens in {tokens}. "
                    f"Found {num_image_tokens}"
                )

            output_ids.append(torch.tensor(tokens, dtype=torch.long))
            max_len = max(max_len, len(tokens))

        if return_tensors is None:
            return {"input_ids": output_ids}

        input_ids = torch.full(
            (len(output_ids), max_len), fill_value=self.tokenizer.pad_token_id
        )
        for index, output in enumerate(output_ids):
            input_ids[index, (-1 * output.shape[0]) :] = output
        if return_tensors == "np":
            input_ids = input_ids.numpy()

        return {"input_ids": input_ids}

    def _process_text(
        self, text: str | list[str], has_image: Sequence[bool] = (True,), **kwargs
    ) -> dict[Literal["input_ids"], torch.Tensor | np.ndarray]:
        return_tensors = kwargs.pop("return_tensors", "pt")
        image_token_sequence: str = self.image_token * self.num_image_tokens

        if isinstance(text, str):
            text = [text]

        for text_index, (text_ids, with_image) in enumerate(zip(text, has_image)):
            if (image_token_sequence not in text_ids) and with_image:
                text_ids = image_token_sequence + text_ids

            text[text_index] = text_ids

        return_ids = self.tokenizer(
            text=text,
            return_tensors=return_tensors,
            padding=kwargs.pop("padding", "longest"),
            padding_side=kwargs.pop("padding_side", "left"),
            **kwargs,
        )

        return {"input_ids": return_ids["input_ids"]}

    def _process_images(
        self, images: ImageData
    ) -> dict[Literal["pixel_values", "cross_pixel_values"], torch.Tensor]:
        """
        Resizes images into two different sizes, then batches them into torch tensors.
        Args:
            Images: Sequence of PIL images, torch tensors, or np arrays.
        Returns:
            dict(
                images = torch.Tensor[batch, 3, size, size],
                cross images = torch.Tensor[batch, 3, cross_size, cross_size]
            )
        """

        if isinstance(images, Image.Image):
            images = [images]
        elif isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 3:
            images = images[None, :, :, :]

        resized_images = self.image_processor(images)
        cross_images = self.cross_image_processor(images)

        if not isinstance(resized_images, torch.Tensor):
            resized_images = torch.stack(resized_images, 0)
        if not isinstance(cross_images, torch.Tensor):
            cross_images = torch.stack(cross_images, 0)

        images = {"pixel_values": resized_images, "cross_pixel_values": cross_images}

        return images

    def __call__(
        self,
        text: str
        | ChatTemplate
        | list[int]
        | list[ChatTemplate]  # single with history
        | list[str]  # batched input
        | list[list[ChatTemplate]]  # batched (with history)
        | list[list[int]],  # batched tokens
        images: ImageData | None = None,
        **kwargs,
    ) -> BatchFeature:
        inputs = {}
        return_tensors = kwargs.get("return_tensors", "pt")
        if isinstance(text, str):
            text = [text]
        if isinstance(text, dict):
            text = [[text]]

        assert all(type(query) == type(text[0]) for query in text[1:]), (  # noqa: E721
            f"{text}: Must be uniform type"
        )

        # Verify if image input is correct format.
        # has_image allows for embedding input.

        has_image = [images is not None]
        if images is not None:
            num_images = len(images)
            num_queries = len(text)
            if isinstance(images, (torch.Tensor, np.ndarray)) and images.ndim <= 3:
                num_images = 1

            if num_images == num_queries:
                if isinstance(text[0], list) and isinstance(text[0][0], dict):
                    text = cast(list[list[ChatTemplate]], text)
                    total_num_images = [
                        sum("image" in content for content in sample["content"])
                        for samples in text
                        for sample in samples
                    ]
                    if max(total_num_images) > 1:
                        raise RuntimeError(
                            "CogAgent does not Support Multi-Image Queries. "
                            "In chat mode, pass the same image only once."
                        )
                    has_image = [num_images > 0 for num_images in total_num_images]
            else:
                raise RuntimeError(
                    "Number images of images does not match text queries."
                )

            # preprocess single or batched non tokenized inputs
            if isinstance(text[0], str) or (
                isinstance(text[0], list) and isinstance(text[0][0], dict)
            ):
                text = cast(list[str] | list[list[ChatTemplate]], text)
                if "chat_template" in kwargs or self.chat_template is not None:
                    text = self.apply_chat_template(
                        text, chat_template=kwargs.get("chat_template"), tokenize=False
                    )
                else:
                    results: list[str] = list()
                    for sample in text:
                        if isinstance(sample, str) and self.applied_template(sample):
                            pass
                        else:
                            results.append(self.default_template(sample).strip())

                    text = cast(list[str], results)

            processed_images = self._process_images(images)
            inputs.update(processed_images)

        text = cast(list[int] | list[list[int]] | list[str] | str, text)
        if isinstance(text[0], list) and isinstance(text[0][0], int):
            text = cast(list[int] | list[list[int]], text)
            processed_text = self._process_token_ids(text, has_image, **kwargs)
        else:
            text = cast(str | list[str], text)
            processed_text = self._process_text(text, has_image, **kwargs)

        inputs.update(processed_text)

        batch_feature = BatchFeature(inputs, tensor_type=return_tensors)
        return batch_feature


AutoProcessor.register("CogAgentProcessor", CogAgentProcessor)
