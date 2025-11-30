# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from OpenVLA/Prismatic processor structure

from typing import Any, ClassVar, List, Optional, Union

import torch
from PIL import Image
from transformers import (
    BatchFeature,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TensorType,
)
from transformers.processing_utils import PaddingStrategy, TruncationStrategy

# Try to import PrismaticProcessor from the model folder
# This will be used when loading the model
try:
    import sys
    import os
    # Add the model directory to path if needed
    # The processor will be loaded dynamically from the model folder
    pass
except ImportError:
    pass


class OpenVLAProcessor(ProcessorMixin):
    """
    Processor for OpenVLA model that wraps PrismaticProcessor.
    This is a minimal wrapper that vLLM can use to interface with the model's processor.
    """
    
    attributes: ClassVar[List[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        image_processor: Optional[Any] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(image_processor, tokenizer)
        self.image_token = "<image>"
        # Get image_token_id from tokenizer if available
        if tokenizer is not None:
            self.image_token_id = tokenizer.vocab.get(self.image_token)
            if self.image_token_id is None:
                # Try to add it
                try:
                    tokenizer.add_special_tokens({"additional_special_tokens": [self.image_token]})
                    self.image_token_id = tokenizer.vocab.get(self.image_token)
                except Exception:
                    pass

    def __call__(
        self,
        text: Union[str, List[str]],
        images: Union[Image.Image, List[Image.Image]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        **kwargs: Any,
    ) -> BatchFeature:
        """
        Preprocess text and images for OpenVLA model.
        
        Args:
            text: Text input(s) to encode
            images: Image(s) to preprocess
            padding: Padding strategy
            truncation: Truncation strategy
            max_length: Maximum sequence length
            return_tensors: Return tensor type
            
        Returns:
            BatchFeature with input_ids, attention_mask, and pixel_values
        """
        # Process images
        if self.image_processor is not None:
            pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        else:
            raise ValueError("image_processor is required")
        
        # Process text
        if self.tokenizer is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                **kwargs,
            )
        else:
            raise ValueError("tokenizer is required")
        
        # Validate batch sizes match
        if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            raise ValueError(
                "Batch is malformed; expected same number of images and text inputs!"
            )
        
        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], torch.Tensor, Any],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Decode token sequences to text."""
        if self.tokenizer is None:
            raise ValueError("tokenizer is required")
        return self.tokenizer.batch_decode(
            sequences=sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor, Any],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: Any,
    ) -> str:
        """Decode token sequence to text."""
        if self.tokenizer is None:
            raise ValueError("tokenizer is required")
        return self.tokenizer.decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self) -> List[str]:
        """Return model input names."""
        tokenizer_input_names = (
            self.tokenizer.model_input_names if self.tokenizer else []
        )
        image_processor_input_names = (
            self.image_processor.model_input_names if self.image_processor else []
        )
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))




