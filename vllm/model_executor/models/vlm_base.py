import torch
from torch import nn

from vllm.config import VisionLanguageConfig
from vllm.multimodal import BatchedTensors


class VisionLanguageModelBase(nn.Module):
    """Base class for all vision language models (VLMs)."""

    def __init__(self, vision_language_config: VisionLanguageConfig) -> None:
        super().__init__()

        self.vision_language_config = vision_language_config

    @classmethod
    def merge_vision_embeddings(cls, input_ids: torch.Tensor,
                                inputs_embeds: torch.Tensor,
                                vision_embeddings: BatchedTensors,
                                image_token_id: int) -> torch.Tensor:
        """In place merges in vision_embeddings with inputs_embeds."""
        mask = (input_ids == image_token_id)
        num_expected_tokens = mask.sum()

        if isinstance(vision_embeddings, torch.Tensor):
            batch_size, batch_tokens, *_, embed_dim = vision_embeddings.shape
            total_tokens = batch_size * batch_tokens
            if num_expected_tokens != total_tokens:
                expr = f"{batch_size} x {batch_tokens}"
                raise ValueError(
                    f"Attempted to assign {expr} = {total_tokens} "
                    f"image tokens to {num_expected_tokens} placeholders")

            inputs_embeds[mask] = vision_embeddings.view(
                total_tokens, embed_dim)
        else:
            size_per_batch = [t.shape[0] for t in vision_embeddings]
            total_tokens = sum(size_per_batch)
            if num_expected_tokens != total_tokens:
                expr = ' + '.join(map(str, size_per_batch))
                raise ValueError(
                    f"Attempted to assign {expr} = {total_tokens} "
                    f"image tokens to {num_expected_tokens} placeholders")

            inputs_embeds[mask] = torch.cat(vision_embeddings)

        return inputs_embeds
