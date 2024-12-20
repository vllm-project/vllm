from collections.abc import Iterable
from typing import Any, TypeVar

import torch
import torch.nn as nn

from .interfaces_base import VllmModelForPooling, is_pooling_model

_T = TypeVar("_T", bound=type[nn.Module])


def as_embedding_model(cls: _T) -> _T:
    """Subclass an existing vLLM model to support embeddings."""
    # Avoid modifying existing embedding models
    if is_pooling_model(cls):
        return cls

    # Lazy import
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.pooler import (Pooler, PoolerOutput,
                                                   PoolingType)
    from vllm.model_executor.pooling_metadata import PoolingMetadata

    from .utils import AutoWeightsLoader, WeightsMapper

    class ModelForEmbedding(cls, VllmModelForPooling):

        def __init__(
            self,
            *,
            vllm_config: "VllmConfig",
            prefix: str = "",
            **kwargs: Any,
        ) -> None:
            super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)

            # These are not used in embedding models
            for attr in ("lm_head", "logits_processor"):
                if hasattr(self, attr):
                    delattr(self, attr)

            pooler_config = vllm_config.model_config.pooler_config
            assert pooler_config is not None

            # If the model already defines a pooler instance, don't overwrite it
            if not getattr(self, "_pooler", None):
                self._pooler = Pooler.from_config_with_defaults(
                    pooler_config,
                    pooling_type=PoolingType.LAST,
                    normalize=True,
                    softmax=False,
                )

        def pooler(
            self,
            hidden_states: torch.Tensor,
            pooling_metadata: PoolingMetadata,
        ) -> PoolerOutput:
            return self._pooler(hidden_states, pooling_metadata)

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
            # TODO: Support uninitialized params tracking

            # We have deleted this attribute, so don't load it
            weights = ((name, data) for name, data in weights
                       if not name.startswith("lm_head."))

            # If `*ForCausalLM` defines `load_weights` on the inner model
            # and there are no other inner modules with parameters,
            # we support loading from both `*Model` and `*ForCausalLM`
            if hasattr(self, "model") and hasattr(self.model, "load_weights"):
                # Whether only `self.model` contains parameters
                model_is_only_param = all(
                    name == "model" or next(child.parameters(), None) is None
                    for name, child in self.named_children())

                if model_is_only_param:
                    mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})
                    weights = mapper.apply(weights)

                    self.model.load_weights(weights)
                    return

            # For most other models
            if hasattr(cls, "load_weights"):
                cls.load_weights(self, weights)  # type: ignore
            # Fallback
            else:
                loader = AutoWeightsLoader(self)
                loader.load_weights(weights)

    ModelForEmbedding.__name__ = cls.__name__ \
        .removesuffix("ForCausalLM") \
        .removesuffix("ForConditionalGeneration") \
        .removesuffix("ChatModel") \
        .removesuffix("LMHeadModel") + "ForEmbedding"

    return ModelForEmbedding  # type: ignore
