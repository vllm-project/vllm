# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Predictable hybrid (Mamba + attention) dummy model for testing
extract_hidden_states with hybrid models.

The key difference from PredictableLlamaForCausalLM is:
  - is_hybrid = True triggers the HMA (Hybrid Memory Allocator) code path
  - FakeMambaLayer registers a MambaSpec in the KV cache, so the model's
    KV cache setup must unify MambaSpec + MLAAttentionSpec page sizes
  - ExampleHiddenStatesConnector.SupportsHMA is exercised via
    request_finished_all_groups() instead of request_finished()
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.models.interfaces import EagleModelMixin, IsHybrid
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


class FakeMambaLayer(MambaBase):
    """Minimal fake Mamba layer for testing hybrid model KV cache setup.

    - Registers in static_forward_context so vLLM discovers its MambaSpec.
    - forward() is a no-op (returns input unchanged); there is no real SSM math.
    - State shapes are chosen so mamba_page_size (1024 B) divides evenly into
      the verifier model's attention page size (= num_kv_heads=4 * head_size=64
      * 2 bytes * 2 [K+V] = 1024 B/token, or 16384 B/block at block_size=16),
      keeping the CacheOnlyAttentionLayer page size consistent.
    """

    # conv_state: (32, 3) -> 96 float32 = 384 bytes
    # ssm_state:  (4, 4, 10) -> 160 float32 = 640 bytes
    # total mamba page_size = 1024 bytes
    CONV_STATE_SHAPE: tuple[int, int] = (32, 3)
    SSM_STATE_SHAPE: tuple[int, int, int] = (4, 4, 10)

    def __init__(self, *, prefix: str, vllm_config: VllmConfig):
        super().__init__()
        self.prefix = prefix
        # kv_cache is set by vLLM's KV cache manager after allocation.
        self.kv_cache: tuple[torch.Tensor, ...] = (
            torch.tensor([]),
            torch.tensor([]),
        )
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate FakeMambaLayer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    # --- MambaBase interface -------------------------------------------------

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.MAMBA2

    def get_state_shape(self) -> list[tuple[int, ...]]:
        return [self.CONV_STATE_SHAPE, self.SSM_STATE_SHAPE]

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (torch.float32, torch.float32)

    # --- forward (no-op) -----------------------------------------------------

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Pass through without any Mamba computation (testing only)."""
        return hidden_states


class PredictableHybridModel(nn.Module, EagleModelMixin):
    """Hybrid model that returns predictable hidden states.

    Has a FakeMambaLayer (for MambaSpec) plus the standard embed_tokens.
    forward() bypasses real Mamba computation and returns constant tensors.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config

        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
        )

        # One fake Mamba layer causes vLLM to treat this as a hybrid model
        # (MambaSpec appears in kv_cache_spec alongside CacheOnlyAttentionLayer
        # MLAAttentionSpec from the draft ExtractHiddenStatesModel).
        # The prefix must contain exactly one integer (vLLM's extract_layer_index
        # requirement), e.g. "model.layers.0.mamba_mixer".
        layer_prefix = (
            f"{prefix}.layers.0.mamba_mixer" if prefix else "layers.0.mamba_mixer"
        )
        self.mamba_layer = FakeMambaLayer(
            prefix=layer_prefix,
            vllm_config=vllm_config,
        )

        from vllm.model_executor.models.utils import (
            make_empty_intermediate_tensors_factory,
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **extra_layer_kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Return predictable hidden states (same pattern as PredictableLlamaModel).

        Layer i returns fill_value=float(i) for auxiliary states.
        Does NOT call mamba_layer.forward(); the Mamba KV cache is allocated
        but not populated, which is acceptable for this integration test.
        """
        if inputs_embeds is not None:
            seq_len = inputs_embeds.shape[0]
            device = inputs_embeds.device
        elif input_ids is not None:
            seq_len = input_ids.shape[0] if input_ids.ndim == 1 else input_ids.shape[-1]
            device = input_ids.device
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        hidden_states = torch.full(
            (seq_len, self.config.hidden_size),
            fill_value=float(self.config.num_hidden_layers),
            device=device,
            dtype=torch.bfloat16,
        )

        if len(self.aux_hidden_state_layers) > 0:
            aux_hidden_states = [
                torch.full(
                    (seq_len, self.config.hidden_size),
                    fill_value=float(layer_idx),
                    device=device,
                    dtype=torch.bfloat16,
                )
                for layer_idx in self.aux_hidden_state_layers
            ]
            return hidden_states, aux_hidden_states

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return set()


class PredictableHybridForCausalLM(LlamaForCausalLM, IsHybrid):
    """Hybrid LM wrapper for testing extract_hidden_states with hybrid models.

    is_hybrid = True causes vLLM to:
      1. Call HybridAttentionMambaModelConfig.verify_and_update_config()
       to set cache_config.mamba_block_size
      2. Call Platform._align_hybrid_block_size()
       to set cache_config.mamba_page_size_padded
      3. Enable HMA (Hybrid Memory Allocator) when the KV connector subclasses
         SupportsHMA (ExampleHiddenStatesConnector does after our fix)
      4. Route through connector.request_finished_all_groups() instead of
       connector.request_finished(), which is the key behavior tested here
    """

    is_hybrid = True

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        return (FakeMambaLayer.CONV_STATE_SHAPE, FakeMambaLayer.SSM_STATE_SHAPE)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return (torch.float32, torch.float32)

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple:
        """Not needed for non-prefix-caching / non-align mode."""
        return ()

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] | None = None,
    ):
        return PredictableHybridModel(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return set()
