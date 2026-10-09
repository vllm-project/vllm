# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-resident Qwen4Exp position-learning enhancement layers."""

from collections.abc import Iterable, Sequence

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
)

from ..common.ngram_embedding import (
    Qwen4ExpPLEDeviceEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
)
from ..common.ops.ple import ple_conv, ple_gate, ple_ngram_ids

logger = init_logger(__name__)


class Qwen4ExpPLEGroupedNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.eps)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (grouped * torch.rsqrt(variance + self.eps)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


class Qwen4ExpNGramEmbedding(nn.Module):
    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PLE_LAYER_PRIME = 10007

    @classmethod
    def _splitmix64(cls, value: int) -> int:
        """Mix an integer into a deterministic unsigned 64-bit value."""
        value = (value + cls._SPLITMIX_GAMMA) & cls._MASK64
        value = ((value ^ (value >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        value = ((value ^ (value >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (value ^ (value >> 31)) & cls._MASK64

    @staticmethod
    def _is_prime_64(value: int) -> bool:
        """Return whether a 64-bit integer is prime."""
        if value < 2:
            return False
        for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if value % prime == 0:
                return value == prime
        exponent = value - 1
        shifts = 0
        while exponent % 2 == 0:
            exponent //= 2
            shifts += 1
        for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
            if base % value == 0:
                continue
            witness = pow(base, exponent, value)
            if witness in (1, value - 1):
                continue
            for _ in range(shifts - 1):
                witness = pow(witness, 2, value)
                if witness == value - 1:
                    break
            else:
                return False
        return True

    @classmethod
    def _nth_prime_after(cls, start: int, count: int) -> int:
        """Return the ``count``-th prime strictly greater than ``start``."""
        prime = int(start)
        for _ in range(count):
            candidate = prime + 1
            if candidate <= 2:
                prime = 2
                continue
            if candidate % 2 == 0:
                candidate += 1
            while not cls._is_prime_64(candidate):
                candidate += 2
            prime = candidate
        return prime

    @classmethod
    def _make_layer_multipliers(
        cls,
        *,
        ngram_size: int,
        unigram_vocab_size: int,
        seed: int,
        ple_dense_layer_id: int,
    ) -> list[int]:
        """Build deterministic hash multipliers for one PLE layer."""
        max_multiplier = ((1 << 63) - 1) // unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = seed + cls._PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(ngram_size):
            value = base_seed + cls._SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (cls._splitmix64(value) % half_bound) + 1)
        return multipliers

    @classmethod
    def _make_vocab_layout(
        cls,
        *,
        ngram_vocab_size_base: int,
        ngram_heads: int,
        ple_dense_layer_id: int,
    ) -> tuple[list[int], list[int], int]:
        """Build per-head vocabulary sizes, offsets, and total row count."""
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(ngram_heads):
            global_head = ple_dense_layer_id * ngram_heads + local_head
            size = cls._nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        return sizes, offsets, offset

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        prefix: str,
        layer_name: str,
        *,
        data_parallel_rank: int = 0,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.layer_name = layer_name
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        multipliers = self._make_layer_multipliers(
            ngram_size=self.ngram_size,
            unigram_vocab_size=self.unigram_vocab_size,
            seed=int(getattr(config, "seed", 1234)),
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        sizes, offsets, total_vocab_size = self._make_vocab_layout(
            ngram_vocab_size_base=int(config.ngram_vocab_size_base),
            ngram_heads=self.ngram_heads,
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = ((total_vocab_size + divisor - 1) // divisor) * divisor
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        embedding_prefix = f"{prefix}.ngram_embedding"
        embedding_quant_method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(
            quant_config,
            embedding_prefix,
            getattr(config, "ple_embedding_dtype", None),
        )
        engram_config = get_current_vllm_config().engram_config
        embedding_cls = (
            Qwen4ExpPLEPinnedHostEmbedding
            if engram_config is not None and engram_config.cpu_offload
            else Qwen4ExpPLEDeviceEmbedding
        )
        self.ngram_embedding = embedding_cls(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=embedding_prefix,
            embedding_method=embedding_quant_method,
            num_ngram_heads=self.ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        logger.info(
            "Initialized AMD PLE embedding %s: quantization_method=%s, "
            "weight_dtype=%s, weight_device=%s, pinned=%s",
            embedding_prefix,
            type(embedding_quant_method).__name__,
            self.ngram_embedding.weight.dtype,
            self.ngram_embedding.weight.device,
            self.ngram_embedding.weight.is_pinned(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        ngram_ids = input_ids.new_empty(
            (input_ids.shape[0], self.ngram_heads), dtype=torch.int64
        )
        torch.ops.vllm.qwen4_exp_amd_ple_ngram_ids(
            input_ids,
            query_start_loc,
            ngram_context,
            ngram_ids,
            self.layer_name,
        )
        embedding = self.ngram_embedding
        if embedding.supports_prefetch:
            output = ngram_ids.new_empty(
                (ngram_ids.shape[0], self.embedding_dim),
                dtype=embedding.weight.dtype,
            )
            torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding_pinned(
                ngram_ids,
                output,
                self.layer_name,
            )
            return output
        output = ngram_ids.new_empty(
            (ngram_ids.shape[0], self.embedding_dim),
            dtype=embedding.params_dtype,
        )
        torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding(
            ngram_ids,
            output,
            self.layer_name,
        )
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""
        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if buffer.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
                loaded.add(name)
                continue
            if name.startswith(shard_prefix) and name.endswith(".weight"):
                shard_text = name[len(shard_prefix) : -len(".weight")]
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                embedding = self.ngram_embedding
                shard_size = (
                    embedding.org_vocab_size + self.split_ngram_parts - 1
                ) // self.split_ngram_parts
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, embedding.org_vocab_size - checkpoint_start),
                )
                expected_shape = (expected_rows, embedding.embedding_dim)
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_index}: "
                        f"expected {expected_shape}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                embedding.weight.weight_loader(
                    embedding.weight,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                )
                loaded.add("ngram_embedding.weight")
                continue
            regular_weights.append((name, loaded_weight))

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        return loaded


class Qwen4ExpPLELayer(nn.Module, MambaBase):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = (
            int(ple_dense_layer_id)
            if ple_dense_layer_id is not None
            else int(layer_idx)
        )
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.activation = "silu"
        self.ple_embedding: nn.Module = Qwen4ExpNGramEmbedding(
            config,
            int(config.ple_embed_dim),
            self.ple_dense_layer_id,
            vllm_config.scheduler_config.max_num_batched_tokens,
            f"{prefix}.ple_embedding",
            prefix,
            data_parallel_rank=vllm_config.parallel_config.data_parallel_rank,
            quant_config=quant_config,
            params_dtype=model_config.dtype,
        )
        # The PLE cache is TP-replicated, so this merged projection is too.
        self.kv_proj = MergedColumnParallelLinear(
            int(config.ple_embed_dim),
            [self.hc_hidden_size, self.hidden_size],
            bias=False,
            params_dtype=model_config.dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
            disable_tp=True,
        )
        norm_args = (
            self.hc_hidden_size,
            config.rms_norm_eps,
            self.hidden_size,
            model_config.dtype,
        )
        self.norm_key = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_query = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_conv = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=model_config.dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True
        self.kv_cache = (torch.tensor([]),)
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    def get_attn_backend(self) -> type[PleShortConvAttentionBackend]:
        return PleShortConvAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> Sequence[tuple[int, ...]]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.hc_hidden_size,
            conv_kernel=self.conv_state_len + 1,
            num_spec=self.num_spec_tokens,
        )

    def _short_conv_dilated_dispatch(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
        outer_residual: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> None:
        num_prefills = metadata.num_prefills
        num_decodes = metadata.num_decodes
        num_decode_tokens = metadata.num_decode_tokens
        num_prefill_tokens = metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        has_spec = metadata.spec_sequence_masks is not None
        has_non_spec = has_prefill or has_decode
        inputs = inputs[: metadata.num_actual_tokens]
        residual = residual[: metadata.num_actual_tokens]
        outer_residual = outer_residual[: metadata.num_actual_tokens]

        spec_token_indices = None
        non_spec_token_indices = None
        if has_spec and has_non_spec:
            assert metadata.spec_token_indx is not None
            assert metadata.non_spec_token_indx is not None
            spec_token_indices = metadata.spec_token_indx
            non_spec_token_indices = metadata.non_spec_token_indx

        if has_spec:
            assert metadata.spec_state_indices_tensor is not None
            query_start_loc = metadata.spec_query_start_loc
            num_accepted_tokens = metadata.num_accepted_tokens
            assert query_start_loc is not None
            assert num_accepted_tokens is not None
            spec_state_indices = metadata.spec_state_indices_tensor[
                : metadata.num_spec_decodes
            ]
            # Mixed batches stay in their original row order; the kernels map
            # logical spec/non-spec rows instead of materializing both groups.
            ple_conv(
                inputs=inputs,
                residual=residual,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=spec_state_indices,
                outer_residual=outer_residual,
                mode="spec",
                dilation=self.short_conv_dilation,
                query_start_loc=query_start_loc,
                num_accepted_tokens=num_accepted_tokens,
                spec_query_len=metadata.spec_query_len,
                token_indices=spec_token_indices,
            )

        if not has_non_spec:
            return

        state_indices = metadata.state_indices_tensor
        assert state_indices is not None
        if has_prefill:
            state_indices_d, state_indices_p = torch.split(
                state_indices, [num_decodes, num_prefills], dim=0
            )
            if non_spec_token_indices is None:
                inputs_d, inputs_p = torch.split(
                    inputs, [num_decode_tokens, num_prefill_tokens], dim=0
                )
                residual_d, residual_p = torch.split(
                    residual, [num_decode_tokens, num_prefill_tokens], dim=0
                )
                outer_residual_d, outer_residual_p = torch.split(
                    outer_residual,
                    [num_decode_tokens, num_prefill_tokens],
                    dim=0,
                )
                token_indices_d = None
                token_indices_p = None
            else:
                inputs_d = inputs_p = inputs
                residual_d = residual_p = residual
                outer_residual_d = outer_residual_p = outer_residual
                token_indices_d, token_indices_p = torch.split(
                    non_spec_token_indices,
                    [num_decode_tokens, num_prefill_tokens],
                    dim=0,
                )

            if has_decode:
                ple_conv(
                    inputs=inputs_d,
                    residual=residual_d,
                    conv_state=conv_state,
                    conv_weights=conv_weights,
                    state_indices=state_indices_d,
                    outer_residual=outer_residual_d,
                    mode="decode",
                    dilation=self.short_conv_dilation,
                    has_initial_states=metadata.has_initial_states_d,
                    token_indices=token_indices_d,
                )

            query_start_loc = metadata.query_start_loc_p
            if query_start_loc is None:
                raise ValueError("query_start_loc is required for prefill short-conv")
            has_initial_states = metadata.has_initial_states_p
            if has_initial_states is None:
                raise ValueError(
                    "has_initial_states_p is required for prefill short-conv"
                )
            ple_conv(
                inputs=inputs_p,
                residual=residual_p,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=state_indices_p,
                outer_residual=outer_residual_p,
                mode="prefill",
                dilation=self.short_conv_dilation,
                query_start_loc=query_start_loc,
                has_initial_states=has_initial_states,
                token_indices=token_indices_p,
            )
        else:
            num_decode_rows = (
                non_spec_token_indices.numel()
                if non_spec_token_indices is not None
                else inputs.size(0)
            )
            ple_conv(
                inputs=inputs,
                residual=residual,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=state_indices[:num_decode_rows],
                outer_residual=outer_residual,
                mode="decode",
                dilation=self.short_conv_dilation,
                has_initial_states=metadata.has_initial_states_d,
                token_indices=non_spec_token_indices,
            )

    def _short_conv(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
        outer_residual: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        # Profiling omits all metadata or this Mamba entry. Short convolution
        # is a no-op there, but preserve the outer residual addition.
        if attn_metadata is None:
            residual.add_(outer_residual)
            return

        if not isinstance(attn_metadata, dict):
            raise RuntimeError(
                "PLE short-conv expects per-layer attention metadata dict "
                f"during inference, got {type(attn_metadata).__name__}."
            )

        layer_attn_metadata = attn_metadata.get(self.prefix)
        if layer_attn_metadata is None:
            residual.add_(outer_residual)
            return
        if not isinstance(layer_attn_metadata, PleShortConvAttentionMetadata):
            raise TypeError(
                "Expected PleShortConvAttentionMetadata for layer "
                f"'{self.prefix}', got "
                f"{type(layer_attn_metadata).__name__}."
            )

        conv_state = self.kv_cache[0]
        # Canonicalize both backend cache layouts to [slot, channel, window].
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = self.conv1d.weight.squeeze(1)

        state_capacity = self.conv_state_len + self.num_spec_tokens
        if state_capacity > 0:
            if conv_state.size(-1) < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache is smaller than expected for "
                    f"dilated convolution: got {conv_state.size(-1)}, "
                    f"expect at least {state_capacity}."
                )
            conv_state = conv_state[..., -state_capacity:]
        self._short_conv_dilated_dispatch(
            inputs=inputs,
            residual=residual,
            outer_residual=outer_residual,
            metadata=layer_attn_metadata,
            conv_state=conv_state,
            conv_weights=conv_weights.to(dtype=inputs.dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same "
                f"token length, got {input_ids.shape[0]} and "
                f"{hidden_states.shape[0]}"
            )
        embeddings = self.ple_embedding(input_ids, query_start_loc, ngram_context)
        embeddings = self.ple_embedding.ngram_embedding.dequantize(
            embeddings, hidden_states.dtype
        )
        kv, _ = self.kv_proj(embeddings)
        key, value = kv.split(self.kv_proj.output_sizes, dim=-1)
        gated_output, conv_input = ple_gate(
            key,
            value,
            hidden_states,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            self.norm_key.eps,
        )
        # The short-conv op is a piecewise-graph splitting op: state routing
        # reads runtime request metadata. It accumulates the convolution and
        # the outer residual into gated_output.
        torch.ops.vllm.qwen4_exp_ple_short_conv(
            conv_input,
            gated_output,
            hidden_states,
            self.prefix,
        )
        return gated_output


def qwen4_exp_amd_ple_ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Hash the current request layout into n-gram embedding indices.

    The launch sizes its request binary search from the request count, which
    is symbolic under Dynamo, so it runs outside Inductor's FX graph.
    """
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpPLELayer):
        raise TypeError(f"{layer_name} is not a Qwen4Exp PLE owner")
    embedding = layer.ple_embedding
    ple_ngram_ids(
        input_ids=input_ids,
        query_start_loc=query_start_loc,
        ngram_context=ngram_context,
        layer_multipliers=embedding.layer_multipliers,
        ngram_heads_vocab_sizes=embedding.ngram_heads_vocab_sizes,
        ngram_heads_offsets=embedding.ngram_heads_offsets,
        eos_token_id=embedding.eos_token_id,
        heads_per_ngram=embedding.heads_per_ngram,
        output=output,
    )


def qwen4_exp_amd_ple_ngram_embedding(
    ngram_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Run the large PLE embedding lookup outside Inductor's FX graph.

    Keeping the embedding weight in ``static_forward_context`` prevents AOT
    compile-time autotuning from materializing a synthetic copy of the weight.
    """
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpPLELayer):
        raise TypeError(f"{layer_name} is not a Qwen4Exp PLE owner")
    result = layer.ple_embedding.ngram_embedding(ngram_ids).flatten(-2)
    output.copy_(result)


def qwen4_exp_amd_ple_ngram_embedding_pinned(
    ngram_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Run the pinned PLE UVA lookup outside Inductor's FX graph.

    Same rationale as the device-path escape: keeping the large embedding weight
    out of the graph prevents AOT compile-time autotuning from materializing a
    synthetic copy of the weight.
    """
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpPLELayer):
        raise TypeError(f"{layer_name} is not a Qwen4Exp PLE owner")
    result = layer.ple_embedding.ngram_embedding.sync_lookup(ngram_ids).flatten(-2)
    output.copy_(result)


def qwen4_exp_ple_short_conv(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    outer_residual: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer._short_conv(inputs, residual, outer_residual)


direct_register_custom_op(
    op_name="qwen4_exp_amd_ple_ngram_ids",
    op_func=qwen4_exp_amd_ple_ngram_ids,
    mutates_args=["output"],
)


direct_register_custom_op(
    op_name="qwen4_exp_amd_ple_ngram_embedding",
    op_func=qwen4_exp_amd_ple_ngram_embedding,
    mutates_args=["output"],
)


direct_register_custom_op(
    op_name="qwen4_exp_amd_ple_ngram_embedding_pinned",
    op_func=qwen4_exp_amd_ple_ngram_embedding_pinned,
    mutates_args=["output"],
)


direct_register_custom_op(
    op_name="qwen4_exp_ple_short_conv",
    op_func=qwen4_exp_ple_short_conv,
    mutates_args=["residual"],
)


__all__ = [
    "Qwen4ExpNGramEmbedding",
    "Qwen4ExpPLEGroupedNorm",
    "Qwen4ExpPLELayer",
]
