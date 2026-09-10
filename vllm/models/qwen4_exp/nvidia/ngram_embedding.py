# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp n-gram embeddings with device and pinned-host storage."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_etp_group, get_tp_group
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptQuantConfigBase,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    _e2m1_inline,
    dequantize_to_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from ..common.ple import PLEVocabParallelEmbedding, compute_ple_shard_overlap
from .ops.ple import ple_ngram_ids

logger = init_logger(__name__)


class Qwen4ExpPLEEmbedding(PLEVocabParallelEmbedding, ABC):
    """ETP-sharded PLE table shared by device and pinned-host backends."""

    supports_prefetch: ClassVar[bool] = False

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: "Qwen4ExpPLEEmbeddingMethod",
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
    ) -> None:
        del num_ngram_heads, max_total_tokens
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            quant_method=embedding_method,
            parallel_group=get_etp_group(),
        )
        self.embedding_method = embedding_method
        self.data_parallel_rank = data_parallel_rank
        tp_size = get_tp_group().world_size
        if self.tp_size % tp_size:
            raise ValueError(
                "ETP size must be divisible by TP size, but got "
                f"ETP={self.tp_size} and TP={tp_size}"
            )
        self.etp_data_parallel_size = self.tp_size // tp_size

    @abstractmethod
    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate storage for the complete embedding weight."""
        raise NotImplementedError

    def dequantize(
        self,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Delegate storage-format conversion to the embedding method."""
        return self.embedding_method.dequantize(self, embeddings, output_dtype)

    def _get_dp_gather_slot(self, local_num_tokens: int) -> tuple[int, int]:
        """Return the per-DP slot size and this rank's slot offset."""
        if self.etp_data_parallel_size == 1:
            return local_num_tokens, 0
        dp_metadata: DPMetadata | None = get_forward_context().dp_metadata
        if dp_metadata is None:
            raise RuntimeError("ETP spanning DP requires DP token metadata")
        group_start = (self.data_parallel_rank // self.etp_data_parallel_size) * (
            self.etp_data_parallel_size
        )
        group_end = group_start + self.etp_data_parallel_size
        token_counts = dp_metadata.num_tokens_across_dp_cpu.tolist()
        group_counts = token_counts[group_start:group_end]
        slot_size = max(group_counts)
        dp_rank = get_dp_group().rank_in_group
        return slot_size, dp_rank * slot_size

    def _gather_dp_ids(
        self,
        ngram_ids: torch.Tensor,
        slot_size: int,
    ) -> torch.Tensor:
        """Gather DP-local IDs that share one ETP-sharded PLE table."""
        if self.etp_data_parallel_size == 1:
            return ngram_ids
        if ngram_ids.shape[0] < slot_size:
            padding = ngram_ids.new_zeros(
                slot_size - ngram_ids.shape[0], ngram_ids.shape[1]
            )
            ngram_ids = torch.cat((ngram_ids, padding), dim=0)
        return get_dp_group().all_gather(ngram_ids, dim=0)

    def _select_embeddings(
        self,
        embeddings: torch.Tensor,
        local_num_tokens: int,
        slot_offset: int,
    ) -> torch.Tensor:
        """Select this DP rank's rows from the ETP-reduced embeddings."""
        if self.etp_data_parallel_size == 1:
            return embeddings
        return embeddings.narrow(0, slot_offset, local_num_tokens)

    @abstractmethod
    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Start an asynchronous lookup when supported."""
        raise NotImplementedError


class Qwen4ExpPLEEmbeddingMethod(QuantizeMethodBase):
    """Quantization interface shared by resident and pinned PLE tables."""

    # PLE post-load processing only validates scales in their current storage.
    requires_device_loading: bool = False

    @staticmethod
    def from_quant_config(
        quant_config: QuantizationConfig | None,
        prefix: str,
        embedding_dtype: str | None = None,
    ) -> "Qwen4ExpPLEEmbeddingMethod":
        """Select the concrete PLE embedding format for a layer."""
        if embedding_dtype == "nvfp4":
            return Qwen4ExpPLENvFp4EmbeddingMethod()
        if embedding_dtype == "float8_e4m3fn":
            return Qwen4ExpPLEFp8EmbeddingMethod()
        if quant_config is None:
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if isinstance(quant_config, ModelOptMixedPrecisionConfig):
            if quant_config._resolve_quant_algo(prefix) == "FP8":
                return Qwen4ExpPLEFp8EmbeddingMethod()
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if isinstance(
            quant_config, ModelOptQuantConfigBase
        ) and quant_config.is_layer_excluded(prefix):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if not isinstance(quant_config, Fp8Config):
            raise NotImplementedError(
                "Qwen4Exp PLE embedding does not support quantization config "
                f"{type(quant_config).__name__}"
            )

        ignored_layers = quant_config.ignored_layers
        if is_layer_skipped(
            prefix,
            ignored_layers,
            quant_config.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        ):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        # PLE checkpoint shards form one runtime embedding parameter.
        shard_prefix = f"{prefix}.shard_"
        if any(name.startswith(shard_prefix) for name in ignored_layers):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if not quant_config.is_checkpoint_fp8_serialized:
            raise NotImplementedError(
                "Qwen4Exp PLE embedding only supports serialized FP8 checkpoints"
            )
        return Qwen4ExpPLEFp8EmbeddingMethod()

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("PLE weights only support embedding lookup")

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)

    def lookup_dtype(self, layer: nn.Module) -> torch.dtype:
        return layer.weight.dtype

    def lookup_from_pinned(
        self,
        layer: "Qwen4ExpPLEPinnedHostEmbedding",
        ids: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        _lookup_ple_embedding_from_pinned_kernel[(ids.numel(),)](
            layer._uva_weight,
            ids,
            output,
            layer.embedding_dim,
            layer.shard_indices.org_vocab_start_index,
            layer.shard_indices.org_vocab_end_index,
            BLOCK_D=layer._block_d,
        )

    @abstractmethod
    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert looked-up PLE rows to the activation dtype."""
        raise NotImplementedError


class Qwen4ExpPLEUnquantizedEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """Unquantized PLE embedding storage and lookup semantics."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        weight = nn.Parameter(
            layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition,
                params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del layer, output_dtype
        return embeddings


class Qwen4ExpPLEFp8EmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """FP8 PLE embedding with one global checkpoint scale."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = ModelWeightParameter(
            data=layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition,
                torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = create_fp8_scale_parameter(
            PerTensorScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            None,
            weight_loader,
            scale_dtype=torch.float32,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Reject FP8 PLE checkpoints without a global scale."""
        sentinel = torch.finfo(torch.float32).min
        if torch.any(layer.weight_scale == sentinel):
            raise ValueError("FP8 PLE checkpoint is missing its global scale")

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        weight_scale = getattr(layer, "weight_scale", None)
        if weight_scale is None:
            raise RuntimeError("FP8 PLE embedding is missing its global scale")
        if weight_scale.device != embeddings.device:
            raise RuntimeError("FP8 PLE embedding scale must be on the output device")
        return embeddings.to(output_dtype) * weight_scale.to(output_dtype)


class Qwen4ExpPLENvFp4EmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """Packed E2M1 rows, E4M3 scales per 16 values, and one global scale."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        if input_size_per_partition % 16:
            raise ValueError("NVFP4 PLE embedding dimension must be divisible by 16")
        self._loaded_ranges: dict[str, set[tuple[int, int]]] = {
            "weight": set(),
            "weight_scale": set(),
        }
        weight_loader = extra_weight_attrs.get("weight_loader")
        for name, width, dtype in (
            ("weight", input_size_per_partition // 2, torch.uint8),
            ("weight_scale", input_size_per_partition // 16, torch.float8_e4m3fn),
        ):
            layer.register_parameter(
                name,
                ModelWeightParameter(
                    data=layer.allocate_embedding_weight(
                        sum(output_partition_sizes), width, dtype
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                ),
            )
        layer.register_parameter(
            "weight_scale_2",
            create_fp8_scale_parameter(
                PerTensorScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                None,
                weight_loader,
                scale_dtype=torch.float32,
            ),
        )
        if layer.weight_scale.is_pinned():
            layer._uva_weight_scale = get_accelerator_view_from_cpu_tensor(
                layer.weight_scale
            )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        scale = layer.weight_scale_2
        if not torch.all(torch.isfinite(scale) & (scale > 0)):
            raise ValueError(
                "NVFP4 PLE checkpoint requires a finite positive global scale"
            )
        expected_rows = (
            layer.shard_indices.org_vocab_end_index
            - layer.shard_indices.org_vocab_start_index
        )
        for name, ranges in self._loaded_ranges.items():
            loaded_end = 0
            for start, end in sorted(ranges):
                if start > loaded_end:
                    break
                loaded_end = max(loaded_end, end)
            if loaded_end != expected_rows:
                raise ValueError(
                    f"NVFP4 PLE checkpoint is missing {name} rows "
                    f"starting at local row {loaded_end}"
                )

    def record_loaded_rows(
        self,
        layer: Qwen4ExpPLEEmbedding,
        name: str,
        checkpoint_start: int,
        checkpoint_rows: int,
    ) -> None:
        overlap = compute_ple_shard_overlap(
            checkpoint_start=checkpoint_start,
            checkpoint_rows=checkpoint_rows,
            tp_start=layer.shard_indices.org_vocab_start_index,
            tp_end=layer.shard_indices.org_vocab_end_index,
        )
        if overlap is not None:
            start = overlap.destination_start
            self._loaded_ranges[name].add((start, start + overlap.row_count))

    def lookup_dtype(self, layer: nn.Module) -> torch.dtype:
        return layer.params_dtype

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        ids = input_.reshape(-1)
        if not input_.is_cuda:
            rows = dequantize_to_dtype(
                F.embedding(ids, layer.weight),
                F.embedding(ids, layer.weight_scale),
                layer.weight_scale_2,
                layer.params_dtype,
                swizzle=False,
            )
            return rows.reshape(*input_.shape, layer.embedding_dim)
        output = torch.empty(
            (*input_.shape, layer.embedding_dim),
            dtype=layer.params_dtype,
            device=input_.device,
        )
        if ids.numel():
            _lookup_nvfp4_ple_embedding_kernel[(ids.numel(),)](
                layer.weight,
                layer.weight_scale,
                layer.weight_scale_2,
                ids,
                output,
                layer.embedding_dim,
                0,
                layer.weight.shape[0],
                BLOCK_D=triton.next_power_of_2(layer.embedding_dim),
            )
        return output

    def lookup_from_pinned(
        self,
        layer: "Qwen4ExpPLEPinnedHostEmbedding",
        ids: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        _lookup_nvfp4_ple_embedding_kernel[(ids.numel(),)](
            layer._uva_weight,
            layer._uva_weight_scale,
            layer.weight_scale_2,
            ids,
            output,
            layer.embedding_dim,
            layer.shard_indices.org_vocab_start_index,
            layer.shard_indices.org_vocab_end_index,
            BLOCK_D=layer._block_d,
        )

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        # NVFP4 rows are dequantized during lookup, before the ETP reduction.
        return embeddings.to(output_dtype)


@triton.jit
def _lookup_nvfp4_ple_embedding_kernel(
    weight_ptr,
    scale_ptr,
    global_scale_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    vocab_start,
    vocab_end,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    owned = (idx >= vocab_start) & (idx < vocab_end)
    local_idx = tl.where(owned, idx - vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = owned & (offsets < embedding_dim)
    packed = tl.load(
        weight_ptr + local_idx * (embedding_dim // 2) + offsets // 2,
        mask=mask,
        other=0,
    )
    codes = (packed >> ((offsets % 2) * 4)) & 0xF
    scales = tl.load(
        scale_ptr + local_idx * (embedding_dim // 16) + offsets // 16,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    values = _e2m1_inline(codes) * scales * global_scale
    tl.store(
        output_ptr + row * embedding_dim + offsets, values, offsets < embedding_dim
    )


class Qwen4ExpPLEDeviceEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table allocated on the active model device."""

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight on the active device."""
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype)

    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Resident embedding prefetch is a no-op."""
        return None

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Gather ETP inputs, look up embeddings, and select local rows."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        embeddings = super().forward(gathered_ids)
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )


@triton.jit
def _lookup_ple_embedding_from_pinned_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    BLOCK_D: tl.constexpr,
):
    """Look up TP-owned PLE rows through a CUDA view of pinned host memory."""
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    store_mask = offsets < embedding_dim
    load_mask = store_mask & in_range
    values = tl.load(
        weight_ptr + local_idx * embedding_dim + offsets,
        mask=load_mask,
        other=0.0,
    )
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        values,
        mask=store_mask,
    )


class Qwen4ExpPLEPinnedHostEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table loaded into pinned CPU memory and looked up through UVA."""

    supports_prefetch: ClassVar[bool] = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
    ) -> None:
        if not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)
        self._block_d = triton.next_power_of_2(self.embedding_dim)
        self._prefetch_stream = torch.cuda.Stream(device=self._uva_weight.device)
        self._prefetch_buffer = torch.empty(
            max_total_tokens * self.etp_data_parallel_size,
            num_ngram_heads,
            self.embedding_dim,
            dtype=self.embedding_method.lookup_dtype(self),
            device=self._uva_weight.device,
        )
        self._output_dim = num_ngram_heads * self.embedding_dim

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight directly in pinned CPU memory."""
        return torch.empty(
            num_embeddings,
            embedding_dim,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )

    def _lookup(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Look up local ETP rows in the embedding method's output format."""
        expected_shape = (*input_ids.shape, self.embedding_dim)
        output_dtype = self.embedding_method.lookup_dtype(self)
        if output is None:
            output = torch.empty(
                expected_shape,
                dtype=output_dtype,
                device=input_ids.device,
            )
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != output_dtype
            or output.device != input_ids.device
        ):
            raise ValueError(
                "PLE prefetch output must match the input shape, lookup dtype, "
                "and input device"
            )

        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            self.embedding_method.lookup_from_pinned(self, flat_ids, output)
        return output

    def _reduce_etp_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Combine pinned lookup results owned by different ETP ranks."""
        if self.tp_size == 1:
            return embeddings
        assert self.parallel_group is not None
        if embeddings.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # Each vocabulary row has one owner, so reduce the raw FP8 bytes.
            reduced = self.parallel_group.all_reduce(embeddings.view(torch.int8))
            return reduced.view(embeddings.dtype)
        return self.parallel_group.all_reduce(embeddings)

    @eager_break_during_capture
    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Gather ETP IDs and launch their UVA lookup on the side stream."""
        slot_size, _ = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        active_output = self._prefetch_buffer[: gathered_ids.shape[0]]
        prefetch_stream = self._prefetch_stream
        prefetch_stream.wait_stream(torch.cuda.current_stream())
        gathered_ids.record_stream(prefetch_stream)
        with torch.cuda.stream(prefetch_stream):
            self._lookup(gathered_ids, output=active_output)

    @eager_break_during_capture
    def _finalize_prefetch(
        self,
        prefetch_output: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Join the side stream, reduce ETP shards, and select local rows."""
        torch.cuda.current_stream().wait_stream(self._prefetch_stream)
        slot_size, slot_offset = self._get_dp_gather_slot(output.shape[0])
        active_output = prefetch_output[: slot_size * self.etp_data_parallel_size]
        embeddings = self._reduce_etp_embeddings(active_output)
        embeddings = self._select_embeddings(
            embeddings,
            output.shape[0],
            slot_offset,
        )
        output.copy_(embeddings.flatten(-2))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Finish the pinned lookup into graph-owned output storage."""
        output = self._prefetch_buffer.new_empty(
            (hidden_states.shape[0], self._output_dim)
        )
        self._finalize_prefetch(self._prefetch_buffer, output)
        return output


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
        *,
        data_parallel_rank: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
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
        embedding_prefix = f"{prefix}.ngram_embedding"
        embedding_quant_method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(
            quant_config,
            embedding_prefix,
            getattr(config, "ple_embedding_dtype", None),
        )
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
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
        weight = self.ngram_embedding.weight
        logger.info(
            "Initialized PLE embedding %s: quantization_method=%s, "
            "weight_dtype=%s, weight_device=%s, pinned=%s",
            embedding_prefix,
            type(embedding_quant_method).__name__,
            weight.dtype,
            weight.device,
            weight.is_pinned(),
        )

    @staticmethod
    def _shift_precompute(
        tokens: torch.Tensor, eos_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor")
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        position_in_segment: torch.Tensor,
        shift: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    def compute_ngram_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute n-gram embedding indices for the current request layout."""
        input_ids = input_ids.reshape(-1)
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]

        if input_ids.is_cuda:
            return ple_ngram_ids(
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
                layer_multipliers=self.layer_multipliers,
                ngram_heads_vocab_sizes=self.ngram_heads_vocab_sizes,
                ngram_heads_offsets=self.ngram_heads_offsets,
                eos_token_id=self.eos_token_id,
                heads_per_ngram=self.heads_per_ngram,
                output=output,
            )
        input_ids = input_ids.long()
        query_start_loc = query_start_loc.long()
        positions = torch.arange(num_tokens, device=input_ids.device, dtype=torch.int64)
        packed = torch.full(
            (num_reqs, num_tokens),
            self.eos_token_id,
            device=input_ids.device,
            dtype=torch.int64,
        )
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc[request_indices]).clamp(
            0, packed.shape[1] - 1
        )
        packed[request_indices, columns] = input_ids
        ngram_context = ngram_context[:num_reqs].to(
            device=input_ids.device, dtype=torch.long
        )

        context = torch.cat([ngram_context, packed], dim=-1)
        positions_2d, position_in_segment = self._shift_precompute(
            context, self.eos_token_id
        )
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(
                    context,
                    positions_2d,
                    position_in_segment,
                    shift,
                    self.eos_token_id,
                )
            )
        adjusted_columns = columns + self.ngram_size - 1
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, shifted[index] * self.layer_multipliers[index]
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(ids[request_indices, adjusted_columns])
        return torch.cat(id_blocks, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.ngram_embedding
        if embedding.supports_prefetch:
            return embedding(hidden_states)
        ngram_ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        return self.ngram_embedding(ngram_ids).flatten(-2)

    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        """Start the pinned lookup while the preceding decoder layer runs."""
        embedding = self.ngram_embedding
        if not embedding.supports_prefetch:
            return
        ngram_ids = self.compute_ngram_ids(
            input_ids,
            query_start_loc,
            ngram_context,
        )
        embedding.start_prefetch(hidden_states, ngram_ids)

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
        embedding = self.ngram_embedding
        method = getattr(embedding, "embedding_method", None)
        nvfp4_method = (
            method if isinstance(method, Qwen4ExpPLENvFp4EmbeddingMethod) else None
        )
        shard_parameters = ("weight", "weight_scale") if nvfp4_method else ("weight",)
        shard_size = (
            embedding.org_vocab_size + self.split_ngram_parts - 1
        ) // self.split_ngram_parts

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
            if name.startswith(shard_prefix):
                shard_text, _, suffix = name[len(shard_prefix) :].partition(".")
                if not shard_text.isdigit() or suffix not in shard_parameters:
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, embedding.org_vocab_size - checkpoint_start),
                )
                parameter = getattr(embedding, suffix)
                expected_shape = (expected_rows, parameter.shape[1])
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_index} "
                        f"{suffix}: "
                        f"expected {expected_shape}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                if nvfp4_method and loaded_weight.dtype != parameter.dtype:
                    raise ValueError(
                        f"NVFP4 PLE shard {shard_index} {suffix} requires "
                        f"{parameter.dtype}, got {loaded_weight.dtype}"
                    )
                parameter.weight_loader(
                    parameter,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                )
                if nvfp4_method:
                    nvfp4_method.record_loaded_rows(
                        embedding, suffix, checkpoint_start, expected_rows
                    )
                loaded.add(f"ngram_embedding.{suffix}")
                continue
            regular_weights.append((name, loaded_weight))

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        return loaded


__all__ = [
    "Qwen4ExpNGramEmbedding",
]
