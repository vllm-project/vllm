# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Qwen4Exp n-gram embedding storage with device and pinned-host backends.

Both the NVIDIA and AMD Qwen4Exp implementations use these classes so the (large)
n-gram embedding table can be kept in pinned host memory and looked up through
Unified Virtual Addressing on any CUDA-alike platform.
"""

import fcntl
import hashlib
import json
import mmap
import os
import tempfile
import threading
import time
import weakref
from abc import ABC, abstractmethod
from contextlib import ExitStack
from pathlib import Path
from typing import ClassVar, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import (
    get_dp_group,
    get_engram_dp_group,
    get_etp_group,
    get_tp_group,
)
from vllm.distributed.device_communicators.shm_broadcast import (
    SHM_PATH,
    check_shm_free_space,
)
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptQuantConfigBase,
)
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    _e2m1_inline,
    _e4m3_inline,
    dequantize_to_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from .ple import PLEVocabParallelEmbedding

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
        reduce_results: bool = True,
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
            reduce_results=reduce_results,
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
    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Start an asynchronous lookup when supported."""
        raise NotImplementedError

    @abstractmethod
    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        """Return prefetched embeddings for the full input token set."""
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
        if embedding_dtype is not None:
            raise ValueError(
                "ple_embedding_dtype must be 'nvfp4' or 'float8_e4m3fn', "
                f"got {embedding_dtype!r}"
            )
        if quant_config is None:
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if isinstance(quant_config, ModelOptMixedPrecisionConfig):
            quant_algo = quant_config._resolve_quant_algo(prefix)
            if quant_algo == "FP8":
                return Qwen4ExpPLEFp8EmbeddingMethod()
            if quant_algo == "NVFP4":
                return Qwen4ExpPLENvFp4EmbeddingMethod()
            if quant_algo is not None:
                raise ValueError(
                    f"Unsupported quant_algo {quant_algo!r} for the PLE embedding "
                    f"{prefix!r}; expected FP8 or NVFP4"
                )
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if isinstance(
            quant_config, ModelOptQuantConfigBase
        ) and quant_config.is_layer_excluded(prefix):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if (
            isinstance(quant_config, CompressedTensorsConfig)
            and quant_config.get_scheme_dict(None, layer_name=prefix) is None
        ):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if (
            isinstance(quant_config, INCConfig)
            and not quant_config.config_parser.resolve(None, prefix).quantized
        ):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        # Quark quantizes only Linear and MoE layers; PLE tables stay BF16.
        if isinstance(quant_config, QuarkConfig):
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
        """Return the dtype of rows that a pinned-host lookup writes."""
        return layer.weight.dtype

    def lookup_from_pinned(
        self,
        layer: "Qwen4ExpPLEPinnedHostEmbedding",
        ids: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Copy the ETP-owned rows for ``ids`` from pinned host memory."""
        row_bytes = layer.embedding_dim * layer.weight.element_size()
        _lookup_ple_embedding_from_pinned_kernel[(ids.numel(),)](
            layer._uva_weight,
            ids,
            output,
            row_bytes,
            layer.shard_indices.org_vocab_start_index,
            layer.shard_indices.org_vocab_end_index,
            BLOCK_D=triton.next_power_of_2(row_bytes),
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """A file-shared table publishes or awaits its ready mark here."""
        finish = getattr(layer, "finish_shared_table", None)
        if finish is not None:
            finish()

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
        super().process_weights_after_loading(layer)
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


# NVFP4 packs two E2M1 values per byte and shares one E4M3 scale per block.
_NVFP4_VALUES_PER_BYTE = 2
_NVFP4_BLOCK_SIZE = 16


class Qwen4ExpPLENvFp4EmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """NVFP4 PLE embedding: packed E2M1 rows, block scales, one global scale.

    Rows stay packed in storage. A lookup decodes only the selected rows to
    the activation dtype, before the ETP reduction.
    """

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
        if input_size_per_partition % _NVFP4_BLOCK_SIZE:
            raise ValueError(
                "NVFP4 PLE embedding dimension must be divisible by "
                f"{_NVFP4_BLOCK_SIZE}, got {input_size_per_partition}"
            )
        weight_loader = extra_weight_attrs.get("weight_loader")
        num_rows = sum(output_partition_sizes)
        for name, width, dtype in (
            (
                "weight",
                input_size_per_partition // _NVFP4_VALUES_PER_BYTE,
                torch.uint8,
            ),
            (
                "weight_scale",
                input_size_per_partition // _NVFP4_BLOCK_SIZE,
                torch.float8_e4m3fn,
            ),
        ):
            parameter = ModelWeightParameter(
                data=layer.allocate_embedding_weight(num_rows, width, dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter(name, parameter)
        weight_scale_2 = create_fp8_scale_parameter(
            PerTensorScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            None,
            weight_loader,
            scale_dtype=torch.float32,
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)
        if layer.weight_scale.is_pinned():
            self._uva_weight_scale = get_accelerator_view_from_cpu_tensor(
                layer.weight_scale
            )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Reject NVFP4 PLE checkpoints without a global scale."""
        super().process_weights_after_loading(layer)
        sentinel = torch.finfo(torch.float32).min
        if torch.any(layer.weight_scale_2 == sentinel):
            raise ValueError("NVFP4 PLE checkpoint is missing its global scale")

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
            # Device rows are already local, so every ID in range is owned.
            _lookup_nvfp4_ple_embedding_kernel[(ids.numel(),)](
                layer.weight,
                layer.weight_scale.view(torch.uint8),
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
            self._uva_weight_scale.view(torch.uint8),
            layer.weight_scale_2,
            ids,
            output,
            layer.embedding_dim,
            layer.shard_indices.org_vocab_start_index,
            layer.shard_indices.org_vocab_end_index,
            BLOCK_D=triton.next_power_of_2(layer.embedding_dim),
        )

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del layer
        # The lookup already decoded the rows to the activation dtype.
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
    """Decode owned NVFP4 PLE rows; write zeros for rows another rank owns."""
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    owned = (idx >= vocab_start) & (idx < vocab_end)
    local_idx = tl.where(owned, idx - vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    in_row = offsets < embedding_dim
    mask = owned & in_row
    packed = tl.load(
        weight_ptr + local_idx * (embedding_dim // 2) + offsets // 2,
        mask=mask,
        other=0,
    )
    # The low nibble holds the even element and the high nibble the odd one.
    codes = (packed >> ((offsets % 2) * 4)) & 0xF
    scales = _e4m3_inline(
        tl.load(
            scale_ptr + local_idx * (embedding_dim // 16) + offsets // 16,
            mask=mask,
            other=0,
        )
    )
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    values = _e2m1_inline(codes) * scales * global_scale
    tl.store(output_ptr + row * embedding_dim + offsets, values, mask=in_row)


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

    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Resident embedding prefetch is a no-op."""
        return None

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        """Reject prefetch finalization for resident embeddings."""
        raise RuntimeError("Device PLE embedding does not support prefetch")

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Gather ETP IDs, optionally reduce lookup results, and select local rows."""
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
    weight_ptr: tl.pointer_type(tl.uint8),  # type: ignore[valid-type]
    ids_ptr,
    output_ptr: tl.pointer_type(tl.uint8),  # type: ignore[valid-type]
    row_bytes,
    tp_vocab_start,
    tp_vocab_end,
    BLOCK_D: tl.constexpr,
):
    """Copy TP-owned PLE rows as raw bytes from a CUDA view of pinned host memory.

    Byte pointers keep the storage dtype out of the kernel signature, so
    dtypes Triton cannot lower on the GPU (FP8 E4M3FN before SM89) still work.
    """
    row_id = tl.program_id(0).to(tl.int64)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    store_mask = offsets < row_bytes
    load_mask = store_mask & in_range
    values = tl.load(
        weight_ptr + local_idx * row_bytes + offsets,
        mask=load_mask,
        other=0,
    )
    tl.store(
        output_ptr + row_id * row_bytes + offsets,
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
        reduce_results: bool = True,
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
            reduce_results=reduce_results,
        )
        self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)
        self._prefetch_stream: torch.cuda.Stream | None = None
        self._prefetch_buffer: torch.Tensor | None = None
        self._prefetch_alloc_lock = threading.Lock()
        self._prefetch_rows = max_total_tokens * self.etp_data_parallel_size
        self._num_ngram_heads = num_ngram_heads
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
        """Look up local ETP rows in the embedding method's lookup dtype."""
        expected_shape = (*input_ids.shape, self.embedding_dim)
        lookup_dtype = self.embedding_method.lookup_dtype(self)
        if output is None:
            output = torch.empty(
                expected_shape,
                dtype=lookup_dtype,
                device=input_ids.device,
            )
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != lookup_dtype
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

    def sync_lookup(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Synchronous UVA lookup for platforms without prefetch wiring."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        embeddings = self._lookup(gathered_ids)
        if self.reduce_results:
            embeddings = self._reduce_etp_embeddings(embeddings)
        else:
            # The outer TP reduce-scatter requires ETP=TP.
            assert self.etp_data_parallel_size == 1
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )

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
    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Gather ETP IDs and launch their UVA lookup on the side stream."""
        buffer = self._prefetch_buffer
        if buffer is None:
            # First use allocates. The eager profile run always precedes
            # cudagraph capture, so allocation never happens mid-capture;
            # the lock keeps concurrent first callers from tearing the
            # stream/buffer pair.
            with self._prefetch_alloc_lock:
                buffer = self._prefetch_buffer
                if buffer is None:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "pinned PLE prefetch buffer must be allocated "
                            "eagerly, before cudagraph capture"
                        )
                    self._prefetch_stream = torch.cuda.Stream(
                        device=self._uva_weight.device
                    )
                    buffer = torch.empty(
                        self._prefetch_rows,
                        self._num_ngram_heads,
                        self.embedding_dim,
                        dtype=self.embedding_method.lookup_dtype(self),
                        device=self._uva_weight.device,
                    )
                    self._prefetch_buffer = buffer
        prefetch_stream = self._prefetch_stream
        if prefetch_stream is None:
            raise RuntimeError("pinned PLE prefetch stream was not allocated")
        slot_size, _ = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        if gathered_ids.shape[0] > buffer.shape[0]:
            raise ValueError(
                f"pinned PLE prefetch buffer holds {buffer.shape[0]} rows, "
                f"but the batch needs {gathered_ids.shape[0]}"
            )
        active_output = buffer[: gathered_ids.shape[0]]
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
        """Join the side stream, optionally reduce ETP shards, and select rows."""
        prefetch_stream = self._prefetch_stream
        if prefetch_stream is None:
            raise RuntimeError("pinned PLE finalize requires a prior start_prefetch")
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        slot_size, slot_offset = self._get_dp_gather_slot(output.shape[0])
        embeddings = prefetch_output[: slot_size * self.etp_data_parallel_size]
        if self.reduce_results:
            embeddings = self._reduce_etp_embeddings(embeddings)
        else:
            # The outer TP reduce-scatter requires ETP=TP.
            assert self.etp_data_parallel_size == 1
        embeddings = self._select_embeddings(
            embeddings,
            output.shape[0],
            slot_offset,
        )
        output.copy_(embeddings.flatten(-2))

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        """Finalize the full-token pinned lookup with optional ETP reduction."""
        buffer = self._prefetch_buffer
        if buffer is None:
            raise RuntimeError("pinned PLE lookup requires a prior start_prefetch")
        # total_tokens counts this DP rank's tokens before SP sharding.
        output = buffer.new_empty((total_tokens, self._output_dim))
        self._finalize_prefetch(buffer, output)
        return output

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "Pinned-host PLE embedding requires "
            "start_prefetch() and finalize_prefetch()."
        )


def _unregister_shared_ple_mapping(mapping: mmap.mmap, pointer: int) -> None:
    # Torch storage retains the numpy owner, including through cached UVA views.
    # Keep its mmap alive until CUDA has released the registration.
    result = torch.cuda.cudart().cudaHostUnregister(pointer)
    if result.value != 0:
        logger.warning("PLE cudaHostUnregister failed: %s", result)


_PLE_SHARE_DECISIONS: dict[tuple[int, int], bool] = {}


def can_share_ple_table(num_bytes: int) -> bool:
    """Whether co-located DP replicas exist and /dev/shm can hold the PLE table.

    Only the leader checks, so every replica follows its decision — a peer
    that decided differently would wait forever on the shared mapping's
    collectives. Decided once per group and table size: every PLE layer asks,
    and one broadcast answers them all. Mirrors ``can_share_engram_tables``.
    """
    group = get_engram_dp_group()
    key = (id(group), num_bytes)
    cached = _PLE_SHARE_DECISIONS.get(key)
    if cached is not None:
        return cached
    decision = _decide_ple_table_sharing(group, num_bytes)
    _PLE_SHARE_DECISIONS[key] = decision
    return decision


def _decide_ple_table_sharing(group, num_bytes: int) -> bool:
    if group is None:
        logger.warning_once(
            "Engram DP replicas are not co-located on one node; "
            "storing the offloaded PLE table per rank instead of sharing it."
        )
        return False
    error = None
    if group.rank_in_group == 0:
        if not os.path.isdir(SHM_PATH):
            error = f"{SHM_PATH} is not mounted"
        else:
            try:
                check_shm_free_space(num_bytes, allocation_name="PLE table")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
    error = group.broadcast_object(error)
    if error is not None:
        logger.warning_once(
            "Storing the offloaded PLE table per rank instead of sharing it: %s",
            error,
        )
    return error is None


def _map_shared_ple_file(
    path: str, num_bytes: int
) -> tuple[torch.Tensor, np.ndarray, mmap.mmap]:
    """Map `path` MAP_SHARED and register it with CUDA for UVA lookups.

    Returns the flat uint8 tensor, the numpy owner and the mapping; the owner's
    finalizer unregisters the pages and keeps the mapping alive until then.
    """
    with open(path, "r+b") as file:
        mapping = mmap.mmap(file.fileno(), num_bytes, flags=mmap.MAP_SHARED)
    owner = np.frombuffer(mapping, dtype=np.uint8)
    pointer = owner.ctypes.data
    flat = torch.from_numpy(owner)
    result = torch.cuda.cudart().cudaHostRegister(pointer, num_bytes, 0)
    if result.value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {result}")
    finalizer = weakref.finalize(
        owner, _unregister_shared_ple_mapping, mapping, pointer
    )
    finalizer.atexit = False  # type: ignore[misc]
    # The UVA helper otherwise allocates a private pinned copy.
    if not flat.is_pinned():
        raise RuntimeError("cudaHostRegister did not pin the shared PLE mapping")
    return flat, owner, mapping


class SharedPLETable(NamedTuple):
    """Where independent processes share a model's PLE tables, and which model.

    ``table_id`` names the model, revision and layer; each process adds the
    shape, dtype and ETP shard it holds (``shared_ple_table_identity``).
    """

    directory: str
    table_id: str


def shared_ple_table_identity(
    table_id: str,
    shape: tuple[int, int],
    dtype: torch.dtype,
    *,
    etp_rank: int,
    etp_size: int,
    vocab_range: tuple[int, int],
) -> dict[str, object]:
    """What one shared table file holds: one ETP shard of one PLE layer.

    The table is vocab-sharded over the ETP group and every shard has the same
    shape, so the shard is part of the identity: equal sizes must never let
    one rank map another rank's rows.
    """
    return {
        "table": table_id,
        "shape": [int(shape[0]), int(shape[1])],
        "dtype": str(dtype),
        "etp_rank": int(etp_rank),
        "etp_size": int(etp_size),
        "vocab_range": [int(vocab_range[0]), int(vocab_range[1])],
    }


def shared_ple_table_path(directory: str, identity: dict[str, object]) -> str:
    """The file independent processes share for one table shard."""
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return os.path.join(directory, f"vllm_ple_{key[:24]}")


# How long a process waits for another, still running, to finish loading a
# shared table. A loader that dies releases its lock, which waiters notice
# within one poll instead.
PLE_SHARED_TABLE_READY_TIMEOUT_S = 3600
PLE_SHARED_TABLE_POLL_S = 0.5


def _try_flock(fd: int, operation: int) -> bool:
    try:
        fcntl.flock(fd, operation | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _shared_table_loader_alive(lock_fd: int) -> bool:
    """Whether a process holds the loader's exclusive lock.

    Probed with a shared lock, which only an exclusive holder refuses, so
    waiters probing at the same time never mistake each other for a loader.
    """
    if not _try_flock(lock_fd, fcntl.LOCK_SH):
        return True
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    return False


def _check_shared_table_ready(
    path: str, identity: dict[str, object], num_bytes: int
) -> None:
    """Refuse a ready table that is not the shard this process holds."""
    mark = Path(path + ".ready").read_text()
    try:
        published = json.loads(mark)
    except ValueError:
        published = mark
    expected = {**identity, "bytes": num_bytes}
    if published != expected or os.stat(path).st_size != num_bytes:
        raise RuntimeError(
            f"shared PLE table {path} holds {published!r}, this process needs "
            f"{expected!r}: remove the file and its .ready and .lock and start "
            "again"
        )


def _claim_shared_table(
    path: str, identity: dict[str, object], num_bytes: int
) -> int | None:
    """Become the process that loads ``path``, or wait until one has sized it.

    The loader holds an exclusive ``flock`` on ``<path>.lock`` from before it
    sizes the file until after it publishes ``<path>.ready``; the kernel
    releases it when the process exits. So a file without a ready mark and
    without a lock holder was left by a loader that died, and the next process
    to take the lock loads it again; exactly one can. Returns the lock's file
    descriptor for the loader (released by ``_publish_shared_table``), None for
    a process that maps the table another loads or has loaded.
    """
    ready = path + ".ready"
    if os.path.exists(ready):
        _check_shared_table_ready(path, identity, num_bytes)
        return None
    lock_fd = os.open(path + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.monotonic() + PLE_SHARED_TABLE_READY_TIMEOUT_S
    try:
        while True:
            if os.path.exists(ready):
                _check_shared_table_ready(path, identity, num_bytes)
                return None
            if _try_flock(lock_fd, fcntl.LOCK_EX):
                # A loader publishes the mark before it unlocks.
                if os.path.exists(ready):
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    continue
                _size_shared_table(path, num_bytes)
                loader_fd, lock_fd = lock_fd, -1
                return loader_fd
            if _shared_table_loader_alive(lock_fd):
                # A live loader: map the table once it is sized.
                try:
                    if os.stat(path).st_size == num_bytes:
                        return None
                except FileNotFoundError:
                    pass
            # Otherwise only waiters' probes held the lock: try again.
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"shared PLE table {path} was not sized within "
                    f"{PLE_SHARED_TABLE_READY_TIMEOUT_S} s by the process "
                    "loading it"
                )
            time.sleep(PLE_SHARED_TABLE_POLL_S)
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)


def _size_shared_table(path: str, num_bytes: int) -> None:
    """Create or resize the table file. Never truncated below its size:
    processes waiting on an interrupted load may already have its pages
    mapped and pinned, and they see the reload in place."""
    created = not os.path.exists(path)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        current = os.fstat(fd).st_size
        if current != num_bytes:
            check_shm_free_space(
                max(num_bytes - current, 0),
                shm_path=os.path.dirname(path),
                allocation_name="shared PLE table",
            )
            os.ftruncate(fd, num_bytes)
    except Exception:
        if created:
            os.unlink(path)
        raise
    finally:
        os.close(fd)


def _publish_shared_table(
    path: str, identity: dict[str, object], num_bytes: int, lock_fd: int
) -> None:
    """Loader: mark the table ready, then release the lock."""
    mark = Path(path + ".ready")
    tmp = mark.with_name(mark.name + ".tmp")
    tmp.write_text(json.dumps({**identity, "bytes": num_bytes}, sort_keys=True))
    os.replace(tmp, mark)
    os.close(lock_fd)


def _await_shared_table(path: str, identity: dict[str, object], num_bytes: int) -> None:
    """Wait for the loader's ready mark; fail at once if the loader died.

    This process has skipped the table's load by then, so it cannot take over:
    it fails with the reason, and its restart (or any other process's) loads
    the table, since nobody holds the lock any more.
    """
    ready = path + ".ready"
    lock_fd = os.open(path + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.monotonic() + PLE_SHARED_TABLE_READY_TIMEOUT_S
    next_log = 0.0
    try:
        while not os.path.exists(ready):
            if not _shared_table_loader_alive(lock_fd):
                if os.path.exists(ready):
                    break
                raise RuntimeError(
                    f"the process loading shared PLE table {path} exited "
                    "before marking it ready; start this process again and "
                    "the first to start loads it"
                )
            now = time.monotonic()
            if now > deadline:
                raise RuntimeError(
                    f"shared PLE table {path} was not marked ready within "
                    f"{PLE_SHARED_TABLE_READY_TIMEOUT_S} s by the process "
                    "still loading it"
                )
            if now >= next_log:
                logger.info(
                    "Waiting for %s to be marked ready by the process loading it",
                    path,
                )
                next_log = now + 30
            time.sleep(PLE_SHARED_TABLE_POLL_S)
    finally:
        os.close(lock_fd)
    _check_shared_table_ready(path, identity, num_bytes)


class Qwen4ExpPLEFileSharedHostEmbedding(Qwen4ExpPLEPinnedHostEmbedding):
    """PLE table in a named host file shared by independent engine processes.

    No process group: each ETP shard of each layer is one file, named by its
    identity (``shared_ple_table_identity``). The first process to take the
    file's lock loads the shard and publishes a ``.ready`` mark after its
    weights are loaded; every other process maps the same file, skips the
    load and waits for the mark in ``process_weights_after_loading``. A load
    interrupted by the loader's exit is taken over by the next process to
    start. Lookups are the pinned backend's. The file outlives the processes
    (see ``EngramConfig.shared_host_table_dir``).
    """

    def __init__(self, *args, shared_table: SharedPLETable, **kwargs) -> None:
        self._shared_table = shared_table
        self._table_path = ""
        self._table_identity: dict[str, object] = {}
        self._table_bytes = 0
        self._table_lock_fd: int | None = None
        self._shared_owner: np.ndarray | None = None
        self._shared_mapping: mmap.mmap | None = None
        super().__init__(*args, **kwargs)

    @property
    def _table_is_writer(self) -> bool:
        return self._table_lock_fd is not None

    @property
    def ready_mark(self) -> str:
        return self._table_path + ".ready"

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._table_bytes:
            # Only the table (``weight``, allocated first) is shared; NVFP4's
            # block scales stay per process, like the global scale.
            return super().allocate_embedding_weight(
                num_embeddings, embedding_dim, dtype
            )
        num_bytes = num_embeddings * embedding_dim * dtype.itemsize
        self._table_bytes = num_bytes
        self._table_identity = shared_ple_table_identity(
            self._shared_table.table_id,
            (num_embeddings, embedding_dim),
            dtype,
            etp_rank=self.tp_rank,
            etp_size=self.tp_size,
            vocab_range=(
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
            ),
        )
        self._table_path = shared_ple_table_path(
            self._shared_table.directory, self._table_identity
        )
        self._table_lock_fd = _claim_shared_table(
            self._table_path, self._table_identity, num_bytes
        )
        flat, owner, mapping = _map_shared_ple_file(self._table_path, num_bytes)
        self._shared_owner, self._shared_mapping = owner, mapping
        logger.info(
            "PLE table shard %d/%d %s %s (%.1f GiB); this process %s it",
            self.tp_rank,
            self.tp_size,
            "created at" if self._table_is_writer else "mapped from",
            self._table_path,
            num_bytes / (1 << 30),
            "loads" if self._table_is_writer else "reads",
        )
        return flat.view(dtype).view(num_embeddings, embedding_dim)

    def weight_loader(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        checkpoint_start: int | None = None,
    ) -> None:
        # Only the table is shared; the scales are every process's own.
        if param is not self.weight or self._table_is_writer:
            super().weight_loader(param, loaded_weight, checkpoint_start)

    def finish_shared_table(self) -> None:
        """Loader: publish the ready mark. Others: wait for it."""
        if self._table_lock_fd is not None:
            lock_fd, self._table_lock_fd = self._table_lock_fd, None
            _publish_shared_table(
                self._table_path, self._table_identity, self._table_bytes, lock_fd
            )
            logger.info("PLE table %s loaded and marked ready", self._table_path)
            return
        _await_shared_table(self._table_path, self._table_identity, self._table_bytes)


class Qwen4ExpPLESharedHostEmbedding(Qwen4ExpPLEPinnedHostEmbedding):
    """PLE table in one pinned host mapping shared by co-located DP replicas.

    The pinned-host table is tens of GiB (48 GiB for Qwen3.8-Flash-Next), so
    two single-card replicas on one node pinning their own copies exceed the
    node's memory. With ``EngramConfig.dp_shared_memory`` the node-local Engram
    DP group maps one ``/dev/shm`` file: rank 0 creates and loads it, every
    rank maps it ``MAP_SHARED`` and registers it with CUDA for UVA lookups, and
    the group's CPU collectives fence creation, mapping and loading. Lookups
    are unchanged from the pinned backend. Mirrors ``DPSharedEngramStorage``
    in the DeepSeek V4.1 Engram implementation.
    """

    def __init__(self, *args, **kwargs) -> None:
        group = get_engram_dp_group()
        if group is None:
            raise RuntimeError(
                "Qwen4ExpPLESharedHostEmbedding requires a node-local Engram DP group"
            )
        self._shared_group = group
        self._shared_owner: np.ndarray | None = None
        self._shared_mapping: mmap.mmap | None = None
        super().__init__(*args, **kwargs)

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Map and register one physical PLE table across the DP group."""
        if self._shared_owner is not None:
            # Only the table (``weight``, allocated first) is shared; NVFP4's
            # block scales stay per rank, like the global scale.
            return super().allocate_embedding_weight(
                num_embeddings, embedding_dim, dtype
            )
        num_bytes = num_embeddings * embedding_dim * dtype.itemsize
        group = self._shared_group
        with ExitStack() as stack:
            path = error = None
            if group.rank_in_group == 0:
                try:
                    check_shm_free_space(num_bytes, allocation_name="shared PLE table")
                    backing_file = stack.enter_context(
                        tempfile.NamedTemporaryFile(prefix="vllm_ple_", dir=SHM_PATH)
                    )
                    backing_file.truncate(num_bytes)
                    path = backing_file.name
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            path, error = group.broadcast_object((path, error))
            if error is not None:
                raise RuntimeError(
                    "shared PLE table creation failed on EDP rank 0: " + error
                )
            mapping = owner = flat = None
            try:
                flat, owner, mapping = _map_shared_ple_file(path, num_bytes)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            # Also fences peer mappings before the leader unlinks the file.
            errors: list[str | None] = [None] * group.world_size
            torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
            failures = "; ".join(
                f"EDP rank {rank}: {err}"
                for rank, err in enumerate(errors)
                if err is not None
            )
            if failures:
                del flat, owner, mapping
                raise RuntimeError(
                    "shared PLE table initialization failed: " + failures
                )
            assert flat is not None and owner is not None
        self._shared_owner = owner
        self._shared_mapping = mapping
        logger.info_once(
            "PLE table shared across %d co-located DP replicas (%.1f GiB in %s)",
            group.world_size,
            num_bytes / (1 << 30),
            SHM_PATH,
        )
        return flat.view(dtype).view(num_embeddings, embedding_dim)

    def weight_loader(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        checkpoint_start: int | None = None,
    ) -> None:
        # Only the table is shared; the per-tensor scale is every rank's own.
        if param is not self.weight:
            super().weight_loader(param, loaded_weight, checkpoint_start)
            return
        if self._shared_group.rank_in_group == 0:
            super().weight_loader(param, loaded_weight, checkpoint_start)
        # Read order may differ across ranks. Equal load counts ensure all shared
        # weights are ready after the last weight-loader call returns.
        torch.distributed.barrier(group=self._shared_group.cpu_group)
