# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Qwen4Exp n-gram embedding storage with device and pinned-host backends.

Both the NVIDIA and AMD Qwen4Exp implementations use these classes so the (large)
n-gram embedding table can be kept in pinned host memory and looked up through
Unified Virtual Addressing on any CUDA-alike platform.
"""

import hashlib
import mmap
import os
import tempfile
import threading
import time
import weakref
from abc import ABC, abstractmethod
from contextlib import ExitStack
from pathlib import Path
from typing import ClassVar

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
        self._row_bytes = self.embedding_dim * self.weight.element_size()
        self._block_d = triton.next_power_of_2(self._row_bytes)
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
        """Look up local ETP rows while preserving the weight storage dtype."""
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if output is None:
            output = torch.empty(
                expected_shape,
                dtype=self.weight.dtype,
                device=input_ids.device,
            )
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != self.weight.dtype
            or output.device != input_ids.device
        ):
            raise ValueError(
                "PLE prefetch output must match the input shape, weight dtype, "
                "and input device"
            )

        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                self._uva_weight,
                flat_ids,
                output,
                self._row_bytes,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                BLOCK_D=self._block_d,
            )
        return output

    def sync_lookup(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Synchronous UVA lookup for platforms without prefetch wiring."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        embeddings = self._lookup(gathered_ids)
        embeddings = self._reduce_etp_embeddings(embeddings)
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
    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
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
                        dtype=self.weight.dtype,
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
        """Join the side stream, reduce ETP shards, and select local rows."""
        prefetch_stream = self._prefetch_stream
        if prefetch_stream is None:
            raise RuntimeError("pinned PLE finalize requires a prior start_prefetch")
        torch.cuda.current_stream().wait_stream(prefetch_stream)
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
        buffer = self._prefetch_buffer
        if buffer is None:
            raise RuntimeError("pinned PLE lookup requires a prior start_prefetch")
        output = buffer.new_empty((hidden_states.shape[0], self._output_dim))
        self._finalize_prefetch(buffer, output)
        return output


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


def shared_ple_table_path(
    directory: str,
    model: str,
    revision: str | None,
    prefix: str,
    shape: tuple[int, int],
    dtype: torch.dtype,
) -> str:
    """The file independent processes of one model share for one PLE table."""
    key = hashlib.sha256(
        f"{model}|{revision}|{prefix}|{shape[0]}x{shape[1]}|{dtype}".encode()
    ).hexdigest()[:24]
    return os.path.join(directory, f"vllm_ple_{key}")


# How long a reader waits for the writer to finish loading the shared table.
PLE_SHARED_TABLE_READY_TIMEOUT_S = 3600


class Qwen4ExpPLEFileSharedHostEmbedding(Qwen4ExpPLEPinnedHostEmbedding):
    """PLE table in a named host file shared by independent engine processes.

    No process group: the first process to create the file (O_EXCL) is the
    writer — it loads the table and writes a `.ready` mark after its weights
    are loaded; every other process maps the same file, skips the load and
    waits for the mark in `process_weights_after_loading`. Lookups are the
    pinned backend's. The file outlives the processes (see
    `EngramConfig.shared_host_table_dir`).
    """

    def __init__(self, *args, table_path: str, **kwargs) -> None:
        self._table_path = table_path
        self._table_is_writer = False
        self._table_bytes = 0
        self._shared_owner: np.ndarray | None = None
        self._shared_mapping: mmap.mmap | None = None
        super().__init__(*args, **kwargs)

    @property
    def ready_mark(self) -> str:
        return self._table_path + ".ready"

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        num_bytes = num_embeddings * embedding_dim * dtype.itemsize
        self._table_bytes = num_bytes
        try:
            fd = os.open(self._table_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                check_shm_free_space(
                    num_bytes,
                    shm_path=os.path.dirname(self._table_path),
                    allocation_name="shared PLE table",
                )
                os.ftruncate(fd, num_bytes)
            except Exception:
                os.close(fd)
                os.unlink(self._table_path)
                raise
            os.close(fd)
            self._table_is_writer = True
        except FileExistsError:
            size = os.stat(self._table_path).st_size
            if size != num_bytes:
                raise RuntimeError(
                    f"shared PLE table {self._table_path} holds {size} bytes, "
                    f"this model needs {num_bytes}: remove the stale file (and "
                    "its .ready mark) and start again"
                ) from None
        flat, owner, mapping = _map_shared_ple_file(self._table_path, num_bytes)
        self._shared_owner, self._shared_mapping = owner, mapping
        logger.info(
            "PLE table %s %s (%.1f GiB); this process %s it",
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
        # Only the table is shared; the per-tensor scale is every process's own.
        if param is not self.weight or self._table_is_writer:
            super().weight_loader(param, loaded_weight, checkpoint_start)

    def finish_shared_table(self) -> None:
        """Writer: publish the ready mark. Reader: wait for it."""
        mark = Path(self.ready_mark)
        if self._table_is_writer:
            tmp = mark.with_name(mark.name + ".tmp")
            tmp.write_text(str(self._table_bytes))
            os.replace(tmp, mark)
            logger.info("PLE table %s loaded and marked ready", self._table_path)
            return
        deadline = time.monotonic() + PLE_SHARED_TABLE_READY_TIMEOUT_S
        waited = 0
        while not mark.exists():
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"shared PLE table {self._table_path} was never marked ready; "
                    "the process loading it may have died — remove the file and "
                    "start again"
                )
            if waited % 30 == 0:
                logger.info(
                    "Waiting for %s to be marked ready by the process loading it",
                    self._table_path,
                )
            time.sleep(1)
            waited += 1


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
