# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any, Generator, List, Optional, Union, cast
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.gpu_connectors import (
    GPUConnectorInterface,
    VLLMPagedMemGPUConnectorV2,
)
from lmcache.v1.gpu_connector.utils import (
    DiscoverableKVCache,
    LayoutHints,
    _get_head_size_view,
    _split_token2d_kv,
    get_block_size,
    get_dtype,
    get_head_size,
    get_hidden_dim_size,
    get_num_blocks,
    get_num_heads,
    get_num_layers,
    get_page_buffer_size,
    is_mla,
    normalize_kv_and_discover_format,
)
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.platform.musa.native_kv_transfer import (
    try_native_from_gpu,
    try_native_to_gpu,
)
import lmcache.c_ops as lmc_ops

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)

_SUPPORTED_MUSA_KV_FORMATS = (
    lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
    lmc_ops.EngineKVFormat.NL_X_NB_BS_HS,
)

ALLOWED_FORMAT_TRANSITIONS = {
    (None, MemoryFormat.KV_MLA_FMT),
    (MemoryFormat.KV_MLA_FMT, MemoryFormat.KV_MLA_FMT),
    (MemoryFormat.KV_T2D, MemoryFormat.KV_MLA_FMT),
}


class VLLMPagedMemMUSAConnectorV2(VLLMPagedMemGPUConnectorV2):
    """Non-layerwise paged KV connector for MUSA devices.

    Follows the same contract as VLLMPagedMemXPUConnectorV2: pure torch ops
    (index_copy_ / index_select) with ``torch.musa`` stream and sync APIs.

    Supported paged KV cache layouts:
      - Non-MLA vLLM flash-attention layout:
        ``NL x [2, NB, BS, NH, HS]`` with LMCache ``KV_2LTD`` memory shaped
        ``[2, NL, T, NH * HS]``.
      - MLA vLLM layout:
        ``NL x [NB, BS, HS]`` with LMCache ``KV_MLA_FMT`` memory shaped
        ``[1, NL, T, HS]``.

    Other vLLM layouts, including flash-infer, HND, cross-layer, connector
    v3, and MP GPU-transfer kernel layouts, are not implemented by this
    connector.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        **kwargs: Any,
    ) -> None:
        self._attributes_initialized = False
        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.use_gpu = use_gpu

    @classmethod
    def from_metadata(
        cls,
        metadata: "LMCacheMetadata",
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemMUSAConnectorV2":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional hints about KV cache layout from the
                serving engine.

        Returns:
            A new instance of VLLMPagedMemMUSAConnectorV2.
        """
        return cls(use_gpu=use_gpu)

    def to_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: Any
    ) -> None:
        """Store KV data from a memory object into MUSA paged KV caches.

        Args:
            memory_obj: The memory object containing KV data.
            start: Starting index in the token sequence.
            end: Ending index in the token sequence.

        Keyword Args:
            kvcaches: Nested tuple of K/V tensors for the whole sequence.
            slot_mapping: Full slot mapping tensor.

        Raises:
            ValueError: If slot_mapping is missing from kwargs.
            AssertionError: If memory_obj has no tensor.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self._initialize_attributes(self.kvcaches)
        self._validate_memory_format(memory_obj)
        self._validate_supported_kv_format()

        vllm_cached = kwargs.get("vllm_cached_tokens", 0)
        skip_prefix_n_tokens = min(end - start, max(0, vllm_cached - start))
        transfer_start = start + skip_prefix_n_tokens
        if transfer_start >= end:
            return
        if try_native_to_gpu(
            use_mla=self.use_mla,
            memory_tensor=memory_obj.tensor,
            kvcaches=self.kvcaches,
            slot_mapping=slot_mapping,
            start=start,
            end=end,
            skip_prefix_n_tokens=skip_prefix_n_tokens,
            block_size=self.block_size,
            num_heads=self.num_heads,
            head_size=self.head_size,
        ):
            return

        slices = slot_mapping[transfer_start:end].to(
            device=self.device, dtype=torch.long, non_blocking=True
        )

        if self.use_mla:
            tmp = memory_obj.tensor[0].to(self.device, non_blocking=True)
            total_blocks = self.num_blocks * self.block_size
            for i, kvcache in enumerate(self.kvcaches):
                kvcache.view(total_blocks, self.head_size).index_copy_(
                    0, slices, tmp[i, skip_prefix_n_tokens:]
                )
        else:
            tmp_k = memory_obj.tensor[0].to(self.device, non_blocking=True)
            tmp_v = memory_obj.tensor[1].to(self.device, non_blocking=True)
            total_blocks = self.num_blocks * self.block_size
            d = self.num_heads * self.head_size
            for i, (kcache, vcache) in enumerate(self.kvcaches):
                kcache.view(total_blocks, d).index_copy_(
                    0, slices, tmp_k[i, skip_prefix_n_tokens:]
                )
                vcache.view(total_blocks, d).index_copy_(
                    0, slices, tmp_v[i, skip_prefix_n_tokens:]
                )

    def from_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: Any
    ) -> None:
        """Load KV data from MUSA paged KV caches into a memory object.

        Args:
            memory_obj: The memory object to populate.
            start: Starting index in the token sequence.
            end: Ending index in the token sequence.

        Keyword Args:
            kvcaches: Nested tuple of K/V tensors for the whole sequence.
            slot_mapping: Full slot mapping tensor.

        Raises:
            ValueError: If slot_mapping is missing from kwargs.
            AssertionError: If memory_obj has no tensor.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self._initialize_attributes(self.kvcaches)
        self._validate_memory_format(memory_obj)
        self._validate_supported_kv_format()
        if start >= end:
            if self.use_mla:
                memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
            return
        if try_native_from_gpu(
            use_mla=self.use_mla,
            memory_tensor=memory_obj.tensor,
            kvcaches=self.kvcaches,
            slot_mapping=slot_mapping,
            start=start,
            end=end,
            block_size=self.block_size,
            num_heads=self.num_heads,
            head_size=self.head_size,
        ):
            if memory_obj.tensor.device.type != "musa" and hasattr(torch, "musa"):
                torch.musa.synchronize()  # type: ignore[attr-defined]
            if self.use_mla:
                memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
            return

        slices = slot_mapping[start:end].to(
            device=self.device, dtype=torch.long, non_blocking=True
        )

        if self.use_mla:
            total_blocks = self.num_blocks * self.block_size
            tmp = torch.stack(
                [
                    kvcache.view(total_blocks, self.head_size).index_select(0, slices)
                    for kvcache in self.kvcaches
                ]
            )
        else:
            total_blocks = self.num_blocks * self.block_size
            d = self.num_heads * self.head_size
            tmp_k = torch.stack(
                [
                    kvcache[0].view(total_blocks, d).index_select(0, slices)
                    for kvcache in self.kvcaches
                ]
            )
            tmp_v = torch.stack(
                [
                    kvcache[1].view(total_blocks, d).index_select(0, slices)
                    for kvcache in self.kvcaches
                ]
            )
            tmp = torch.stack([tmp_k, tmp_v])
        memory_obj.tensor.copy_(tmp, non_blocking=True)

        if memory_obj.tensor.device.type != "musa":
            torch.musa.synchronize()  # type: ignore[attr-defined]

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_to_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> None:
        if memory_objs is None or starts is None or ends is None:
            raise ValueError("memory_objs, starts, and ends should be provided.")

        typed_memory_objs = cast(List[MemoryObj], memory_objs)
        for memory_obj, start, end in zip(
            typed_memory_objs, starts, ends, strict=False
        ):
            self.to_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data given the number of tokens.

        Args:
            num_tokens: The number of tokens in the data.

        Returns:
            The shape of the KV cache data.

        Raises:
            RuntimeError: If attributes have not been initialized yet.
        """
        if not self._attributes_initialized:
            raise RuntimeError(
                "Cannot determine shape before attributes are initialized. "
                "Call to_gpu or from_gpu first so that _initialize_attributes "
                "can discover the KV cache layout."
            )
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])

    def _validate_memory_format(self, memory_obj: MemoryObj) -> None:
        """Validate that the memory object has the expected format.

        Args:
            memory_obj: The memory object to validate.

        Raises:
            ValueError: If the memory format does not match.
        """
        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemMUSAConnectorV2"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemMUSAConnectorV2"
                )

    def _validate_supported_kv_format(self) -> None:
        """Validate that this connector implements the discovered engine KV format.

        Raises:
            ValueError: If the active KV cache layout is unsupported by the
                non-layerwise torch-based MUSA path.
        """
        if self.engine_kv_format not in _SUPPORTED_MUSA_KV_FORMATS:
            supported = ", ".join(fmt.name for fmt in _SUPPORTED_MUSA_KV_FORMATS)
            raise ValueError(
                "VLLMPagedMemMUSAConnectorV2 supports only vLLM MUSA layouts "
                f"{supported}; got {self.engine_kv_format.name}. Unsupported "
                "layouts include flash-infer, HND, cross-layer, connector v3, "
                "and MP GPU-transfer kernel layouts."
            )

    def _initialize_attributes(self, kv_caches: List[torch.Tensor]) -> None:
        """Initialize attributes from KV caches using utils functions.

        Args:
            kv_caches: The KV cache tensors from which to discover layout.
        """
        if self._attributes_initialized:
            return

        self.device = kv_caches[0].device
        assert self.device.type == "musa", "The device should be MUSA."

        discoverable_kv_caches = cast(DiscoverableKVCache, kv_caches)
        normalized_kv_caches: DiscoverableKVCache
        self.engine_kv_format, normalized_kv_caches = normalize_kv_and_discover_format(
            discoverable_kv_caches, EngineType.VLLM
        )
        self.num_layers = get_num_layers(normalized_kv_caches, self.engine_kv_format)
        self.num_blocks = get_num_blocks(normalized_kv_caches, self.engine_kv_format)
        self.block_size = get_block_size(normalized_kv_caches, self.engine_kv_format)
        self.page_buffer_size = get_page_buffer_size(
            normalized_kv_caches, self.engine_kv_format
        )
        self.hidden_dim_size = get_hidden_dim_size(
            normalized_kv_caches, self.engine_kv_format
        )
        self.head_size = get_head_size(normalized_kv_caches, self.engine_kv_format)
        self.use_mla = is_mla(self.engine_kv_format)
        self.dtype = get_dtype(normalized_kv_caches, self.engine_kv_format)
        self.num_heads = (
            1
            if self.use_mla
            else get_num_heads(normalized_kv_caches, self.engine_kv_format)
        )

        self._attributes_initialized = True
        logger.info(
            "MUSA: attributes initialized - format: %s, "
            "num_layers: %d, num_blocks: %d, block_size: %d, "
            "page_buffer_size: %d, hidden_dim_size: %d, head_size: %d, "
            "use_mla: %s, dtype: %s, num_heads: %d",
            self.engine_kv_format,
            self.num_layers,
            self.num_blocks,
            self.block_size,
            self.page_buffer_size,
            self.hidden_dim_size,
            self.head_size,
            self.use_mla,
            self.dtype,
            self.num_heads,
        )


class VLLMPagedMemLayerwiseMUSAConnector(GPUConnectorInterface):
    """Layerwise paged KV connector for MUSA devices.

    Implements the same generator contract as VLLMPagedMemLayerwiseXPUConnector:
      - batched_to_gpu(...) yields num_layers + 2 times
      - batched_from_gpu(...) yields num_layers + 1 times

    Transfer is implemented with pure torch ops (index_copy_ / index_select).
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_musa: bool = False,
        **kwargs: Any,
    ) -> None:
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_musa = use_musa

        assert "chunk_size" in kwargs, "chunk_size should be provided."
        assert "dtype" in kwargs, "dtype should be provided."
        assert "device" in kwargs, "device should be provided."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.kvcaches: Optional[List[torch.Tensor]] = None

        self._load_stream: Optional[Any] = None
        self._store_stream: Optional[Any] = None

        self.gpu_buffer_allocator: Optional[GPUMemoryAllocator] = None

    @property
    def load_stream(self) -> Any:
        """Return the lazily-created MUSA load stream."""
        self._ensure_streams()
        return self._load_stream

    @property
    def store_stream(self) -> Any:
        """Return the lazily-created MUSA store stream."""
        self._ensure_streams()
        return self._store_stream

    @classmethod
    def from_metadata(
        cls,
        metadata: "LMCacheMetadata",
        use_musa: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemLayerwiseMUSAConnector":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model
                configuration.
            use_musa: Whether to use MUSA intermediate buffer.
            device: The device to use for the connector.

        Returns:
            A new instance of VLLMPagedMemLayerwiseMUSAConnector.
        """
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size
        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_musa=use_musa,
            chunk_size=metadata.kv_shape[2],
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def _validate_format_transition(
        self, mem: MemoryObj, target_fmt: MemoryFormat
    ) -> None:
        current_fmt = mem.metadata.fmt
        if (current_fmt, target_fmt) not in ALLOWED_FORMAT_TRANSITIONS:
            raise ValueError(
                f"Invalid KV format transition: {current_fmt} -> {target_fmt}"
            )

    def _lazy_initialize_buffer(self, kv_caches: List[torch.Tensor]) -> None:
        if self.use_musa and self.gpu_buffer_allocator is None:
            layer0 = kv_caches[0]
            derived_bytes = layer0.numel() * layer0.element_size()
            staging_bytes = int(
                os.getenv("LMCACHE_GPU_STAGING_BUFFER_BYTES", derived_bytes)
            )
            logger.info(
                "Initializing MUSA staging buffer (derived=%d bytes, final=%d bytes)",
                derived_bytes,
                staging_bytes,
            )
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                size=staging_bytes, device=self.device
            )

    def to_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: Any
    ) -> None:
        raise NotImplementedError("Layerwise uses batched_to_gpu(generator).")

    def from_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: Any
    ) -> None:
        raise NotImplementedError("Layerwise uses batched_from_gpu(generator).")

    def _batched_to_gpu_gen(
        self, starts: List[int], ends: List[int], **kwargs: Any
    ) -> Generator[Any, Any, None]:
        """Generator: CPU token2d -> (optional staging) -> MUSA paged KV."""
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        def _ensure_musa(t: torch.Tensor) -> torch.Tensor:
            if t.device != self.device:
                return t.to(self.device, non_blocking=True)
            return t

        slot_mapping_chunks = [
            slot_mapping[s:e] for s, e in zip(starts, ends, strict=False)
        ]
        if not slot_mapping_chunks:
            for _ in range(self.num_layers):
                _ = yield
            yield
            if sync:
                torch.musa.current_stream().wait_stream(self.load_stream)  # type: ignore[attr-defined]
            yield
            return

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)
        slot_mapping_full = _ensure_musa(slot_mapping_full)

        num_tokens = int(slot_mapping_full.numel())
        if num_tokens <= 0:
            for _ in range(self.num_layers):
                _ = yield
            yield
            if sync:
                torch.musa.current_stream().wait_stream(self.load_stream)  # type: ignore[attr-defined]
            yield
            return

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_musa:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            if tmp_gpu_buffer_obj is None or tmp_gpu_buffer_obj.tensor is None:
                raise RuntimeError(
                    "Failed to allocate MUSA staging buffer for batched_to_gpu."
                )

        current_stream = torch.musa.current_stream()  # type: ignore[attr-defined]

        try:
            for layer_id in range(self.num_layers):
                memory_objs_layer = yield

                if sync:
                    current_stream.wait_stream(self.load_stream)

                with torch.musa.stream(self.load_stream):  # type: ignore[attr-defined]
                    dst_layer = self.kvcaches[layer_id]
                    if self.use_mla:
                        dst_flat = cast(
                            torch.Tensor,
                            _get_head_size_view(dst_layer, use_mla=True),
                        )
                    else:
                        dst_k_flat, dst_v_flat = _get_head_size_view(  # type: ignore[misc]
                            dst_layer, use_mla=False
                        )

                    cursor = 0

                    if self.use_musa:
                        assert tmp_gpu_buffer_obj is not None
                        staged = tmp_gpu_buffer_obj.tensor
                        assert staged is not None

                        for s, e, mem in zip(
                            starts, ends, memory_objs_layer, strict=False
                        ):
                            assert mem.tensor is not None
                            n = int(e - s)
                            if n <= 0:
                                continue
                            src = _ensure_musa(mem.tensor)
                            staged[cursor : cursor + n].copy_(src, non_blocking=True)
                            cursor += n

                        sl = _ensure_musa(slot_mapping_full)

                        if self.use_mla:
                            staged_dev = _ensure_musa(staged)
                            if staged_dev.dim() == 2:
                                dst_flat.index_copy_(0, sl, staged_dev)
                            elif staged_dev.dim() == 3 and staged_dev.shape[0] == 1:
                                dst_flat.index_copy_(0, sl, staged_dev[0])
                            else:
                                raise ValueError(
                                    f"Unexpected MLA staged tensor: {staged_dev.shape}"
                                )
                        else:
                            k_tok, v_tok = _split_token2d_kv(staged)
                            k_tok = _ensure_musa(k_tok)
                            v_tok = _ensure_musa(v_tok)

                            if (
                                k_tok.dim() == 2
                                and dst_k_flat.dim() == 3
                                and k_tok.shape[1]
                                == dst_k_flat.shape[1] * dst_k_flat.shape[2]
                            ):
                                k_tok = k_tok.reshape(
                                    k_tok.shape[0],
                                    dst_k_flat.shape[1],
                                    dst_k_flat.shape[2],
                                )
                            if (
                                v_tok.dim() == 2
                                and dst_v_flat.dim() == 3
                                and v_tok.shape[1]
                                == dst_v_flat.shape[1] * dst_v_flat.shape[2]
                            ):
                                v_tok = v_tok.reshape(
                                    v_tok.shape[0],
                                    dst_v_flat.shape[1],
                                    dst_v_flat.shape[2],
                                )

                            dst_k_flat.index_copy_(0, sl, k_tok)
                            dst_v_flat.index_copy_(0, sl, v_tok)

                    else:
                        for s, e, mem in zip(
                            starts, ends, memory_objs_layer, strict=False
                        ):
                            assert mem.tensor is not None
                            n = int(e - s)
                            if n <= 0:
                                continue
                            src = _ensure_musa(mem.tensor)
                            sl = slot_mapping_full[cursor : cursor + n]
                            sl = _ensure_musa(sl)
                            cursor += n

                            if self.use_mla:
                                if src.dim() == 2:
                                    dst_flat.index_copy_(0, sl, src)
                                elif src.dim() == 3 and src.shape[0] == 1:
                                    dst_flat.index_copy_(0, sl, src[0])
                                else:
                                    raise ValueError(
                                        f"Unexpected MLA token tensor: {src.shape}"
                                    )
                            else:
                                k_tok, v_tok = _split_token2d_kv(src)
                                k_tok = _ensure_musa(k_tok)
                                v_tok = _ensure_musa(v_tok)

                                if (
                                    k_tok.dim() == 2
                                    and dst_k_flat.dim() == 3
                                    and k_tok.shape[1]
                                    == dst_k_flat.shape[1] * dst_k_flat.shape[2]
                                ):
                                    k_tok = k_tok.reshape(
                                        k_tok.shape[0],
                                        dst_k_flat.shape[1],
                                        dst_k_flat.shape[2],
                                    )
                                if (
                                    v_tok.dim() == 2
                                    and dst_v_flat.dim() == 3
                                    and v_tok.shape[1]
                                    == dst_v_flat.shape[1] * dst_v_flat.shape[2]
                                ):
                                    v_tok = v_tok.reshape(
                                        v_tok.shape[0],
                                        dst_v_flat.shape[1],
                                        dst_v_flat.shape[2],
                                    )

                                dst_k_flat.index_copy_(0, sl, k_tok)
                                dst_v_flat.index_copy_(0, sl, v_tok)

            yield

            if sync:
                current_stream.wait_stream(self.load_stream)
        finally:
            if tmp_gpu_buffer_obj is not None:
                tmp_gpu_buffer_obj.ref_count_down()

        yield

    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs: Any,
    ) -> Generator[Any, Any, None]:
        """Generator: MUSA paged KV -> CPU token2d (per layer)."""
        typed_memory_objs = cast(List[List[MemoryObj]], memory_objs)
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        current_stream = torch.musa.current_stream()  # type: ignore[attr-defined]

        slot_mapping_on_device = slot_mapping.to(self.device)

        for layer_id in range(self.num_layers):
            mem_layer = typed_memory_objs[layer_id]

            with torch.musa.stream(self.store_stream):  # type: ignore[attr-defined]
                self.store_stream.wait_stream(current_stream)

                src_layer = self.kvcaches[layer_id]

                if self.use_mla:
                    src_flat = cast(
                        torch.Tensor,
                        _get_head_size_view(src_layer, use_mla=True),
                    )
                    for s, e, mem in zip(starts, ends, mem_layer, strict=False):
                        assert mem.tensor is not None
                        sl = slot_mapping_on_device[s:e]
                        gathered = src_flat.index_select(0, sl)
                        mem.tensor.copy_(
                            gathered.to(mem.tensor.device),
                            non_blocking=True,
                        )

                    target_fmt = MemoryFormat.KV_MLA_FMT
                    for mem in mem_layer:
                        self._validate_format_transition(mem, target_fmt)
                        mem.metadata.fmt = target_fmt
                else:
                    src_k_flat, src_v_flat = _get_head_size_view(
                        src_layer, use_mla=False
                    )
                    for s, e, mem in zip(starts, ends, mem_layer, strict=False):
                        assert mem.tensor is not None
                        sl = slot_mapping_on_device[s:e]
                        k = src_k_flat.index_select(0, sl)
                        v = src_v_flat.index_select(0, sl)

                        if mem.tensor.shape[0] == 2:
                            mem.tensor[0].copy_(
                                k.to(mem.tensor.device), non_blocking=True
                            )
                            mem.tensor[1].copy_(
                                v.to(mem.tensor.device), non_blocking=True
                            )
                        elif mem.tensor.dim() >= 2 and mem.tensor.shape[1] == 2:
                            mem.tensor[:, 0].copy_(
                                k.to(mem.tensor.device), non_blocking=True
                            )
                            mem.tensor[:, 1].copy_(
                                v.to(mem.tensor.device), non_blocking=True
                            )
                        else:
                            raise ValueError(
                                f"Unrecognized KV tensor layout: {mem.tensor.shape}"
                            )

            if sync:
                self.store_stream.synchronize()
            yield

        yield

    def batched_to_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Generator[Any, Any, None]:
        return self._batched_to_gpu_gen(starts=starts or [], ends=ends or [], **kwargs)

    def _ensure_streams(self) -> None:
        """Lazily create MUSA streams on first transfer."""
        if self._load_stream is None:
            self._load_stream = torch.musa.Stream()  # type: ignore[attr-defined]
            self._store_stream = torch.musa.Stream()  # type: ignore[attr-defined]

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data for a single layer.

        Args:
            num_tokens: The number of tokens in the data.

        Returns:
            The shape of the KV cache data for one layer.
        """
        if self.use_mla:
            return torch.Size([num_tokens, self.hidden_dim_size])
        return torch.Size([num_tokens, 2, self.hidden_dim_size])
