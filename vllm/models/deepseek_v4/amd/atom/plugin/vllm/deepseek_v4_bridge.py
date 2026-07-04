from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from aiter import dtypes

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME = "model.layers.0.atom_deepseek_v4_proxy"
ATOM_DEEPSEEK_V4_BLOCK_SIZE = 128


def _aligned_index_dim(index_head_dim: int) -> int:
    # extra 4 Bytes for scale.
    # 16 Bytes aligned.
    return ((index_head_dim + 4 + 15) // 16) * 16


def _layer_counts(hf_config) -> tuple[list[int], int, int, int]:
    ratios = [int(r) for r in (getattr(hf_config, "compress_ratios", []) or [])]
    csa = sum(1 for r in ratios if r == 4)
    hca = sum(1 for r in ratios if r == 128)
    dense = sum(1 for r in ratios if r == 0)
    return ratios, dense, csa, hca


def _classical_block_bytes(hf_config) -> int:
    ratios, _dense, csa_layers, hca_layers = _layer_counts(hf_config)
    head_dim = int(getattr(hf_config, "head_dim", 512))
    index_head_dim = int(getattr(hf_config, "index_head_dim", 128))
    index_dim = _aligned_index_dim(index_head_dim)
    csa_main = (ATOM_DEEPSEEK_V4_BLOCK_SIZE // 4) * head_dim * 2
    csa_index = (ATOM_DEEPSEEK_V4_BLOCK_SIZE // 4) * index_dim
    hca_main = (ATOM_DEEPSEEK_V4_BLOCK_SIZE // 128) * head_dim * 2
    return csa_layers * (csa_main + csa_index) + hca_layers * hca_main


def _proxy_page_bytes(vllm_config) -> int:
    hf = vllm_config.model_config.hf_config
    ratios, _dense, _csa, _hca = _layer_counts(hf)
    head_dim = int(getattr(hf, "head_dim", 512))
    win = int(getattr(hf, "sliding_window", 128))
    max_num_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 1))
    max_model_len = int(vllm_config.model_config.max_model_len)
    min_blocks = max(
        1,
        (max_model_len + ATOM_DEEPSEEK_V4_BLOCK_SIZE - 1)
        // ATOM_DEEPSEEK_V4_BLOCK_SIZE,
    )
    swa_bytes = len(ratios) * max_num_seqs * win * head_dim * 2
    # Amortize fixed SWA state into every vLLM page so total proxy storage can
    # hold both SWA prefix and classical paged KV while remaining a vLLM KV cache.
    return _classical_block_bytes(hf) + ((swa_bytes + min_blocks - 1) // min_blocks)


def slice_deepseek_v4_proxy_cache_views(
    proxy_kv_cache: torch.Tensor,
    *,
    compress_ratios: list[int] | tuple[int, ...] | None = None,
    csa_layer_count: int | None = None,
    hca_layer_count: int | None = None,
    num_slots: int = 1,
    window_size: int = 128,
    head_dim: int = 512,
    index_head_dim: int = 128,
) -> dict[str, list[torch.Tensor]]:
    """Carve ATOM V4 KV views from vLLM-managed proxy KV storage.

    Storage layout is linear over the raw proxy bytes:
      per layer: SWA prefix [num_slots*window, D] BF16, then optional classical
      tail [num_blocks*k, D] BF16 for CSA/HCA; CSA indexer blocks are stored
      after each CSA main tail as FP8 bytes.
    """
    if compress_ratios is None:
        assert csa_layer_count is not None and hca_layer_count is not None
        compress_ratios = [4] * csa_layer_count + [128] * hca_layer_count
    ratios = [int(r) for r in compress_ratios]
    index_dim = _aligned_index_dim(index_head_dim)
    physical = proxy_kv_cache.permute(1, 0, 2, 3, 4)
    if not physical.is_contiguous():
        raise ValueError("DeepSeek V4 proxy cache must be block-major contiguous")
    num_blocks = int(physical.shape[0])
    raw = physical.reshape(-1)
    offset = 0
    unified: list[torch.Tensor] = []
    swa: list[torch.Tensor] = []
    csa_main: list[torch.Tensor] = []
    csa_indexer: list[torch.Tensor] = []
    hca_main: list[torch.Tensor] = []

    def take_bytes(n: int) -> torch.Tensor:
        nonlocal offset
        if offset + n > raw.numel():
            raise ValueError(
                f"DeepSeek V4 proxy cache too small: need {offset+n}, have {raw.numel()}"
            )
        out = raw[offset : offset + n]
        offset += n
        return out

    for ratio in ratios:
        swa_bytes = num_slots * window_size * head_dim * 2
        layer_start = offset
        swa_view = (
            take_bytes(swa_bytes)
            .view(torch.bfloat16)
            .view(num_slots, window_size, head_dim)
        )
        swa.append(swa_view)
        if ratio == 4:
            k = ATOM_DEEPSEEK_V4_BLOCK_SIZE // 4
            main_bytes = num_blocks * k * head_dim * 2
            main = (
                take_bytes(main_bytes)
                .view(torch.bfloat16)
                .as_strided(
                    size=(num_blocks, k, head_dim), stride=(k * head_dim, head_dim, 1)
                )
            )
            unified_bytes = raw[layer_start:offset]
            unified.append(
                unified_bytes.view(torch.bfloat16).view(
                    num_slots * window_size + num_blocks * k, head_dim
                )
            )
            idx = (
                take_bytes(num_blocks * k * index_dim)
                .view(dtypes.fp8)
                .as_strided(
                    size=(num_blocks, k, index_dim),
                    stride=(k * index_dim, index_dim, 1),
                )
            )
            csa_main.append(main)
            csa_indexer.append(idx)
        elif ratio == 128:
            k = ATOM_DEEPSEEK_V4_BLOCK_SIZE // 128
            main_bytes = num_blocks * k * head_dim * 2
            main = (
                take_bytes(main_bytes)
                .view(torch.bfloat16)
                .as_strided(
                    size=(num_blocks, k, head_dim), stride=(k * head_dim, head_dim, 1)
                )
            )
            unified_bytes = raw[layer_start:offset]
            unified.append(
                unified_bytes.view(torch.bfloat16).view(
                    num_slots * window_size + num_blocks * k, head_dim
                )
            )
            hca_main.append(main)
        else:
            unified.append(swa_view.view(num_slots * window_size, head_dim))

    return {
        "unified": unified,
        "swa": swa,
        "csa_main": csa_main,
        "csa_indexer": csa_indexer,
        "hca_main": hca_main,
    }


class AtomDeepseekV4ProxyMetadataBuilder(AttentionMetadataBuilder):
    # Decode (query_len == 1) is the only shape we capture in a full CUDA/HIP
    # graph: it has a fixed per-step kernel grid and the per-fwd index/indptr/
    # slot/compress-plan tensors are staged into persistent fixed-address
    # buffers here in build() (outside the captured region) so replay re-reads
    # the same addresses. Prefill/mixed batches stay eager (FULL_DECODE_ONLY).
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.device = device
        # Pure decodes (query_len == 1) get pulled to the front of the batch so
        # vLLM can classify a uniform-decode batch and dispatch the captured
        # decode graph. The plugin reads the (reordered) CommonAttentionMetadata
        # transparently; the per-request state slot is keyed on the first block
        # id, so it is invariant to reordering.
        self.reorder_batch_threshold = 1

    def build(
        self, common_prefix_len: int, common_attn_metadata, fast_build: bool = False
    ):
        if common_prefix_len:
            raise ValueError(
                "ATOM DeepSeek V4 proxy does not support cascade attention"
            )
        return self._build_and_attach_atom_v4_md(common_attn_metadata, capturing=False)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        # vLLM builds the metadata for a synthetic uniform-decode batch here,
        # OUTSIDE the captured region, then captures the model forward. Stage
        # into the persistent decode buffers with arange slots (the dummy
        # batch's NULL block ids must not pollute the real slot allocator).
        return self._build_and_attach_atom_v4_md(common_attn_metadata, capturing=True)

    def _build_and_attach_atom_v4_md(self, common_attn_metadata, *, capturing):
        """Build the ATOM V4 attention metadata OUTSIDE the captured graph and
        attach it to the vLLM ``CommonAttentionMetadata`` the model forward reads.

        vLLM calls ``builder.build()`` / ``build_for_cudagraph_capture()`` once
        per step, before ``set_forward_context`` + the (possibly
        CUDA/HIP-graph-wrapped) model forward. Building here -- rather than
        inside the forward -- is what makes a captured decode graph correct:
        for decode this refreshes the per-fwd index/indptr/slot/compress-plan
        tensors *in place* in persistent fixed-address buffers (allocated at
        cache-bind time), so the captured kernels replay against stable
        addresses. The per-request selective state reset also runs here
        (outside any capture). Prefill stays on the eager fresh-tensor path and
        is never captured.

        Returns the same ``common_attn_metadata`` (now carrying ``atom_v4_md``)
        so it flows through vLLM's per-layer attn-metadata dict to the forward,
        which consumes the prebuilt metadata instead of rebuilding it.
        """
        if common_attn_metadata is None:
            return common_attn_metadata
        sfc = self.vllm_config.compilation_config.static_forward_context
        proxy = sfc.get(ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME)
        model = getattr(proxy, "_atom_v4_model", None) if proxy is not None else None
        meta_params = getattr(model, "_atom_v4_meta_params", None)
        if model is None or meta_params is None:
            # Pre-bind (profiling / first warmup forward, before the proxy cache is
            # bound): leave common untouched. The forward detects the missing
            # atom_v4_md and falls back to an inline eager build (force_dummy).
            return common_attn_metadata
        slot_allocator = (
            None if capturing else getattr(model, "_atom_v4_slot_allocator", None)
        )
        decode_bufs = getattr(model, "_atom_v4_decode_bufs", None)
        # Batch-ordered req_ids exposed by the ATOM vLLM patch for this step;
        # used as the host-resident state-slot key (no block-table D2H). None
        # when the patch isn't applied (standalone/tests) -> build falls back.
        req_ids = None
        if not capturing:
            try:
                from vllm.models.deepseek_v4.amd.atom.plugin.vllm.req_id_passthrough_patch import (
                    get_current_req_ids,
                )

                req_ids = get_current_req_ids()
            except Exception:
                req_ids = None
        md = build_atom_v4_attention_metadata(
            common_attn_metadata,
            meta_params=meta_params,
            slot_allocator=slot_allocator,
            decode_bufs=decode_bufs,
            capturing=capturing,
            req_ids=req_ids,
        )
        # Native ATOM enables V4 compressor side-stream launches only while the
        # forward is being captured into a HIP/CUDA graph. vLLM builds this metadata
        # on the capture path, so carry the signal into ATOM's forward context.
        md.in_hipgraph = bool(capturing)
        # Selective per-slot reset OUTSIDE the captured region. For decode this
        # is empty (no fresh slots are bound mid-generation); it fires for the
        # prefill chunk that first allocates a request's slot, which is eager.
        reset_slots = getattr(md, "reset_slots", None)
        if reset_slots:
            reset_deepseek_v4_state_slots(model, reset_slots)
        common_attn_metadata.atom_v4_md = md
        return common_attn_metadata


class AtomDeepseekV4ProxyBackend(AttentionBackend):
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name() -> str:
        return "ATOM_DEEPSEEK_V4_PROXY"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [ATOM_DEEPSEEK_V4_BLOCK_SIZE]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return ATOM_DEEPSEEK_V4_BLOCK_SIZE

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        return (
            (1, 0, 2, 3, 4) if not include_num_layers_dimension else (1, 0, 2, 3, 4, 5)
        )

    @staticmethod
    def get_impl_cls():
        return nn.Identity

    @staticmethod
    def get_builder_cls():
        return AtomDeepseekV4ProxyMetadataBuilder

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class AtomDeepseekV4ProxyAttention(nn.Module, AttentionLayerBase):
    def __init__(self, prefix: str = ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME):
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        self.impl = nn.Identity()

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AtomDeepseekV4ProxyBackend

    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec:
        page_bytes = _proxy_page_bytes(vllm_config)
        head_size = (page_bytes + 2 * ATOM_DEEPSEEK_V4_BLOCK_SIZE - 1) // (
            2 * ATOM_DEEPSEEK_V4_BLOCK_SIZE
        )
        return FullAttentionSpec(
            block_size=ATOM_DEEPSEEK_V4_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=head_size,
            dtype=torch.uint8,
        )


def register_deepseek_v4_proxy_layer(vllm_config) -> AtomDeepseekV4ProxyAttention:
    sfc = vllm_config.compilation_config.static_forward_context
    existing = sfc.get(ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME)
    if isinstance(existing, AtomDeepseekV4ProxyAttention):
        return existing
    if existing is not None:
        raise ValueError(f"Duplicate layer name: {ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME}")
    proxy = AtomDeepseekV4ProxyAttention()
    sfc[ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME] = proxy
    return proxy


def _bind_compressor_state(
    compressor,
    kv_cache: torch.Tensor,
    num_slots: int,
    head_dim: int,
    *,
    is_indexer: bool = False,
) -> None:
    compressor.kv_state = torch.zeros(
        (num_slots, *compressor.kv_state.shape[1:]),
        dtype=torch.float32,
        device=kv_cache.device,
    )
    compressor.score_state = torch.full(
        (num_slots, *compressor.score_state.shape[1:]),
        float("-inf"),
        dtype=torch.float32,
        device=kv_cache.device,
    )
    compressor.kv_cache = kv_cache
    if is_indexer:
        nb, k1, aligned_dim = kv_cache.shape
        block_fp32_stride = (k1 * aligned_dim) // 4
        scale_fp32_offset = (k1 * head_dim) // 4
        compressor.cache_scale = (
            kv_cache.view(torch.float32)
            .view(-1)
            .as_strided(
                size=(nb, k1),
                stride=(block_fp32_stride, 1),
                storage_offset=scale_fp32_offset,
            )
        )
    else:
        compressor.cache_scale = None


def _v4_max_spec_steps(vllm_config) -> int:
    """Speculative draft length per decode step (0 when spec decode is off).

    Sets the decode CG bucket: the per-fwd token count for a uniform decode
    batch of ``bs`` requests is ``bs * (1 + max_spec_steps)``.
    """
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None:
        return 0
    n = getattr(spec, "num_speculative_tokens", None)
    return int(n) if n else 0


class _V4DecodeMetaBuffers:
    """Persistent, fixed-address scratch for the V4 *decode* attention metadata.

    A captured decode HIP/CUDA graph re-runs recorded kernels that read these
    tensors by address; the per-step ``build()`` refreshes their *contents* in
    place (numpy -> ``copy_to_gpu`` / ``write_v4_paged_decode_indices``) before
    replay -- mirroring native ATOM's ``forward_vars`` decode buffers. Sized
    once to the worst-case decode shape (``num_slots`` seqs,
    ``num_slots * (1 + max_spec_steps)`` tokens, ``max_committed_hca`` HCA
    entries per seq for the native ``max_model_len``). Index/indptr views are
    always sliced from the buffer base so their data pointer is stable across
    builds even as the logical length changes. Prefill never touches these.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        max_decode_tokens: int,
        window: int,
        index_topk: int,
        max_committed_hca: int,
        ratios_overlap,
        device: torch.device,
    ):
        from vllm.models.deepseek_v4.amd.atom.utils import CpuGpuBuffer

        self.device = device
        self.window = int(window)
        S = max(1, int(num_slots))
        T = max(1, int(max_decode_tokens))
        win = int(window)
        topk = int(index_topk)
        hca = int(max_committed_hca)
        self.num_slots = S
        self.max_decode_tokens = T

        def i32(*shape):
            return CpuGpuBuffer(*shape, dtype=torch.int32, device=device)

        # Per-seq scalars (sized to padded request count == num_slots).
        self.state_slot = i32(S)
        self.n_csa = i32(S)
        self.n_hca = i32(S)
        # Per-token mapping (sized to padded token count). int32: accepted by
        # torch advanced-indexing AND by the fused flydsl SWA scatter (which
        # loads batch_id as int32); matches the in-tree model_runner path.
        self.batch_id = CpuGpuBuffer(T, dtype=torch.int32, device=device)
        # Ragged cumsums (T + 1) and ragged index pools (worst-case per-token
        # slot counts): SWA = win, CSA = win + index_topk, HCA = win + hca.
        self.indptr_swa = i32(T + 1)
        self.indptr_csa = i32(T + 1)
        self.indptr_hca = i32(T + 1)
        self.idx_swa = i32(T * max(1, win))
        self.idx_csa = i32(T * max(1, win + topk))
        self.idx_hca = i32(T * max(1, win + hca))
        # Native compress-plan buffers (one pair per compress ratio present).
        # Decode worst case: each seq contributes ceil((1 + spec) / ratio)
        # compression boundaries. The write plan is a subset of the per-fwd
        # ragged tokens (a token is written iff its position falls in the per-seq
        # "last K_pool" window), so for decode it has at most `total` rows
        # (<= T == max_decode_tokens). Sizing the write buffer to T instead of
        # the prefill-style S*K_pool worst case keeps the per-step sentinel fill,
        # the H2D copy, AND the write-kernel grid (== write_plan.shape[0]) bounded
        # to the decode token count -- the prior S*K_pool sizing filled/copied an
        # almost-entirely-sentinel buffer every decode step (up to 128x for the
        # HCA ratio). CUDAGraph-safe: shape[0]==T is fixed across capture/replay.
        from vllm.models.deepseek_v4.amd.atom.model_ops.v4_kernels.compress_plan import (  # noqa: F401
            make_compress_plans as _mcp,
        )

        self.plan_buffers: dict[int, dict] = {}
        self.decode_compress_cap: dict[int, int] = {}
        spec_plus_one = max(1, T // S)
        for ratio, is_overlap in ratios_overlap:
            ratio = int(ratio)
            per_seq = (spec_plus_one + ratio - 1) // ratio
            cap = max(1, S * per_seq)
            self.plan_buffers[ratio] = {
                "compress": i32(cap, 4),
                "write": i32(max(1, T), 4),
            }
            self.decode_compress_cap[ratio] = cap

    def stage(self, buf, arr_np):
        """Copy ``arr_np`` into the head of CpuGpuBuffer ``buf`` and return the
        from-base GPU view (stable data pointer)."""
        n = int(arr_np.shape[0]) if getattr(arr_np, "ndim", 1) else 1
        assert (
            n <= buf.np.shape[0]
        ), f"V4 decode buffer too small: need {n}, have {buf.np.shape[0]}"
        if n:
            buf.np[:n] = arr_np
        return buf.copy_to_gpu(n)


def bind_deepseek_v4_proxy_cache_views(model, vllm_config) -> bool:
    sfc = vllm_config.compilation_config.static_forward_context
    proxy = sfc.get(ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME)
    if proxy is None or not isinstance(proxy, AtomDeepseekV4ProxyAttention):
        return False
    if not isinstance(proxy.kv_cache, torch.Tensor) or proxy.kv_cache.numel() == 0:
        return False
    ptr = proxy.kv_cache.untyped_storage().data_ptr()
    if getattr(model, "_atom_vllm_v4_proxy_cache_ptr", None) == ptr:
        return True
    ratios = [int(r) for r in model.args.compress_ratios]
    num_slots = max(1, int(vllm_config.scheduler_config.max_num_seqs))
    # Stash the per-request state-slot allocator + the metadata params the
    # bridge needs but cannot read from common_attn_metadata (the SWA ring pool
    # size, window, ring stride, and indexer topk). `num_slots == max_num_seqs`
    # is the actual SWA ring boundary in `unified_kv` (see slicing); the bridge
    # must use it for `swa_pages`, not the per-forward request count.
    if not hasattr(model, "_atom_v4_slot_allocator"):
        model._atom_v4_slot_allocator = _V4StateSlotAllocator(num_slots)
    model._atom_v4_meta_params = SimpleNamespace(
        num_slots=num_slots,
        window_size=int(model.args.window_size),
        # Plugin SWA ring is sized by window_size (no MTP spec steps folded in,
        # matching the slicing above), so the ring stride cs == window_size.
        cs=int(model.args.window_size),
        index_topk=int(getattr(model.args, "index_topk", 1024)),
    )
    views = slice_deepseek_v4_proxy_cache_views(
        proxy.kv_cache,
        compress_ratios=ratios,
        num_slots=num_slots,
        window_size=int(model.args.window_size),
        head_dim=int(model.args.head_dim),
        index_head_dim=int(model.args.index_head_dim),
    )
    csa_i = 0
    hca_i = 0
    for layer_id, block in enumerate(model.model.layers):
        attn = block.attn
        ratio = int(attn.compress_ratio)
        attn.unified_kv = views["unified"][layer_id]
        attn.swa_kv = views["swa"][layer_id]
        if ratio == 4:
            _bind_compressor_state(
                attn.compressor,
                views["csa_main"][csa_i],
                num_slots,
                int(model.args.head_dim),
            )
            attn.indexer.kv_cache = views["csa_indexer"][csa_i]
            attn.indexer._max_model_len_idx = max(
                1, int(vllm_config.model_config.max_model_len) // 4
            )
            _bind_compressor_state(
                attn.indexer.compressor,
                views["csa_indexer"][csa_i],
                num_slots,
                int(model.args.index_head_dim),
                is_indexer=True,
            )
            csa_i += 1
        elif ratio == 128:
            _bind_compressor_state(
                attn.compressor,
                views["hca_main"][hca_i],
                num_slots,
                int(model.args.head_dim),
            )
            hca_i += 1
    # Persistent decode-metadata buffers for the FULL decode CUDA/HIP graph.
    # Allocated once (sized to the worst-case decode shape) so build() can
    # refresh them in place each step; the captured kernels read stable
    # addresses. Eager prefill never touches them. Stash the model on the proxy
    # so the metadata builder (which only sees vllm_config) can reach it.
    proxy._atom_v4_model = model
    if not hasattr(model, "_atom_v4_decode_bufs"):
        max_spec = _v4_max_spec_steps(vllm_config)
        max_model_len = int(vllm_config.model_config.max_model_len)
        max_committed_hca = max(1, (max_model_len + 127) // 128)
        # CSA (ratio 4) compress windows overlap; HCA (ratio 128) does not.
        ratios_overlap = [(r, r == 4) for r in sorted(set(ratios)) if r > 0]
        model._atom_v4_decode_bufs = _V4DecodeMetaBuffers(
            num_slots=num_slots,
            max_decode_tokens=num_slots * (1 + max_spec),
            window=int(model.args.window_size),
            index_topk=int(getattr(model.args, "index_topk", 1024)),
            max_committed_hca=max_committed_hca,
            ratios_overlap=ratios_overlap,
            device=proxy.kv_cache.device,
        )
    model._atom_vllm_v4_proxy_cache_ptr = ptr
    return True


def reset_deepseek_v4_state_slots(model, slots) -> None:
    """Clear V4 per-request SWA + compressor state for specific state slots.

    Chunk-aware analogue of `reset_deepseek_v4_state_caches`: rather than wiping
    every slot whenever a batch happens to start at position 0, reset only the
    slots the allocator just (re)assigned to a fresh request. This preserves a
    long prompt's accumulated SWA window and compressor state across its prefill
    chunks while still guaranteeing a brand-new request (or a slot left dirty by
    a finished request or a profiling forward) starts from clean state.
    """
    if not slots:
        return
    layers = getattr(getattr(model, "model", None), "layers", [])
    if not layers:
        return
    device = None
    for block in layers:
        swa = getattr(getattr(block, "attn", None), "swa_kv", None)
        if isinstance(swa, torch.Tensor):
            device = swa.device
            break
    if device is None:
        return
    idx = torch.as_tensor(
        sorted(int(s) for s in slots), dtype=torch.long, device=device
    )
    for block in layers:
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        swa = getattr(attn, "swa_kv", None)
        if isinstance(swa, torch.Tensor):
            swa[idx] = 0
        for compressor in (
            getattr(attn, "compressor", None),
            getattr(getattr(attn, "indexer", None), "compressor", None),
        ):
            if compressor is None:
                continue
            if isinstance(getattr(compressor, "kv_state", None), torch.Tensor):
                compressor.kv_state[idx] = 0
            if isinstance(getattr(compressor, "score_state", None), torch.Tensor):
                compressor.score_state[idx] = float("-inf")


def _infer_atom_attn_state(common_attn_metadata):
    from vllm.models.deepseek_v4.amd.atom.utils.forward_context import AttnState

    if common_attn_metadata is None:
        return AttnState.PREFILL_NATIVE
    if getattr(common_attn_metadata, "max_query_len", 0) == 1:
        return AttnState.DECODE
    num_computed = getattr(common_attn_metadata, "_num_computed_tokens_cpu", None)
    if num_computed is not None and bool((num_computed > 0).any().item()):
        return AttnState.PREFILL_PREFIX
    return AttnState.PREFILL_NATIVE


def _counts_to_indptr(counts: np.ndarray) -> np.ndarray:
    out = np.zeros(len(counts) + 1, dtype=np.int32)
    out[1:] = np.cumsum(counts, dtype=np.int32)
    return out


def _make_compress_plans(
    extend_lens_cpu, context_lens_cpu, ratios, device, decode: bool
):
    total = int(extend_lens_cpu.sum())
    from vllm.models.deepseek_v4.amd.atom.model_ops.v4_kernels import make_compress_plans
    from vllm.models.deepseek_v4.amd.atom.utils import CpuGpuBuffer

    capacity = max(1, total)
    plan_buffers = {
        int(ratio): {
            "compress": CpuGpuBuffer(capacity, 4, dtype=torch.int32, device=device),
            "write": CpuGpuBuffer(capacity, 4, dtype=torch.int32, device=device),
        }
        for ratio, _ in ratios
    }
    plans = make_compress_plans(
        extend_lens_cpu,
        context_lens_cpu,
        ratios,
        plan_buffers=plan_buffers,
        decode_capacity_per_ratio=None,
    )
    # Preserve the bridge eager path's old variable-grid behavior. Native
    # make_compress_plans returns a sentinel-padded write buffer, while the
    # previous bridge helper launched update_compressor_states with exactly
    # num_write rows. Graph decode uses _make_decode_compress_plans instead.
    for plan in plans.values():
        plan.write_plan_gpu = plan.write_plan_gpu[: plan.num_write]
    return plans


class _V4StateSlotAllocator:
    """Stable per-request state-slot allocator over ``[0, num_slots)``.

    Keyed by each request's id (``req_id``), the canonical, host-resident
    request identity from vLLM's ``InputBatch``. This hands back the same state
    slot for every chunked-prefill step and every decode step of a request, so
    its SWA ring and compressor state accumulate in one place -- matching native
    ATOM's per-request cache slots.

    Keying on ``req_id`` (rather than the first KV block id, which lived on the
    GPU block table) removes the per-step D2H copy + host<->device sync that the
    block-id key required, and is immune to vLLM recycling a finished request's
    blocks to a new request within the same step.

    A slot is reported as freshly allocated (caller resets it) when it is newly
    bound to an unseen ``req_id``, or when a known ``req_id`` reappears with
    ``num_computed == 0`` -- vLLM recomputes preempted requests from scratch
    under the same id, so the slot's accumulated state must be cleared on resume.

    Slots are reclaimed lazily on exhaustion by evicting the least-recently-seen
    slot whose ``req_id`` is absent from the current step (its request finished
    or was preempted). vLLM caps concurrency at ``num_slots`` (max_num_seqs), so
    a request that is live this step never has its slot evicted.
    """

    def __init__(self, num_slots: int):
        self.num_slots = max(1, int(num_slots))
        self._key_to_slot: dict[object, int] = {}
        self._slot_to_key: list[object] = [None] * self.num_slots
        self._free: list[int] = list(range(self.num_slots - 1, -1, -1))
        self._last_seen: list[int] = [-1] * self.num_slots
        self._step = 0

    def assign(self, req_keys, num_computed):
        """Return ``(slots: np.int32[num_reqs], reset_slots: set[int])``.

        ``req_keys`` is a per-request sequence of stable, hashable keys (the
        ``req_id`` strings), aligned with the batch rows.
        """
        self._step += 1
        # Pull num_computed to a Python list in one C call (per-element
        # numpy-scalar -> int was the dominant cost of this per-decode-step
        # loop). req_keys is already a host-side list[str]. Local-bind the
        # dict/list fields too -- attribute lookups inside the bs-length loop
        # add up at large batch (profiled #1 build cost).
        keys = list(req_keys)
        nc = (
            num_computed.tolist()
            if hasattr(num_computed, "tolist")
            else list(num_computed)
        )
        n = len(keys)
        active = set(keys)
        key_to_slot = self._key_to_slot
        slot_to_key = self._slot_to_key
        last_seen = self._last_seen
        step = self._step
        slots = [0] * n
        reset: set[int] = set()
        for i in range(n):
            k = keys[i]
            slot = key_to_slot.get(k)
            if slot is None:
                slot = self._acquire(active)
                key_to_slot[k] = slot
                slot_to_key[slot] = k
                reset.add(slot)
            elif nc[i] == 0:
                # Known request recomputed from scratch (preemption resume).
                reset.add(slot)
            slots[i] = slot
            last_seen[slot] = step
        return np.asarray(slots, dtype=np.int32), reset

    def _acquire(self, active: set) -> int:
        if self._free:
            return self._free.pop()
        victim = -1
        victim_seen = None
        for s in range(self.num_slots):
            if self._slot_to_key[s] in active:
                continue
            if victim_seen is None or self._last_seen[s] < victim_seen:
                victim = s
                victim_seen = self._last_seen[s]
        if victim < 0:
            # All slots belong to requests active this step: only possible if
            # concurrency exceeds num_slots, which vLLM forbids. Fall back to
            # slot 0 rather than crash.
            victim = 0
        old = self._slot_to_key[victim]
        if old is not None:
            self._key_to_slot.pop(old, None)
        self._slot_to_key[victim] = None
        return victim


def build_atom_v4_attention_metadata(
    common_attn_metadata,
    *,
    meta_params=None,
    slot_allocator=None,
    decode_bufs=None,
    capturing=False,
    req_ids=None,
):
    """Translate a vLLM ``CommonAttentionMetadata`` into ATOM's V4
    ``AttentionMetaData``.

    When ``decode_bufs`` is provided and the batch is a pure decode, the per-fwd
    index/indptr/slot/compress-plan tensors are staged into those persistent
    fixed-address buffers (CUDA/HIP-graph replay safety) and the token count is
    padded to the captured bucket (``num_actual_tokens``) with a ``batch_id ==
    -1`` sentinel tail + repeating indptr tail for the padded slots. Otherwise
    (prefill, or decode without buffers) it falls back to fresh per-fwd tensors
    (eager-only). ``capturing`` forces ``arange`` state slots so a CUDA-graph
    capture dummy batch (whose block ids are NULL) does not pollute the real
    per-request slot allocator.

    ``req_ids`` (batch-ordered, host-resident) is the slot-allocation key,
    threaded in by the req_id passthrough patch with no device sync. The decode
    slot-assignment path requires it: if it is missing/short there (patch not
    applied or out of sync) the build raises rather than reading the device
    block table.
    """
    from vllm.models.deepseek_v4.amd.atom.utils.forward_context import AttentionMetaData

    if common_attn_metadata is None:
        return AttentionMetaData()
    state = _infer_atom_attn_state(common_attn_metadata)
    is_decode = state.value == "decode"
    device = common_attn_metadata.seq_lens.device
    num_reqs = int(common_attn_metadata.num_reqs)
    q_cpu = getattr(common_attn_metadata, "query_start_loc_cpu", None)
    if q_cpu is None:
        q_cpu = common_attn_metadata.query_start_loc.cpu()
    q_np = q_cpu[: num_reqs + 1].numpy().astype(np.int32)
    lens = np.diff(q_np).astype(np.int32)
    total = int(lens.sum())  # real tokens (CG-padded reqs contribute 0)
    # Per-seq lengths on the HOST without a device sync. This vLLM build does
    # not expose an eager `seq_lens_cpu`, so `seq_lens.cpu()` is a blocking D2H
    # that drains the prior decode step's GPU work -> a large per-step bubble.
    # Prefer, in order: a future `seq_lens_cpu`; the (deprecated but exact)
    # `_seq_lens_cpu`; vLLM's CPU-resident `seq_lens_cpu_upper_bound` (exact for
    # prefill and for every decode row outside async spec-decode, which this
    # integration does not use). Fall back to the D2H only if none exist.
    # NOTE: test each for None explicitly -- `a or b` on a multi-element tensor
    # raises "Boolean value of Tensor ... is ambiguous" (e.g. CG-capture warmup).
    # IMPORTANT: read the RAW backing attributes, never the `seq_lens_cpu`
    # property -- that property lazily does `seq_lens.to("cpu")` (a blocking
    # D2H) whenever `_seq_lens_cpu` is unset, which is exactly the bubble we are
    # removing. `_seq_lens_cpu` is the exact CPU tensor when present;
    # `seq_lens_cpu_upper_bound` is a CPU tensor that is always populated and is
    # exact for prefill and every decode row outside async spec-decode (which
    # this integration does not use). Only as a last resort do the D2H.
    seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = getattr(common_attn_metadata, "seq_lens_cpu_upper_bound", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
    seq_np = seq_lens_cpu[:num_reqs].numpy().astype(np.int32)
    batch_np = np.repeat(np.arange(num_reqs, dtype=np.int32), lens)
    md = AttentionMetaData(
        cu_seqlens_q=common_attn_metadata.query_start_loc,
        cu_seqlens_k=common_attn_metadata.query_start_loc,
        max_seqlen_q=int(common_attn_metadata.max_query_len),
        max_seqlen_k=int(common_attn_metadata.max_seq_len),
        slot_mapping=getattr(common_attn_metadata, "slot_mapping", None),
        context_lens=common_attn_metadata.seq_lens,
        block_tables=common_attn_metadata.block_table_tensor,
        state=state,
    )
    if meta_params is not None:
        md.swa_num_slots = int(meta_params.num_slots)
        md.swa_window = int(meta_params.window_size)
        md.swa_cs = int(meta_params.cs)
        md.index_topk = int(meta_params.index_topk)
    else:
        # Standalone/test fallback: per-forward request count is the ring pool,
        # default window/topk. Production always passes meta_params (bound at
        # cache-bind time) so swa_pages tracks the real max_num_seqs boundary.
        md.swa_num_slots = num_reqs
        md.swa_window = 128
        md.swa_cs = 128
        md.index_topk = 512
    # chunk_start == num_computed_tokens (== global position of each seq's first
    # token this forward); 0 for a fresh prompt / single-shot prefill.
    chunk_start_np = np.maximum(seq_np - lens, 0).astype(np.int32)
    md.chunk_start_per_seq_cpu = chunk_start_np

    decode_persistent = is_decode and decode_bufs is not None
    # Real reqs are contiguous at the front of a (reordered) decode batch; CG
    # padding appends zero-query-len reqs at the tail.
    scheduled_bs = int((lens > 0).sum()) if is_decode else num_reqs
    # T_pad: the captured per-fwd token count (== num_actual_tokens, which vLLM
    # sets to the cudagraph bucket size on capture/replay; == total in eager).
    T_pad = int(getattr(common_attn_metadata, "num_actual_tokens", total) or total)
    if T_pad < total:
        T_pad = total

    # ---- per-request state slot ----
    # Real per-request state slots are assigned only for genuine (non-capture)
    # builds that carry a live allocator and real scheduled rows. The slot key
    # is vLLM's batch-ordered req_ids (the canonical, host-resident request
    # identity), threaded in by the ATOM req_id passthrough patch with no device
    # sync (installed at register.apply_vllm_req_id_passthrough_patch).
    real_slots = not capturing and slot_allocator is not None and scheduled_bs > 0
    if real_slots and req_ids is None:
        # Patch contract violated: a real build with a live allocator must
        # receive batch-ordered req_ids. None means the passthrough patch did
        # not run (not installed / out of sync) -> fail fast rather than
        # silently degrading to the old block-id key, which needed a per-step
        # D2H sync and was not immune to vLLM recycling a finished request's
        # blocks to a new request within the same step.
        raise RuntimeError(
            "ATOM V4 decode slot assignment requires batch-ordered req_ids "
            f"from the vLLM passthrough patch (scheduled_bs={scheduled_bs}), "
            "but none were threaded in. Ensure "
            "apply_vllm_req_id_passthrough_patch() ran at model registration "
            "and is still active."
        )
    if not real_slots or len(req_ids) < scheduled_bs:
        # Capture / profiling / warmup / empty synthetic batch (patch ran but
        # there are no -- or too few -- real request ids): throwaway arange
        # slots. The batch's results are discarded, and its NULL block ids /
        # absent req ids must not pollute the real per-request slot allocator.
        slot_arr = np.arange(num_reqs, dtype=np.int32)
        reset_slots: set = set()
    else:
        slot_real, reset_slots = slot_allocator.assign(
            req_ids[:scheduled_bs], chunk_start_np[:scheduled_bs]
        )
        # Padded reqs get slot 0 (a valid slot); their tokens carry batch_id ==
        # -1 so the per-token decode kernels never read them.
        slot_arr = np.zeros(num_reqs, dtype=np.int32)
        slot_arr[:scheduled_bs] = slot_real
    md.reset_slots = reset_slots

    n_csa_cpu = (seq_np // 4).astype(np.int32)
    n_hca_cpu = (seq_np // 128).astype(np.int32)
    md.n_committed_csa_per_seq_cpu = n_csa_cpu
    md.n_committed_hca_per_seq_cpu = n_hca_cpu
    md.batch_id_per_token_cpu = batch_np
    index_topk = int(md.index_topk)

    if decode_persistent:
        bufs = decode_bufs
        md.state_slot_mapping_cpu = slot_arr
        md.state_slot_mapping = bufs.stage(bufs.state_slot, slot_arr)
        # Per-token seq map padded to T_pad with the -1 sentinel tail.
        if total:
            bufs.batch_id.np[:total] = batch_np
        if T_pad > total:
            bufs.batch_id.np[total:T_pad] = -1
        md.batch_id_per_token = bufs.batch_id.copy_to_gpu(T_pad)
        # Pad CSA committed count with index_topk (aiter top_k_per_row_decode
        # derives a per-row length from this for the whole captured grid; a
        # stale/zero value on a pad row can make that length negative -> hang).
        bufs.n_csa.np[:num_reqs] = n_csa_cpu
        if num_reqs > scheduled_bs:
            bufs.n_csa.np[scheduled_bs:num_reqs] = index_topk
        bufs.n_hca.np[:num_reqs] = n_hca_cpu
        md.n_committed_csa_per_seq = bufs.n_csa.copy_to_gpu(num_reqs)
        md.n_committed_hca_per_seq = bufs.n_hca.copy_to_gpu(num_reqs)
        md.compress_plans = _make_decode_compress_plans(
            lens[:scheduled_bs], seq_np[:scheduled_bs], bufs
        )
        positions = getattr(common_attn_metadata, "positions", None)
        if positions is None:
            positions = torch.arange(max(total, 1), dtype=torch.int64, device=device)
        # Per-token global position from chunk_start + within-seq offset (no
        # D2H; equals vLLM's `positions[:total]` for a decode batch).
        if total:
            cu_real = np.zeros(scheduled_bs + 1, dtype=np.int32)
            np.cumsum(lens[:scheduled_bs], out=cu_real[1:], dtype=np.int32)
            within = np.arange(total, dtype=np.int32) - cu_real[batch_np]
            pos_np = (chunk_start_np[batch_np] + within).astype(np.int32)
        else:
            pos_np = np.zeros(0, dtype=np.int32)
        _populate_decode_persistent(
            md,
            common_attn_metadata,
            batch_np,
            pos_np,
            bufs,
            scheduled_bs,
            total,
            T_pad,
            positions,
        )
        # Decode indexer (CUDAGraph-friendly path) reads only the per-seq
        # committed count; the prefill-only fields stay unset.
        md.indexer_meta = {
            "total_committed": 0,
            "cu_committed_gpu": None,
            "n_committed_per_seq_gpu": md.n_committed_csa_per_seq,
            "batch_id_per_token_gpu": md.batch_id_per_token,
            "seq_base_per_token_gpu": None,
            "cu_starts_gpu": None,
            "cu_ends_gpu": None,
        }
        return md

    # ---- eager path: prefill, or decode without persistent buffers ----
    md.state_slot_mapping = torch.from_numpy(slot_arr).to(device)
    md.state_slot_mapping_cpu = slot_arr
    md.batch_id_per_token = torch.from_numpy(batch_np).to(device)
    md.n_committed_csa_per_seq = torch.from_numpy(n_csa_cpu).to(device)
    md.n_committed_hca_per_seq = torch.from_numpy(n_hca_cpu).to(device)
    md.compress_plans = _make_compress_plans(
        lens, seq_np, [(4, True), (128, False)], device, is_decode
    )
    positions = getattr(common_attn_metadata, "positions", None)
    if positions is None:
        positions = torch.arange(total, dtype=torch.int64, device=device)
    pos_np = positions[:total].detach().cpu().numpy().astype(np.int32)
    if is_decode:
        _populate_decode(md, common_attn_metadata, batch_np, pos_np, positions)
    else:
        _populate_prefill(md, common_attn_metadata, batch_np, pos_np, q_np, positions)
    _populate_indexer(md, common_attn_metadata, batch_np, positions[:total], device)
    return md


def _make_decode_compress_plans(extend_lens_cpu, context_lens_cpu, bufs):
    """Decode compress plans via native ``make_compress_plans`` into the
    persistent per-ratio plan buffers (fixed capacity per ratio so capture and
    replay dispatch identically shaped compress kernels)."""
    from vllm.models.deepseek_v4.amd.atom.model_ops.v4_kernels.compress_plan import make_compress_plans

    ratios_overlap = [(int(r), int(r) == 4) for r in bufs.plan_buffers]
    return make_compress_plans(
        np.ascontiguousarray(extend_lens_cpu, dtype=np.int32),
        np.ascontiguousarray(context_lens_cpu, dtype=np.int32),
        ratios_overlap,
        plan_buffers=bufs.plan_buffers,
        decode_capacity_per_ratio=bufs.decode_compress_cap,
    )


def _populate_decode_persistent(
    md, common, batch_np, pos_np, bufs, scheduled_bs, total, T_pad, positions_gpu
):
    """Decode index/indptr build into persistent fixed-address buffers.

    Faithful port of ATOM's ``_attach_v4_paged_decode_meta`` for plugin mode:
    three ragged ``indptr`` cumsums sized to the captured (padded) token count
    with a repeating tail (kv_len == 0 for padded slots), the HCA compress tail
    scattered on CPU, then ``write_v4_paged_decode_indices`` fills the SWA / CSA
    / HCA window-prefix offsets. All index/indptr views are sliced from the
    buffer base so their data pointers are stable across builds (the captured
    decode-attention kernels read these addresses on replay).
    """
    from vllm.models.deepseek_v4.amd.atom.plugin.vllm.deepseek_v4_ops import write_v4_decode_indices_fused

    win = int(md.swa_window)
    cs = int(md.swa_cs)
    index_topk = int(md.index_topk)
    swa_pages = int(md.swa_num_slots) * cs
    md.swa_pages = swa_pages
    n_csa_cpu = md.n_committed_csa_per_seq_cpu
    n_hca_cpu = md.n_committed_hca_per_seq_cpu

    # Per-token slot counts over the real tokens [0:total].
    actual_swa = np.minimum(pos_np + 1, win).astype(np.int32)
    csa_valid_k = np.minimum(
        np.minimum((pos_np + 1) // 4, n_csa_cpu[batch_np]), index_topk
    ).astype(np.int32)
    n_h_per_token = n_hca_cpu[batch_np].astype(np.int32)

    def _indptr(counts):
        out = np.zeros(T_pad + 1, dtype=np.int32)
        out[1 : total + 1] = np.cumsum(counts, dtype=np.int32)
        if T_pad > total:
            out[total + 1 :] = out[total]
        return out

    swa_indptr = _indptr(actual_swa)
    csa_indptr = _indptr(actual_swa + csa_valid_k)
    hca_indptr = _indptr(actual_swa + n_h_per_token)
    swa_indptr_gpu = bufs.stage(bufs.indptr_swa, swa_indptr)
    csa_indptr_gpu = bufs.stage(bufs.indptr_csa, csa_indptr)
    hca_indptr_gpu = bufs.stage(bufs.indptr_hca, hca_indptr)
    hca_total = int(hca_indptr[total]) if total else 0

    # Build the whole decode index set on-GPU with one fused Triton kernel
    # writing directly into the persistent idx buffers. Each token's program
    # writes both its SWA window prefix (slice tail of SWA / CSA / HCA) and its
    # HCA compress section (slice head of HCA: `swa_pages + block_tables[seq, j]`,
    # read straight from GPU). The two segments are disjoint and together cover
    # the full HCA segment `[hca_indptr[t], hca_indptr[t+1])`, so no `-1`
    # pre-fill is needed. This replaces the prior CPU HCA-tail scatter (a
    # per-step block-table D2H + numpy repeat/cumsum/fancy-index + H2D). T ==
    # real tokens; the `-1` batch_id pad tail is skipped natively by the kernel.
    swa_indices_gpu = bufs.idx_swa.gpu
    csa_indices_gpu = bufs.idx_csa.gpu
    write_v4_decode_indices_fused(
        state_slot_per_seq=md.state_slot_mapping,
        batch_id_per_token=md.batch_id_per_token,
        positions=positions_gpu,
        swa_indptr=swa_indptr_gpu,
        csa_indptr=csa_indptr_gpu,
        hca_indptr=hca_indptr_gpu,
        swa_indices=swa_indices_gpu,
        csa_indices=csa_indices_gpu,
        hca_indices=bufs.idx_hca.gpu,
        n_committed_hca_per_seq=md.n_committed_hca_per_seq,
        block_tables=common.block_table_tensor,
        T=total,
        win=win,
        cs=cs,
        swa_pages=swa_pages,
    )
    md.kv_indices_swa = swa_indices_gpu[: int(swa_indptr[total])]
    md.kv_indices_csa = csa_indices_gpu[: int(csa_indptr[total])]
    md.kv_indices_hca = bufs.idx_hca.gpu[: max(hca_total, 0)]
    md.kv_indptr_swa = swa_indptr_gpu
    md.kv_indptr_csa = csa_indptr_gpu
    md.kv_indptr_hca = hca_indptr_gpu
    md.swa_pages = swa_pages


def _populate_indexer(md, common, batch_np, positions, device):
    n_csa = md.n_committed_csa_per_seq_cpu
    cu = np.concatenate([np.zeros(1, dtype=np.int32), np.cumsum(n_csa, dtype=np.int32)])
    cu[-1] = max(int(cu[-1]), 1)
    cu_gpu = torch.from_numpy(cu).to(device)
    bid = md.batch_id_per_token
    base = cu_gpu[bid].to(torch.int32)
    end = base + torch.minimum(
        (positions + 1) // 4, md.n_committed_csa_per_seq[bid]
    ).to(torch.int32)
    md.indexer_meta = {
        "total_committed": int(cu[-1]),
        "cu_committed_gpu": cu_gpu,
        "n_committed_per_seq_gpu": md.n_committed_csa_per_seq,
        "batch_id_per_token_gpu": bid,
        "seq_base_per_token_gpu": base,
        "cu_starts_gpu": base,
        "cu_ends_gpu": end,
    }


def _populate_prefill(md, common, batch_np, pos_np, q_np, positions_gpu):
    """Chunk-aware paged-prefill index build.

    Mirrors native ATOM's ``_build_prefill_paged_indices`` (deepseek_v4_attn.py):
    per-token counts/indptrs on CPU, then one ``write_v4_paged_prefill_indices``
    Triton kernel scatters the SWA-prefix, extend, and HCA-compress index
    segments. Handles both a single full prefill (chunk_start == 0, reduces to
    the old behavior) and any later chunk (chunk_start > 0): each token's SWA
    window splits into a paged "prefix" part (positions before this chunk, read
    from the ring) and an "extend" part (positions in this chunk, read from the
    freshly written K/V). The per-layer ``csa_translate_pack`` later fills the
    CSA topk section of the prefix_csa buffer (sized exactly to match, so no
    ``-1`` sentinel fill is needed).
    """
    device = md.state_slot_mapping.device
    T = len(batch_np)
    num_reqs = int(common.num_reqs)
    win = int(md.swa_window)
    cs = int(md.swa_cs)
    index_topk = int(md.index_topk)
    swa_pages = int(md.swa_num_slots) * cs
    md.swa_pages = swa_pages
    if T == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        zero1 = torch.zeros(1, dtype=torch.int32, device=device)
        md.kv_indices_extend = empty
        md.kv_indptr_extend = zero1
        md.kv_indices_prefix_swa = empty
        md.kv_indptr_prefix_swa = zero1.clone()
        md.kv_indices_prefix_csa = empty.clone()
        md.kv_indptr_prefix_csa = zero1.clone()
        md.kv_indices_prefix_hca = empty.clone()
        md.kv_indptr_prefix_hca = zero1.clone()
        md.skip_prefix_len_csa = empty.clone()
        return

    # ----- Per-token counts (CPU numpy; cumsum gives indptr totals w/o D2H) ---
    chunk_start_pt = md.chunk_start_per_seq_cpu[batch_np]
    token_pos_in_chunk = pos_np - chunk_start_pt
    swa_low = np.maximum(pos_np - win + 1, 0)
    extend_count = np.minimum(token_pos_in_chunk + 1, win).astype(np.int32)
    prefix_swa_count = np.maximum(chunk_start_pt - swa_low, 0).astype(np.int32)
    n_csa_pt = md.n_committed_csa_per_seq_cpu[batch_np]
    csa_valid_k = np.minimum(
        np.minimum((pos_np + 1) // 4, n_csa_pt), index_topk
    ).astype(np.int32)
    # Per-token causal cap, mirroring CSA above and the kernel
    # (write_v4_paged_prefill_indices: n_hca = min((pos+1)//128, committed)).
    # Without it the indptr reserves `committed` HCA slots but the kernel only
    # writes min((pos+1)//128, committed), leaving uninitialized tail garbage.
    n_hca_pt = np.minimum(
        (pos_np + 1) // 128, md.n_committed_hca_per_seq_cpu[batch_np]
    ).astype(np.int32)

    ext_indptr_np = _counts_to_indptr(extend_count)
    swa_indptr_np = _counts_to_indptr(prefix_swa_count)
    csa_indptr_np = _counts_to_indptr(prefix_swa_count + csa_valid_k)
    hca_indptr_np = _counts_to_indptr(prefix_swa_count + n_hca_pt)
    ext_total = int(ext_indptr_np[-1])
    swa_total = int(swa_indptr_np[-1])
    csa_total = int(csa_indptr_np[-1])
    hca_total = int(hca_indptr_np[-1])

    ext_indptr = torch.from_numpy(ext_indptr_np).to(device)
    swa_indptr = torch.from_numpy(swa_indptr_np).to(device)
    csa_indptr = torch.from_numpy(csa_indptr_np).to(device)
    hca_indptr = torch.from_numpy(hca_indptr_np).to(device)

    # scatter on-GPU with native ATOM's Triton kernel (one
    # program per token; avoids an O(T) Python loop).
    from vllm.models.deepseek_v4.amd.atom.model_ops.v4_kernels import write_v4_paged_prefill_indices

    ext_indices = torch.empty(max(ext_total, 1), dtype=torch.int32, device=device)
    swa_indices = torch.empty(max(swa_total, 1), dtype=torch.int32, device=device)
    csa_indices = torch.empty(max(csa_total, 1), dtype=torch.int32, device=device)
    hca_indices = torch.empty(max(hca_total, 1), dtype=torch.int32, device=device)
    chunk_start_g = torch.from_numpy(
        np.ascontiguousarray(md.chunk_start_per_seq_cpu[:num_reqs])
    ).to(device)
    cu_q_g = torch.from_numpy(np.ascontiguousarray(q_np[:num_reqs])).to(device)
    n_hca_seq_g = torch.from_numpy(
        np.ascontiguousarray(md.n_committed_hca_per_seq_cpu[:num_reqs])
    ).to(device)
    write_v4_paged_prefill_indices(
        positions=positions_gpu[:T].to(torch.int32),
        bid_per_token=md.batch_id_per_token[:T],
        chunk_start_per_seq=chunk_start_g,
        cu_seqlens_q_per_seq=cu_q_g,
        state_slot_per_seq=md.state_slot_mapping[:num_reqs],
        n_committed_hca_per_seq=n_hca_seq_g,
        block_tables=common.block_table_tensor[:num_reqs],
        extend_indptr=ext_indptr,
        prefix_swa_indptr=swa_indptr,
        prefix_csa_indptr=csa_indptr,
        prefix_hca_indptr=hca_indptr,
        extend_indices=ext_indices,
        prefix_swa_indices=swa_indices,
        prefix_csa_indices=csa_indices,
        prefix_hca_indices=hca_indices,
        T=T,
        win=win,
        cs=cs,
        swa_pages=swa_pages,
    )
    md.kv_indices_extend = ext_indices[:ext_total]
    md.kv_indices_prefix_swa = swa_indices[:swa_total]
    md.kv_indices_prefix_csa = csa_indices[:csa_total]
    md.kv_indices_prefix_hca = hca_indices[:hca_total]

    md.kv_indptr_extend = ext_indptr
    md.kv_indptr_prefix_swa = swa_indptr
    md.kv_indptr_prefix_csa = csa_indptr
    md.kv_indptr_prefix_hca = hca_indptr
    md.skip_prefix_len_csa = torch.from_numpy(prefix_swa_count).to(device)


def _populate_decode(md, common, batch_np, pos_np, positions_gpu):
    device = md.state_slot_mapping.device
    win = int(md.swa_window)
    cs = int(md.swa_cs)
    # SWA ring boundary in unified_kv is num_slots*cs (the real pool size, ==
    # max_num_seqs), not the per-forward request count -- the HCA compress tail
    # (swa_pages + block_id) lands in the wrong region otherwise once a sequence
    # is long enough to commit HCA entries (>=128 tokens).
    swa_pages = int(md.swa_num_slots) * cs
    index_topk = int(md.index_topk)
    swa_counts = np.minimum(pos_np + 1, win).astype(np.int32)
    csa_counts = np.minimum(
        np.minimum((pos_np + 1) // 4, index_topk),
        md.n_committed_csa_per_seq_cpu[batch_np],
    ).astype(np.int32)
    hca_counts = md.n_committed_hca_per_seq_cpu[batch_np].astype(np.int32)
    swa_indptr_np = _counts_to_indptr(swa_counts)
    csa_indptr_np = _counts_to_indptr(swa_counts + csa_counts)
    hca_indptr_np = _counts_to_indptr(swa_counts + hca_counts)
    swa_total = int(swa_indptr_np[-1])
    csa_total = int(csa_indptr_np[-1])
    hca_total = int(hca_indptr_np[-1])
    swa_indptr = torch.from_numpy(swa_indptr_np).to(device)
    csa_indptr = torch.from_numpy(csa_indptr_np).to(device)
    hca_indptr = torch.from_numpy(hca_indptr_np).to(device)
    T = len(batch_np)

    # On-GPU build (mirrors the persistent decode path): one kernel writes
    # the shared SWA window prefix into all three buffers, a second appends
    # the HCA compress tail straight from the GPU block table
    from vllm.models.deepseek_v4.amd.atom.model_ops.v4_kernels import write_v4_paged_decode_indices

    from vllm.models.deepseek_v4.amd.atom.plugin.vllm.deepseek_v4_ops import (
        write_v4_decode_hca_compress_tail,
    )

    swa_indices = torch.empty(max(swa_total, 1), dtype=torch.int32, device=device)
    csa_indices = torch.empty(max(csa_total, 1), dtype=torch.int32, device=device)
    hca_indices = torch.empty(max(hca_total, 1), dtype=torch.int32, device=device)
    write_v4_paged_decode_indices(
        state_slot_per_seq=md.state_slot_mapping,
        batch_id_per_token=md.batch_id_per_token,
        positions=positions_gpu,
        swa_indptr=swa_indptr,
        csa_indptr=csa_indptr,
        hca_indptr=hca_indptr,
        swa_indices=swa_indices,
        csa_indices=csa_indices,
        hca_indices=hca_indices,
        T=T,
        win=win,
        cs=cs,
    )
    write_v4_decode_hca_compress_tail(
        batch_id_per_token=md.batch_id_per_token,
        positions=positions_gpu,
        hca_indptr=hca_indptr,
        n_committed_hca_per_seq=md.n_committed_hca_per_seq,
        block_tables=common.block_table_tensor,
        hca_indices=hca_indices,
        T=T,
        win=win,
        swa_pages=swa_pages,
    )
    md.kv_indices_swa = swa_indices[:swa_total]
    md.kv_indices_csa = csa_indices[:csa_total]
    md.kv_indices_hca = hca_indices[:hca_total]

    md.kv_indptr_swa = swa_indptr
    md.kv_indptr_csa = csa_indptr
    md.kv_indptr_hca = hca_indptr
    md.swa_pages = swa_pages


def get_deepseek_v4_proxy_metadata_from_vllm_context():
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    meta = get_forward_context().attn_metadata
    if isinstance(meta, dict):
        return meta.get(ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME)
    if isinstance(meta, list) and meta and isinstance(meta[0], dict):
        return meta[0].get(ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME)
    return None


@contextmanager
def atom_deepseek_v4_forward_context(
    *,
    atom_config,
    input_ids,
    positions,
    common_attn_metadata=None,
    force_dummy: bool = False,
    state_model=None,
    meta_params=None,
    slot_allocator=None,
):
    from vllm.models.deepseek_v4.amd.atom.utils.forward_context import (
        Context,
        reset_forward_context,
        set_forward_context,
    )

    if common_attn_metadata is None:
        common_attn_metadata = get_deepseek_v4_proxy_metadata_from_vllm_context()
    # Fast path: the proxy metadata builder already built the ATOM metadata into
    # persistent buffers (outside any captured region) and attached it. This is
    # the only path that is CUDA/HIP-graph safe -- the captured forward merely
    # reads it. The per-slot reset was already applied in build().
    attn_metadata = getattr(common_attn_metadata, "atom_v4_md", None)
    if attn_metadata is None:
        # Fallback (profiling / dummy / standalone, before the proxy cache is
        # bound): build inline with fresh tensors. Never captured.
        if common_attn_metadata is not None:
            common_attn_metadata.positions = positions
        attn_metadata = build_atom_v4_attention_metadata(
            common_attn_metadata,
            meta_params=meta_params,
            slot_allocator=slot_allocator,
        )
        # Selective per-slot reset: clear only the slots the allocator just
        # bound to a fresh request (replaces the old global position-0 reset,
        # which corrupted in-flight requests in a mixed prefill/decode batch).
        if state_model is not None:
            reset_slots = getattr(attn_metadata, "reset_slots", None)
            if reset_slots:
                reset_deepseek_v4_state_slots(state_model, reset_slots)
    in_hipgraph = bool(getattr(attn_metadata, "in_hipgraph", False))
    is_prefill = attn_metadata.state.value.startswith("prefill")
    batch_size = int(
        getattr(common_attn_metadata, "num_reqs", 0)
        or (input_ids.shape[0] if input_ids is not None else 0)
    )
    context = Context(
        positions=positions,
        is_prefill=is_prefill,
        is_dummy_run=force_dummy or common_attn_metadata is None,
        batch_size=batch_size,
        graph_bs=batch_size,
        input_ids=input_ids,
    )
    set_forward_context(
        attn_metadata=attn_metadata,
        atom_config=atom_config,
        context=context,
        num_tokens=int(positions.numel()),
        in_hipgraph=in_hipgraph,
    )
    try:
        yield
    finally:
        reset_forward_context()
