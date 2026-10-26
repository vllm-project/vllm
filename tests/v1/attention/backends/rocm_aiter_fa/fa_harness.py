# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive ROCM_AITER_FA the way the model runner does, one layer at a time.

``FaCase`` owns a VllmConfig shaped like one model's attention layer, real
``Attention`` layers bound to ``AiterFlashAttentionBackend``, the metadata
builder, and a KV cache allocated through the same view helpers the runner
uses. ``run`` writes the batch's K/V through ``do_kv_cache_update`` and calls
``forward``; ``reference`` recomputes the result in float32 from logical K/V.
"""

import sys
from dataclasses import dataclass, field

import torch

from tests.v1.attention.backends.harness import (
    KernelSpy,
    PagedBatch,
    make_paged_batch,
    ref_attention,
)
from tests.v1.attention.model_shapes import AttnShape
from tests.v1.attention.utils import (
    dense_kv_cache_views,
)
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.v1.attention.backends import rocm_aiter_fa
from vllm.v1.attention.backends.rocm_aiter_fa import (
    AiterFlashAttentionBackend,
    AiterFlashAttentionMetadata,
    AiterFlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheLayout,
    get_kv_quant_mode,
    is_quantized_kv_cache,
)

DEVICE = torch.device("cuda")
# Non-unit so a missing scale shows up as a numerical error.
FP8_K_SCALE = 0.5
FP8_V_SCALE = 0.25

FP8_ATOL = 6e-2
FP8_RTOL = 1e-1
FP8_MAX_BAD_FRACTION = 2e-3
FP8_MAX_ABS = 0.3

# Kernel labels used by the dispatch tests.
VARLEN = "flash_attn_varlen_func"
GATHER = "cp_mha_gather_cache"
MERGE = "merge_attn_states"
PAGED_V1 = "paged_attention_v1"
PAGED_COMMON = "paged_attention_common"
UNIFIED = "unified_attention"
MHA_V3 = "flash_attn_with_kvcache"
WRITE_FLASH = "reshape_and_cache_flash"
WRITE_SHUFFLE = "reshape_and_cache_shuffle_triton"


def tolerance(dtype: torch.dtype, kv_cache_dtype: str) -> dict[str, float]:
    if is_quantized_kv_cache(kv_cache_dtype):
        return {"atol": 6e-2, "rtol": 1e-1}
    if dtype == torch.float16:
        return {"atol": 1e-2, "rtol": 1e-2}
    return {"atol": 2e-2, "rtol": 2e-2}


def make_config(
    shape: AttnShape,
    *,
    dtype: torch.dtype = torch.bfloat16,
    kv_cache_dtype: str = "auto",
    block_size: int = 16,
    num_blocks: int = 1024,
    max_model_len: int = 16384,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 16384,
) -> VllmConfig:
    """A VllmConfig whose attention dims are the given model's."""
    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        dtype=dtype,
        seed=0,
        max_model_len=max_model_len,
        hf_overrides={
            "num_attention_heads": shape.num_heads,
            "num_key_value_heads": shape.num_kv_heads,
            "head_dim": shape.head_size,
            "max_position_embeddings": max_model_len,
        },
    )
    cache_config = CacheConfig(block_size=block_size, cache_dtype=kv_cache_dtype)
    cache_config.num_gpu_blocks = num_blocks
    cache_config.num_cpu_blocks = 0
    config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            is_encoder_decoder=False,
        ),
    )
    assert model_config.get_num_kv_heads(config.parallel_config) == shape.num_kv_heads
    assert model_config.get_head_size() == shape.head_size
    return config


def watch_all_kernels(spy: KernelSpy) -> None:
    """Spy on every kernel AiterFlashAttentionImpl can dispatch to."""
    import importlib

    import aiter  # noqa: F401  # registers torch.ops.aiter

    # import_module, not `import a.b.c as m`: aiter's package __init__ re-exports
    # functions under the same names as their modules.
    mha_v3 = importlib.import_module("aiter.ops.triton.attention.mha_v3")
    # The backend imports unified_attention by its old path, which aiter's
    # import shim maps to attention.unified_attention by loading the file a
    # second time under the old name, so there are two module objects with
    # their own function. Patch both.
    importlib.import_module("aiter.ops.triton.unified_attention")
    unified_modules = {
        id(m): m
        for m in (
            sys.modules["aiter.ops.triton.unified_attention"],
            importlib.import_module("aiter.ops.triton.attention.unified_attention"),
        )
    }.values()

    spy.watch(rocm_aiter_ops, "flash_attn_varlen_func", VARLEN)
    spy.watch(rocm_aiter_ops, "paged_attention_common", PAGED_COMMON)
    spy.watch(rocm_aiter_fa, "cp_mha_gather_cache", GATHER)
    spy.watch(rocm_aiter_fa, "merge_attn_states", MERGE)
    spy.watch(rocm_aiter_fa, "reshape_and_cache_shuffle_triton", WRITE_SHUFFLE)
    spy.watch(torch.ops.aiter, "paged_attention_v1", PAGED_V1)
    spy.watch(torch.ops._C_cache_ops, "reshape_and_cache_flash", WRITE_FLASH)
    for unified in unified_modules:
        spy.watch(unified, "unified_attention", UNIFIED)
    spy.watch(mha_v3, "flash_attn_with_kvcache", MHA_V3)


@dataclass
class FaCase:
    shape: AttnShape
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: str = "auto"
    block_size: int = 16
    sliding_window: int | None = None
    logits_soft_cap: float | None = None
    # None: take it from the shape (GPT-OSS layers have sinks).
    use_sinks: bool | None = None
    use_alibi: bool = False
    kv_sharing: bool = False
    layout: KVCacheLayout | None = None
    max_model_len: int = 16384
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 16384
    num_blocks: int = 0
    seed: int = 0

    config: VllmConfig = field(init=False)
    layers: dict[str, Attention] = field(init=False)
    builder: AiterFlashAttentionMetadataBuilder = field(init=False)
    kv_cache: torch.Tensor = field(init=False)
    k_all: torch.Tensor = field(init=False)
    v_all: torch.Tensor = field(init=False)

    @property
    def scale(self) -> float:
        return self.shape.head_size**-0.5

    @property
    def layer(self) -> Attention:
        """The layer under test: the sharing layer when KV is shared."""
        return self.layers["shared" if self.kv_sharing else "target"]

    def setup(self, num_blocks: int) -> "FaCase":
        torch.manual_seed(self.seed)
        if self.use_sinks is None:
            self.use_sinks = self.shape.sinks
        self.num_blocks = num_blocks
        self.config = make_config(
            self.shape,
            dtype=self.dtype,
            kv_cache_dtype=self.kv_cache_dtype,
            block_size=self.block_size,
            num_blocks=num_blocks,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
        )
        h, kv, d = self.shape.num_heads, self.shape.num_kv_heads, self.shape.head_size
        extra = {}
        if self.use_sinks:
            extra["sinks"] = torch.randn(h, dtype=torch.float32, device=DEVICE)
        alibi = [2 ** (-8 * (i + 1) / h) for i in range(h)] if self.use_alibi else None
        self.layers = {}
        # The model loader builds layers under the target device.
        with set_current_vllm_config(self.config), torch.device(DEVICE):
            names = ["target", "shared"] if self.kv_sharing else ["target"]
            for name in names:
                self.layers[name] = Attention(
                    h,
                    d,
                    self.scale,
                    kv,
                    alibi_slopes=alibi,
                    cache_config=self.config.cache_config,
                    logits_soft_cap=self.logits_soft_cap,
                    per_layer_sliding_window=self.sliding_window,
                    prefix=name,
                    kv_sharing_target_layer_name=(
                        "target" if name == "shared" else None
                    ),
                    attn_backend=AiterFlashAttentionBackend,
                    **extra,
                )
            spec = AiterFlashAttentionBackend.customize_spec(
                FullAttentionSpec(
                    block_size=self.block_size,
                    num_kv_heads=kv,
                    head_size=d,
                    dtype=self.layers["target"].kv_cache_torch_dtype,
                    kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
                )
            )
            self.builder = AiterFlashAttentionMetadataBuilder(
                spec, names, self.config, DEVICE
            )
        layout = (
            self.layout or AiterFlashAttentionBackend.supported_kv_cache_layouts()[0]
        )
        raw = torch.zeros(
            num_blocks * spec.page_size_bytes, dtype=torch.int8, device=DEVICE
        )
        (self.kv_cache,) = dense_kv_cache_views(
            raw, spec, num_blocks, num_layers=1, layout=layout
        )
        quantized = is_quantized_kv_cache(self.kv_cache_dtype)
        for layer in self.layers.values():
            layer.kv_cache = self.kv_cache
            k_scale = FP8_K_SCALE if quantized else 1.0
            v_scale = FP8_V_SCALE if quantized else 1.0
            layer._k_scale.fill_(k_scale)
            layer._v_scale.fill_(v_scale)
            layer._k_scale_float = k_scale
            layer._v_scale_float = v_scale
            layer._k_scale_cpu = torch.tensor(k_scale, dtype=torch.float32)
            layer._v_scale_cpu = torch.tensor(v_scale, dtype=torch.float32)
        self._fill_cache()
        return self

    def _representable(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """Round K/V to values the cache stores exactly, so the reference and
        the kernel see the same K/V and only accumulation error remains."""
        if not is_quantized_kv_cache(self.kv_cache_dtype):
            return x
        fp8 = current_platform.fp8_dtype()
        return ((x / scale).to(fp8).to(x.dtype) * scale).to(x.dtype)

    def _fill_cache(self) -> None:
        num_slots = self.num_blocks * self.block_size
        kv, d = self.shape.num_kv_heads, self.shape.head_size
        k = torch.randn(num_slots, kv, d, dtype=self.dtype, device=DEVICE)
        v = torch.randn(num_slots, kv, d, dtype=self.dtype, device=DEVICE)
        self.k_all = self._representable(k, FP8_K_SCALE)
        self.v_all = self._representable(v, FP8_V_SCALE)
        target = self.layers["target"]
        target.impl.do_kv_cache_update(
            target,
            self.k_all,
            self.v_all,
            self.kv_cache,
            torch.arange(num_slots, dtype=torch.int64, device=DEVICE),
        )

    def batch(
        self,
        query_lens: list[int],
        seq_lens: list[int],
        *,
        num_padded_tokens: int | None = None,
        causal: bool = True,
    ) -> PagedBatch:
        return make_paged_batch(
            query_lens,
            seq_lens,
            self.block_size,
            DEVICE,
            num_blocks=self.num_blocks,
            num_padded_tokens=num_padded_tokens,
            causal=causal,
            seed=self.seed,
        )

    def build(self, batch: PagedBatch) -> AiterFlashAttentionMetadata:
        return self.builder.build(0, batch.common)

    def query(self, num_tokens: int) -> torch.Tensor:
        return torch.randn(
            num_tokens,
            self.shape.num_heads,
            self.shape.head_size,
            dtype=self.dtype,
            device=DEVICE,
        )

    def run(
        self,
        batch: PagedBatch,
        query: torch.Tensor,
        metadata: AiterFlashAttentionMetadata | None = None,
    ) -> torch.Tensor:
        """Forward the layer under test; returns [num_tokens, H, D]."""
        metadata = metadata if metadata is not None else self.build(batch)
        n = query.shape[0]
        key = value = None
        if not self.kv_sharing:
            key = torch.zeros(
                n,
                self.shape.num_kv_heads,
                self.shape.head_size,
                dtype=self.dtype,
                device=DEVICE,
            )
            value = torch.zeros_like(key)
            key[: batch.num_tokens] = self.k_all[batch.query_slots]
            value[: batch.num_tokens] = self.v_all[batch.query_slots]
        output = torch.full_like(query, float("nan"))
        layer = self.layer
        layer.impl.forward(layer, query, key, value, self.kv_cache, metadata, output)
        return output

    def reference(self, batch: PagedBatch, query: torch.Tensor) -> torch.Tensor:
        impl = self.layer.impl
        return ref_attention(
            query[: batch.num_tokens],
            self.k_all,
            self.v_all,
            batch,
            self.scale,
            sliding_window=self.sliding_window,
            causal=bool(batch.common.causal),
            logits_soft_cap=self.logits_soft_cap or 0.0,
            sinks=impl.sinks,
            alibi_slopes=impl.alibi_slopes,
        )

    def check(self, batch: PagedBatch, query: torch.Tensor, output: torch.Tensor):
        expected = self.reference(batch, query)
        actual = output[: batch.num_tokens].float()
        assert torch.isfinite(actual).all(), "non-finite attention output"
        if not is_quantized_kv_cache(self.kv_cache_dtype):
            torch.testing.assert_close(
                actual, expected, **tolerance(self.dtype, self.kv_cache_dtype)
            )
            return
        # The fp8 decode kernels lose more precision than the K/V rounding
        # the reference already models (up to ~0.2 abs at 2-3 token contexts
        # on MI355, shrinking with context length). Bound the bulk tightly and
        # the worst element loosely, so a broken scale still fails.
        err = (actual - expected).abs()
        bad = (err > FP8_ATOL + FP8_RTOL * expected.abs()).float().mean().item()
        assert bad <= FP8_MAX_BAD_FRACTION, f"{bad:.2%} elements out of tolerance"
        assert err.max().item() <= FP8_MAX_ABS, f"max abs error {err.max():.3f}"


def blocks_for(seq_lens: list[int], block_size: int, slack: int = 8) -> int:
    return sum(-(-s // block_size) for s in seq_lens) + slack


def set_shuffle(monkeypatch, enabled: bool) -> None:
    monkeypatch.setattr(rocm_aiter_ops, "_SHUFFLE_KV_CACHE_ENABLED", enabled)
