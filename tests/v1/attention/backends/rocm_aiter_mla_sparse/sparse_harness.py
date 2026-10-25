# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive ROCM_AITER_MLA_SPARSE the way the model runner does.

``SparseCase`` owns a VllmConfig shaped like one sparse-MLA model's attention
layer, real ``MLAAttention`` layers bound to ``ROCMAiterMLASparseBackend``
(layer 0 owns the indexer and its top-k buffer; later layers reuse that
buffer, like GLM-5.2's "shared" indexer layers), the metadata builder, and a
KV cache allocated through the runner's view helpers. ``run`` writes the
batch's latent rows through the layer and calls ``MLAAttention.forward``;
``reference`` recomputes the output in float32 from the logical rows.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.backends.harness import PagedBatch, make_paged_batch
from tests.v1.attention.backends.mla_harness import (
    fp8_representable,
    logical_to_slots,
    ref_sparse_mla,
    sparse_topk_indices,
)
from tests.v1.attention.model_shapes import MLAShape
from tests.v1.attention.utils import dense_kv_cache_views
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
    ROCMAiterMLASparseMetadata,
    ROCMAiterMLASparseMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheLayout, is_quantized_kv_cache

DEVICE = torch.device("cuda")
# Non-unit so a missing scale shows up as a numerical error.
FP8_K_SCALE = 0.5
FP8_Q_SCALE = 0.25

FP8_ATOL = 6e-2
FP8_RTOL = 1e-1
FP8_MAX_BAD_FRACTION = 2e-3
FP8_MAX_ABS = 0.3

# Kernel labels used by the dispatch tests.
MLA_DECODE = "mla_decode_fwd"
MLA_DECODE_LSE = "mla_decode_fwd_lse"
AITER_TRITON = "triton_sparse_mla_fwd"
TRITON_PREFILL = "rocm_sparse_attn_prefill"
TRITON_DECODE = "rocm_sparse_attn_decode_bf16"
REMAP = "triton_convert_req_index_to_global_index"
FIT_KPOOL = "fit_kpool_indices_to_aiter"
MLA_METADATA = "get_mla_metadata_v1"
CONCAT_Q = "concat_mla_q"
QUANT_Q = "scaled_fp8_quant"
KV_WRITE = "concat_and_cache_mla"
DENSE_PREFILL = "forward_mha"
SPARSE_MQA = "forward_mqa"


# AITER's non-persistent fp8 LSE decode (the sink path with an fp8 cache)
# returns rows of stale or NaN output for KV lengths around 65-128 on gfx950,
# depending on what the allocator hands it; it cannot be strict because clean
# memory hides it. Not reproduced in a fresh process.
AITER_FP8_LSE_STALE_ROWS = pytest.mark.xfail(
    strict=False,
    reason="AITER mla_decode_fwd(return_lse=True) with fp8 Q/KV leaves output "
    "rows depending on uninitialized memory (gfx950)",
)


def tolerance(dtype: torch.dtype) -> dict[str, float]:
    if dtype == torch.float16:
        return {"atol": 1e-2, "rtol": 1e-2}
    return {"atol": 2e-2, "rtol": 2e-2}


def make_config(
    shape: MLAShape,
    *,
    topk: int,
    dtype: torch.dtype = torch.bfloat16,
    kv_cache_dtype: str = "auto",
    block_size: int = 64,
    num_blocks: int = 1024,
    max_model_len: int = 16384,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 8192,
) -> VllmConfig:
    """A VllmConfig whose MLA and indexer dims are the given model's."""
    model_config = ModelConfig(
        model="deepseek-ai/DeepSeek-V2-Lite-Chat",
        dtype=dtype,
        seed=0,
        max_model_len=max_model_len,
        hf_overrides={
            "num_attention_heads": shape.num_heads,
            "num_key_value_heads": shape.num_heads,
            "q_lora_rank": shape.q_lora_rank,
            "kv_lora_rank": shape.kv_lora_rank,
            "qk_nope_head_dim": shape.qk_nope_head_dim,
            "qk_rope_head_dim": shape.qk_rope_head_dim,
            "v_head_dim": shape.v_head_dim,
            "index_topk": topk,
            "index_n_heads": shape.index_n_heads or 64,
            "index_head_dim": shape.index_head_dim or 128,
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
    assert model_config.get_num_attention_heads(config.parallel_config) == (
        shape.num_heads
    )
    return config


def watch_all_kernels(spy) -> None:
    """Spy on every kernel the sparse layer and impl can dispatch to."""
    spy.watch(rocm_aiter_ops, "mla_decode_fwd", MLA_DECODE)
    spy.watch(rocm_aiter_ops, "mla_decode_fwd_lse", MLA_DECODE_LSE)
    if hasattr(rocm_aiter_ops, "triton_sparse_mla_fwd"):
        spy.watch(rocm_aiter_ops, "triton_sparse_mla_fwd", AITER_TRITON)
    spy.watch(rocm_aiter_mla_sparse, "rocm_sparse_attn_prefill", TRITON_PREFILL)
    spy.watch(rocm_aiter_mla_sparse, "rocm_sparse_attn_decode_bf16", TRITON_DECODE)
    spy.watch(rocm_aiter_mla_sparse, "triton_convert_req_index_to_global_index", REMAP)
    spy.watch(rocm_aiter_mla_sparse, "fit_kpool_indices_to_aiter", FIT_KPOOL)
    spy.watch(rocm_aiter_mla_sparse.ops, "concat_mla_q", CONCAT_Q)
    spy.watch(rocm_aiter_mla_sparse.ops, "scaled_fp8_quant", QUANT_Q)
    import aiter

    spy.watch(aiter, "get_mla_metadata_v1", MLA_METADATA)


@dataclass
class SparseCase:
    shape: MLAShape
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: str = "auto"
    block_size: int = 64
    # The indexer width; the models use 2048, smaller keeps rows sparse at
    # test-sized contexts.
    topk: int = 256
    # Shared top-k buffer width; > topk models a wider (kpool) indexer.
    topk_width: int | None = None
    use_sinks: bool = False
    # Layer 0 owns the indexer; the others reuse its buffer.
    num_layers: int = 1
    layout: KVCacheLayout | None = None
    force_mqa: bool = False
    # Speculative tokens; widens the builder's decode threshold to 1 + this.
    spec_tokens: int = 0
    max_model_len: int = 16384
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    num_blocks: int = 0
    seed: int = 0

    config: VllmConfig = field(init=False)
    layers: dict[str, MLAAttention] = field(init=False)
    builder: ROCMAiterMLASparseMetadataBuilder = field(init=False)
    topk_buffer: torch.Tensor = field(init=False)
    kv_rows: torch.Tensor = field(init=False)

    @property
    def names(self) -> list[str]:
        return [f"layers.{i}.attn" for i in range(self.num_layers)]

    @property
    def layer(self) -> MLAAttention:
        return self.layers[self.names[0]]

    @property
    def scale(self) -> float:
        return (self.shape.qk_nope_head_dim + self.shape.qk_rope_head_dim) ** -0.5

    @property
    def quantized(self) -> bool:
        return is_quantized_kv_cache(self.kv_cache_dtype)

    def setup(self, num_blocks: int) -> "SparseCase":
        torch.manual_seed(self.seed)
        self.num_blocks = num_blocks
        self.config = make_config(
            self.shape,
            topk=self.topk,
            dtype=self.dtype,
            kv_cache_dtype=self.kv_cache_dtype,
            block_size=self.block_size,
            num_blocks=num_blocks,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
        )
        self.config.attention_config.sparse_mla_force_mqa = self.force_mqa
        if self.spec_tokens:
            self.config.speculative_config = SpeculativeConfig(
                method="ngram", num_speculative_tokens=self.spec_tokens
            )
        s = self.shape
        h, lora = s.num_heads, s.kv_lora_rank
        self.topk_buffer = torch.full(
            (self.max_num_batched_tokens, self.topk_width or self.topk),
            -1,
            dtype=torch.int32,
            device=DEVICE,
        )
        indexer = SimpleNamespace(topk_indices_buffer=self.topk_buffer)
        self.layers = {}
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        try:
            # The model loader builds layers under the target device.
            with set_current_vllm_config(self.config), torch.device(DEVICE):
                for i, name in enumerate(self.names):
                    kv_b_proj = ColumnParallelLinear(
                        lora,
                        h * (s.qk_nope_head_dim + s.v_head_dim),
                        bias=False,
                        params_dtype=self.dtype,
                        prefix=f"{name}.kv_b_proj",
                    )
                    with torch.no_grad():
                        kv_b_proj.weight.normal_(0, lora**-0.5)
                    extra = {}
                    if self.use_sinks:
                        extra["sinks"] = torch.randn(h, dtype=torch.float32)
                    layer = MLAAttention(
                        num_heads=h,
                        scale=self.scale,
                        qk_nope_head_dim=s.qk_nope_head_dim,
                        qk_rope_head_dim=s.qk_rope_head_dim,
                        v_head_dim=s.v_head_dim,
                        q_lora_rank=s.q_lora_rank,
                        kv_lora_rank=lora,
                        kv_b_proj=kv_b_proj,
                        cache_config=self.config.cache_config,
                        prefix=name,
                        attn_backend=ROCMAiterMLASparseBackend,
                        use_sparse=True,
                        indexer=indexer if i == 0 else None,
                        topk_indices_buffer=self.topk_buffer,
                        **extra,
                    )
                    layer.process_weights_after_loading(self.dtype)
                    self.layers[name] = layer
                spec = ROCMAiterMLASparseBackend.customize_spec(
                    self.layer.get_kv_cache_spec(self.config)
                )
                self.spec = spec
                self.builder = ROCMAiterMLASparseMetadataBuilder(
                    spec, self.names, self.config, DEVICE
                )
        finally:
            torch.set_default_dtype(old_dtype)
        layout = (
            self.layout or ROCMAiterMLASparseBackend.supported_kv_cache_layouts()[0]
        )
        raw = torch.zeros(
            num_blocks * spec.page_size_bytes * self.num_layers,
            dtype=torch.int8,
            device=DEVICE,
        )
        views = dense_kv_cache_views(
            raw, spec, num_blocks, num_layers=self.num_layers, layout=layout
        )
        for layer, view in zip(self.layers.values(), views):
            layer.bind_kv_cache(view)
            if self.quantized:
                layer._k_scale.fill_(FP8_K_SCALE)
                layer._q_scale.fill_(FP8_Q_SCALE)
                layer._k_scale_float = FP8_K_SCALE
                layer._q_scale_float = FP8_Q_SCALE
        self._fill_cache()
        return self

    def _fill_cache(self) -> None:
        """Write a random latent row to every slot of every layer's cache."""
        num_slots = self.num_blocks * self.block_size
        rows = torch.randn(
            num_slots, self.shape.head_size, dtype=self.dtype, device=DEVICE
        )
        if self.quantized:
            rows = fp8_representable(rows, FP8_K_SCALE)
        self.kv_rows = rows
        slots = torch.arange(num_slots, dtype=torch.int64, device=DEVICE)
        lora = self.shape.kv_lora_rank
        for layer in self.layers.values():
            layer.impl.do_kv_cache_update(
                rows[:, :lora],
                rows[:, lora:].unsqueeze(1),
                layer.kv_cache,
                slots,
                layer.kv_cache_dtype,
                layer._k_scale,
            )

    def batch(
        self,
        query_lens: list[int],
        seq_lens: list[int],
        *,
        num_padded_tokens: int | None = None,
    ) -> PagedBatch:
        return make_paged_batch(
            query_lens,
            seq_lens,
            self.block_size,
            DEVICE,
            num_blocks=self.num_blocks,
            num_padded_tokens=num_padded_tokens,
            seed=self.seed,
        )

    def indices(self, batch: PagedBatch, *, dense: bool = False) -> torch.Tensor:
        return sparse_topk_indices(
            batch,
            self.topk,
            width=self.topk_width,
            dense=dense,
            seed=self.seed,
            device=DEVICE,
        )

    def build(self, batch: PagedBatch) -> ROCMAiterMLASparseMetadata:
        return self.builder.build(0, batch.common)

    def query(self, num_tokens: int) -> torch.Tensor:
        s = self.shape
        return torch.randn(
            num_tokens,
            s.num_heads,
            s.qk_nope_head_dim + s.qk_rope_head_dim,
            dtype=self.dtype,
            device=DEVICE,
        )

    def run(
        self,
        batch: PagedBatch,
        query: torch.Tensor,
        topk_indices: torch.Tensor,
        metadata: ROCMAiterMLASparseMetadata | None = None,
        *,
        layer_names: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward the layers in order; returns {name: [tokens, H * V]}."""
        metadata = metadata if metadata is not None else self.build(batch)
        n = query.shape[0]
        self.topk_buffer[: topk_indices.shape[0]] = topk_indices
        lora = self.shape.kv_lora_rank
        rows = torch.zeros(n, self.shape.head_size, dtype=self.dtype, device=DEVICE)
        rows[: batch.num_tokens] = self.kv_rows[batch.query_slots]
        kv_c, k_pe = rows[:, :lora], rows[:, lora:].unsqueeze(1)
        names = layer_names or self.names
        out_shape = torch.Size((n, self.shape.num_heads * self.shape.v_head_dim))
        outputs = {}
        with set_forward_context(
            {name: metadata for name in names},
            self.config,
            slot_mapping={name: batch.common.slot_mapping for name in names},
        ):
            for name in names:
                outputs[name] = self.layers[name].forward(
                    query, kv_c, k_pe, output_shape=out_shape
                )
        return outputs

    def absorbed_query(
        self, query: torch.Tensor, layer: MLAAttention | None = None
    ) -> torch.Tensor:
        """[tokens, H, kv_lora_rank + rope] in float32, as the MQA path sees it."""
        s = self.shape
        q_nope, q_pe = query.float().split(
            [s.qk_nope_head_dim, s.qk_rope_head_dim], dim=-1
        )
        w_uk_t = (layer or self.layer).W_UK_T.float()  # [H, nope, L]
        ql_nope = torch.einsum("thn,hnl->thl", q_nope, w_uk_t)
        q = torch.cat([ql_nope, q_pe], dim=-1)
        if self.quantized:
            q = fp8_representable(q, FP8_Q_SCALE).float()
        return q

    def reference(
        self,
        batch: PagedBatch,
        query: torch.Tensor,
        topk_indices: torch.Tensor,
        *,
        layer: MLAAttention | None = None,
    ) -> torch.Tensor:
        """Float32 [tokens, H * V] for the batch's real tokens."""
        layer = layer or self.layer
        n = batch.num_tokens
        out_latent, _ = ref_sparse_mla(
            self.absorbed_query(query[:n], layer),
            self.kv_rows,
            logical_to_slots(batch, topk_indices[:n]),
            self.scale,
            self.shape.kv_lora_rank,
            sinks=layer.impl.sinks,
        )
        w_uv = layer.W_UV.float()  # [H, L, V]
        out = torch.einsum("thl,hlv->thv", out_latent, w_uv)
        return out.reshape(n, -1)

    def check(
        self,
        batch: PagedBatch,
        query: torch.Tensor,
        topk_indices: torch.Tensor,
        output: torch.Tensor,
        *,
        layer: MLAAttention | None = None,
    ) -> None:
        expected = self.reference(batch, query, topk_indices, layer=layer)
        actual = output[: batch.num_tokens].float()
        assert torch.isfinite(actual).all(), "non-finite attention output"
        if not self.quantized:
            torch.testing.assert_close(actual, expected, **tolerance(self.dtype))
            return
        err = (actual - expected).abs()
        bad = (err > FP8_ATOL + FP8_RTOL * expected.abs()).float().mean().item()
        assert bad <= FP8_MAX_BAD_FRACTION, f"{bad:.2%} elements out of tolerance"
        assert err.max().item() <= FP8_MAX_ABS, f"max abs error {err.max():.3f}"


def blocks_for(seq_lens: list[int], block_size: int, slack: int = 8) -> int:
    return sum(-(-s // block_size) for s in seq_lens) + slack


def make_case(shape: MLAShape, seq_lens: list[int], **kw) -> SparseCase:
    case = SparseCase(shape, **kw)
    return case.setup(blocks_for(seq_lens, case.block_size))
