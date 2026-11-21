# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_ATTN pure decode on the AITER asm paged attention with an fp8 KV cache."""

import importlib
from dataclasses import replace

import pytest
import torch

import vllm.envs as envs
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.platforms import current_platform

if not (current_platform.is_rocm() and is_aiter_found_and_supported()):
    pytest.skip("Requires ROCm with AITER on CDNA3+.", allow_module_level=True)

from vllm.utils.platform_utils import num_compute_units  # noqa: E402
from vllm.v1.attention.backends.rocm_attn import (  # noqa: E402
    RocmAttentionImpl,
    RocmAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.paged_attn import PagedAttention  # noqa: E402

DEVICE = torch.device("cuda")
HEAD_SIZE = 128
BLOCK_SIZE = 16
LAYER = "model.layers.0.self_attn.attn"


@pytest.fixture
def set_aiter(monkeypatch):
    def set_env(enabled: bool) -> None:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1" if enabled else "0")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MHA", "1")
        importlib.reload(envs)
        rocm_aiter_ops.refresh_env_variables()

    yield set_env
    monkeypatch.undo()
    importlib.reload(envs)
    rocm_aiter_ops.refresh_env_variables()


class _Layer:
    """Attention layer state the backend reads, as in test_attention_backends."""

    def __init__(self, kv_cache: torch.Tensor, k_scale: float, v_scale: float):
        self.layer_name = LAYER
        self.kv_cache = kv_cache
        self._q_scale_float = 1.0
        self._k_scale = torch.tensor(k_scale, dtype=torch.float32, device=DEVICE)
        self._v_scale = torch.tensor(v_scale, dtype=torch.float32, device=DEVICE)
        self._k_scale_float = k_scale
        self._v_scale_float = v_scale


def _seq_lens(num_seqs: int) -> list[int]:
    # One token, block and 256-token partition edges, then a spread up to 3000.
    edges = [1, 16, 17, 256, 257, 3000]
    return [edges[i] if i < len(edges) else 1 + i * 397 % 3000 for i in range(num_seqs)]


class _Case:
    """A filled fp8 paged cache in the ROCM_ATTN layout and decode metadata."""

    def __init__(self, num_heads, num_kv_heads, seq_lens, query_len=1, **impl_kw):
        torch.manual_seed(0)
        self.num_heads, self.num_kv_heads, self.impl_kw = (
            num_heads,
            num_kv_heads,
            impl_kw,
        )
        self.vllm_config = create_vllm_config(block_size=BLOCK_SIZE, max_model_len=4096)
        self.vllm_config.cache_config.cache_dtype = "fp8"
        # The KV cache group's spec carries the head layout the backend sees.
        self.spec = replace(
            create_standard_kv_cache_spec(self.vllm_config), num_kv_heads=num_kv_heads
        )
        self.common = create_common_attn_metadata(
            BatchSpec(seq_lens=seq_lens, query_lens=[query_len] * len(seq_lens)),
            BLOCK_SIZE,
            DEVICE,
            arange_block_indices=True,
        )
        blocks = self.common.block_table_tensor
        row = BLOCK_SIZE * num_kv_heads * HEAD_SIZE
        self.kv_cache = (
            torch.zeros(2, blocks.numel(), row, dtype=torch.uint8, device=DEVICE)
            .view(2, blocks.numel(), BLOCK_SIZE, num_kv_heads * HEAD_SIZE)
            .transpose(0, 1)
        )
        self.layer = _Layer(self.kv_cache, 0.37, 2.5)
        self.vllm_config.compilation_config.static_forward_context[LAYER] = self.layer
        n = len(seq_lens)
        self.query = (torch.randn(n * query_len, num_heads, HEAD_SIZE) * 0.5).to(
            device=DEVICE, dtype=torch.bfloat16
        )
        kv_shape = (n * query_len, num_kv_heads, HEAD_SIZE)
        self.key = torch.randn(kv_shape, device=DEVICE, dtype=torch.bfloat16)
        self.value = torch.randn(kv_shape, device=DEVICE, dtype=torch.bfloat16)
        writer = self.arm()[1]
        for i, s in enumerate(seq_lens):
            pos = torch.arange(s, device=DEVICE)
            slots = blocks[i, pos // BLOCK_SIZE].long() * BLOCK_SIZE + pos % BLOCK_SIZE
            k = torch.randn(s, num_kv_heads, HEAD_SIZE, device=DEVICE).bfloat16()
            v = torch.randn(s, num_kv_heads, HEAD_SIZE, device=DEVICE).bfloat16()
            writer.do_kv_cache_update(self.layer, k, v, self.kv_cache, slots)

    def arm(self) -> tuple[RocmAttentionMetadataBuilder, RocmAttentionImpl]:
        builder = RocmAttentionMetadataBuilder(
            self.spec, [LAYER], self.vllm_config, DEVICE
        )
        impl = RocmAttentionImpl(
            num_heads=self.num_heads,
            head_size=HEAD_SIZE,
            scale=HEAD_SIZE**-0.5,
            num_kv_heads=self.num_kv_heads,
            alibi_slopes=None,
            sliding_window=self.impl_kw.get("sliding_window"),
            kv_cache_dtype="fp8",
        )
        return builder, impl

    def run(self, builder, impl, query=None, metadata=None):
        query = self.query if query is None else query
        md = builder.build(0, self.common) if metadata is None else metadata
        out = torch.empty_like(query)
        impl.forward(
            self.layer, query, self.key, self.value, self.kv_cache, md, output=out
        )
        return out

    def direct_asm(self, metadata):
        key_cache, value_cache = PagedAttention.split_kv_cache(
            self.kv_cache.transpose(0, 1), self.num_kv_heads, HEAD_SIZE
        )
        fp8 = current_platform.fp8_dtype()
        out = torch.empty_like(self.query)
        rocm_aiter_ops.pa_fwd_asm(
            Q=self.query,
            K=key_cache.view(fp8),
            V=value_cache.view(fp8).unsqueeze(2),
            block_tables=metadata.block_table,
            context_lens=metadata.seq_lens,
            block_tables_stride0=metadata.block_table.stride(0),
            K_QScale=metadata.k_scale[LAYER],
            V_QScale=metadata.v_scale[LAYER],
            out_=out,
            high_precision=2,
        )
        return out


def _below_cut(num_heads: int) -> int:
    """Largest decode batch that stays on the HIP kernel."""
    return 2 * num_compute_units() // num_heads


@pytest.mark.parametrize("heads", [(16, 16), (32, 8), (40, 8), (64, 8)])
@pytest.mark.parametrize("above_cut", [False, True])
@torch.inference_mode()
def test_decode_switches_to_asm_above_the_cut(set_aiter, heads, above_cut):
    num_seqs = _below_cut(heads[0]) + int(above_cut)
    case = _Case(heads[0], heads[1], _seq_lens(num_seqs))
    set_aiter(False)
    hip = case.run(*case.arm())
    set_aiter(True)
    builder, impl = case.arm()
    md = builder.build(0, case.common)
    out = case.run(builder, impl, metadata=md)
    if not above_cut:
        torch.testing.assert_close(out, hip, atol=0, rtol=0)
        return
    torch.testing.assert_close(out, case.direct_asm(md), atol=0, rtol=0)
    # The asm kernel rounds more than the HIP one on an fp8 cache.
    err = (out.float() - hip.float()).abs()
    assert err.max().item() < 5e-2 and err.mean().item() < 3e-3


@torch.inference_mode()
def test_asm_decode_replays_in_a_cuda_graph(set_aiter):
    num_seqs = 2 * (_below_cut(40) + 1)
    case = _Case(40, 8, [3000] * num_seqs)
    set_aiter(True)
    builder, impl = case.arm()
    md = builder.build(0, case.common)
    out = torch.empty_like(case.query)

    def step():
        impl.forward(
            case.layer, case.query, case.key, case.value, case.kv_cache, md, output=out
        )

    step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    lens = torch.tensor(_seq_lens(num_seqs), dtype=torch.int32, device=DEVICE)
    case.common.seq_lens.copy_(lens)
    order = torch.randperm(num_seqs, device=DEVICE)
    case.common.block_table_tensor.copy_(case.common.block_table_tensor[order])
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(
        out, case.run(builder, impl, metadata=md), atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "variant",
    ["gqa_16", "sliding_window", "fp16_query", "multi_token_query", "kv_not_bound"],
)
@torch.inference_mode()
def test_decode_unchanged_where_asm_does_not_apply(set_aiter, variant):
    heads = (64, 4) if variant == "gqa_16" else (32, 8)
    impl_kw = {"sliding_window": 64} if variant == "sliding_window" else {}
    query_len = 2 if variant == "multi_token_query" else 1
    # Each sequence holds at least query_len tokens.
    lens = [s + query_len - 1 for s in _seq_lens(2 * (_below_cut(heads[0]) + 1))]
    case = _Case(*heads, lens, query_len, **impl_kw)
    query = case.query.half() if variant == "fp16_query" else case.query
    set_aiter(False)
    hip = case.run(*case.arm(), query=query)
    set_aiter(True)
    builder, impl = case.arm()
    md = None
    if variant == "kv_not_bound":
        # Until the KV cache is bound the builder has no scale tables.
        case.layer.kv_cache = torch.tensor([])
        md = builder.build(0, case.common)
        assert md.k_scale is None
    out = case.run(builder, impl, query=query, metadata=md)
    torch.testing.assert_close(out, hip, atol=0, rtol=0)


@torch.inference_mode()
def test_scale_tables_follow_the_bound_kv_cache(set_aiter):
    case = _Case(32, 8, _seq_lens(4))
    set_aiter(True)
    builder = case.arm()[0]
    case.layer.kv_cache = torch.tensor([])
    assert builder.build(0, case.common).k_scale is None
    case.layer.kv_cache = case.kv_cache
    md = builder.build(0, case.common)
    assert md.k_scale is not None and md.v_scale is not None
    tables = (md.k_scale[LAYER], md.v_scale[LAYER])
    assert tables[0].shape == (case.kv_cache.shape[0], 8, BLOCK_SIZE)
    assert torch.all(tables[0] == 0.37) and torch.all(tables[1] == 2.5)
    rebuilt = builder.build(0, case.common).k_scale
    assert rebuilt is not None and rebuilt[LAYER] is tables[0]
