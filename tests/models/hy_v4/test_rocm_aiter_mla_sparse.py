# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 ROCm draft rows: AITER sparse MLA reads the first `paged_kv_indptr`
delta entries of each row and maps -1 to slot 0, so the DFly draft's rows must
be front-packed and counted before every layer reads them.

The real AITER kernels must then match sink-softmax attention over each row's
live keys, and the ROCm attention layer must bind a sink-capable backend.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from vllm.config import (
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("HYV4 ROCm sparse MLA", allow_module_level=True)

from vllm.models.hy_v4.amd import aiter_mla_sparse
from vllm.models.hy_v4.amd.model import HYV4DecoderLayer
from vllm.models.hy_v4.amd.model import HYV4ForCausalLM as AMDHYV4ForCausalLM
from vllm.transformers_utils.configs.hy_v4 import HYV4Config
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseImpl,
    ROCMAiterMLASparseMetadata,
)

DEVICE = "cuda"
KV_LORA_RANK, ROPE_DIM = 512, 64
HEAD_DIM = KV_LORA_RANK + ROPE_DIM
BLOCK_SIZE = 64


@pytest.fixture
def hyv4_vllm_config(tmp_path, dist_init, workspace_init):
    """The current `VllmConfig` of a one-layer sparse HY4 checkpoint."""
    config = HYV4Config(
        hidden_size=256,
        num_hidden_layers=1,
        q_lora_rank=128,
        max_position_embeddings=4096,
        layer_types=["deepseek_sparse_attention"],
    )
    config.architectures = ["HYV4ForCausalLM"]
    config.save_pretrained(tmp_path)
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), dtype="bfloat16", skip_tokenizer_init=True
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=4096,
            is_encoder_decoder=False,
            max_num_batched_tokens=64,
            max_num_seqs=8,
        ),
    )
    with set_current_vllm_config(vllm_config):
        yield vllm_config


def _impl(num_heads, topk_indices_buffer, sinks=None):
    """The DFly draft impl: its rows come from DFly, so it owns no indexer."""
    extra = {} if sinks is None else {"sinks": sinks}
    return aiter_mla_sparse.HYV4AiterMLASparseImpl(
        num_heads=num_heads,
        head_size=HEAD_DIM,
        scale=HEAD_DIM**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        topk_indices_buffer=topk_indices_buffer,
        q_lora_rank=None,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=KV_LORA_RANK,
        qk_rope_head_dim=ROPE_DIM,
        qk_head_dim=HEAD_DIM,
        v_head_dim=KV_LORA_RANK,
        kv_b_proj=None,
        **extra,
    )


def _decode_metadata(req_ids, seq_lens, block_table, width):
    """Decode-only sparse MLA metadata with one query token per row."""
    num_tokens = len(req_ids)
    starts = torch.arange(num_tokens + 1, dtype=torch.int32, device=DEVICE)
    return ROCMAiterMLASparseMetadata(
        num_reqs=num_tokens,
        max_query_len=1,
        max_seq_len=max(seq_lens),
        num_actual_tokens=num_tokens,
        query_start_loc=starts,
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64, device=DEVICE),
        block_table=block_table.int().to(DEVICE),
        req_id_per_token=torch.tensor(req_ids, dtype=torch.int32, device=DEVICE),
        block_size=BLOCK_SIZE,
        topk_tokens=width,
        num_decodes=num_tokens,
        num_decode_tokens=num_tokens,
        decode_max_query_len=1,
        qo_indptr=starts,
        paged_kv_last_page_len=torch.ones(num_tokens, dtype=torch.int32, device=DEVICE),
        paged_kv_indices=torch.zeros(
            num_tokens * width, dtype=torch.int32, device=DEVICE
        ),
        paged_kv_indptr=torch.zeros(num_tokens + 1, dtype=torch.int32, device=DEVICE),
        attn_out_dtype=torch.bfloat16,
    )


def _assert_front_packed(rows: torch.Tensor, counts: torch.Tensor) -> None:
    columns = torch.arange(rows.shape[1], device=rows.device)
    head = columns < counts[:, None].long()
    assert torch.all((rows >= 0) == head)


def _stable_pack_reference(rows: torch.Tensor) -> torch.Tensor:
    packed = torch.full_like(rows, -1)
    for i, row in enumerate(rows):
        live = row[row >= 0]
        packed[i, : live.numel()] = live
    return packed


@pytest.mark.parametrize("width", [2176, 300])
def test_pack_live_indices_is_stable_compaction(width):
    torch.manual_seed(0)
    storage = torch.randint(0, 1 << 20, (9, width + 32), device=DEVICE).int()
    storage.masked_fill_(torch.rand_like(storage, dtype=torch.float) < 0.6, -1)
    storage[3] = -1
    storage[5] = torch.arange(width + 32, device=DEVICE)
    original = storage.clone()
    rows = storage[:, :width]

    counts = aiter_mla_sparse.pack_live_indices_(rows)

    expected = _stable_pack_reference(original[:, :width])
    torch.testing.assert_close(rows, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        counts, (original[:, :width] >= 0).sum(-1).int(), rtol=0, atol=0
    )
    torch.testing.assert_close(storage[:, width:], original[:, width:], rtol=0, atol=0)


def test_packed_impl_rebuilds_indptr_from_live_counts(monkeypatch, hyv4_vllm_config):
    monkeypatch.setattr(
        ROCMAiterMLASparseImpl, "forward_mqa", lambda *args: "forwarded"
    )
    torch.manual_seed(0)
    buffer = torch.randint(0, 4096, (8, 256), device=DEVICE).int()
    buffer.masked_fill_(torch.rand_like(buffer, dtype=torch.float) < 0.5, -1)
    buffer[6:] = 7
    original = buffer.clone()
    num_tokens = 5
    impl = _impl(16, buffer)
    metadata = _decode_metadata(
        [0] * 9, [256], torch.zeros(1, 4, dtype=torch.int32), width=256
    )
    indptr = metadata.paged_kv_indptr
    indptr.fill_(-3)
    indptr[0] = 0
    q = torch.empty(num_tokens, 16, HEAD_DIM, device=DEVICE)

    result = impl.forward_mqa(q, None, metadata, None)

    assert result == "forwarded"
    counts = (original[:num_tokens] >= 0).sum(-1).int()
    expected = torch.zeros(10, dtype=torch.int32, device=DEVICE)
    expected[1 : num_tokens + 1] = torch.cumsum(counts, 0)
    expected[num_tokens + 1 :] = expected[num_tokens]
    torch.testing.assert_close(indptr, expected, rtol=0, atol=0)
    _assert_front_packed(buffer[:num_tokens], counts)
    torch.testing.assert_close(buffer[num_tokens:], original[num_tokens:])


def test_registered_text_model_uses_amd_attention():
    """The ROCm text model, also built by VL, shares the attention DFly reuses."""
    from vllm.models.hy_v4 import HYV4ForCausalLM
    from vllm.models.hy_v4.amd.attention import HYV4MLAAttention

    assert HYV4ForCausalLM is AMDHYV4ForCausalLM
    assert HYV4DecoderLayer.attention_cls is HYV4MLAAttention


# Output is bf16; the reference accumulates in fp32. Errors stay below 6e-3.
BF16_TOL = dict(atol=2e-2, rtol=2e-2)


@pytest.fixture
def triton_sparse_mla(monkeypatch, request):
    """Run with `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA` set to `request.param`
    (AITER itself on, as the recipes require), restoring the gate afterwards.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA", str(request.param))
    rocm_aiter_ops.refresh_env_variables()
    yield bool(request.param)
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


def _draft_rows(seq_lens, queries_per_req, width, gen):
    """Request-local top-k rows with live entries scattered among -1, as the
    DFly draft composes them; covers a full row, a single live entry, live
    entries only at the tail and an all-dead (inactive padding) row."""
    rows, req_ids = [], []
    for req, (seq_len, num_queries) in enumerate(zip(seq_lens, queries_per_req)):
        for _ in range(num_queries):
            row = torch.full((width,), -1, dtype=torch.int32)
            live = min(seq_len, width)
            columns = torch.randperm(width, generator=gen)[:live]
            row[columns] = torch.randperm(seq_len, generator=gen)[:live].int()
            rows.append(row)
            req_ids.append(req)
    rows[1][:] = -1
    rows[2][:] = -1
    rows[2][width // 2] = 3
    rows[3][:] = -1
    rows[3][-4:] = torch.tensor([0, 2, 4, 1])
    return torch.stack(rows), req_ids


def _sparse_mla_case(
    num_heads, width, seed, block_stride=BLOCK_SIZE, rows=None, seq_lens=None
):
    """A HYV4 draft sparse-MLA call: impl, inputs, and an fp32 reference.

    `rows` defaults to `_draft_rows`; given rows split evenly over `seq_lens`.
    """
    gen = torch.Generator().manual_seed(seed)
    if rows is None:
        seq_lens = [5, 300, width + 300]
        rows, req_ids = _draft_rows(seq_lens, [4, 2, 3], width, gen)
    else:
        assert seq_lens is not None
        per_req = rows.shape[0] // len(seq_lens)
        req_ids = [token // per_req for token in range(rows.shape[0])]
    num_tokens, width = rows.shape
    blocks_per_req = -(-max(seq_lens) // BLOCK_SIZE)
    num_blocks = len(seq_lens) * blocks_per_req + 3
    block_table = torch.randperm(num_blocks, generator=gen)[
        : len(seq_lens) * blocks_per_req
    ].view(len(seq_lens), blocks_per_req)
    # A block stride past BLOCK_SIZE puts other layers' rows between blocks.
    storage = torch.randn(num_blocks, block_stride, HEAD_DIM, generator=gen) * 0.5
    kv_cache = storage.bfloat16()[:, :BLOCK_SIZE]
    q = torch.randn(num_tokens, num_heads, HEAD_DIM, generator=gen).bfloat16()
    sinks = torch.randn(num_heads, generator=gen)
    scale = HEAD_DIM**-0.5

    # Two spare rows past this step's tokens, as in the persistent buffer.
    buffer = torch.cat([rows, torch.randint(0, 99, (2, width), generator=gen)])
    impl = _impl(num_heads, buffer.int().to(DEVICE), sinks.to(DEVICE))
    # forward_mqa reads only the layer's attention-quant scales.
    layer = SimpleNamespace(
        _q_scale=torch.ones(1, device=DEVICE), _k_scale=torch.ones(1, device=DEVICE)
    )
    metadata = _decode_metadata(req_ids, seq_lens, block_table, width)

    def reference(rows, q):
        """Softmax over each row's live keys plus the per-head sink logit."""
        out = torch.zeros(num_tokens, num_heads, KV_LORA_RANK)
        cache = kv_cache.float()
        for token, row in enumerate(rows.cpu()):
            live = row[row >= 0].long()
            if live.numel() == 0:
                continue
            blocks = block_table[req_ids[token], live // BLOCK_SIZE]
            keys = cache[blocks, live % BLOCK_SIZE]
            scores = q[token].float().cpu() @ keys.T * scale
            lse = torch.logsumexp(torch.cat([scores, sinks[:, None]], dim=1), dim=1)
            out[token] = torch.exp(scores - lse[:, None]) @ keys[:, :KV_LORA_RANK]
        return out

    return impl, q.to(DEVICE), kv_cache.to(DEVICE), metadata, layer, reference


def _requires_gfx950():
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("AITER Triton sparse MLA is gfx950-only")


@pytest.mark.parametrize("triton_sparse_mla", [1], indirect=True)
@pytest.mark.parametrize(
    ("num_heads", "width", "block_stride"),
    [(8, 128, 64), (4, 128, 64), (64, 2176, 64), (16, 256, 192), (128, 384, 64)],
    ids=["tp8_minwidth", "flash_tp8", "tp1_draftwidth", "strided_cache", "128heads"],
)
@torch.inference_mode()
def test_triton_sparse_mla_matches_torch_reference(
    triton_sparse_mla, hyv4_vllm_config, num_heads, width, block_stride
):
    """With the recipe flag, the draft rows run through the real AITER Triton
    sparse MLA and equal sink-softmax attention over each row's live keys,
    across pages; dead rows carry only sink mass, so their output is zero."""
    _requires_gfx950()
    impl, q, kv_cache, metadata, layer, reference = _sparse_mla_case(
        num_heads, width, width, block_stride
    )
    rows = impl.topk_indices_buffer[: q.shape[0]].clone()
    spare = impl.topk_indices_buffer[q.shape[0] :].clone()
    assert impl.use_aiter_sparse_mla

    out, lse = impl.forward_mqa(q, kv_cache, metadata, layer)

    assert lse is None
    torch.testing.assert_close(out.float().cpu(), reference(rows, q), **BF16_TOL)
    assert not out[1].any()
    torch.testing.assert_close(impl.topk_indices_buffer[q.shape[0] :], spare)


@pytest.mark.parametrize("triton_sparse_mla", [1], indirect=True)
@torch.inference_mode()
def test_triton_sparse_mla_graph_replay_follows_new_rows(
    triton_sparse_mla, hyv4_vllm_config
):
    """Captured once, the pack, indptr rebuild and Triton read must follow
    rows whose live counts change between replays, with no host sync."""
    _requires_gfx950()
    impl, q, kv_cache, metadata, layer, reference = _sparse_mla_case(16, 256, 0)
    num_tokens = q.shape[0]
    rows = impl.topk_indices_buffer[:num_tokens]
    # Warm up on a copy: each step's metadata starts unremapped, as built.
    impl.forward_mqa(q, kv_cache, dataclasses.replace(metadata), layer)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, _ = impl.forward_mqa(q, kv_cache, metadata, layer)

    gen = torch.Generator().manual_seed(1)
    for live_fraction in (0.9, 0.1):
        new_rows = torch.randint(0, 5, rows.shape, generator=gen).int()
        dead = torch.rand(rows.shape, generator=gen) > live_fraction
        new_rows.masked_fill_(dead, -1)
        new_rows[0] = -1
        new_q = torch.randn(q.shape, generator=gen).bfloat16()
        rows.copy_(new_rows)
        q.copy_(new_q)
        graph.replay()
        torch.testing.assert_close(
            out.float().cpu(), reference(new_rows, new_q), **BF16_TOL
        )


@pytest.mark.parametrize("triton_sparse_mla", [1], indirect=True)
@torch.inference_mode()
def test_triton_sparse_mla_reads_dfly_reuse_rows(triton_sparse_mla, hyv4_vllm_config):
    """Rows from the real DFly composition (filtered target history, local
    window, causal query block; inactive requests dead) attend exactly their
    live keys once packed."""
    from vllm.models.hy_v4.common.dfly_inputs import build_dfly_reuse_indices

    _requires_gfx950()
    query_width, target_topk, local_window, width = 4, 120, 4, 128
    anchors = torch.tensor([7, 500, 300], device=DEVICE)
    active = torch.tensor([True, True, False], device=DEVICE)
    gen = torch.Generator().manual_seed(3)
    snapshots = torch.randint(-1, 520, (1, 3, target_topk), generator=gen)
    rows = torch.full((1, 3 * query_width, width), -7, dtype=torch.int32)
    rows = rows.to(DEVICE)
    build_dfly_reuse_indices(
        rows,
        snapshots.int().to(DEVICE),
        torch.zeros(1, dtype=torch.int32, device=DEVICE),
        active,
        torch.arange(3, device=DEVICE),
        anchors,
        target_topk,
        local_window,
        query_width,
        padded_batch=3,
    )
    rows = rows[0].cpu()
    assert (rows >= 0).any(dim=1).tolist() == [True] * 8 + [False] * 4
    seq_lens = (anchors + query_width).tolist()
    impl, q, kv_cache, metadata, layer, reference = _sparse_mla_case(
        16, width, 3, rows=rows, seq_lens=seq_lens
    )

    out, _ = impl.forward_mqa(q, kv_cache, metadata, layer)

    torch.testing.assert_close(out.float().cpu(), reference(rows, q), **BF16_TOL)
    assert not out[8:].any()


@pytest.mark.parametrize("triton_sparse_mla", [0], indirect=True)
@torch.inference_mode()
def test_default_sparse_mla_matches_torch_reference(
    triton_sparse_mla, hyv4_vllm_config
):
    """Without the flag, the AITER default kernel still reads the packed rows
    correctly while the KV cache is under 4 GiB (ROCm/aiter#6277)."""
    from vllm.platforms.rocm import on_mi3xx

    if not on_mi3xx():
        pytest.skip("AITER MLA attention sinks need MI3xx")
    impl, q, kv_cache, metadata, layer, reference = _sparse_mla_case(16, 256, 2)
    rows = impl.topk_indices_buffer[: q.shape[0]].clone()
    assert not impl.use_aiter_sparse_mla

    out, _ = impl.forward_mqa(q, kv_cache, metadata, layer)

    torch.testing.assert_close(out.float().cpu(), reference(rows, q), **BF16_TOL)
    assert not out[1].any()


@pytest.mark.parametrize("triton_sparse_mla", [0, 1], indirect=True)
def test_warns_unless_triton_sparse_mla_enabled(monkeypatch, triton_sparse_mla):
    """HYV4 on ROCm warns that the default decode kernel misreads KV caches
    past 4 GiB unless the Triton sparse MLA kernel is enabled."""
    _requires_gfx950()
    warnings: list[str] = []
    monkeypatch.setattr(
        aiter_mla_sparse.logger,
        "warning_once",
        lambda msg, *args, **kwargs: warnings.append(msg % args),
    )

    aiter_mla_sparse.warn_if_aiter_triton_sparse_mla_disabled()

    if triton_sparse_mla:
        assert not warnings
    else:
        (warning,) = warnings
        assert "VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1" in warning
        assert "ROCm/aiter#6277" in warning


def _amd_attention(vllm_config: VllmConfig, num_local_heads: int):
    """A sparse ROCm HY4 attention layer; TP1, so all heads are local."""
    from vllm.models.hy_v4.amd.attention import HYV4MLAAttention
    from vllm.platforms.rocm import on_mi3xx

    if not on_mi3xx():
        pytest.skip("ROCm sparse MLA sinks need MI3xx")

    with set_default_torch_dtype(torch.bfloat16), torch.device(DEVICE):
        return HYV4MLAAttention(
            vllm_config=vllm_config,
            config=vllm_config.model_config.hf_text_config,
            hidden_size=256,
            num_heads=num_local_heads,
            q_lora_rank=128,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=128,
            qk_rope_head_dim=ROPE_DIM,
            v_head_dim=128,
            cache_config=vllm_config.cache_config,
            prefix="model.layers.0.self_attn",
            topk_indices_buffer=torch.empty(64, 2048, dtype=torch.int32),
        )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("num_local_heads", [8, 16, 64])
def test_amd_sink_backend_is_rocm_sparse_mla(
    hyv4_vllm_config, kv_cache_dtype, num_local_heads
):
    """HY4's 64 heads at TP1..TP8 bind the sink-capable ROCm sparse MLA
    backend, never the CUDA-only FLASHMLA_SPARSE fallback."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    layer = _amd_attention(hyv4_vllm_config, num_local_heads)
    with set_default_torch_dtype(torch.bfloat16):
        backend = layer._resolve_sink_backend(kv_cache_dtype)

    assert issubclass(backend, ROCMAiterMLASparseBackend)
    assert backend.supports_sink()


@pytest.mark.parametrize("selection", ["no_sink", "raises"])
def test_amd_sink_backend_disables_sink_without_fallback(
    monkeypatch, hyv4_vllm_config, selection
):
    """A selected backend that cannot apply sinks, or a failed selection,
    disables the sink bias instead of binding a CUDA-only backend."""
    from vllm.models.hy_v4.amd import attention

    layer = _amd_attention(hyv4_vllm_config, 16)

    class NoSinkBackend:
        @staticmethod
        def get_name():
            return "NO_SINK"

        @staticmethod
        def supports_sink():
            return False

    def select(**kwargs):
        if selection == "raises":
            raise ValueError("no backend")
        return NoSinkBackend

    monkeypatch.setattr(attention, "get_attn_backend", select)
    warnings: list[str] = []
    monkeypatch.setattr(
        attention.logger,
        "warning_once",
        lambda msg, *args, **kwargs: warnings.append(msg % args),
    )

    assert layer._resolve_sink_backend("auto") is None
    (warning,) = warnings
    assert "sink bias is disabled" in warning
