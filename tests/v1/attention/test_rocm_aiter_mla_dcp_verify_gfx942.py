# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal target verify under DCP on gfx942 (MI300/MI325).

Reproduces the attention shape of

    vllm serve moonshotai/Kimi-K3 --tensor-parallel-size 8 \
        --decode-context-parallel-size 8 \
        --speculative-config '{"method": "dspark", "num_speculative_tokens": 5}'

96 gathered heads, bf16 KV, qlen 6 and the 768-token blocks the hybrid KDA
layers impose. Segmented MLA tiles those blocks at 128 tokens and needs 128 KiB
of LDS (gfx942 has 64 KiB). DeepSeek-V3 with MTP and an fp8 KV cache at DCP 8
hits the same limit through its 128 gathered heads. By default every gfx942
verify shape runs on AITER's single-token asm decode. With
VLLM_ROCM_AITER_MLA_GFX942_DCP_VERIFY=triton, shapes segmented MLA fits stay on
it and the rest take the generic Triton split-KV kernel.

The real metadata builder and ``forward_mqa`` run on one GPU with the DCP ranks
simulated one after another. That is sound because this path has no
collective: the query all-gather happens in the layer, before ``forward_mqa``.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

if not current_platform.is_rocm():
    pytest.skip("ROCm AITER MLA tests", allow_module_level=True)

from vllm._aiter_ops import is_aiter_found  # noqa: E402

# DeepSeek-R1 supplies the MLA dims, which Kimi-K3 and DeepSeek-V3 share; only
# the head count is overridden.
MODEL = "deepseek-ai/DeepSeek-R1"
KV_LORA_RANK, ROPE_DIM, NOPE_DIM = 512, 64, 128
HEAD_SIZE = KV_LORA_RANK + ROPE_DIM
SCALE = (NOPE_DIM + ROPE_DIM) ** -0.5
DCP = 8
# Each rank holds ~800 tokens per request, so its shard spans several blocks.
NUM_REQS, CTX = 4, 6400
FP8_KV_SCALE = 0.05
FP8_Q_SCALE = 0.02


def _on_gfx942() -> bool:
    if not (is_aiter_found() and torch.accelerator.is_available()):
        return False
    from vllm.platforms.rocm import on_gfx942

    return on_gfx942()


pytestmark = pytest.mark.skipif(not _on_gfx942(), reason="needs ROCm + AITER on gfx942")


class _FakeGroup:
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank_in_group = self.rank = self.local_rank = rank

    def all_gather(self, tensor, dim=0):
        raise RuntimeError("DCP verify must not issue a collective")


@pytest.fixture
def fake_dcp_groups(monkeypatch):
    """Point every importer of get_dcp_group at a settable rank stub."""
    import vllm.distributed.parallel_state as ps
    import vllm.model_executor.layers.attention.mla_attention as mla_attention
    import vllm.v1.attention.backends.mla.rocm_aiter_mla as backend

    group = _FakeGroup(DCP, 0)
    for module in (ps, mla_attention, backend):
        monkeypatch.setattr(module, "get_dcp_group", lambda: group)
    monkeypatch.setattr(ps, "get_tp_group", lambda: _FakeGroup(1, 0))
    return group


def _vllm_config(heads_per_rank: int, qlen: int, fp8_kv: bool):
    from tests.v1.attention.utils import create_vllm_config

    config = create_vllm_config(
        model_name=MODEL,
        max_model_len=CTX + qlen,
        max_num_seqs=NUM_REQS,
        max_num_batched_tokens=4096,
    )
    config.model_config.model_arch_config.total_num_attention_heads = heads_per_rank
    # The reorder threshold reads only these two fields; without them a
    # multi-token row is classified as prefill and never reaches decode.
    config.speculative_config = SimpleNamespace(
        num_speculative_tokens=qlen - 1, parallel_drafting=False
    )
    config.parallel_config.decode_context_parallel_size = DCP
    config.parallel_config.cp_kv_cache_interleave_size = 1
    if fp8_kv:
        config.cache_config.cache_dtype = "fp8"
    return config


def _ctx_layer():
    """The registered-layer fields MLACommonMetadataBuilder.__init__ reads."""
    from vllm.v1.attention.ops.dcp import MLADCPManager

    dcp_manager = object.__new__(MLADCPManager)
    dcp_manager.init_kv_gather = lambda *args, **kwargs: None
    # Only chunked-context prefill uses the cloned backend.
    prefill_backend = SimpleNamespace()
    prefill_backend.clone = lambda: prefill_backend
    return SimpleNamespace(
        non_causal_multi_token_decode=False,
        q_lora_rank=None,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=NOPE_DIM,
        qk_rope_head_dim=ROPE_DIM,
        v_head_dim=NOPE_DIM,
        dcp_manager=dcp_manager,
        prefill_backend=prefill_backend,
    )


def _impl(dcp_rank: int, heads_per_rank: int, fp8_kv: bool):
    """forward_mqa needs these attributes, not the weight-loading __init__."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAImpl

    impl = object.__new__(AiterMLAImpl)
    impl.num_heads = heads_per_rank
    impl.kv_lora_rank = KV_LORA_RANK
    impl.qk_rope_head_dim = ROPE_DIM
    impl.dcp_world_size = DCP
    impl.dcp_rank = dcp_rank
    impl.pcp_world_size = 1
    impl.kv_cache_dtype = "fp8" if fp8_kv else "auto"
    impl.scale = SCALE
    impl._sm_count = current_platform.num_compute_units()
    return impl


def _run_rank(config, spec, layer_name, group, rank, q, q_scale, kv_cache_src, k_scale):
    """Shard the KV round-robin onto ``rank`` and run its verify step."""
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAMetadataBuilder

    device = q.device
    qlen, heads = q.shape[1], q.shape[2] // DCP
    seq_len = CTX + qlen
    positions = torch.arange(rank, seq_len, DCP, device=device)
    local_len = positions.numel()
    # Production block tables span max_model_len, so every rank's table is at
    # least as wide as the largest shard needs.
    blocks_per_req = cdiv(cdiv(seq_len, DCP), spec.block_size)
    kv_cache = torch.zeros(
        NUM_REQS * blocks_per_req,
        spec.block_size,
        HEAD_SIZE,
        dtype=kv_cache_src.dtype,
        device=device,
    )
    kv_cache.view(NUM_REQS, -1, HEAD_SIZE)[:, :local_len] = kv_cache_src[:, positions]
    block_table = torch.arange(
        NUM_REQS * blocks_per_req, dtype=torch.int32, device=device
    ).view(NUM_REQS, blocks_per_req)

    query_start_loc = torch.arange(
        0, NUM_REQS * qlen + 1, qlen, dtype=torch.int32, device=device
    )
    local_lens = torch.full((NUM_REQS,), local_len, dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.full((NUM_REQS,), seq_len, dtype=torch.int32, device=device),
        num_reqs=NUM_REQS,
        num_actual_tokens=NUM_REQS * qlen,
        max_query_len=qlen,
        max_seq_len=seq_len,
        block_table_tensor=block_table,
        slot_mapping=torch.arange(NUM_REQS * qlen, dtype=torch.int64, device=device),
        causal=True,
        dcp_local_seq_lens=local_lens.to(device),
        dcp_local_seq_lens_cpu_upper_bound=local_lens,
    )

    group.rank_in_group = group.rank = group.local_rank = rank
    builder = AiterMLAMetadataBuilder(spec, [layer_name], config, device)
    metadata = builder.build(0, common)
    layer = SimpleNamespace(
        _q_scale=torch.tensor(q_scale, device=device),
        _k_scale=torch.tensor(k_scale, device=device),
    )
    impl = _impl(rank, heads, kv_cache_src.dtype != torch.bfloat16)
    output, lse = impl.forward_mqa(q.flatten(0, 1), kv_cache, metadata, layer)
    return metadata.decode.dcp_route.name, output, lse


@pytest.mark.parametrize(
    "heads_per_rank, qlen, block_size, fp8_kv, route",
    [
        (12, 6, 768, False, "ASM_ROWS"),
        (12, 6, 768, False, "TRITON"),
        (12, 6, 64, False, "ASM_ROWS"),
        (12, 6, 64, False, "SEGMENTED"),
        (8, 4, 16, False, "ASM_ROWS"),
        (16, 2, 16, True, "ASM_ROWS"),
        (16, 2, 16, True, "TRITON"),
    ],
    ids=[
        "kimi-k3-hybrid-kda-blocks-asm",
        "kimi-k3-hybrid-kda-blocks-triton",
        "segmented-fits-asm",
        "segmented-fits-triton",
        "64-heads-bf16-padded-asm",
        "deepseek-v3-mtp-fp8-kv-asm",
        "deepseek-v3-mtp-fp8-kv-triton",
    ],
)
@torch.inference_mode()
def test_dcp_verify_matches_attention(
    fake_dcp_groups, monkeypatch, heads_per_rank, qlen, block_size, fp8_kv, route
):
    monkeypatch.setenv(
        "VLLM_ROCM_AITER_MLA_GFX942_DCP_VERIFY",
        "asm" if route == "ASM_ROWS" else "triton",
    )
    from vllm.config import set_current_vllm_config
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        is_workspace_manager_initialized,
    )

    device = torch.device("cuda", torch.accelerator.current_device_index())
    if not is_workspace_manager_initialized():
        init_workspace_manager(device)
    torch.manual_seed(0)
    seq_len = CTX + qlen
    q = torch.randn(
        NUM_REQS,
        qlen,
        heads_per_rank * DCP,
        HEAD_SIZE,
        dtype=torch.bfloat16,
        device=device,
    )
    kv = torch.randn(NUM_REQS, seq_len, HEAD_SIZE, device=device).clamp_(-4, 4)
    if fp8_kv:
        # MLAAttention hands this backend an fp8 query whenever the KV cache is.
        q_scale, k_scale = FP8_Q_SCALE, FP8_KV_SCALE
        q = (q.clamp(-4, 4) / q_scale).to(current_platform.fp8_dtype())
        kv_cache_src = (kv / k_scale).to(current_platform.fp8_dtype())
    else:
        q_scale = k_scale = 1.0
        kv_cache_src = kv.to(torch.bfloat16)
    q_ref = q.float() * q_scale
    kv_ref = kv_cache_src.float() * k_scale

    config = _vllm_config(heads_per_rank, qlen, fp8_kv)
    layer_name = "model.layers.0.self_attn.attn"
    with set_current_vllm_config(config):
        config.compilation_config.static_forward_context[layer_name] = _ctx_layer()
        spec = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=HEAD_SIZE,
            dtype=kv_cache_src.dtype,
            cache_dtype_str="fp8" if fp8_kv else None,
            non_causal_multi_token_decode=False,
        )
        shards = [
            _run_rank(
                config,
                spec,
                layer_name,
                fake_dcp_groups,
                rank,
                q,
                q_scale,
                kv_cache_src,
                k_scale,
            )
            for rank in range(DCP)
        ]

    assert {shard_route for shard_route, _, _ in shards} == {route}
    lses = torch.stack([lse.float() for _, _, lse in shards])
    weights = torch.exp(lses - lses.max(dim=0).values)
    merged = (
        weights.unsqueeze(-1) * torch.stack([out.float() for _, out, _ in shards])
    ).sum(dim=0) / weights.sum(dim=0).unsqueeze(-1)

    # Query token t of a request sees the context plus verify tokens 0..t.
    scores = torch.einsum("rthd,rld->rthl", q_ref, kv_ref) * SCALE
    visible = torch.arange(seq_len, device=device) <= CTX + torch.arange(
        qlen, device=device
    ).view(-1, 1)
    scores.masked_fill_(~visible[None, :, None, :], float("-inf"))
    reference = torch.einsum(
        "rthl,rld->rthd", scores.softmax(dim=-1), kv_ref[..., :KV_LORA_RANK]
    )
    torch.testing.assert_close(
        merged.view_as(reference), reference, rtol=2e-2, atol=2e-2
    )


def test_asm_rows_split_count():
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import _asm_rows_num_kv_splits

    mi325x_cus = 304
    # One request verifying 2 tokens: KV is the only parallelism, so split past
    # the 16 AITER's own heuristic stops at.
    assert _asm_rows_num_kv_splits(2, 16384, mi325x_cus) == 64
    # Short rows keep at least 128 tokens per split.
    assert _asm_rows_num_kv_splits(2, 1024, mi325x_cus) == 8
    # Rows alone fill the GPU, but one split would skip the LSE reducer.
    assert _asm_rows_num_kv_splits(192, 16384, mi325x_cus) == 2
    assert _asm_rows_num_kv_splits(2, 16, mi325x_cus) == 2


@pytest.mark.parametrize(
    "fp8_kv, num_heads, block_size",
    [
        (False, 96, 768),
        (False, 128, 64),
        (True, 64, 128),
        (True, 128, 16),
    ],
)
def test_segmented_lds_gate_matches_the_kernel(fp8_kv, num_heads, block_size):
    """The gate must flag exactly the shapes AITER segmented MLA cannot compile."""
    from triton.runtime.errors import OutOfResources

    from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
        _get_segmented_mla_decode,
        _segmented_mla_fits_lds,
        _segmented_mla_page_size,
    )

    page = _segmented_mla_page_size(block_size)
    kv_dtype = current_platform.fp8_dtype() if fp8_kv else torch.bfloat16
    q = torch.zeros(1, num_heads, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
    kv = torch.zeros(1, page, 1, HEAD_SIZE, dtype=kv_dtype, device="cuda")
    one_row = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    try:
        _get_segmented_mla_decode()(
            q,
            kv,
            None,
            one_row,
            torch.tensor([page], dtype=torch.int32, device="cuda"),
            page,
            torch.zeros(1, 1, dtype=torch.int32, device="cuda"),
            SCALE,
            KV_LORA_RANK,
            ROPE_DIM,
            causal=True,
            q_descale=None,
            kv_descale=torch.tensor(1.0, device="cuda"),
            skip_reduce=True,
        )
        compiles = True
    except OutOfResources:
        compiles = False
    assert _segmented_mla_fits_lds(num_heads, block_size, fp8_kv) == compiles
