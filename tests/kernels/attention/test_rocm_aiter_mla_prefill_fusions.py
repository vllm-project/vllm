# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ROCm AITER MLA prefill fusions.

The fused chunked-context gather (``gather_kv_b_proj`` fed with the per-token
KV indices the metadata builder expands) and the fused ``[k_nope | k_pe]``
concat are each checked against the unfused ``MLACommonImpl`` path, using the
production builder and a bare ``AiterMLAImpl``.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv


def _fused_gather_available() -> bool:
    if not (current_platform.is_rocm() and torch.cuda.is_available()):
        return False
    from vllm._aiter_ops import is_aiter_found_and_supported

    return is_aiter_found_and_supported()


pytestmark = pytest.mark.skipif(
    not _fused_gather_available(),
    reason="needs ROCm and AITER",
)


def _on_gfx950() -> bool:
    if not _fused_gather_available():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


requires_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="the fused gather is only enabled on gfx950"
)

KV_LORA_RANK = 512  # gather_kv_b_proj asserts KV_CDim == 4 * 128
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = 128
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM
NUM_HEADS = 16
SCALE = 1.0 / math.sqrt(QK_HEAD_DIM)
BLOCK_SIZE = 16
MAX_MODEL_LEN = 512
DTYPE = torch.bfloat16
QUERY_LEN = 32

# The context workspace holds 8 * MAX_MODEL_LEN rows: ten near-full contexts
# (one empty) span two chunks, three short ones fit in one.
MULTI_CHUNK_CONTEXTS = [480] * 9 + [0]
SINGLE_CHUNK_CONTEXTS = [200, 64, 333]

# k/v come out of a different GEMM than torch's and can differ by one bf16 ulp;
# measured output error on MI355 is 2e-3.
ATOL, RTOL = 2e-2, 2e-2


@pytest.fixture(autouse=True)
def _workspace_manager():
    """The builder draws its context workspace from the global manager."""
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        reset_workspace_manager,
    )

    init_workspace_manager(torch.device("cuda"))
    yield
    reset_workspace_manager()


class _KVBProj(torch.nn.Module):
    """Stand-in for kv_b_proj; vLLM linear layers return (output, bias)."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(
                NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
                KV_LORA_RANK,
                dtype=DTYPE,
                device=device,
            )
            * KV_LORA_RANK**-0.5
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.nn.functional.linear(x, self.weight), None


def _make_impl(device: torch.device):
    """Minimal AiterMLAImpl exposing what the prefill-context paths read."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAImpl

    impl = object.__new__(AiterMLAImpl)
    impl.num_heads = NUM_HEADS
    impl.kv_lora_rank = KV_LORA_RANK
    impl.qk_nope_head_dim = QK_NOPE_HEAD_DIM
    impl.qk_rope_head_dim = QK_ROPE_HEAD_DIM
    impl.v_head_dim = V_HEAD_DIM
    impl.kv_cache_dtype = "auto"
    impl.kv_b_proj = _KVBProj(device)
    impl._use_flashinfer_concat_mla_k = False
    impl._use_fused_mla_kv_concat = hasattr(torch.ops._C, "fused_kimi_k3_mla_kv_concat")
    return impl


def _build_metadata(context_lens: list[int], device: torch.device):
    """Run the production builder; returns (metadata, num_blocks)."""
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_vllm_config,
    )
    from vllm.config.vllm import set_current_vllm_config
    from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
        AiterMLAMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    seq_lens = [context + QUERY_LEN for context in context_lens]
    num_blocks = len(seq_lens) * cdiv(max(seq_lens), BLOCK_SIZE)
    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-R1",
        max_model_len=MAX_MODEL_LEN,
        num_gpu_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        max_num_seqs=len(seq_lens),
        hf_config_override={"num_attention_heads": NUM_HEADS},
    )
    spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
        cache_dtype_str=vllm_config.cache_config.cache_dtype,
    )
    prefill_backend = get_mla_prefill_backend(vllm_config)(
        num_heads=NUM_HEADS,
        scale=SCALE,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        vllm_config=vllm_config,
    )
    vllm_config.compilation_config.static_forward_context["placeholder"] = (
        SimpleNamespace(
            prefill_backend=prefill_backend,
            q_lora_rank=None,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
        )
    )
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=[QUERY_LEN] * len(seq_lens))
    with set_current_vllm_config(vllm_config):
        builder = AiterMLAMetadataBuilder(spec, ["placeholder"], vllm_config, device)
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, BLOCK_SIZE, device, arange_block_indices=True
        )
        metadata = builder.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )
    return metadata, num_blocks


def _reference_chunk_kv_indices(prefill, chunk) -> torch.Tensor:
    """Flat per-token cache rows of one chunk, straight from the block table."""
    block_table = prefill.block_table[chunk.request_slice]
    cu_seq_lens = chunk.cu_seq_lens.tolist()
    starts = chunk.starts.tolist()
    rows = []
    for req, (lo, hi) in enumerate(zip(cu_seq_lens, cu_seq_lens[1:])):
        positions = torch.arange(
            starts[req], starts[req] + hi - lo, device=block_table.device
        )
        rows.append(
            block_table[req, positions // BLOCK_SIZE] * BLOCK_SIZE
            + positions % BLOCK_SIZE
        )
    return torch.cat(rows).to(torch.int32)


@requires_gfx950
@pytest.mark.parametrize("context_lens", [SINGLE_CHUNK_CONTEXTS, MULTI_CHUNK_CONTEXTS])
def test_context_chunk_kv_indices_follow_the_block_table(context_lens):
    """The builder's per-token indices match a block-table walk per chunk."""
    device = torch.device("cuda")
    metadata, _ = _build_metadata(context_lens, device)
    chunks = metadata.prefill.chunked_context.chunks
    kv_indices = metadata.context_chunk_kv_indices

    assert kv_indices is not None and len(kv_indices) == len(chunks)
    for chunk in chunks:
        expected = _reference_chunk_kv_indices(metadata.prefill, chunk)
        assert kv_indices[chunk.index].dtype == torch.int32
        assert torch.equal(kv_indices[chunk.index], expected)


@requires_gfx950
@pytest.mark.parametrize(
    ("context_lens", "expected_chunks"),
    [(SINGLE_CHUNK_CONTEXTS, 1), (MULTI_CHUNK_CONTEXTS, 2)],
)
def test_fused_prefill_context_matches_unfused(
    context_lens, expected_chunks, monkeypatch
):
    """gather_kv_b_proj + LSE merge reproduce the gather -> GEMM -> copy path."""
    from vllm.model_executor.layers.attention.mla_attention import MLACommonImpl

    torch.manual_seed(0)
    device = torch.device("cuda")
    metadata, num_blocks = _build_metadata(context_lens, device)
    assert len(metadata.prefill.chunked_context.chunks) == expected_chunks

    impl = _make_impl(device)
    impl._use_fused_mla_kv_concat = False  # reference stays unfused
    kv_cache = torch.randn(
        num_blocks, BLOCK_SIZE, HEAD_SIZE, dtype=DTYPE, device=device
    )
    q = torch.randn(
        metadata.num_actual_tokens, NUM_HEADS, QK_HEAD_DIM, dtype=DTYPE, device=device
    )
    k_scale = torch.ones(1, dtype=torch.float32, device=device)

    # A silent fallback must not pass as base-vs-base.
    import aiter.ops.triton.gather_kv_b_proj as aiter_gather

    gather_kv_b_proj = aiter_gather.gather_kv_b_proj
    launches = 0

    def counted(*args, **kwargs):
        nonlocal launches
        launches += 1
        return gather_kv_b_proj(*args, **kwargs)

    monkeypatch.setattr(aiter_gather, "gather_kv_b_proj", counted)

    output, lse = impl._compute_prefill_context(q, kv_cache, metadata, k_scale)
    assert launches == expected_chunks
    expected_output, expected_lse = MLACommonImpl._compute_prefill_context(
        impl, q, kv_cache, metadata, k_scale
    )

    torch.testing.assert_close(output, expected_output, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(lse, expected_lse, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "unsupported",
    [
        "kv_lora_rank",
        "fp8_query",
        "fp8_ds_mla_cache",
        "quantized_kv_b_proj",
        "noncontiguous_cache",
    ],
)
def test_fused_gather_declines_unsupported_inputs(unsupported):
    """Anything gather_kv_b_proj cannot take falls back to the base path."""
    device = torch.device("cuda")
    impl = _make_impl(device)
    prefill = SimpleNamespace(q_data_type=DTYPE)
    kv_cache = torch.empty(4, BLOCK_SIZE, HEAD_SIZE, dtype=DTYPE, device=device)
    q = torch.empty(QUERY_LEN, NUM_HEADS, QK_HEAD_DIM, dtype=DTYPE, device=device)
    assert impl._can_fuse_context_gather(q, prefill, kv_cache)

    if unsupported == "kv_lora_rank":
        impl.kv_lora_rank = 2 * KV_LORA_RANK
    elif unsupported == "fp8_query":
        prefill.q_data_type = current_platform.fp8_dtype()
    elif unsupported == "fp8_ds_mla_cache":
        impl.kv_cache_dtype = "fp8_ds_mla"
    elif unsupported == "quantized_kv_b_proj":
        impl.kv_b_proj.weight_scale = torch.ones(1, device=device)
    else:
        kv_cache = kv_cache.transpose(0, 1)
    assert not impl._can_fuse_context_gather(q, prefill, kv_cache)


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "fused_kimi_k3_mla_kv_concat"),
    reason="this build has no fused_kimi_k3_mla_kv_concat op",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_k_concat_matches_torch_cat(dtype):
    """The fused (bf16) and fallback (fp32) concat both equal torch.cat."""
    device = torch.device("cuda")
    impl = _make_impl(device)
    k_nope = torch.randn(
        QUERY_LEN, NUM_HEADS, QK_NOPE_HEAD_DIM, dtype=dtype, device=device
    )
    k_pe = torch.randn(QUERY_LEN, 1, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)

    k = impl._concat_k_nope_k_pe(k_nope, k_pe)

    expected = torch.cat([k_nope, k_pe.expand(-1, NUM_HEADS, -1)], dim=-1)
    assert torch.equal(k, expected)


@requires_gfx950
@pytest.mark.parametrize("tokens", [0, 1, 37, 2048])
def test_flydsl_fp8_quantization_strides_and_zero(tokens):
    """Strided inputs obey E4M3 rounding bounds, including zero/empty tensors."""
    from vllm.v1.attention.ops.rocm_aiter_mla_fp8_quant import (
        fused_qkv_per_tensor_quant,
    )

    torch.manual_seed(11)
    q = torch.randn(tokens, 12, 384, device="cuda", dtype=DTYPE)[..., ::2]
    kv = torch.randn(tokens, 12, 320, device="cuda", dtype=DTYPE)
    k, v = kv[..., :192], kv[..., 192:]
    for zero in (False, True):
        if zero:
            q.zero_()
            kv.zero_()
        result = fused_qkv_per_tensor_quant(q, k, v)
        for x, x8, descale in zip((q, k, v), result[:3], result[3:6]):
            amax = x.float().abs().amax() if tokens else x.new_zeros(()).float()
            expected_scale = torch.where(amax > 0, amax / 448, 1e-6)
            torch.testing.assert_close(descale[0], expected_scale)
            assert x8.dtype == torch.float8_e4m3fn and x8.is_contiguous()
            error = (x8.float() * descale - x.float()).abs()
            # Half an E4M3 ULP, plus FP32 reciprocal/multiply rounding.
            bound = x.float().abs() * (1 / 16 + 2e-7) + descale / 1024
            assert torch.all(error <= bound)
        for descale, gather_scale in zip(result[4:6], result[6:]):
            torch.testing.assert_close(gather_scale, descale.clamp_min(1e-6) * 2)


@requires_gfx950
@pytest.mark.parametrize("preshuffled", [False, True])
def test_flydsl_fp8_projection_and_fmha_graph_replay(preshuffled):
    """Check row selection, FP8 output, descales/LSE, and changed graph inputs."""
    from aiter.ops.shuffle import shuffle_weight

    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_fp8_prefill import (
        FP8PrefillContext,
        ProjectionWeight,
        project,
    )

    torch.manual_seed(7)
    device = torch.device("cuda")
    heads = 12
    cache = torch.randn(512, 576, device=device).to(torch.float8_e4m3fn)
    cache_scale = torch.tensor([0.25], device=device)
    indices = torch.randperm(512, device=device)[:386].to(torch.int32)
    weight = torch.randn(heads * 256, 512, device=device).to(torch.float8_e4m3fn)
    weight_scale = torch.linspace(0.005, 0.02, heads * 256, device=device)
    params = ProjectionWeight(
        shuffle_weight(weight, (16, 16)) if preshuffled else weight,
        weight_scale,
        preshuffled,
    )
    q = torch.randn(20, heads, 192, device=device, dtype=DTYPE) * 2
    new_k = torch.randn_like(q) * 2
    new_v = torch.randn(20, heads, 128, device=device, dtype=DTYPE) * 3
    chunk = MLACommonPrefillMetadata.ContextChunk(
        index=0,
        request_slice=slice(0, 2),
        token_slice=slice(0, 20),
        continuation_token_end=0,
        is_continuation=False,
        num_context_tokens=386,
        query_start_loc=torch.tensor([0, 7, 20], device=device, dtype=torch.int32),
        max_query_len=13,
        cu_seq_lens=torch.tensor([0, 129, 386], device=device, dtype=torch.int32),
        starts=torch.zeros(2, device=device, dtype=torch.int32),
        max_seq_len=257,
        seq_lens=torch.tensor([129, 257], device=device, dtype=torch.int32),
        token_to_seq=torch.repeat_interleave(
            torch.arange(2, device=device, dtype=torch.int32),
            torch.tensor([129, 257], device=device),
        ),
        all_rows_active=True,
    )

    def run():
        state = FP8PrefillContext()
        state.prepare(q, new_k, new_v)
        kv8 = project(cache, cache_scale, indices, params, heads, state.gather_scales)
        output, lse = state.run(chunk, q, kv8, SCALE)
        key = (q.data_ptr(), tuple(q.shape), tuple(q.stride()))
        q8, qs = state.queries[key]
        return output, lse, q8, qs, kv8

    def check(result):
        output, lse, q8, qs, (k8, v8, ks, vs) = result
        assert k8.dtype == v8.dtype == torch.float8_e4m3fn
        latent = cache.double()[indices.long()] * cache_scale.double()
        expanded = (
            latent[:, :512] @ (weight.double() * weight_scale.double()[:, None]).t()
        ).view(-1, heads, 256)
        expected_k = torch.cat(
            (expanded[..., :128], latent[:, None, 512:].expand(-1, heads, -1)),
            dim=-1,
        )
        for value, expected, descale in (
            (k8, expected_k, ks),
            (v8, expanded[..., 128:], vs),
        ):
            error = value.double() * descale.double() - expected
            assert torch.isfinite(error).all()
            assert error.square().mean().sqrt() / expected.square().mean().sqrt() < 0.04
            assert error.abs().max() / expected.abs().max() < 0.07
            assert torch.all(expected.abs() <= 448 * descale.double())
        qf = q8.float() * qs
        kf = k8.float() * ks
        vf = v8.float() * vs
        for qslice, kslice in (
            (slice(0, 7), slice(0, 129)),
            (slice(7, 20), slice(129, 386)),
        ):
            logits = torch.einsum("qhd,khd->hqk", qf[qslice], kf[kslice]) * SCALE
            expected_out = torch.einsum("hqk,khd->qhd", logits.softmax(-1), vf[kslice])
            torch.testing.assert_close(
                output[qslice].float(), expected_out, atol=2e-3, rtol=2e-3
            )
            torch.testing.assert_close(
                lse[:, qslice], logits.logsumexp(-1), atol=1e-3, rtol=0
            )

    for _ in range(2):
        check(run())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    # Reuse every allocation while changing Q and the scale-producing K/V.
    q.mul_(0.5)
    new_k.mul_(1.5)
    new_v.mul_(0.75)
    graph.replay()
    check(captured)


@requires_gfx950
def test_flydsl_fp8_native_weight_layout(default_vllm_config):
    """The native online layer's stored weight layout reaches the same matmul."""
    from vllm.config.model import ModelConfig
    from vllm.config.quantization import QuantizationConfigArgs
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )
    from vllm.model_executor.utils import replace_parameter
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.ops.rocm_aiter_mla_fp8_prefill import (
        project,
        projection_weight,
    )

    torch.manual_seed(23)
    default_vllm_config.model_config = ModelConfig(dtype="bfloat16")
    quant_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={"*.self_attn.kv_b_proj": "fp8_per_channel"})
    )
    with set_default_torch_dtype(DTYPE), torch.device("cuda"):
        layer = ColumnParallelLinear(
            512,
            12 * 256,
            bias=False,
            params_dtype=DTYPE,
            quant_config=quant_config,
            prefix="model.layers.0.self_attn.kv_b_proj",
            disable_tp=True,
        )
        original = torch.randn(12 * 256, 512, dtype=DTYPE) * 0.02
        replace_parameter(layer, "weight", original)
        layer.quant_method.process_weights_after_loading(layer)
        params = projection_weight(layer, 12)
        # Adapting the layout must not create another copy of the weights.
        assert params.weight.data_ptr() == layer.weight.data_ptr()
        cache = torch.randn(31, 576).to(torch.float8_e4m3fn)
        cache_scale = torch.tensor([0.125], dtype=torch.float32)
        indices = torch.arange(30, -1, -1, dtype=torch.int32)
        k, v = project(cache, cache_scale, indices, params, 12)
        latent = cache.float()[indices.long()] * cache_scale
        expected = (latent[:, :512] @ original.float().t()).view(-1, 12, 256)
        for value, reference in (
            (k[..., :128], expected[..., :128]),
            (v, expected[..., 128:]),
        ):
            error = value.float() - reference
            assert (
                error.square().mean().sqrt() / reference.square().mean().sqrt() < 0.04
            )
        torch.testing.assert_close(
            k[..., 128:],
            latent[:, None, 512:].expand(-1, 12, -1).to(DTYPE),
            rtol=0,
            atol=0,
        )


@requires_gfx950
def test_flydsl_fp8_forward_scope_clears_after_error(default_vllm_config):
    """An exception cannot leave Q or descales alive on a shared backend."""
    from vllm.v1.attention.backends.mla.prefill.aiter_flash_attn import (
        AiterFlashAttnPrefillBackend,
    )

    backend = AiterFlashAttnPrefillBackend(
        12,
        SCALE,
        512,
        128,
        64,
        128,
        default_vllm_config,
    )
    q = torch.ones(1, 12, 192, device="cuda", dtype=DTYPE)
    v = torch.ones(1, 12, 128, device="cuda", dtype=DTYPE)
    with (
        pytest.raises(RuntimeError, match="sentinel"),
        backend.fp8_context_forward() as state,
    ):
        state.prepare(q, q, v)
        assert state.gather_scales is not None
        old_scale = state.gather_scales.clone()
        raise RuntimeError("sentinel")
    assert backend.fp8_context is None
    q.mul_(2)
    with backend.fp8_context_forward() as state:
        state.prepare(q, q, v)
        assert state.gather_scales is not None
        assert state.gather_scales[0] == 2 * old_scale[0]
    assert backend.fp8_context is None
