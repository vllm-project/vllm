# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm._custom_ops as ops
import vllm.models.muse_spark.common.ops.fused_qk_norm_rope_kv as fused_ops
from vllm.platforms import current_platform

HEAD_DIM = 64


pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike() or not torch.cuda.is_available(),
    reason="Muse Spark kernel requires a CUDA or ROCm GPU",
)


def make_cos_sin_cache(max_pos, rotary_dim, base, device, dtype=torch.float32):
    """The table `RotaryEmbedding` builds: cos||sin, fp32 for Muse-Spark."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def qk_norm_rope_ref(x, positions, cos_sin_cache, rotary_dim, eps):
    """fp32 reference for one q or k branch, rounded once; no table means NoPE."""
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(dim=-1, keepdim=True) + eps)
    if cos_sin_cache is not None:
        half = rotary_dim // 2
        cs = cos_sin_cache[positions].float()
        cos = cs[..., :half].unsqueeze(1)
        sin = cs[..., half:].unsqueeze(1)
        # GPT-J interleaved: the rotated pair is (2i, 2i+1).
        x0 = out[..., :rotary_dim:2]
        x1 = out[..., 1:rotary_dim:2]
        out = out.clone()
        out[..., :rotary_dim:2] = x0 * cos - x1 * sin
        out[..., 1:rotary_dim:2] = x0 * sin + x1 * cos
    return out.to(x.dtype)


def assert_fp8_cache_close(got, expected, exact_fraction=0.999):
    """Compare raw cache codes in the platform's FP8 encoding."""
    diff = (got.int() - expected.int()).abs()
    within = diff <= 1
    assert bool(within.all()), (
        f"fp8 cache differs by more than one code in {int((~within).sum())} "
        f"of {diff.numel()} elements"
    )
    exact = float((diff == 0).float().mean())
    assert exact >= exact_fraction, (
        f"only {exact:.5f} of the fp8 cache is bit-identical to "
        f"reshape_and_cache_flash, expected at least {exact_fraction}"
    )


def make_case(num_tokens, nq, nkv, dtype, device, with_gate=True, seed=0):
    torch.manual_seed(seed)
    qsz, kvsz = nq * HEAD_DIM, nkv * HEAD_DIM
    gate = qsz if with_gate else 0
    qkv = torch.randn(num_tokens, qsz + 2 * kvsz + gate, dtype=dtype, device=device)
    return qkv, qsz, kvsz, gate


def split_qkv(qkv, qsz, kvsz, gate):
    return qkv.split([qsz, kvsz, kvsz, gate], dim=-1)


def check_qk_norm_rope(
    num_tokens,
    nq,
    nkv,
    dtype,
    use_rope,
    rotary_dim=HEAD_DIM,
    with_gate=True,
    rope_dtype=torch.float32,
):
    device, eps, max_pos = "cuda", 1e-5, 4096

    qkv, qsz, kvsz, gate = make_case(num_tokens, nq, nkv, dtype, device, with_gate)
    qkv_orig = qkv.clone()
    cos_sin = (
        make_cos_sin_cache(max_pos, rotary_dim, 500_000.0, device, rope_dtype)
        if use_rope
        else None
    )
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=device
    )

    ops.fused_muse_spark_qknorm_rope_kv_insert(
        qkv,
        positions,
        nq,
        nkv,
        HEAD_DIM,
        eps,
        cos_sin_cache=cos_sin,
        rotary_dim=rotary_dim if use_rope else 0,
    )

    q_out, k_out, v_out, og_out = split_qkv(qkv, qsz, kvsz, gate)
    q_in, k_in, v_in, og_in = split_qkv(qkv_orig, qsz, kvsz, gate)

    q_ref = qk_norm_rope_ref(
        q_in.view(num_tokens, nq, HEAD_DIM), positions, cos_sin, rotary_dim, eps
    ).view(num_tokens, qsz)
    k_ref = qk_norm_rope_ref(
        k_in.view(num_tokens, nkv, HEAD_DIM), positions, cos_sin, rotary_dim, eps
    ).view(num_tokens, kvsz)

    # rsqrtf and lane summation order cost a couple of fp32 ulp, well inside
    # one ulp of the stored dtype.
    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_out, v_in, rtol=0, atol=0)
    torch.testing.assert_close(og_out, og_in, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_rope", [True, False])
def test_qk_norm_rope(dtype, use_rope):
    check_qk_norm_rope(17, 32, 4, dtype, use_rope)


@pytest.mark.parametrize(
    "num_tokens,variant",
    [
        pytest.param(64, {"rotary_dim": 32}, id="partial_rotary_dim"),
        # A [q | k | v] row (no o-gate) is addressed by its own stride.
        pytest.param(33, {"with_gate": False}, id="no_gate_row"),
        pytest.param(128, {"rope_dtype": torch.bfloat16}, id="bf16_cos_sin_cache"),
    ],
)
def test_qk_norm_rope_layout_variants(num_tokens, variant):
    check_qk_norm_rope(num_tokens, 32, 4, torch.bfloat16, True, **variant)


# Cover narrow/wide head groups for TP4/TP8 and the one-chunk-per-token case.
@pytest.mark.parametrize(
    "num_tokens,nq,nkv,kv_cache_dtype,use_rope",
    [
        pytest.param(1, 32, 4, "auto", True, id="tp4-decode"),
        pytest.param(1, 16, 2, "fp8", False, id="tp8-decode-fp8-nope"),
        pytest.param(1025, 32, 4, "auto", True, id="tp4-group2"),
        pytest.param(1025, 32, 4, "fp8", True, id="tp4-group2-fp8"),
        pytest.param(1537, 32, 4, "auto", True, id="tp4-group4"),
        pytest.param(1537, 32, 4, "auto", False, id="tp4-nope"),
        pytest.param(3073, 16, 2, "auto", True, id="tp8-group4"),
        pytest.param(3073, 16, 2, "fp8", False, id="tp8-fp8-nope"),
        pytest.param(40000, 2, 1, "auto", True, id="single-chunk-per-token"),
    ],
)
def test_kv_insert(num_tokens, nq, nkv, kv_cache_dtype, use_rope):
    device, dtype, eps = "cuda", torch.bfloat16, 1e-5
    rotary_dim, max_pos, block_size = HEAD_DIM, 4096, 16

    qkv, qsz, kvsz, gate = make_case(num_tokens, nq, nkv, dtype, device, seed=1)
    qkv_orig = qkv.clone()
    cos_sin = (
        make_cos_sin_cache(max_pos, rotary_dim, 500_000.0, device) if use_rope else None
    )
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=device
    )

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    storage = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    kv_cache = torch.zeros(
        num_blocks, nkv, block_size, 2 * HEAD_DIM, dtype=storage, device=device
    )
    slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device=device
    )[:num_tokens]
    if num_tokens > 1:
        # -1 marks a token the scheduler did not place; the kernel must skip it
        # and leave the cache alone.
        slot_mapping[::3] = -1

    # Non-unit scales, so a kernel that ignored them would fail loudly.
    k_scale = torch.tensor(0.37, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.83, dtype=torch.float32, device=device)

    ops.fused_muse_spark_qknorm_rope_kv_insert(
        qkv,
        positions,
        nq,
        nkv,
        HEAD_DIM,
        eps,
        cos_sin_cache=cos_sin,
        rotary_dim=rotary_dim if use_rope else 0,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
        block_size=block_size,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale if kv_cache_dtype != "auto" else None,
        v_scale=v_scale if kv_cache_dtype != "auto" else None,
    )

    q_out, k_out, v_out, og_out = split_qkv(qkv, qsz, kvsz, gate)
    q_in, k_in, v_in, og_in = split_qkv(qkv_orig, qsz, kvsz, gate)

    q_ref = qk_norm_rope_ref(
        q_in.view(num_tokens, nq, HEAD_DIM), positions, cos_sin, rotary_dim, eps
    ).view(num_tokens, qsz)
    k_ref = qk_norm_rope_ref(
        k_in.view(num_tokens, nkv, HEAD_DIM), positions, cos_sin, rotary_dim, eps
    ).view(num_tokens, kvsz)
    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_out, v_in, rtol=0, atol=0)
    torch.testing.assert_close(og_out, og_in, rtol=0, atol=0)

    expected = torch.zeros_like(kv_cache)
    exp_k, exp_v = expected.transpose(1, 2).split(HEAD_DIM, dim=-1)
    ops.reshape_and_cache_flash(
        k_out.view(num_tokens, nkv, HEAD_DIM),
        v_out.view(num_tokens, nkv, HEAD_DIM),
        exp_k,
        exp_v,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    if kv_cache_dtype == "auto":
        torch.testing.assert_close(kv_cache, expected, rtol=0, atol=0)
    else:
        assert_fp8_cache_close(kv_cache, expected)


# The fused Q output requires E4M3FN; scaled_fp8_quant uses the platform dtype.
@pytest.mark.skipif(
    current_platform.is_rocm() and current_platform.fp8_dtype() != torch.float8_e4m3fn,
    reason="Fused E4M3FN Q output cannot be compared with ROCm FNUZ quantization",
)
@pytest.mark.parametrize(
    "head_dim,num_tokens,rotary_fraction,use_cuda_graph",
    [
        pytest.param(64, 17, 0, False, id="head64-nope"),
        pytest.param(128, 17, 0.5, False, id="head128-partial-rope"),
        pytest.param(256, 17, 0, False, id="head256-nope"),
        pytest.param(256, 3073, 0.5, False, id="head256-partial-rope-large"),
        pytest.param(64, 3073, 0.5, True, id="graph-rope"),
        pytest.param(64, 3073, 0, True, id="graph-nope"),
    ],
)
def test_fused_query_quant_matches_separate_quant(
    head_dim, num_tokens, rotary_fraction, use_cuda_graph
):
    """Preserve the rounding boundary, scales, padded Q, and KV/gate writes."""
    torch.manual_seed(42)
    nq, nkv, block_size = 16, 2, 16
    qsz, kvsz = nq * head_dim, nkv * head_dim
    qkv = torch.randn(num_tokens, 2 * (qsz + kvsz), device="cuda", dtype=torch.bfloat16)
    original = qkv.clone()
    reference = qkv.clone()
    q_fp8 = torch.empty(num_tokens, qsz, device="cuda", dtype=torch.float8_e4m3fn)
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    rotary_dim = int(head_dim * rotary_fraction)
    cos_sin = (
        make_cos_sin_cache(num_tokens, rotary_dim, 500_000.0, "cuda")
        if rotary_dim
        else None
    )
    cache = torch.zeros(
        (num_tokens + block_size - 1) // block_size,
        nkv,
        block_size,
        2 * head_dim,
        device="cuda",
        dtype=torch.uint8,
    )
    expected_cache = torch.zeros_like(cache)
    slots = positions.clone()
    slots[::3] = -1
    # Small enough that part of Q saturates the e4m3 range.
    scale = torch.tensor(0.001, device="cuda", dtype=torch.float32)
    kwargs = dict(
        cos_sin_cache=cos_sin,
        rotary_dim=rotary_dim,
        slot_mapping=slots,
        block_size=block_size,
        kv_cache_dtype="fp8",
        k_scale=torch.tensor(0.37, device="cuda"),
        v_scale=torch.tensor(1.83, device="cuda"),
    )
    ops.fused_muse_spark_qknorm_rope_kv_insert(
        reference,
        positions,
        nq,
        nkv,
        head_dim,
        1e-5,
        kv_cache=expected_cache,
        **kwargs,
    )
    expected_q, _ = ops.scaled_fp8_quant(reference[:, :qsz], scale=scale)

    def run():
        qkv.copy_(original)
        ops.fused_muse_spark_qknorm_rope_kv_insert(
            qkv,
            positions,
            nq,
            nkv,
            head_dim,
            1e-5,
            kv_cache=cache,
            q_fp8_out=q_fp8,
            q_scale=scale,
            **kwargs,
        )

    run()
    if use_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        qkv.zero_()
        cache.zero_()
        q_fp8.zero_()
        graph.replay()
    torch.testing.assert_close(
        q_fp8.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
    torch.testing.assert_close(qkv[:, :qsz], original[:, :qsz], rtol=0, atol=0)
    torch.testing.assert_close(qkv[:, qsz:], reference[:, qsz:], rtol=0, atol=0)


def test_fused_kv_insert_requires_backend_bypass_support(monkeypatch):
    monkeypatch.setattr(
        fused_ops,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(decode_context_parallel_size=1)
        ),
    )
    backend = SimpleNamespace(
        forward_includes_kv_cache_update=False,
        supports_kv_cache_update_bypass=False,
    )
    attn_layer = SimpleNamespace(
        attn_backend=backend,
        kv_sharing_target_layer_name=None,
        kv_cache_dtype="auto",
    )

    assert not fused_ops.supports_fused_kv_insert(attn_layer)
    backend.supports_kv_cache_update_bypass = True
    assert fused_ops.supports_fused_kv_insert(attn_layer)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm backend gate")
def test_rocm_aiter_fa_takes_the_fused_kv_insert(monkeypatch):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend

    monkeypatch.setattr(
        fused_ops,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(decode_context_parallel_size=1)
        ),
    )
    attn_layer = SimpleNamespace(
        attn_backend=AiterFlashAttentionBackend,
        kv_sharing_target_layer_name=None,
        kv_cache_dtype="auto",
    )
    # The kernel writes the packed [B, H, N, 2*hd] page, so the backend only
    # accepts the bypass outside the shuffle layout.
    shuffle = rocm_aiter_ops.is_shuffle_kv_cache_enabled()
    assert AiterFlashAttentionBackend.supports_kv_cache_update_bypass is not shuffle
    assert fused_ops.supports_fused_kv_insert(attn_layer) is not shuffle


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm compile boundary")
def test_rocm_compiled_caller_keeps_the_kv_insert(monkeypatch):
    from vllm.models.muse_spark.amd import model as amd_model

    kv_cache = torch.zeros(4, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    attn_layer = SimpleNamespace(
        attn_backend=SimpleNamespace(get_name=lambda: "ROCM_AITER_FA"),
        kv_cache_dtype="auto",
    )
    slot_mapping: list[torch.Tensor | None] = [None]
    inserted: list[bool] = []

    monkeypatch.setattr(
        fused_ops,
        "get_attention_context",
        lambda layer_name: (None, attn_layer, kv_cache, slot_mapping[0]),
    )
    monkeypatch.setattr(
        fused_ops.ops,
        "fused_muse_spark_qknorm_rope_kv_insert",
        lambda *args, **kwargs: inserted.append("slot_mapping" in kwargs),
    )

    @torch.compile(backend="eager", fullgraph=True)
    def forward(qkv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        amd_model.rocm_fused_qk_norm_rope_kv_insert(
            qkv, positions, None, "layer", 1, 1, HEAD_DIM, 0, 1e-5
        )
        return qkv + 1

    qkv = torch.zeros(2, 4 * HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    positions = torch.zeros(2, device="cuda", dtype=torch.int64)
    forward(qkv, positions)
    slot_mapping[0] = torch.arange(2, device="cuda", dtype=torch.int64)
    with torch.compiler.set_stance("fail_on_recompile"):
        forward(qkv, positions)

    assert inserted == [False, True]


@pytest.mark.parametrize(
    "prefill_tokens,decode_tokens,prefill_dtype,decode_dtype,expected_fp8",
    [
        (0, 2, torch.bfloat16, torch.float8_e4m3fn, True),
        (2, 0, torch.float8_e4m3fn, torch.bfloat16, True),
        (1, 1, torch.float8_e4m3fn, torch.float8_e4m3fn, True),
        (1, 1, torch.float8_e4m3fn, torch.bfloat16, False),
        (0, 2, torch.float8_e4m3fn, torch.bfloat16, False),
    ],
)
def test_query_fusion_respects_attention_metadata(
    monkeypatch,
    prefill_tokens,
    decode_tokens,
    prefill_dtype,
    decode_dtype,
    expected_fp8,
):
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadata

    metadata = FlashInferMetadata(
        kv_cache_layout=None,
        num_actual_tokens=2,
        slot_mapping=None,
        q_data_type_prefill=prefill_dtype,
        q_data_type_decode=decode_dtype,
        num_decodes=decode_tokens,
        num_decode_tokens=decode_tokens,
        num_prefills=int(prefill_tokens > 0),
        num_prefill_tokens=prefill_tokens,
        causal=True,
        prefill=None,
        decode=None,
    )
    layer = SimpleNamespace(
        attn_backend=SimpleNamespace(get_name=lambda: "FLASHINFER"),
        kv_cache_dtype="fp8",
        _q_scale=torch.tensor(0.37, device="cuda"),
    )
    monkeypatch.setattr(
        fused_ops,
        "get_attention_context",
        lambda _: (metadata, layer, torch.empty(0, device="cuda"), None),
    )
    qkv = torch.randn(2, 20 * HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    ref = qkv.clone()
    positions = torch.arange(2, device="cuda", dtype=torch.int64)
    ops.fused_muse_spark_qknorm_rope_kv_insert(ref, positions, 16, 2, HEAD_DIM, 1e-5)
    q_fp8 = fused_ops.fused_qk_norm_rope_kv_insert(
        qkv,
        positions,
        None,
        "layer",
        16,
        2,
        HEAD_DIM,
        0,
        1e-5,
    )
    assert (q_fp8 is not None) == expected_fp8
    if expected_fp8:
        assert q_fp8 is not None
        expected_q, _ = ops.scaled_fp8_quant(ref[:, : 16 * HEAD_DIM], layer._q_scale)
        torch.testing.assert_close(q_fp8.float(), expected_q.float(), rtol=0, atol=0)
    else:
        torch.testing.assert_close(qkv, ref, rtol=0, atol=0)


def test_attention_accepts_fused_fp8_query_without_requantizing(monkeypatch):
    """The model's FP8 Q must reach attention unchanged with BF16 output."""
    from vllm.model_executor.layers.attention import attention as attention_module

    layer = attention_module.Attention.__new__(attention_module.Attention)
    torch.nn.Module.__init__(layer)
    layer.kv_cache_dtype = "fp8"
    layer.impl = SimpleNamespace(supports_quant_query_input=True)
    layer.num_heads = 16
    layer.head_size = layer.head_size_v = HEAD_DIM
    layer.layer_name = "layer"
    layer.use_direct_call = True
    layer.attn_backend = SimpleNamespace(forward_includes_kv_cache_update=False)
    layer.kv_sharing_target_layer_name = None

    def unexpected_quant(*args):
        pytest.fail("an already quantized query must not be quantized again")

    layer.query_quant = unexpected_quant
    query = torch.randn(2, 16 * HEAD_DIM, device="cuda").to(torch.float8_e4m3fn)

    def attend(q, k, v, output, *args, **kwargs):
        assert q.data_ptr() == query.data_ptr()
        assert q.dtype == torch.float8_e4m3fn
        assert output.dtype == torch.bfloat16
        output.copy_(q)

    monkeypatch.setattr(attention_module, "unified_attention_with_output", attend)
    result = layer(query, None, None, output_dtype=torch.bfloat16)
    torch.testing.assert_close(result, query.bfloat16(), rtol=0, atol=0)


requires_sm100 = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode's attention kernels run on SM100-family GPUs",
)


@requires_sm100
@pytest.mark.parametrize("rope", [True, False])
@pytest.mark.parametrize("m", range(1, 9))
@torch.inference_mode()
def test_specialized_qkv_front_matches_projection_norm_rope_and_insert(m, rope):
    """At the TP8 shard (16 q and 2 KV heads per rank): q and k normalized and
    rotated, v and the o-gate as projected (FP32 references), the k | v rows
    in the cache at the valid slots and the rest of the cache untouched."""
    from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.attention import (
        qkv_front as front,
    )

    torch.manual_seed(620 + m)
    hq, hkv, d, pages = 16, 2, front.HEAD_DIM, 4
    n, qn, kn = (2 * hq + 2 * hkv) * d, hq * d, hkv * d
    with torch.device("cuda"):
        x = torch.randn(m, front.HIDDEN).bfloat16()
        weight = (torch.randn(n, front.HIDDEN) / front.HIDDEN**0.5).bfloat16()
        positions = torch.randint(0, 1 << 16, (m,))
        cos_sin = make_cos_sin_cache(1 << 16, d, 10000.0, "cuda") if rope else None
        slots = torch.randperm(pages * front.BLOCK_SIZE)[:m]
        slots[1::3] = -1
        cache = torch.full((pages, hkv, front.BLOCK_SIZE, 2 * d), 7.0).bfloat16()
        out = torch.empty(m, n, dtype=torch.bfloat16)
    front.prepare(n, [m])
    front.qkv_front(x, weight, positions, cos_sin, slots, cache, out, 1e-6)

    q, k, v, gate = (x.float() @ weight.float().T).split([qn, kn, kn, qn], dim=1)
    got_q, got_k, got_v, got_gate = out.float().split([qn, kn, kn, qn], dim=1)
    for got, ref, heads in ((got_q, q, hq), (got_k, k, hkv)):
        ref = qk_norm_rope_ref(ref.view(m, heads, d), positions, cos_sin, d, 1e-6)
        torch.testing.assert_close(got, ref.view(m, -1), rtol=1e-2, atol=1e-2)
    for got, ref in ((got_v, v), (got_gate, gate)):
        torch.testing.assert_close(got, ref, rtol=2**-8, atol=1e-3)

    entries = cache.permute(0, 2, 1, 3).reshape(-1, hkv, 2 * d)
    valid = slots >= 0
    written = out[:, qn : qn + 2 * kn].view(m, 2, hkv, d).transpose(1, 2)
    torch.testing.assert_close(
        entries[slots[valid]], written.flatten(2)[valid], rtol=0, atol=0
    )
    untouched = torch.ones(entries.shape[0], dtype=torch.bool, device="cuda")
    untouched[slots[valid]] = False
    assert bool((entries[untouched] == 7.0).all())


def _attn2_reference(q, gate, kv, pages, length, window):
    """FP32 attn2 of one request: its query rows are the last `q.shape[0]` of
    its `length` cache entries, causal and windowed (0: no window), then the
    weightless per-head RMS norm and the sigmoid gate."""
    from vllm.models.muse_spark.nvidia.specialized_decode.runtime import cute_ops

    rows, hkv, d = q.shape[0], kv.shape[1], cute_ops.HEAD_DIM
    hq = q.shape[1] // d
    cache = kv[pages.long()].permute(1, 0, 2, 3).reshape(hkv, -1, 2 * d)
    keys, values = cache[:, :length, :d].float(), cache[:, :length, d:].float()
    pos = torch.arange(length - rows, length, device=q.device)[:, None]
    key = torch.arange(length, device=q.device)[None]
    visible = (key <= pos) & ((key > pos - window) if window else True)
    qh = q.float().view(rows, hkv, hq // hkv, d)
    scores = torch.einsum("tkhd,ksd->tkhs", qh, keys) * cute_ops.SM_SCALE
    scores = scores.masked_fill(~visible[:, None, None], float("-inf"))
    o = torch.einsum("tkhs,ksd->tkhd", scores.softmax(-1), values).reshape(rows, hq, d)
    o = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + cute_ops.ATTN2_EPS)
    return (o * gate.float().view(rows, hq, d).sigmoid()).view(rows, -1)


@requires_sm100
@pytest.mark.parametrize(
    "max_len,window,lengths",
    [
        (2048, 0, (8, 2048, 17, 2048, 300, 9, 1000, 2048)),
        (1 << 20, 0, (8, 4000, 17, 2100, 300, 9, 1000, 2048)),
        (1 << 20, 2048, (8, 4000, 17, 2100, 300, 9, 1000, 2048)),
        (1 << 20, 0, (9000, 16384, 8193, 12000, 8192, 20000, 4000, 12345)),
    ],
    ids=["one_cluster", "split_full", "split_sliding", "split_full_page_ring"],
)
@pytest.mark.parametrize("requests", range(1, 9))
@torch.inference_mode()
def test_specialized_attn2_matches_fp32_verify_attention(
    requests, max_len, window, lengths
):
    """The verify step at the TP8 shard's 2 KV heads: per request its 8 query
    rows are the last 8 of its cache entries, for 1-8 requests of lengths from
    one page to past the window, the one-cluster limit and (page ring) 8192."""
    from vllm.models.muse_spark.nvidia.specialized_decode.runtime import cute_ops

    torch.manual_seed(620 + requests)
    hkv, d, page = 2, cute_ops.HEAD_DIM, cute_ops.BLOCK_SIZE
    hq, rows = cute_ops.HEADS_PER_KV_HEAD * hkv, cute_ops.QUERY_LEN
    lengths = lengths[:requests]
    per_request = -(-max(lengths) // page)
    splits = cute_ops.attn2_splits(max_len, window, hkv)
    cute_ops.prepare_attn2(max_len, window, hkv, splits)
    with torch.device("cuda"):
        kv = torch.randn(requests * per_request, hkv, page, 2 * d).bfloat16()
        qkv = torch.randn(requests * rows, 2 * hq * d + 2 * hkv * d).bfloat16()
        table = torch.randperm(requests * per_request, dtype=torch.int32)
        table = table.view(requests, per_request)
        seq_lens = torch.tensor(lengths, dtype=torch.int32)
        out = torch.empty(requests * rows, hq * d, dtype=torch.bfloat16)
        workspace = torch.zeros(
            cute_ops.attn2_workspace_bytes(requests, hkv, splits), dtype=torch.uint8
        )
    q, gate = cute_ops.split_qkv(qkv, hkv)
    cute_ops.attn2(
        q,
        kv,
        table,
        seq_lens,
        gate,
        out,
        hkv,
        max_seq_len=max_len,
        window=window,
        num_splits=splits,
        workspace=workspace,
    )
    for r, length in enumerate(lengths):
        span = slice(r * rows, (r + 1) * rows)
        ref = _attn2_reference(q[span], gate[span], kv, table[r], length, window)
        got = out[span].float()
        torch.testing.assert_close(got, ref, rtol=0, atol=6e-2)
        assert (got - ref).abs().mean() < 4e-3


def test_attn2_layers_share_one_workspace_sized_for_every_layer():
    """The attn2 layers of a model share one zeroed workspace, sized once for
    the largest of them, and a binding it is too small for raises."""
    from vllm.models.muse_spark.nvidia.specialized_decode.runtime import cute_ops

    def layer(num_kv_heads, window=None):
        return SimpleNamespace(
            num_kv_heads=num_kv_heads,
            attn=SimpleNamespace(sliding_window=window),
            qkv_proj=SimpleNamespace(weight=torch.empty(0)),
        )

    max_len, requests = 8192, 2
    layers = [layer(1), layer(2), layer(2, window=128)]
    workspace = cute_ops.attn2_workspace(layers, max_len, requests)
    size = max(
        cute_ops.attn2_workspace_bytes(
            requests, hkv, cute_ops.attn2_splits(max_len, 0, hkv)
        )
        for hkv in (1, 2)
    )
    assert workspace is not None and workspace.numel() == size
    assert not workspace.any()
    assert cute_ops.attn2_workspace(layers[:2], 2048, requests) is None
    with pytest.raises(ValueError, match="workspace"):
        cute_ops.bind_attn2(layer(2), max_len, requests, workspace[:-1])
