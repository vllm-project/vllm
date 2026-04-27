# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel tests for ROCm AITER MLA.

This file owns the dense ROCm MLA execution path:
- custom-op registration and fake-tensor support
- MLA enablement gating via rocm_aiter_ops
- BF16 and FP8-capable decode execution
- accuracy and determinism of the decode path

Static backend metadata and variant wiring live in the ROCm selector and MLA
variant tests instead of being duplicated here.
"""

import importlib
import warnings
from typing import TypedDict

import pytest
import torch
from tests.kernels.rocm.utils import _assert_accurate, _assert_deterministic

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_mi3xx
from vllm.utils.torch_utils import set_random_seed

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific tests"),
    pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only"),
]


# Helpers -----------------------------------------------------------------


class DecodeCase(TypedDict):
    q: torch.Tensor
    kv_buffer: torch.Tensor
    o: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_lens: torch.Tensor
    sm_scale: float
    v_head_dim: int


def _reload_envs():
    import vllm.envs as envs

    return importlib.reload(envs)


def _assert_aiter_supported() -> None:
    from vllm._aiter_ops import is_aiter_found_and_supported

    assert is_aiter_found_and_supported(), (
        "aiter is required on supported ROCm hardware for this test"
    )


def _print_close_stats(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    pass_rate: float = 0.99999,
    max_atol: float | None = None,
) -> None:
    abs_diff = (actual - expected).abs().float().flatten()
    expected_abs = expected.abs().float().flatten()
    allowed = atol + rtol * expected_abs
    within = abs_diff <= allowed

    total = abs_diff.numel()
    passed = int(within.sum().item())
    failed = total - passed
    allowed_fail_rate = 1.0 - pass_rate

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    p99_abs = torch.quantile(abs_diff, 0.99).item()
    p999_abs = torch.quantile(abs_diff, 0.999).item()
    worst_ratio = (abs_diff / allowed.clamp_min(1e-12)).max().item()

    msg = (
        "[rocm_aiter_mla] "
        f"{label}: "
        f"pass={passed / total:.4%} ({passed}/{total}) "
        f"fail={failed / total:.4%} ({failed}/{total}) "
        f"allowed_fail={allowed_fail_rate:.4%} "
        f"atol={atol:g} "
        f"rtol={rtol:g} "
    )
    if max_atol is not None:
        above_max = int((abs_diff > max_atol).sum().item())
        msg += f"abs>{max_atol:g}={above_max / total:.4%} ({above_max}/{total}) "
    msg += (
        f"max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} "
        f"p99_abs={p99_abs:.6g} "
        f"p999_abs={p999_abs:.6g} "
        f"worst_ratio={worst_ratio:.6g}"
    )
    print(msg)
    if failed > 0:
        warnings.warn(msg, stacklevel=2)


def _make_decode_case(
    *,
    batch_size: int,
    nhead: int,
    kv_seq_len_per_seq: int,
    seed: int,
) -> DecodeCase:
    torch.set_default_device("cuda")
    set_random_seed(seed)

    q_head_dim = 576
    v_head_dim = 512
    num_kv_heads = 1
    kv_seq_len = batch_size * kv_seq_len_per_seq
    sm_scale = q_head_dim**-0.5

    q = torch.randn(batch_size, nhead, q_head_dim, dtype=torch.bfloat16)
    kv_buffer = torch.randn(kv_seq_len, num_kv_heads, q_head_dim, dtype=torch.bfloat16)
    o = torch.zeros(batch_size, nhead, v_head_dim, dtype=torch.bfloat16)

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32)
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * kv_seq_len_per_seq
    kv_indices = torch.arange(0, kv_seq_len, dtype=torch.int32)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32)

    return {
        "q": q,
        "kv_buffer": kv_buffer,
        "o": o,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_last_page_lens": kv_last_page_lens,
        "sm_scale": sm_scale,
        "v_head_dim": v_head_dim,
    }


def _run_decode(case: DecodeCase) -> torch.Tensor:
    torch.ops.vllm.rocm_aiter_mla_decode_fwd(
        case["q"],
        case["kv_buffer"],
        case["o"],
        case["qo_indptr"],
        1,
        kv_indptr=case["kv_indptr"],
        kv_indices=case["kv_indices"],
        kv_last_page_lens=case["kv_last_page_lens"],
        sm_scale=case["sm_scale"],
    )
    return case["o"]


def _clone_decode_case(case: DecodeCase) -> DecodeCase:
    return {
        "q": case["q"],
        "kv_buffer": case["kv_buffer"],
        "o": case["o"],
        "qo_indptr": case["qo_indptr"],
        "kv_indptr": case["kv_indptr"],
        "kv_indices": case["kv_indices"],
        "kv_last_page_lens": case["kv_last_page_lens"],
        "sm_scale": case["sm_scale"],
        "v_head_dim": case["v_head_dim"],
    }


# Custom op tests ---------------------------------------------------------


def test_aiter_mla_custom_op_registered():
    """The dense ROCm MLA custom op should stay registered for runtime use."""
    _assert_aiter_supported()
    # Import to trigger op registration
    import vllm._aiter_ops as aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_mla_decode_fwd")
    assert callable(torch.ops.vllm.rocm_aiter_mla_decode_fwd)


def test_aiter_mla_fake_tensor_support():
    """The dense ROCm MLA custom op should keep its fake-tensor support for
    torch.compile-style tracing."""
    _assert_aiter_supported()
    import vllm._aiter_ops  # noqa: F401

    # Create representative tensors for opcheck.
    # nhead=128, q_head_dim=576, v_head_dim=512: DeepSeek MLA dimensions.
    # The kernel uses q_head_dim for attention scores and writes v_head_dim output.
    batch_size = 4
    nhead = 128
    q_head_dim = 576  # kv_lora_rank + qk_rope_head_dim
    v_head_dim = 512  # kv_lora_rank (output dimension)
    num_kv_heads = 1
    kv_seq_len = 128

    q = torch.randn(batch_size, nhead, q_head_dim, dtype=torch.bfloat16, device="cuda")
    kv_buffer = torch.randn(
        kv_seq_len, num_kv_heads, q_head_dim, dtype=torch.bfloat16, device="cuda"
    )
    o = torch.zeros(batch_size, nhead, v_head_dim, dtype=torch.bfloat16, device="cuda")
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * (
        kv_seq_len // batch_size
    )
    kv_indices = torch.arange(0, kv_seq_len, dtype=torch.int32, device="cuda")
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    # max_seqlen_qo=1: decode mode has exactly 1 query token per sequence
    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_mla_decode_fwd,
        (q, kv_buffer, o, qo_indptr, 1),
        kwargs={
            "kv_indptr": kv_indptr,
            "kv_indices": kv_indices,
            "kv_last_page_lens": kv_last_page_lens,
            "sm_scale": q_head_dim**-0.5,
        },
        test_utils=("test_faketensor",),
    )


# Env gating tests --------------------------------------------------------


@pytest.mark.parametrize(
    ("use_aiter", "use_mla", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_aiter_mla_enablement_follows_env(use_aiter, use_mla, expected, monkeypatch):
    """MLA enablement should follow the explicit ROCm AITER env toggles."""
    _assert_aiter_supported()
    import vllm._aiter_ops as aiter_ops
    from vllm._aiter_ops import rocm_aiter_ops

    with monkeypatch.context() as mp:
        mp.setenv("VLLM_ROCM_USE_AITER", "1" if use_aiter else "0")
        mp.setenv("VLLM_ROCM_USE_AITER_MLA", "1" if use_mla else "0")
        _reload_envs()
        rocm_aiter_ops.refresh_env_variables()
        assert rocm_aiter_ops.is_mla_enabled() is expected

    _reload_envs()
    aiter_ops.rocm_aiter_ops.refresh_env_variables()


# Dense MLA decode tests --------------------------------------------------
#
# Kernel constraints on gfx942 (precompiled ASM):
#   - num_heads (gqa_ratio) supported: 16 and 128.
#     Kernel files: mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co (nhead=16)
#                   mla_dec_stage1_bf16_a16w16_subQ128_mqa128.co (nhead=128)
#     Other values raise: "get_heuristic_kernel_mla: cannot get heuristic kernel!"
#   - max_seqlen_qo must be 1 for decode (backend passes qo_len.max()=1 in decode).
#     Passing batch_size raises: "get_heuristic_kernel_mla: causal:0 qseqlen:N".
#   - block_size=1 always (each page holds exactly 1 KV token).
#   - No FP4 MLA kernel exists; FP4 in aiter is GEMM-only (gfx950).
#   Source: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (max_qo_len computation)
#           aiter/mla.py (nhead dispatch logic)


def test_aiter_mla_decode_unsupported_nhead_raises():
    """Unsupported nhead values raise RuntimeError from the C++ kernel selector.

    MI3xx precompiled ASM kernels only expose nhead=16 and nhead=128.
    nhead=1 (or other unsupported values) fail at the C++ heuristic kernel
    selection step with: "get_heuristic_kernel_mla: cannot get heuristic kernel!"
    """
    _assert_aiter_supported()

    case = _make_decode_case(
        batch_size=2,
        nhead=1,
        kv_seq_len_per_seq=8,
        seed=11,
    )

    with pytest.raises(
        RuntimeError,
        match=r"get_heuristic_kernel_mla: cannot get heuristic kernel!",
    ):
        _run_decode(case)


def test_aiter_mla_decode_bf16_basic():
    """A representative BF16 MLA decode launch should produce finite,
    non-trivial output with the expected shape."""
    _assert_aiter_supported()

    case = _make_decode_case(
        batch_size=4,
        nhead=128,
        kv_seq_len_per_seq=64,
        seed=0,
    )

    out = _run_decode(case)

    assert out.shape == (4, 128, 512)
    assert out.dtype == torch.bfloat16
    # Output should be non-trivial (not all zeros)
    assert not torch.all(out == 0)
    assert torch.isfinite(out).all()


# Reference MLA decode implementation ------------------------------------


def _ref_mla_decode(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    v_head_dim: int,
) -> torch.Tensor:
    """Pure PyTorch reference for MLA decode (absorbed formulation).

    In absorbed MLA, attention scores use the full kv_buffer dimension (K),
    but the output weighted sum uses only the first v_head_dim dims (V = kv_lora_rank).

    Args:
        q: Query tensor [batch_size, num_heads, q_head_dim] in BF16.
            q_head_dim = kv_lora_rank + qk_rope_head_dim.
        kv_buffer: KV buffer [total_tokens, num_kv_heads, q_head_dim] in BF16.
        kv_indptr: KV sequence start/end indices [batch_size + 1] int32.
        kv_indices: Token indices into kv_buffer [total_tokens] int32.
        sm_scale: Attention scale factor (typically 1/sqrt(q_head_dim)).
        v_head_dim: Output dimension = kv_lora_rank
            (first v_head_dim dims of kv_buffer).

    Returns:
        Output tensor [batch_size, num_heads, v_head_dim] in BF16.
    """
    batch_size, num_heads, q_head_dim = q.shape
    output = torch.zeros(
        batch_size, num_heads, v_head_dim, dtype=q.dtype, device=q.device
    )

    for b in range(batch_size):
        start = kv_indptr[b].item()
        end = kv_indptr[b + 1].item()
        token_indices = kv_indices[start:end]  # [seq_len]

        # K uses full q_head_dim (for attention scores)
        # V uses first v_head_dim dims (kv_lora_rank, for output)
        kv_seq = kv_buffer[token_indices]  # [seq_len, num_kv_heads, q_head_dim]
        k = kv_seq[:, 0, :].float()  # [seq_len, q_head_dim]
        v = kv_seq[:, 0, :v_head_dim].float()  # [seq_len, v_head_dim]

        for h in range(num_heads):
            q_h = q[b, h, :].float()  # [q_head_dim]
            scores = torch.mv(k, q_h) * sm_scale  # [seq_len]
            attn_weights = torch.softmax(scores, dim=0)  # [seq_len]
            output[b, h, :] = torch.mv(v.t(), attn_weights).to(q.dtype)

    return output


# Accuracy and determinism tests -----------------------------------------


def test_aiter_mla_decode_bf16_accuracy():
    """AITER MLA decode BF16 output matches PyTorch reference.

    Compares the AITER custom op against _ref_mla_decode with
    allow_close tolerance for BF16 attention operations.
    """
    _assert_aiter_supported()

    case = _make_decode_case(
        batch_size=4,
        nhead=128,
        kv_seq_len_per_seq=16,
        seed=0,
    )
    out = _run_decode(case)

    # Reference: K=kv_buffer[:,0,:] (full q_head_dim), V=kv_buffer[:,0,:v_head_dim]
    ref = _ref_mla_decode(
        case["q"],
        case["kv_buffer"],
        case["kv_indptr"],
        case["kv_indices"],
        case["sm_scale"],
        case["v_head_dim"],
    )

    assert out.shape == ref.shape
    _print_close_stats(
        "bf16_accuracy",
        out.float(),
        ref.float(),
        atol=0.01,
        rtol=0.0,
        max_atol=0.03,
    )
    _assert_accurate(out.float(), ref.float(), atol=0.01, rtol=0.0)


@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 not supported on this hardware",
)
def test_aiter_mla_decode_fp8_accuracy():
    """AITER MLA decode with BF16 KV on FP8-capable hardware: output close to reference.

    This test exercises the FP8-capable code path with BF16 inputs to verify
    the kernel runs without error on FP8-supporting hardware. The op's FP8 KV
    path (passing uint8 kv_buffer with q_scale/kv_scale) is gated separately in
    ``tests/kernels/quantization/rocm/aiter/test_mla_fp8_support_check.py``.
    """
    _assert_aiter_supported()

    case = _make_decode_case(
        batch_size=2,
        nhead=128,
        kv_seq_len_per_seq=16,
        seed=1,
    )
    out = _run_decode(case)

    assert not torch.any(torch.isnan(out))
    assert not torch.any(torch.isinf(out))

    ref = _ref_mla_decode(
        case["q"],
        case["kv_buffer"],
        case["kv_indptr"],
        case["kv_indices"],
        case["sm_scale"],
        case["v_head_dim"],
    )
    _print_close_stats(
        "fp8_capable_bf16_accuracy",
        out.float(),
        ref.float(),
        atol=0.01,
        rtol=0.0,
        max_atol=0.03,
    )
    _assert_accurate(out.float(), ref.float(), atol=0.01, rtol=0.0)


def test_aiter_mla_decode_determinism():
    """AITER MLA decode produces bitwise-identical results across N runs."""
    _assert_aiter_supported()

    case = _make_decode_case(
        batch_size=4,
        nhead=128,
        kv_seq_len_per_seq=16,
        seed=2,
    )

    def run_mla():
        local_case = _clone_decode_case(case)
        local_case["o"] = torch.zeros_like(case["o"])
        return _run_decode(local_case)

    _assert_deterministic(run_mla, n_runs=4)


# Parametrized parity tests (parity with NVIDIA FlashMLA / CutlassMLA) ----
#
# NVIDIA tests parametrize over:
#   h_q=[16,32,64,128], batch=[1,16,128], seq_len=[4096,8192,16384], FP8+BF16
#
# ROCm parity: we test both supported nhead values (16 and 128) across a range
# of batch sizes and sequence lengths. FP8 KV path is a separate capability check.
#
# Note: nhead=16 uses mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co
#       nhead=128 uses mla_dec_stage1_bf16_a16w16_subQ128_mqa128.co


@pytest.mark.parametrize("nhead", [16, 128])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("kv_seq_len_per_seq", [16, 256])
def test_aiter_mla_decode_parametrized_accuracy(nhead, batch_size, kv_seq_len_per_seq):
    """AITER MLA decode accuracy across supported nhead values,
    batch sizes and seq lens.

    Tests both gfx942 precompiled ASM kernels:
    - nhead=16:  mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co
    - nhead=128: mla_dec_stage1_bf16_a16w16_subQ128_mqa128.co

    Parity reference: NVIDIA FlashMLA tests use h_q=[16,32,64,128] across
    batch=[1..128] and mean_sk=[4096,8192,16384].
    """
    _assert_aiter_supported()

    case = _make_decode_case(
        batch_size=batch_size,
        nhead=nhead,
        kv_seq_len_per_seq=kv_seq_len_per_seq,
        seed=nhead + batch_size * 100 + kv_seq_len_per_seq,
    )
    out = _run_decode(case)

    ref = _ref_mla_decode(
        case["q"],
        case["kv_buffer"],
        case["kv_indptr"],
        case["kv_indices"],
        case["sm_scale"],
        case["v_head_dim"],
    )

    assert out.shape == (batch_size, nhead, case["v_head_dim"])
    _print_close_stats(
        f"param_accuracy nhead={nhead} batch={batch_size} seq={kv_seq_len_per_seq}",
        out.float(),
        ref.float(),
        atol=0.01,
        rtol=0.0,
        max_atol=0.03,
    )
    _assert_accurate(out.float(), ref.float(), atol=0.01, rtol=0.0)
