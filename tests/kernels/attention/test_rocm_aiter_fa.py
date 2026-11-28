# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm-specific tests for the AITER Flash Attention backend.

This file owns the ROCm backend contract and the ROCm-specific execution paths
wired through `vllm.v1.attention.backends.rocm_aiter_fa`:
- backend contract and env gating
- extend-kernel warmup registration
- representative prefill, multi-batch, decode, and FP8-KV execution
- direct kernel stress for large block tables and sliding-window masking
"""

import importlib
import math
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import get_cdna_version, on_mi3xx
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


@pytest.fixture(autouse=True)
def _cuda_default_device():
    torch.set_default_device("cuda")
    try:
        yield
    finally:
        torch.set_default_device("cpu")


DTYPES = [torch.bfloat16, torch.float16]
HEAD_SIZES = [64, 128, 256]
NUM_HEADS_PAIRS = [(8, 8), (16, 4)]  # (num_q_heads, num_kv_heads) - tests GQA
BLOCK_SIZE = 16
NUM_BLOCKS = 2048
DIRECT_NUM_BLOCKS = [2048, 32768]
DIRECT_NUM_HEADS = (8, 2)
DIRECT_HEAD_SIZE = 128
DIRECT_DTYPE = torch.bfloat16
HEAD_SIZE_TEST_NUM_HEADS_PAIRS = [(16, 16), (16, 4)]
# Prefill seq lens: (query_len, kv_len). q=1 is not a production shape for this
# kernel: sort_batch_req_ids orders requests by scheduled token count, so a
# 1-token request leads the batch and is split off as a decode, and the decode
# path takes no min_seqlen_q. It is kept as a regression guard -- CK returns
# early for seqlen_q <= min_seqlen_q and leaves `out` unwritten, so this entry
# fails loudly if min_seqlen_q ever goes back to 1. See
# test_aiter_mha_decode_single_token for the decode path itself.
SEQ_LENS = [(1, 512), (8, 512), (32, 1024)]
DIRECT_SEQ_LENS = [(10, 1328), (5, 18), (129, 463)]


# Reference implementation -------------------------------------------------
def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Naive reference paged attention using einsum."""
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len] * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(
            torch.ones(query_len, kv_len), diagonal=kv_len - query_len + 1
        ).bool()
        if sliding_window is not None:
            window_mask = (
                torch.triu(
                    torch.ones(query_len, kv_len),
                    diagonal=kv_len - (query_len + sliding_window) + 1,
                )
                .bool()
                .logical_not()
            )
            mask |= window_mask
        attn.masked_fill_(mask, float("-inf"))
        if sinks is not None:
            sink = sinks.float()[:, None, None].expand(-1, query_len, 1)
            attn = torch.cat([attn, sink], dim=-1)
        attn = torch.softmax(attn, dim=-1)[..., :kv_len].to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


# Small test helpers ------------------------------------------------------
def _reload_envs():
    import vllm.envs as envs

    return importlib.reload(envs)


def _assert_aiter_supported() -> None:
    from vllm._aiter_ops import is_aiter_found_and_supported

    assert is_aiter_found_and_supported(), (
        "aiter is required on supported ROCm hardware for this test"
    )


def _format_observed_rate(count: int, total: int) -> str:
    return f"{count / total:.4%} ({count}/{total})"


def _format_allowed_rate(rate: float, total: int) -> str:
    allowed_count = int(rate * total)
    return f"{rate:.4%} (<= {allowed_count}/{total})"


def _print_close_stats(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    abs_diff = (actual - expected).abs().float().flatten()
    expected_abs = expected.abs().float().flatten()
    allowed = atol + rtol * expected_abs
    within = abs_diff <= allowed

    total = abs_diff.numel()
    passed = int(within.sum().item())
    failed = total - passed

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    p99_abs = torch.quantile(abs_diff, 0.99).item()
    p999_abs = torch.quantile(abs_diff, 0.999).item()
    worst_ratio = (abs_diff / allowed.clamp_min(1e-12)).max().item()

    print(
        "[rocm_aiter_fa] "
        f"{label}: "
        f"pass={passed / total:.4%} ({passed}/{total}) "
        f"fail={_format_observed_rate(failed, total)} "
        f"allowed_fail={_format_allowed_rate(0.0, total)} "
        f"atol={atol:g} "
        f"rtol={rtol:g} "
        f"max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} "
        f"p99_abs={p99_abs:.6g} "
        f"p999_abs={p999_abs:.6g} "
        f"worst_ratio={worst_ratio:.6g}"
    )


# Direct kernel helper ----------------------------------------------------
def _run_direct_flash_attn_case(
    *,
    seq_lens: list[tuple[int, int]],
    num_blocks: int,
    sliding_window: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    import aiter

    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    set_random_seed(0)

    query_lens = [query_len for query_len, _ in seq_lens]
    kv_lens = [kv_len for _, kv_len in seq_lens]
    num_query_heads, num_kv_heads = DIRECT_NUM_HEADS
    total_query_tokens = sum(query_lens)
    total_kv_tokens = sum(kv_lens)
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = DIRECT_HEAD_SIZE**-0.5

    query = torch.randn(
        total_query_tokens,
        num_query_heads,
        DIRECT_HEAD_SIZE,
        dtype=DIRECT_DTYPE,
    )
    key_cache = torch.randn(
        num_blocks,
        BLOCK_SIZE,
        num_kv_heads,
        DIRECT_HEAD_SIZE,
        dtype=DIRECT_DTYPE,
    )
    value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seq_lens = torch.tensor([0] + kv_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    max_num_blocks_per_seq = (max_kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(
        0, num_blocks, (len(seq_lens), max_num_blocks_per_seq), dtype=torch.int32
    )

    token_to_batch = torch.empty(total_kv_tokens, dtype=torch.int32)
    seq_starts = torch.zeros(len(seq_lens), dtype=torch.int32)
    token_index = 0
    for batch_index, kv_len in enumerate(kv_lens):
        token_to_batch[token_index : token_index + kv_len] = batch_index
        token_index += kv_len

    gathered_key = torch.empty(
        total_kv_tokens,
        num_kv_heads,
        DIRECT_HEAD_SIZE,
        dtype=DIRECT_DTYPE,
    )
    gathered_value = torch.empty_like(gathered_key)
    cp_mha_gather_cache(
        key_cache=key_cache,
        value_cache=value_cache,
        key=gathered_key,
        value=gathered_value,
        block_tables=block_tables,
        k_scales=torch.ones(1, dtype=torch.float32),
        v_scales=torch.ones(1, dtype=torch.float32),
        cu_seqlens_kv=cu_seq_lens,
        token_to_batch=token_to_batch,
        seq_starts=seq_starts,
        dequant=False,
        kv_cache_layout="NHD",
        total_tokens=total_kv_tokens,
    )

    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    output = torch.empty_like(query)
    aiter.flash_attn_varlen_func(
        q=query,
        k=gathered_key,
        v=gathered_value,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        min_seqlen_q=0,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        alibi_slopes=None,
        return_lse=False,
        out=output,
    )

    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
    )
    return output, ref


# Backend contract tests --------------------------------------------------
def test_aiter_mha_backend_contract():
    """The ROCm backend advertises the dtypes, shapes, and attention types it
    is designed to handle."""
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend

    assert AiterFlashAttentionBackend.get_name() == "FLASH_ATTN"
    assert AiterFlashAttentionBackend.supported_dtypes == [
        torch.float16,
        torch.bfloat16,
    ]
    assert AiterFlashAttentionBackend.supported_kv_cache_dtypes == [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]
    assert AiterFlashAttentionBackend.get_supported_kernel_block_sizes() == [16, 32]
    assert AiterFlashAttentionBackend.get_supported_head_sizes() == [64, 128, 256]
    assert AiterFlashAttentionBackend.supports_attn_type(AttentionType.DECODER)
    assert not AiterFlashAttentionBackend.supports_attn_type(AttentionType.ENCODER)
    assert not AiterFlashAttentionBackend.supports_attn_type(
        AttentionType.ENCODER_DECODER
    )


def test_aiter_mha_backend_validates_kv_cache_block_size():
    """The backend should reject KV cache specs that cannot be gathered
    correctly by the ROCm kernel."""
    from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend
    from vllm.v1.kv_cache_interface import AttentionSpec

    with pytest.raises(ValueError, match="Block size must be a multiple of 16"):
        AiterFlashAttentionBackend.customize_spec(
            AttentionSpec(
                block_size=15, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
            )
        )
    # block_size == 1 is the per-token page-size probe and must pass through.
    AiterFlashAttentionBackend.customize_spec(
        AttentionSpec(block_size=1, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)
    )


def test_aiter_mha_backend_supports_compute_capability_matches_mi3xx_probe():
    """The backend should trust the ROCm MI3xx probe instead of the raw torch
    capability tuple."""
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend

    assert (
        AiterFlashAttentionBackend.supports_compute_capability(DeviceCapability(0, 0))
        is on_mi3xx()
    )


# Env and platform gate tests ---------------------------------------------
@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize(
    ("use_aiter", "use_mha", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_aiter_mha_env_flags_control_enablement(
    use_aiter, use_mha, expected, monkeypatch
):
    """Both the global AITER flag and the MHA-specific flag must be enabled
    before the ROCm MHA path is considered active."""
    from vllm._aiter_ops import rocm_aiter_ops

    _assert_aiter_supported()

    with monkeypatch.context() as mp:
        mp.setenv("VLLM_ROCM_USE_AITER", "1" if use_aiter else "0")
        mp.setenv("VLLM_ROCM_USE_AITER_MHA", "1" if use_mha else "0")
        _reload_envs()
        rocm_aiter_ops.refresh_env_variables()
        assert rocm_aiter_ops.is_mha_enabled() is expected

    _reload_envs()
    rocm_aiter_ops.refresh_env_variables()


def test_aiter_mha_platform_gate_matches_install_and_arch():
    """The global AITER availability check should only open on ROCm MI3xx when
    the aiter package is installed."""
    from vllm._aiter_ops import IS_AITER_FOUND, is_aiter_found_and_supported

    assert is_aiter_found_and_supported() is (
        current_platform.is_rocm() and on_mi3xx() and IS_AITER_FOUND
    )


# Extend warmup -----------------------------------------------------------
def test_extend_warmup_skips_sliding_window_layers():
    """A windowed extend uses the same call as a fresh prompt, so nothing is warmed."""
    from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionExtendKernel

    keys = AiterFlashAttentionExtendKernel().get_warmup_keys(
        dtype=torch.bfloat16,
        head_dim=64,
        num_qo_heads=8,
        num_kv_heads=2,
        sliding_window=(1023, 0),
        has_alibi=False,
        has_sinks=False,
    )
    assert keys == []


def test_extend_warmup_compiles_the_varlen_function(monkeypatch):
    """Each non-window key reaches AITER's varlen entry. The arguments are AITER's."""
    from unittest.mock import MagicMock

    from vllm.v1.attention.backends import rocm_aiter_fa
    from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionExtendKernel

    ops = MagicMock()
    monkeypatch.setattr(rocm_aiter_fa, "rocm_aiter_ops", ops)
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    kernel = AiterFlashAttentionExtendKernel()
    keys = kernel.get_warmup_keys(
        dtype=torch.bfloat16,
        head_dim=64,
        num_qo_heads=8,
        num_kv_heads=2,
        sliding_window=(-1, -1),
        has_alibi=True,
        has_sinks=True,
    )
    kernel.compile_many(keys)

    assert keys
    assert ops.flash_attn_varlen_func.call_count == len(keys)


@pytest.mark.parametrize("enable_jit_warmup", [True, False])
def test_extend_warmup_registers_when_enabled(monkeypatch, enable_jit_warmup):
    """Selecting the backend registers a compile when JIT warmup is on."""
    from unittest.mock import MagicMock

    from tests.v1.attention.utils import (
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.v1.attention.backends import rocm_aiter_fa

    config = create_vllm_config(dtype="bfloat16")
    config.kernel_config.enable_jit_warmup = enable_jit_warmup
    model_config = config.model_config
    head_size = model_config.get_head_size()
    registry = JitWarmupRegistry(config)
    with set_current_vllm_config(config):
        Attention(
            model_config.get_num_attention_heads(config.parallel_config),
            head_size,
            head_size**-0.5,
            model_config.get_num_kv_heads(config.parallel_config),
            cache_config=config.cache_config,
            prefix="layer",
            attn_backend=rocm_aiter_fa.AiterFlashAttentionBackend,
        )
        with registry.activate():
            rocm_aiter_fa.AiterFlashAttentionMetadataBuilder(
                create_standard_kv_cache_spec(config),
                ["layer"],
                config,
                torch.device(current_platform.device_type),
            )

    ops = MagicMock()
    monkeypatch.setattr(rocm_aiter_fa, "rocm_aiter_ops", ops)
    registry.warmup()

    assert ops.flash_attn_varlen_func.called is enable_jit_warmup


# Kernel path tests -------------------------------------------------------
def _run_single_seq_flash_attn_case(
    *,
    head_size: int,
    num_heads: tuple[int, int],
    seq_lens: tuple[int, int],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    import aiter

    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    set_random_seed(0)

    num_q_heads, num_kv_heads = num_heads
    query_len, kv_len = seq_lens
    scale = head_size**-0.5

    query = torch.randn(query_len, num_q_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor([0, query_len], dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seq_lens = torch.tensor([0, kv_len], dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    max_num_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(0, NUM_BLOCKS, (1, max_num_blocks), dtype=torch.int32)

    token_to_batch = torch.zeros(kv_len, dtype=torch.int32)
    seq_starts = torch.zeros(1, dtype=torch.int32)
    gathered_key = torch.empty(kv_len, num_kv_heads, head_size, dtype=dtype)
    gathered_value = torch.empty_like(gathered_key)

    cp_mha_gather_cache(
        key_cache=key_cache,
        value_cache=value_cache,
        key=gathered_key,
        value=gathered_value,
        block_tables=block_tables,
        k_scales=torch.ones(1, dtype=torch.float32),
        v_scales=torch.ones(1, dtype=torch.float32),
        cu_seqlens_kv=cu_seq_lens,
        token_to_batch=token_to_batch,
        seq_starts=seq_starts,
        dequant=False,
        kv_cache_layout="NHD",
        total_tokens=kv_len,
    )

    output = torch.empty_like(query)
    aiter.flash_attn_varlen_func(
        q=query,
        k=gathered_key,
        v=gathered_value,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=query_len,
        max_seqlen_k=kv_len,
        min_seqlen_q=0,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        return_lse=False,
        out=output,
    )

    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[query_len],
        kv_lens=[kv_len],
        block_tables=block_tables,
        scale=scale,
    )
    return output, ref


@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_heads", HEAD_SIZE_TEST_NUM_HEADS_PAIRS)
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
def test_aiter_fa_head_sizes(head_size, dtype, num_heads, seq_lens):
    """AITER flash attention should stay accurate across supported head sizes."""
    atol = 1.5e-2
    rtol = 1e-2
    _assert_aiter_supported()

    output, ref = _run_single_seq_flash_attn_case(
        head_size=head_size,
        num_heads=num_heads,
        seq_lens=seq_lens,
        dtype=dtype,
    )

    _print_close_stats(
        "head_sizes "
        f"dtype={dtype} head_size={head_size} "
        f"num_heads={num_heads} seq_lens={seq_lens}",
        output,
        ref,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS_PAIRS)
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_aiter_mha_varlen_paged_kv(head_size, num_heads, seq_lens, dtype):
    """Test AITER flash attention varlen with paged KV cache.

    Exercises: VLLM_ROCM_USE_AITER, VLLM_ROCM_USE_AITER_MHA
    """
    atol = 1.5e-2
    rtol = 1e-2
    _assert_aiter_supported()

    output, ref = _run_single_seq_flash_attn_case(
        head_size=head_size,
        num_heads=num_heads,
        seq_lens=seq_lens,
        dtype=dtype,
    )

    _print_close_stats(
        "varlen_paged_kv "
        f"dtype={dtype} head_size={head_size} "
        f"num_heads={num_heads} seq_lens={seq_lens}",
        output,
        ref,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize("num_heads", NUM_HEADS_PAIRS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_aiter_mha_multi_batch(num_heads, head_size, dtype):
    """Test AITER flash attention with multiple sequences in a batch."""
    atol = 1.5e-2
    rtol = 1e-2
    _assert_aiter_supported()
    import aiter

    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    set_random_seed(42)

    num_q_heads, num_kv_heads = num_heads
    seq_lens = [(4, 128), (2, 256), (8, 64)]
    query_lens = [q for q, _ in seq_lens]
    kv_lens = [k for _, k in seq_lens]
    num_seqs = len(seq_lens)
    scale = head_size**-0.5

    total_q = sum(query_lens)
    total_kv = sum(kv_lens)

    query = torch.randn(total_q, num_q_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seq_lens = torch.tensor([0] + kv_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )

    max_kv_len = max(kv_lens)
    max_num_blocks = (max_kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks), dtype=torch.int32
    )

    token_to_batch = torch.empty(total_kv, dtype=torch.int32)
    seq_starts = torch.zeros(num_seqs, dtype=torch.int32)
    tok_idx = 0
    for b, kl in enumerate(kv_lens):
        token_to_batch[tok_idx : tok_idx + kl] = b
        tok_idx += kl

    gathered_key = torch.empty(total_kv, num_kv_heads, head_size, dtype=dtype)
    gathered_value = torch.empty_like(gathered_key)

    cp_mha_gather_cache(
        key_cache=key_cache,
        value_cache=value_cache,
        key=gathered_key,
        value=gathered_value,
        block_tables=block_tables,
        k_scales=torch.ones(1, dtype=torch.float32),
        v_scales=torch.ones(1, dtype=torch.float32),
        cu_seqlens_kv=cu_seq_lens,
        token_to_batch=token_to_batch,
        seq_starts=seq_starts,
        dequant=False,
        kv_cache_layout="NHD",
        total_tokens=total_kv,
    )

    output = torch.empty_like(query)
    aiter.flash_attn_varlen_func(
        q=query,
        k=gathered_key,
        v=gathered_value,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max_kv_len,
        min_seqlen_q=0,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        return_lse=False,
        out=output,
    )

    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
    )

    _print_close_stats(
        "multi_batch "
        f"dtype={dtype} head_size={head_size} "
        f"num_heads={num_heads} seq_lens={seq_lens}",
        output,
        ref,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize("num_blocks", DIRECT_NUM_BLOCKS)
def test_aiter_fa_large_block_table_matches_reference(num_blocks):
    """The direct paged-KV path should stay stable for both normal and very
    large block tables."""
    atol = 2e-2
    rtol = 2e-2
    _assert_aiter_supported()

    output, ref = _run_direct_flash_attn_case(
        seq_lens=DIRECT_SEQ_LENS,
        num_blocks=num_blocks,
    )
    _print_close_stats(
        f"direct_varlen_paged_kv num_blocks={num_blocks}",
        output,
        ref,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
def test_aiter_fa_sliding_window_matches_reference():
    """The direct kernel should respect the same sliding-window causal mask as
    the naive reference implementation."""
    atol = 2e-2
    rtol = 2e-2
    _assert_aiter_supported()

    output, ref = _run_direct_flash_attn_case(
        seq_lens=[(8, 523), (24, 37), (3, 2011)],
        num_blocks=2048,
        sliding_window=256,
    )
    _print_close_stats(
        "sliding_window num_blocks=2048 window=256",
        output,
        ref,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize(
    "shuffle,sliding_window", [(False, None), (False, 16), (True, None)]
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("batch", ["prefill", "extend", "mixed"])
def test_aiter_fa_shared_kv_matches_reference(
    monkeypatch, shuffle, sliding_window, kv_cache_dtype, batch
):
    """Shared layers read cached K/V for prefill and extend without modifying it."""
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backends.rocm_aiter_fa import (
        AiterFlashAttentionBackend,
        AiterFlashAttentionMetadataBuilder,
    )

    _assert_aiter_supported()
    monkeypatch.setattr(rocm_aiter_ops, "_SHUFFLE_KV_CACHE_ENABLED", shuffle)
    set_random_seed(42)
    batch_spec = {
        "prefill": BatchSpec(seq_lens=[32, 24, 1], query_lens=[32, 24, 1]),
        "extend": BatchSpec(seq_lens=[80, 56], query_lens=[32, 24]),
        "mixed": BatchSpec(seq_lens=[48, 80, 24], query_lens=[1, 32, 24]),
    }[batch]
    config = create_vllm_config(
        model_name="Qwen/Qwen3-0.6B",
        dtype="bfloat16",
        max_num_batched_tokens=1024,
    )
    config.cache_config.cache_dtype = kv_cache_dtype
    num_heads = config.model_config.get_num_attention_heads(config.parallel_config)
    num_kv_heads = config.model_config.get_num_kv_heads(config.parallel_config)
    head_size = config.model_config.get_head_size()
    scale = head_size**-0.5
    with set_current_vllm_config(config):
        target = Attention(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            cache_config=config.cache_config,
            per_layer_sliding_window=sliding_window,
            prefix="target",
            attn_backend=AiterFlashAttentionBackend,
        )
        layer = Attention(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            cache_config=config.cache_config,
            per_layer_sliding_window=sliding_window,
            kv_sharing_target_layer_name="target",
            prefix="shared",
            attn_backend=AiterFlashAttentionBackend,
        )
        builder = AiterFlashAttentionMetadataBuilder(
            create_standard_kv_cache_spec(config),
            ["target", "shared"],
            config,
            torch.device("cuda"),
        )
        common = create_common_attn_metadata(
            batch_spec, BLOCK_SIZE, torch.device("cuda"), max_block_idx=32
        )
        cache_dtype = (
            current_platform.fp8_dtype() if kv_cache_dtype == "fp8" else torch.bfloat16
        )
        kv_cache = (
            torch.empty(
                2, 32, BLOCK_SIZE, num_kv_heads * head_size, dtype=cache_dtype
            ).transpose(0, 1)
            if shuffle
            else torch.empty(
                32, num_kv_heads, BLOCK_SIZE, 2 * head_size, dtype=cache_dtype
            )
        )
        target.kv_cache = kv_cache
        layer.kv_cache = kv_cache
        k_scale = 0.5 if kv_cache_dtype == "fp8" and not shuffle else 1.0
        v_scale = 0.25 if kv_cache_dtype == "fp8" and not shuffle else 1.0
        for attn_layer in (target, layer):
            attn_layer._k_scale.fill_(k_scale)
            attn_layer._v_scale.fill_(v_scale)
        key = torch.randn(
            32 * BLOCK_SIZE, num_kv_heads, head_size, dtype=torch.bfloat16
        )
        value = torch.randn_like(key)
        from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionImpl

        assert isinstance(target.impl, AiterFlashAttentionImpl)
        target.impl.do_kv_cache_update(
            target,
            key,
            value,
            kv_cache,
            torch.arange(32 * BLOCK_SIZE, dtype=torch.int64),
        )
        key_cache = ((key / k_scale).to(cache_dtype).to(key.dtype) * k_scale).reshape(
            32, BLOCK_SIZE, num_kv_heads, head_size
        )
        value_cache = (
            (value / v_scale).to(cache_dtype).to(value.dtype) * v_scale
        ).reshape(32, BLOCK_SIZE, num_kv_heads, head_size)
        with torch.device("cpu"):
            metadata = builder.build(0, common)
        cache_before = kv_cache.view(torch.uint8).clone()
        query = torch.randn(
            batch_spec.compute_num_tokens(),
            num_heads,
            head_size,
            dtype=torch.bfloat16,
        )
        output = torch.empty_like(query)
        layer.impl.forward(layer, query, None, None, kv_cache, metadata, output)
        expected = ref_paged_attn(
            query,
            key_cache.contiguous(),
            value_cache.contiguous(),
            batch_spec.query_lens,
            batch_spec.seq_lens,
            common.block_table_tensor,
            scale,
            sliding_window,
        )
        num_decode_tokens = metadata.num_decode_tokens
        if kv_cache_dtype == "fp8" and num_decode_tokens:
            # Use the established FP8 decode tolerance; gathered prefill and
            # extend tokens must still satisfy the tighter BF16 tolerance.
            torch.testing.assert_close(
                output[:num_decode_tokens],
                expected[:num_decode_tokens],
                atol=6e-2,
                rtol=1e-1,
            )
            output = output[num_decode_tokens:]
            expected = expected[num_decode_tokens:]
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            kv_cache.view(torch.uint8), cache_before, atol=0, rtol=0
        )


@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize("use_sinks", [True, False])
def test_aiter_fa_chunked_extend_with_sinks_matches_reference(monkeypatch, use_sinks):
    """Extends whose cached context spans several chunks count each sink once."""
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backends import rocm_aiter_fa
    from vllm.v1.attention.backends.rocm_aiter_fa import (
        AiterFlashAttentionBackend,
        AiterFlashAttentionImpl,
        AiterFlashAttentionMetadataBuilder,
    )

    _assert_aiter_supported()
    monkeypatch.setattr(rocm_aiter_ops, "_SHUFFLE_KV_CACHE_ENABLED", False)
    # 32 context tokens per request per chunk: context lens 128, 32 and 69 give
    # 4 chunks, with empty chunks 1-3 for the second request and 3 for the third.
    monkeypatch.setattr(rocm_aiter_fa, "_CP_TOKENS_PER_ITER_ROCM", 96)
    set_random_seed(42)
    batch_spec = BatchSpec(seq_lens=[160, 48, 77], query_lens=[32, 16, 8])
    num_blocks = 64
    # head_dim 64, as in gpt-oss: at 128, aiter < 0.1.28 routes bf16 varlen calls
    # to its v3 kernel, which ignores sinks (ROCm/aiter#6906).
    config = create_vllm_config(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="bfloat16",
        max_num_batched_tokens=1024,
    )
    num_heads = config.model_config.get_num_attention_heads(config.parallel_config)
    num_kv_heads = config.model_config.get_num_kv_heads(config.parallel_config)
    head_size = config.model_config.get_head_size()
    scale = head_size**-0.5
    # Sinks comparable to the softmax mass of the real keys.
    sinks = torch.randn(num_heads, dtype=torch.float32) + 4.0 if use_sinks else None
    with set_current_vllm_config(config):
        layer = Attention(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            cache_config=config.cache_config,
            prefix="layer",
            attn_backend=AiterFlashAttentionBackend,
            sinks=sinks,
        )
        builder = AiterFlashAttentionMetadataBuilder(
            create_standard_kv_cache_spec(config),
            ["layer"],
            config,
            torch.device("cuda"),
        )
        common = create_common_attn_metadata(
            batch_spec, BLOCK_SIZE, torch.device("cuda"), max_block_idx=num_blocks
        )
        with torch.device("cpu"):
            metadata = builder.build(0, common)
    assert isinstance(layer.impl, AiterFlashAttentionImpl)
    assert metadata.num_extends == 3
    assert metadata.extend_metadata is not None
    assert metadata.extend_metadata.chunk_context_metadata.num_chunks == 4

    kv_cache = torch.empty(
        num_blocks, num_kv_heads, BLOCK_SIZE, 2 * head_size, dtype=torch.bfloat16
    )
    key_cache, value_cache = layer.impl._split_kv_cache(kv_cache)
    key_cache.copy_(torch.randn_like(key_cache))
    value_cache.copy_(torch.randn_like(value_cache))
    key_cache, value_cache = key_cache.contiguous(), value_cache.contiguous()

    # The new tokens' K/V must match what the cache holds at their positions.
    keys, values = [], []
    for i, (seq_len, query_len) in enumerate(
        zip(batch_spec.seq_lens, batch_spec.query_lens)
    ):
        pos = torch.arange(seq_len - query_len, seq_len)
        blocks = common.block_table_tensor[i, pos // BLOCK_SIZE]
        keys.append(key_cache[blocks, pos % BLOCK_SIZE])
        values.append(value_cache[blocks, pos % BLOCK_SIZE])
    query = torch.randn(
        batch_spec.compute_num_tokens(), num_heads, head_size, dtype=torch.bfloat16
    )
    output = torch.empty_like(query)
    layer.impl.forward(
        layer, query, torch.cat(keys), torch.cat(values), kv_cache, metadata, output
    )
    expected = ref_paged_attn(
        query,
        key_cache,
        value_cache,
        batch_spec.query_lens,
        batch_spec.seq_lens,
        common.block_table_tensor,
        scale,
        sinks=sinks,
    )
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


# Layers the flag must leave correct: every head shape for full attention, one
# shape for each layer kind or cache that has its own prefill path. The latter
# use the test model's own head shape (Qwen3-0.6B), which the metadata builder
# sizes its BF16 prefill workspace from.
_FLYDSL = pytest.mark.skipif(get_cdna_version() != 4, reason="gfx950 FlyDSL kernel")
_PREFILL_QUERY_QUANT_CASES = (
    [
        pytest.param(
            head_size,
            num_heads,
            "full_attention",
            False,
            False,
            id=f"full-{head_size}-{nh}",
        )
        for head_size in HEAD_SIZES
        for nh, num_heads in enumerate(NUM_HEADS_PAIRS)
    ]
    + [
        pytest.param(128, (16, 8), case, False, False, id=case)
        for case in ("sliding_window", "sinks", "bf16_kv_cache", "shuffled_kv_layout")
    ]
    + [
        # gfx950: the FlyDSL kernel (head size 64, 8 query heads per KV head), also
        # on sliding-window layers and with the exp-free softmax.
        pytest.param(
            64,
            (16, 2),
            case,
            True,
            exp_free,
            id=f"{case}-64-gqa8-{'exp_free' if exp_free else 'exp2'}",
            marks=_FLYDSL,
        )
        for case in ("full_attention", "sliding_window")
        for exp_free in (False, True)
    ]
)


@pytest.mark.skipif(get_cdna_version() not in (3, 4), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize(
    "head_size, num_heads, layer_kind, flydsl, exp_free", _PREFILL_QUERY_QUANT_CASES
)
@pytest.mark.parametrize("q_scale", [1.0, 4.0])
@pytest.mark.parametrize("prefill_first", [False, True])
def test_aiter_fa_fp8_query_extend_matches_reference(
    monkeypatch,
    head_size,
    num_heads,
    layer_kind,
    flydsl,
    exp_free,
    q_scale,
    prefill_first,
):
    """With `use_prefill_query_quantization` on, a batch with a decode, extends
    and a pure prefill matches the reference attention of the layer: full
    attention, sliding window, sinks, a BF16 KV cache or the shuffled KV
    layout. Each extend and prefill matches, by relative L2, a float32
    reference computed from the BF16 query or from the query quantized to FP8
    with the layer's static q scale (an FP8 query also rounds the probabilities
    to FP8, so single elements of rows that attend over a few keys differ by
    more than an elementwise tolerance would allow). On gfx950, head size 64
    with 8 query heads per KV head runs the FlyDSL kernel, also for sliding-window
    layers and with `use_prefill_exp_free_softmax` (then against the query the
    kernel casts in-kernel)."""
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backends.rocm_aiter_fa import (
        AiterFlashAttentionBackend,
        AiterFlashAttentionImpl,
        AiterFlashAttentionMetadataBuilder,
    )

    _assert_aiter_supported()
    shuffle = layer_kind == "shuffled_kv_layout"
    monkeypatch.setattr(rocm_aiter_ops, "_SHUFFLE_KV_CACHE_ENABLED", shuffle)
    kv_cache_dtype = "auto" if layer_kind == "bf16_kv_cache" else "fp8"
    sliding_window = 32 if layer_kind == "sliding_window" else None
    set_random_seed(0)
    # A decode, the extends and a pure prefill, which may precede the extends.
    if prefill_first:
        batch_spec = BatchSpec(
            seq_lens=[500, 64, 700, 333, 90], query_lens=[1, 64, 96, 40, 24]
        )
    else:
        batch_spec = BatchSpec(
            seq_lens=[500, 700, 333, 90, 64], query_lens=[1, 96, 40, 24, 64]
        )
    num_blocks = 256
    num_q_heads, num_kv_heads = num_heads
    # Sinks comparable to the softmax mass of the real keys.
    sinks = (
        torch.randn(num_q_heads, dtype=torch.float32) + 4.0
        if layer_kind == "sinks"
        else None
    )
    config = create_vllm_config(
        dtype="bfloat16",
        max_num_batched_tokens=1024,
        hf_config_override={
            "num_attention_heads": num_q_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_size,
        },
    )
    config.cache_config.cache_dtype = kv_cache_dtype
    config.attention_config.use_prefill_query_quantization = True
    config.attention_config.use_prefill_exp_free_softmax = exp_free
    scale = head_size**-0.5

    with set_current_vllm_config(config):
        layer = Attention(
            num_q_heads,
            head_size,
            scale,
            num_kv_heads,
            cache_config=config.cache_config,
            per_layer_sliding_window=sliding_window,
            prefix="layer",
            attn_backend=AiterFlashAttentionBackend,
            sinks=sinks,
        )
        builder = AiterFlashAttentionMetadataBuilder(
            create_standard_kv_cache_spec(config),
            ["layer"],
            config,
            torch.device("cuda"),
        )
        common = create_common_attn_metadata(
            batch_spec, BLOCK_SIZE, torch.device("cuda"), arange_block_indices=True
        )
        with torch.device("cpu"):
            metadata = builder.build(0, common)
    impl = layer.impl
    assert isinstance(impl, AiterFlashAttentionImpl)
    layer._q_scale.fill_(q_scale)

    cache_dtype = (
        current_platform.fp8_dtype() if kv_cache_dtype == "fp8" else torch.bfloat16
    )
    kv_cache = (
        torch.empty(
            2, num_blocks, BLOCK_SIZE, num_kv_heads * head_size, dtype=cache_dtype
        ).transpose(0, 1)
        if shuffle
        else torch.empty(
            num_blocks, num_kv_heads, BLOCK_SIZE, 2 * head_size, dtype=cache_dtype
        )
    )
    shape = (num_blocks, BLOCK_SIZE, num_kv_heads, head_size)
    key = torch.randn(shape).to(cache_dtype).to(torch.bfloat16)
    value = torch.randn(shape).to(cache_dtype).to(torch.bfloat16)
    # Keys past each request's last token must be masked out of the scores.
    for i, seq_len in enumerate(batch_spec.seq_lens):
        last = common.block_table_tensor[i, (seq_len - 1) // BLOCK_SIZE]
        key[last, (seq_len - 1) % BLOCK_SIZE + 1 :] = 200.0
    impl.do_kv_cache_update(
        layer,
        key.view(-1, num_kv_heads, head_size),
        value.view(-1, num_kv_heads, head_size),
        kv_cache,
        torch.arange(num_blocks * BLOCK_SIZE, dtype=torch.int64),
    )
    key_ref, value_ref = key.float(), value.float()
    # The new tokens' K/V as the forward receives them: what the cache holds.
    keys, values = [], []
    for i, (seq_len, query_len) in enumerate(
        zip(batch_spec.seq_lens, batch_spec.query_lens)
    ):
        pos = torch.arange(seq_len - query_len, seq_len)
        blocks = common.block_table_tensor[i, pos // BLOCK_SIZE]
        keys.append(key[blocks, pos % BLOCK_SIZE])
        values.append(value[blocks, pos % BLOCK_SIZE])
    query = torch.randn(
        batch_spec.compute_num_tokens(), num_q_heads, head_size, dtype=torch.bfloat16
    )
    fp8_dtype = current_platform.fp8_dtype()
    references = [
        ref_paged_attn(
            q,
            key_ref,
            value_ref,
            batch_spec.query_lens,
            batch_spec.seq_lens,
            common.block_table_tensor,
            scale,
            sliding_window,
            sinks,
        )
        for q in (
            query.float(),
            (query.float() / q_scale).to(fp8_dtype).float() * q_scale,
            *(
                [_exp_free_query(query, scale, q_scale, exp_free=exp_free)]
                if flydsl
                else []
            ),
        )
    ]

    from aiter.ops.triton.unified_attention import unified_attention

    from vllm.v1.attention.ops.flydsl_kernels import fp8_extend

    output = torch.empty_like(query)
    with (
        patch(
            "aiter.ops.triton.unified_attention.unified_attention",
            wraps=unified_attention,
        ) as spy,
        patch.object(
            fp8_extend, "fp8_extend_attention", wraps=fp8_extend.fp8_extend_attention
        ) as fly_spy,
    ):
        impl.forward(
            layer, query, torch.cat(keys), torch.cat(values), kv_cache, metadata, output
        )
    if layer_kind == "full_attention" or flydsl:
        # The FP8-query output is within the bound of the BF16-query reference
        # too, so check that the extends and the prefill took the FP8-query path.
        fp8_rows = sum(
            call.kwargs["q"].shape[0]
            for call in spy.call_args_list
            if call.kwargs["q"].dtype == fp8_dtype
        ) + sum(call.args[0].shape[0] for call in fly_spy.call_args_list)
        assert fp8_rows == sum(batch_spec.query_lens[1:])
        assert fly_spy.called is flydsl
    # The decode does not take the flag; it keeps the established FP8-cache
    # decode tolerance (the shuffled layout's decode kernel needs it).
    torch.testing.assert_close(
        output[:1].float(), references[0][:1], atol=6e-2, rtol=1e-1
    )
    start = 1
    for i, query_len in enumerate(batch_spec.query_lens[1:], start=1):
        rows = slice(start, start + query_len)
        errors = [
            ((output[rows].float() - ref[rows]).norm() / ref[rows].norm()).item()
            for ref in references
        ]
        assert min(errors) < FP8_EXTEND_REL_L2[exp_free], (
            f"request {i}: relative L2 {errors}"
        )
        start += query_len


# Relative L2 bound per softmax mode of the FlyDSL FP8 extend kernel vs float32
# attention on the query it sees.
FP8_EXTEND_REL_L2 = {False: 3e-2, True: 4e-2}


def _exp_free_query(
    query: torch.Tensor,
    scale: float,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    exp_free: bool = True,
) -> torch.Tensor:
    """The query the kernel sees: cast in-kernel to FP8 pre-multiplied by
    c = log2(e) scale k_scale (times 8 for exp-free codes) and a power-of-two
    2^-e (e = ceil(log2(c q_scale))) that the QK MFMA undoes."""
    c = (8 if exp_free else 1) * math.log2(math.e) * scale * k_scale
    pre = c * 2.0 ** -math.ceil(math.log2(c * q_scale))
    cast = (query.float() * pre).clamp(-448, 448).to(torch.float8_e4m3fn)
    return cast.float() / pre


FP8_EXTEND_CASES = [
    # (q heads, kv heads, page, [(cached, new) per request], window)
    (32, 4, 32, [(5000, 1000), (0, 777), (123, 1), (12000, 1100)], 0),
    (32, 4, 16, [(5000, 1000), (0, 777), (123, 1), (12000, 1100)], 2048),
    (32, 4, 32, [(5000, 1000), (0, 777), (123, 1), (12000, 1100)], 2048),
    (16, 2, 16, [(33, 300), (12288, 1500)], 2048),
    (32, 4, 16, [(3000, 517), (0, 300), (77, 5)], 100),
    (32, 4, 16, [(3000, 517), (0, 40), (77, 5)], 16),
    (32, 4, 16, [(300, 51), (0, 40)], 1),
    (32, 4, 32, [(0, 16), (0, 17), (31, 1)], 0),
    # More than 255 requests (the kernel's request search is sized for the batch).
    (8, 1, 16, [(i % 37, 1 + i % 19) for i in range(300)], 0),
]


@pytest.mark.skipif(get_cdna_version() != 4, reason="gfx950 FlyDSL kernel only")
@pytest.mark.parametrize("exp_free", [False, True], ids=["exp2", "exp_free"])
@pytest.mark.parametrize("q_scale", [1.0, 256.0])
@pytest.mark.parametrize("case", FP8_EXTEND_CASES)
def test_rocm_fp8_extend_matches_reference(case, q_scale, exp_free):
    """The FlyDSL FP8 extend kernel on varlen batches with prefixes and new
    lengths that are not multiples of 16 or the page, an empty prefix, a
    1-token extend and sliding windows down to 1. Cache slots past each
    request's length hold NaN bytes, and with a window every block wholly before
    a request's first window points at a NaN page, as a freed block would;
    neither may reach the output. With `q_scale` 256 the queries exceed the
    FP8 range but not the q scale's (K's scale is 1/256, so the scores keep
    their distribution); they must not be clipped. V's scale is 0.5 there.
    Each request matches float32 attention on the query the kernel sees
    (FP8-quantized by `q_scale`, or the exp-free in-kernel cast) by relative
    L2."""
    from vllm.v1.attention.ops import rocm_fp8_extend

    num_q_heads, num_kv_heads, page, reqs, window = case
    fp8 = torch.float8_e4m3fn
    set_random_seed(0)
    head_size = rocm_fp8_extend.HEAD_SIZE
    scale = head_size**-0.5
    lens = [p + n for p, n in reqs]
    num_pages_per_req = [(n + page - 1) // page for n in lens]
    num_pages = sum(num_pages_per_req) + 64
    # NaN bytes everywhere; page 0 stands in for unmapped blocks.
    pool = torch.full(
        (num_pages, num_kv_heads, page, 2 * head_size), 0x7F, dtype=torch.uint8
    ).view(fp8)
    perm = torch.randperm(num_pages - 1) + 1
    block_table = torch.zeros((len(reqs), max(num_pages_per_req)), dtype=torch.int32)
    offset = 0
    for b, seq_len in enumerate(lens):
        ids = perm[offset : offset + num_pages_per_req[b]]
        offset += num_pages_per_req[b]
        kv = torch.randn((seq_len, num_kv_heads, 2 * head_size)).to(fp8)
        for i, pid in enumerate(ids.tolist()):
            rows = slice(i * page, min(seq_len, (i + 1) * page))
            pool[pid, :, : rows.stop - rows.start] = kv[rows].transpose(0, 1)
        block_table[b, : len(ids)] = ids.to(torch.int32)
    key_cache = pool[..., :head_size].transpose(1, 2)
    value_cache = pool[..., head_size:].transpose(1, 2)
    ref_table = block_table.clone()
    if window:
        for b, (cached, _) in enumerate(reqs):
            first = max(0, cached - (window - 1))
            block_table[b, : first // page] = 0
    new = [n for _, n in reqs]
    cu_seqlens_q = torch.tensor([0, *torch.tensor(new).cumsum(0).tolist()])
    query = (torch.randn((sum(new), num_q_heads, head_size)) * q_scale).to(
        torch.bfloat16
    )
    one = torch.ones(1)
    k_scale = 1.0 / q_scale
    v_scale = 1.0 if q_scale == 1.0 else 0.5
    output = torch.full_like(query, float("nan"))
    rocm_fp8_extend.extend(
        query,
        key_cache,
        value_cache,
        output,
        cu_seqlens_q.to(torch.int32),
        torch.tensor(lens, dtype=torch.int32),
        block_table,
        one * q_scale,
        one * k_scale,
        one * v_scale,
        scale,
        window,
        exp_free,
    )
    assert torch.isfinite(output).all()
    query_ref = _exp_free_query(query, scale, q_scale, k_scale, exp_free)
    for b, (cached, n) in enumerate(reqs):
        rows = slice(int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1]))
        expected = ref_paged_attn(
            query_ref[rows],
            key_cache.float().contiguous() * k_scale,
            value_cache.float().contiguous() * v_scale,
            [n],
            [cached + n],
            ref_table[b : b + 1],
            scale,
            sliding_window=window or None,
        )
        diff = output[rows].float() - expected
        assert diff.norm() < FP8_EXTEND_REL_L2[exp_free] * expected.norm(), (
            b,
            diff.norm() / expected.norm(),
        )


def _fp8_extend_single_query(
    key_rows: torch.Tensor, value_rows: torch.Tensor, window: int, exp_free: bool
) -> torch.Tensor:
    """One new token (the last key) over `key_rows` / `value_rows` ([L, 64] float,
    rounded to FP8 in a 16-token-page cache, 1 KV head, query = e0), returned as
    the output row of query head 0 (float)."""
    from vllm.v1.attention.ops import rocm_fp8_extend

    fp8 = torch.float8_e4m3fn
    seq_len, head_size = key_rows.shape
    pages = -(-seq_len // 16)
    pool = torch.zeros((pages, 1, 16, 2 * head_size)).to(fp8)
    flat = pool.view(pages * 16, 2 * head_size)
    flat[:seq_len, :head_size] = key_rows.to(fp8)
    flat[:seq_len, head_size:] = value_rows.to(fp8)
    query = torch.zeros((1, 8, head_size), dtype=torch.bfloat16)
    query[..., 0] = 1.0
    output = torch.empty_like(query)
    one = torch.ones(1)
    rocm_fp8_extend.extend(
        query,
        pool[..., :head_size].transpose(1, 2),
        pool[..., head_size:].transpose(1, 2),
        output,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([seq_len], dtype=torch.int32),
        torch.arange(pages, dtype=torch.int32)[None],
        one,
        one,
        one,
        1.0,
        window,
        exp_free,
    )
    return output[0, 0].float()


@pytest.mark.skipif(get_cdna_version() != 4, reason="gfx950 FlyDSL kernel only")
@pytest.mark.parametrize("exp_free", [False, True], ids=["exp2", "exp_free"])
@pytest.mark.parametrize("window", [0, 4096], ids=["full", "swa"])
def test_rocm_fp8_extend_far_keys_by_layer_type(exp_free, window):
    """Keys 11-13 octaves below the row maximum: full-attention layers drop them
    (as the FP8-query unified_attention path drops keys about 10 octaves down),
    sliding-window layers keep them (they replace BF16 attention). One key at the
    maximum (V = 0) and 2048 far keys (V = 1): the output is the far keys' share
    of the softmax, about 0.2 with them and 0 without."""
    set_random_seed(0)
    far = 2048
    head_size = 64
    keys = torch.zeros((far + 2, head_size))
    values = torch.zeros((far + 2, head_size))
    # Scores are K[:, 0] (query e0, scale 1); P = 2^((s - m) log2(e)).
    keys[0, 0] = 16.0
    octaves = 11.0 + 2.0 * torch.rand(far)
    keys[1 : far + 1, 0] = 16.0 - octaves / math.log2(math.e)
    keys[far + 1, 0] = -448.0  # the new token's own key, negligible
    values[1 : far + 1, 0] = 1.0
    q0 = _exp_free_query(torch.ones(1), 1.0, exp_free=exp_free).item()
    scores = q0 * keys.to(torch.float8_e4m3fn).float()[:, 0]
    p = torch.exp(scores - scores.max())
    with_far = (p * values[:, 0]).sum() / p.sum()
    got = _fp8_extend_single_query(keys, values, window, exp_free)[0]
    assert with_far > 0.1
    if window:
        # FP8 rounding of P (exp2) or the codes' approximation of 2^x (exp_free);
        # the far keys' scores take few distinct FP8 values, so it does not average
        tol = 0.05
        torch.testing.assert_close(
            got.cpu(), with_far.cpu(), atol=tol * with_far, rtol=0
        )
    else:
        assert got.abs() < 1e-3


# Decode path test --------------------------------------------------------
@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize("dtype", DTYPES)
def test_aiter_mha_decode_single_token(dtype):
    """Test AITER MHA for decode (single query token per sequence)."""
    atol = 1.5e-2
    rtol = 1e-2
    _assert_aiter_supported()
    import aiter

    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    set_random_seed(0)

    num_q_heads, num_kv_heads = 8, 8
    head_size = 128
    kv_len = 512
    scale = head_size**-0.5

    query = torch.randn(1, num_q_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor([0, 1], dtype=torch.int32)
    cu_seq_lens = torch.tensor([0, kv_len], dtype=torch.int32)

    max_num_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(0, NUM_BLOCKS, (1, max_num_blocks), dtype=torch.int32)

    token_to_batch = torch.zeros(kv_len, dtype=torch.int32)
    seq_starts = torch.zeros(1, dtype=torch.int32)
    gathered_key = torch.empty(kv_len, num_kv_heads, head_size, dtype=dtype)
    gathered_value = torch.empty_like(gathered_key)

    cp_mha_gather_cache(
        key_cache=key_cache,
        value_cache=value_cache,
        key=gathered_key,
        value=gathered_value,
        block_tables=block_tables,
        k_scales=torch.ones(1, dtype=torch.float32),
        v_scales=torch.ones(1, dtype=torch.float32),
        cu_seqlens_kv=cu_seq_lens,
        token_to_batch=token_to_batch,
        seq_starts=seq_starts,
        dequant=False,
        kv_cache_layout="NHD",
        total_tokens=kv_len,
    )

    output = torch.empty_like(query)
    aiter.flash_attn_varlen_func(
        q=query,
        k=gathered_key,
        v=gathered_value,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=1,
        max_seqlen_k=kv_len,
        min_seqlen_q=0,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        return_lse=False,
        out=output,
    )

    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1],
        kv_lens=[kv_len],
        block_tables=block_tables,
        scale=scale,
    )

    _print_close_stats(
        f"decode_single_token dtype={dtype} kv_len={kv_len}",
        output,
        ref,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


# FP8 KV cache test -------------------------------------------------------
@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize("dtype", DTYPES)
def test_aiter_mha_varlen_fp8_kv(dtype):
    """AITER flash attention with FP8 KV cache matches reference on BF16-cast KV.

    The reference dequantizes the same FP8 KV tensors the kernel reads, so
    e4m3 rounding cancels on both sides; the residual is bf16/fp16 kernel
    rounding (~few millidecimals), not fp8-scale quantization error.
    """
    if dtype == torch.bfloat16:
        atol, rtol = 6e-3, 1e-2
    else:
        atol, rtol = 7e-4, 1e-3
    _assert_aiter_supported()
    if not current_platform.supports_fp8():
        pytest.skip("FP8 not supported on this hardware")

    import aiter

    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    FP8_DTYPE = current_platform.fp8_dtype()

    set_random_seed(10)

    num_q_heads, num_kv_heads = 8, 8
    head_size = 128
    query_len, kv_len = 4, 128
    scale = head_size**-0.5

    query = torch.randn(query_len, num_q_heads, head_size, dtype=dtype)
    # FP8 KV cache; clamp to stay in FP8 representable range
    key_cache_fp8 = torch.clamp(
        torch.randn(NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size), -1.0, 1.0
    ).to(FP8_DTYPE)
    value_cache_fp8 = torch.clamp(
        torch.randn(NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size), -1.0, 1.0
    ).to(FP8_DTYPE)

    cu_query_lens = torch.tensor([0, query_len], dtype=torch.int32)
    cu_seq_lens = torch.tensor([0, kv_len], dtype=torch.int32)
    max_num_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(0, NUM_BLOCKS, (1, max_num_blocks), dtype=torch.int32)
    token_to_batch = torch.zeros(kv_len, dtype=torch.int32)
    seq_starts = torch.zeros(1, dtype=torch.int32)

    # Gather and dequantize FP8 KV to dtype
    gathered_key = torch.empty(kv_len, num_kv_heads, head_size, dtype=dtype)
    gathered_value = torch.empty_like(gathered_key)
    k_scales = torch.ones(1, dtype=torch.float32)
    v_scales = torch.ones(1, dtype=torch.float32)

    cp_mha_gather_cache(
        key_cache=key_cache_fp8,
        value_cache=value_cache_fp8,
        key=gathered_key,
        value=gathered_value,
        block_tables=block_tables,
        k_scales=k_scales,
        v_scales=v_scales,
        cu_seqlens_kv=cu_seq_lens,
        token_to_batch=token_to_batch,
        seq_starts=seq_starts,
        dequant=True,
        kv_cache_layout="NHD",
        total_tokens=kv_len,
    )

    output = torch.empty_like(query)
    aiter.flash_attn_varlen_func(
        q=query,
        k=gathered_key,
        v=gathered_value,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=query_len,
        max_seqlen_k=kv_len,
        min_seqlen_q=0,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        return_lse=False,
        out=output,
    )

    # Reference: ref_paged_attn on dtype-cast KV (simulates perfect dequant)
    key_cache_ref = key_cache_fp8.to(dtype)
    value_cache_ref = value_cache_fp8.to(dtype)
    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache_ref,
        value_cache=value_cache_ref,
        query_lens=[query_len],
        kv_lens=[kv_len],
        block_tables=block_tables,
        scale=scale,
    )

    # FP8 quantization + dequantization cancels on the reference side; compare
    # kernel rounding at the working precision.
    _print_close_stats(
        f"varlen_fp8_kv dtype={dtype} query_len={query_len} kv_len={kv_len}",
        output,
        ref,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
def test_aiter_paged_decode_over_window_trimmed_block_table_matches_reference():
    """Sliding-window decode on the paged kernel, fed the trimmed block table.

    The metadata builder keeps only the blocks that overlap the window and
    rebases the sequence lengths onto the first kept block. That must give the
    same attention as the full table, for contexts below, at and above the
    window, including a window start that falls inside a block.
    """
    _assert_aiter_supported()
    from types import SimpleNamespace

    import aiter  # noqa: F401

    from vllm.v1.attention.backends.rocm_aiter_fa import (
        AiterFlashAttentionMetadataBuilder,
    )

    set_random_seed(0)
    device, dtype = "cuda", torch.bfloat16
    num_q_heads, num_kv_heads, head_size, block_size, window = 16, 2, 64, 16, 256
    seq_lens_list = [100, 256, 257, 300, 1000, 2011]
    num_seqs = len(seq_lens_list)
    max_blocks = (max(seq_lens_list) + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks + 4
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, device=device, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    block_table = (
        torch.randperm(num_blocks - 4, device=device)[: num_seqs * max_blocks]
        .reshape(num_seqs, max_blocks)
        .to(torch.int32)
    )
    seq_lens = torch.tensor(seq_lens_list, device=device, dtype=torch.int32)
    query = torch.randn(num_seqs, num_q_heads, head_size, device=device, dtype=dtype)
    scale = head_size**-0.5

    num_window_blocks = (window + block_size - 1) // block_size + 1
    builder = SimpleNamespace(
        aot_sliding_window=(window - 1, 0),
        block_size=block_size,
        _swa_block_table=torch.zeros(
            (num_seqs, num_window_blocks), dtype=torch.int32, device=device
        ),
        _swa_seq_lens=torch.zeros(num_seqs, dtype=torch.int32, device=device),
        _swa_block_offsets=torch.arange(
            num_window_blocks, dtype=torch.int64, device=device
        ),
    )
    swa_table, swa_lens, swa_max_len = (
        AiterFlashAttentionMetadataBuilder._build_swa_decode_tables(
            builder, block_table, seq_lens, num_seqs, max(seq_lens_list)
        )
    )
    assert swa_table is not None and swa_lens is not None
    assert swa_table.shape == (num_seqs, num_window_blocks)
    assert swa_max_len == window + block_size - 1
    assert int(swa_lens.max()) <= swa_max_len
    # Below the window nothing is dropped.
    assert int(swa_lens[0]) == seq_lens_list[0]

    partition = 256
    num_partitions = (swa_max_len + partition - 1) // partition
    workspace = torch.empty(
        num_seqs * num_q_heads * num_partitions * head_size * 2
        + 2 * num_seqs * num_q_heads * num_partitions * 4,
        dtype=torch.uint8,
        device=device,
    )
    one = torch.ones(1, device=device, dtype=torch.float32)
    output = torch.empty_like(query)
    torch.ops.aiter.paged_attention_v1(
        output,
        workspace,
        query,
        key_cache,
        value_cache,
        scale,
        swa_table,
        # The kernel reads query_start_loc[i + 1], so slice it like the backend.
        torch.arange(num_seqs + 1, device=device, dtype=torch.int32)[:num_seqs],
        swa_lens,
        swa_max_len,
        None,
        "auto",
        "NHD",
        0.0,
        one,
        one,
        None,
        partition,
        1,
        window,
    )

    ref = torch.empty_like(query)
    for i, length in enumerate(seq_lens_list):
        blocks = block_table[i, : (length + block_size - 1) // block_size].long()
        keys = key_cache[blocks].reshape(-1, num_kv_heads, head_size)[:length]
        values = value_cache[blocks].reshape(-1, num_kv_heads, head_size)[:length]
        keys, values = (
            keys[max(0, length - window) :],
            values[max(0, length - window) :],
        )
        keys = keys.float().repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        values = values.float().repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        weights = (torch.einsum("hd,lhd->hl", query[i].float(), keys) * scale).softmax(
            -1
        )
        ref[i] = torch.einsum("hl,lhd->hd", weights, values).to(dtype)
    torch.testing.assert_close(output, ref, atol=1.5e-2, rtol=1e-2)


def test_asm_kv_scale_tables_shared_across_builders():
    """Builders share one AsmKvScaleTables; equal (shape, scale) requests must
    return the same dense table, a different scale a different one."""
    from vllm.v1.attention.backends.rocm_aiter_fa import (
        AiterFlashAttentionMetadataBuilder,
        AsmKvScaleTables,
    )

    tables = AsmKvScaleTables()
    device = torch.device("cuda")
    table = tables.get(device, 8, 2, 16, 0.5)
    assert tables.get(device, 8, 2, 16, 0.5) is table
    assert tables.get(device, 8, 2, 16, 1.0) is not table
    assert table.shape == (8, 2, 16)
    assert table.dtype == torch.float32
    assert torch.all(table == 0.5)
    # The builder class holds the single process-wide instance.
    assert isinstance(
        AiterFlashAttentionMetadataBuilder.asm_kv_scale_tables, AsmKvScaleTables
    )
