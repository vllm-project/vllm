# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.test_attention_backends import (
    BATCH_SPECS,
    _test_backend_correctness,
)
from tests.v1.attention.utils import BatchSpec
from vllm.config import ModelConfig
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.b12x import get_b12x_paged_attention
from vllm.v1.attention.backends import b12x
from vllm.v1.attention.backends.b12x import (
    B12xPagedAttentionBackend,
    B12xPagedAttentionImpl,
    _kv_page_size,
    _max_page_table_width,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout


def _require_b12x_paged_attention() -> None:
    capability = current_platform.get_device_capability()
    if (
        not current_platform.is_cuda()
        or capability is None
        or not B12xPagedAttentionBackend.supports_compute_capability(capability)
    ):
        pytest.skip("b12x paged attention requires SM120 or SM121.")

    paged_attention = get_b12x_paged_attention()
    if paged_attention is None or not paged_attention.is_supported():
        pytest.skip("b12x paged attention is not available.")


def _causal_mask(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    *,
    context_len: int,
):
    return q_idx + context_len >= kv_idx


def _causal_sliding_window_mask(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    *,
    context_len: int,
    sliding_window: int,
):
    causal_mask = q_idx + context_len >= kv_idx
    window_mask = q_idx + context_len - kv_idx < sliding_window
    return causal_mask & window_mask


@pytest.mark.parametrize(
    ("dtype", "kv_cache_dtype", "paged_attention", "expected_reason"),
    [
        pytest.param(
            torch.float16,
            "fp8_e4m3",
            None,
            "b12x currently requires bfloat16 queries",
            id="query-dtype",
        ),
        pytest.param(
            torch.bfloat16,
            "float16",
            None,
            "b12x does not support float16 KV cache",
            id="kv-cache-dtype",
        ),
        pytest.param(
            torch.bfloat16,
            "auto",
            None,
            "Install the b12x backend with `pip install vllm[b12x]`",
            id="package-not-installed",
        ),
        pytest.param(
            torch.bfloat16,
            "auto",
            SimpleNamespace(is_supported=lambda: False),
            "b12x paged attention is not supported on the current device",
            id="device-api",
        ),
        pytest.param(
            torch.bfloat16,
            "auto",
            SimpleNamespace(is_supported=lambda: True),
            None,
            id="supported",
        ),
    ],
)
def test_b12x_attention_config_support(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    paged_attention,
    expected_reason: str | None,
) -> None:
    monkeypatch.setattr(
        b12x,
        "get_b12x_paged_attention",
        lambda: paged_attention,
    )

    reason = B12xPagedAttentionBackend.supports_combination(
        head_size=128,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        block_size=128,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        device_capability=DeviceCapability(12, 0),
    )

    assert reason == expected_reason


def test_b12x_attention_uses_two_plane_nhd_cache() -> None:
    spec = B12xPagedAttentionBackend.customize_spec(
        FullAttentionSpec(
            block_size=128,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.bfloat16,
        )
    )

    assert spec.num_head_slots == 2
    assert spec.state_content_bytes == 4 * 128 * 2
    assert B12xPagedAttentionBackend.supported_kv_cache_layouts() == (
        KVCacheLayout.LBHNC,
        KVCacheLayout.BLHNC,
    )
    assert B12xPagedAttentionBackend.supports_block_size(128)
    assert not B12xPagedAttentionBackend.supports_block_size(32)


def test_b12x_attention_hybrid_cache_capacity_includes_expansion() -> None:
    assert _max_page_table_width(4096, 128, 4096, False) == 32
    assert _max_page_table_width(4096, 128, 4096, True) == 64


def test_b12x_attention_runtime_page_size_comes_from_cache() -> None:
    key_cache = torch.empty((3, 64, 4, 128), device="meta")
    value_cache = torch.empty_like(key_cache)

    assert _kv_page_size(key_cache, value_cache) == 64
    with pytest.raises(ValueError, match="matching K/V page sizes"):
        _kv_page_size(key_cache, torch.empty((3, 128, 4, 128), device="meta"))


def test_b12x_attention_lazily_prepares_decode_bucket(monkeypatch) -> None:
    impl = object.__new__(B12xPagedAttentionImpl)
    plan = SimpleNamespace(layout=SimpleNamespace(nbytes=96))
    created: list[tuple[int, int]] = []

    def create_plan(page_size: int, batch_size: int) -> SimpleNamespace:
        created.append((page_size, batch_size))
        return plan

    impl._decode_plans = {}
    impl._create_decode_plan = create_plan
    impl._scratch_nbytes = 128
    impl._extend_plans = {}
    impl._verify_q_per_req = 0
    metadata = SimpleNamespace(max_query_len=1)
    monkeypatch.setattr(b12x, "_capture_alloc_forbidden", lambda: False)

    assert impl._select_plan(metadata, 7, 7, 7, 64) is plan
    assert impl._select_plan(metadata, 7, 7, 7, 64) is plan
    assert created == [(64, 7)]


def test_b12x_attention_fp8_descales_follow_request_batch() -> None:
    impl = object.__new__(B12xPagedAttentionImpl)
    impl.kv_cache_dtype = "fp8_e4m3"
    layer = SimpleNamespace(
        _k_scale=torch.tensor(2.0),
        _v_scale=torch.tensor([3.0, 4.0, 5.0]),
    )

    k_descale, v_descale = impl._prepare_fp8_descales(
        layer, num_reqs=2, device=torch.device("cpu")
    )

    torch.testing.assert_close(k_descale, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(v_descale, torch.tensor([3.0, 4.0]))
    assert k_descale.stride() == (0,)
    assert v_descale.stride() == (1,)


def test_b12x_attention_sinks_refresh_in_place_after_reload() -> None:
    impl = object.__new__(B12xPagedAttentionImpl)
    source = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    impl._sinks_source = source
    impl.sinks = None

    impl.process_weights_after_loading(torch.bfloat16)
    assert impl.sinks is not None
    sinks_ptr = impl.sinks.data_ptr()
    source.copy_(torch.tensor([3.0, 4.0], dtype=torch.bfloat16))
    impl.process_weights_after_loading(torch.bfloat16)

    assert impl.sinks.data_ptr() == sinks_ptr
    torch.testing.assert_close(impl.sinks, source.float())


@pytest.mark.parametrize(
    "batch_spec_name",
    ["small_decode", "small_prefill", "mixed_small", "medium_decode"],
)
@pytest.mark.parametrize(
    ("kv_cache_dtype", "model_dtype"),
    [
        ("auto", None),
        ("bfloat16", torch.bfloat16),
        ("fp8_e4m3", torch.bfloat16),
    ],
)
@pytest.mark.parametrize("block_size", [64, 128])
def test_b12x_causal_backend_correctness(
    default_vllm_config,
    workspace_init,
    batch_spec_name: str,
    kv_cache_dtype: str,
    model_dtype: torch.dtype | None,
    block_size: int,
) -> None:
    """b12x causal paged attention matches the shared SDPA reference."""
    _require_b12x_paged_attention()
    batch_spec = BATCH_SPECS[batch_spec_name]

    _test_backend_correctness(
        batch_spec,
        "Qwen/Qwen3-0.6B",
        [AttentionBackendEnum.B12X],
        _causal_mask,
        block_size=block_size,
        kv_cache_dtype=kv_cache_dtype,
        model_dtype=model_dtype,
        max_num_seqs=batch_spec.batch_size,
        max_num_batched_tokens=max(sum(batch_spec.query_lens), 64),
    )


@pytest.mark.parametrize(
    "batch_spec",
    [
        pytest.param(BatchSpec(seq_lens=[2080, 2200], query_lens=[1, 1]), id="decode"),
        pytest.param(BatchSpec(seq_lens=[2080, 2200], query_lens=[8, 8]), id="prefill"),
    ],
)
def test_b12x_causal_sliding_window(
    default_vllm_config,
    workspace_init,
    batch_spec: BatchSpec,
) -> None:
    """b12x causal sliding-window attention matches the shared reference."""
    _require_b12x_paged_attention()

    model = "microsoft/Phi-tiny-MoE-instruct"
    sliding_window = ModelConfig(
        model=model, max_model_len=max(batch_spec.seq_lens)
    ).get_sliding_window()
    assert sliding_window is not None
    mask = partial(_causal_sliding_window_mask, sliding_window=sliding_window)

    _test_backend_correctness(
        batch_spec,
        model,
        [AttentionBackendEnum.B12X],
        mask,
        block_size=64,
        atol=3e-2,
        rtol=3e-2,
        max_num_seqs=batch_spec.batch_size,
        max_num_batched_tokens=max(sum(batch_spec.query_lens), 64),
    )


def test_b12x_attention_sinks(
    default_vllm_config,
    workspace_init,
) -> None:
    """b12x attention sinks match the explicit sink reference."""
    _require_b12x_paged_attention()
    batch_spec = BATCH_SPECS["small_prefill"]

    _test_backend_correctness(
        batch_spec,
        "Qwen/Qwen3-0.6B",
        [AttentionBackendEnum.B12X],
        _causal_mask,
        block_size=64,
        atol=3e-2,
        rtol=3e-2,
        use_sinks=True,
        max_num_seqs=batch_spec.batch_size,
        max_num_batched_tokens=max(sum(batch_spec.query_lens), 64),
    )


def test_b12x_decode_cuda_graph_replay(
    default_vllm_config,
    workspace_init,
) -> None:
    """b12x decode output remains correct after CUDA graph replay."""
    _require_b12x_paged_attention()
    batch_spec = BATCH_SPECS["small_decode"]

    _test_backend_correctness(
        batch_spec,
        "Qwen/Qwen3-0.6B",
        [AttentionBackendEnum.B12X],
        _causal_mask,
        block_size=64,
        use_cuda_graph=True,
        max_num_seqs=batch_spec.batch_size,
        max_num_batched_tokens=max(sum(batch_spec.query_lens), 64),
    )


def test_b12x_speculative_verification_uses_cuda_graph_plan(
    default_vllm_config,
    workspace_init,
) -> None:
    """Exercise the verifier plan rather than the general extend plan."""
    _require_b12x_paged_attention()
    batch_spec = BatchSpec(seq_lens=[32, 40], query_lens=[4, 4])

    _test_backend_correctness(
        batch_spec,
        "Qwen/Qwen3-0.6B",
        [AttentionBackendEnum.B12X],
        _causal_mask,
        block_size=128,
        num_speculative_tokens=3,
        use_cuda_graph=True,
        max_num_seqs=batch_spec.batch_size,
        max_num_batched_tokens=max(sum(batch_spec.query_lens), 64),
    )
