# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import unittest.mock

import pytest

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm.config import CUDAGraphMode, SpeculativeConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import PerLayerParameters

try:
    import flashinfer
except ImportError:
    if current_platform.is_rocm():
        pytest.skip(
            "flashinfer is not supported for vLLM on ROCm.", allow_module_level=True
        )

import torch

NUM_HEADS = [(32, 8), (6, 1)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 32]
DTYPES = [torch.bfloat16]
NUM_BLOCKS = 32768  # Large enough to test overflow in index calculation.
SOFT_CAPS = [None, 30.0]
SLIDING_WINDOWS = [None, 64]
SPEC_DECODE_BLOCK_SIZE = 16
SPEC_DECODE_MODEL = "Qwen/Qwen2.5-0.5B"
NUM_SPEC_TOKENS = 4
DEVICE = torch.device("cuda:0")


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def _make_paged_kv_metadata(
    kv_lens: list[int],
    block_size: int,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build paged-KV metadata tensors for fast_plan_decode tests.

    Returns:
        kv_indptr          – CPU int32, shape [num_seqs + 1]
        kv_indices         – CUDA int32, shape [total_blocks]
        kv_last_page_lens  – CPU int32, shape [num_seqs]
        block_tables       – CUDA int32, shape [num_seqs, max_blocks_per_seq]
    """
    num_seqs = len(kv_lens)
    max_blocks = (max(kv_lens) + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_blocks), dtype=torch.int32, device="cuda"
    )

    indptr_list = [0]
    indices_list: list[int] = []
    last_lens_list: list[int] = []
    for i, seq_len in enumerate(kv_lens):
        n = (seq_len + block_size - 1) // block_size
        indices_list.extend(block_tables[i, :n].cpu().tolist())
        indptr_list.append(indptr_list[-1] + n)
        last_lens_list.append(seq_len % block_size or block_size)

    return (
        torch.tensor(indptr_list, dtype=torch.int32, device="cpu"),
        torch.tensor(indices_list, dtype=torch.int32, device="cuda"),
        torch.tensor(last_lens_list, dtype=torch.int32, device="cpu"),
        block_tables,
    )


def _make_cg_decode_wrapper(
    num_seqs: int,
    kv_indices_buffer: torch.Tensor,
    workspace_buffer: torch.Tensor,
    use_tensor_cores: bool = True,
) -> "flashinfer.BatchDecodeWithPagedKVCacheWrapper":
    """Create a cudagraph-enabled BatchDecodeWithPagedKVCacheWrapper.

    *kv_indices_buffer* is shared with the caller so that fast_plan_decode
    can avoid the device-to-device index copy on subsequent (cudagraph) calls.
    """
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        "NHD",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=torch.zeros(
            num_seqs + 1, dtype=torch.int32, device="cuda"
        ),
        paged_kv_indices_buffer=kv_indices_buffer,
        paged_kv_last_page_len_buffer=torch.zeros(
            num_seqs, dtype=torch.int32, device="cuda"
        ),
        use_tensor_cores=use_tensor_cores,
    )


def _mock_flashinfer_layer_params(vllm_config, layer_names, impl_cls):
    head_size = vllm_config.model_config.get_head_size()
    return {
        name: PerLayerParameters(
            window_left=-1,
            logits_soft_cap=0.0,
            sm_scale=1.0 / (head_size**0.5),
        )
        for name in layer_names
    }


class _RecordingPrefillWrapper:
    def __init__(self):
        self.plan_kwargs: dict | None = None
        self.requested_batch_size: int | None = None
        self.requested_use_cudagraph: bool | None = None
        self._window_left = -1
        self._logits_soft_cap = 0.0
        self._sm_scale = 0.0
        self._causal = False

    def plan(self, **kwargs):
        self.plan_kwargs = {
            k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        self._window_left = kwargs.get("window_left", self._window_left)
        self._logits_soft_cap = kwargs.get("logits_soft_cap", self._logits_soft_cap)
        self._sm_scale = kwargs.get("sm_scale", self._sm_scale)
        self._causal = kwargs.get("causal", self._causal)


def _make_flashinfer_spec_decode_builder():
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    vllm_config = create_vllm_config(
        model_name=SPEC_DECODE_MODEL,
        max_model_len=512,
        block_size=SPEC_DECODE_BLOCK_SIZE,
        num_gpu_blocks=512,
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=NUM_SPEC_TOKENS,
        prompt_lookup_max=4,
        prompt_lookup_min=2,
    )
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    with (
        set_current_vllm_config(vllm_config),
        unittest.mock.patch(
            "vllm.v1.attention.backends.flashinfer.can_use_trtllm_attention",
            return_value=False,
        ),
        unittest.mock.patch(
            "vllm.v1.attention.backends.flashinfer.get_per_layer_parameters",
            _mock_flashinfer_layer_params,
        ),
    ):
        builder = FlashInferMetadataBuilder(
            kv_cache_spec, ["layer.0"], vllm_config, DEVICE
        )

    assert builder.reorder_batch_threshold == 1 + NUM_SPEC_TOKENS
    return builder


def _build_flashinfer_spec_decode_metadata(
    builder,
    batch_spec: BatchSpec,
    num_actual_tokens_override: int | None = None,
):
    common = create_common_attn_metadata(batch_spec, SPEC_DECODE_BLOCK_SIZE, DEVICE)
    if num_actual_tokens_override is not None:
        common.num_actual_tokens = num_actual_tokens_override
    fake_wrapper = _RecordingPrefillWrapper()

    def _fake_get_spec_decode_prefill_wrapper(batch_size, use_cudagraph):
        fake_wrapper.requested_batch_size = batch_size
        fake_wrapper.requested_use_cudagraph = use_cudagraph
        return fake_wrapper

    with unittest.mock.patch.object(
        builder,
        "_get_spec_decode_prefill_wrapper",
        side_effect=_fake_get_spec_decode_prefill_wrapper,
    ):
        attn_metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)

    return attn_metadata, fake_wrapper


def test_fast_decode_plan_importable() -> None:
    """fast_decode_plan must be importable from flashinfer.decode.

    This is a forward-compatibility smoke test: if FlashInfer reorganises its
    public API the import will fail before any other test does.
    """
    from flashinfer.decode import fast_decode_plan  # noqa: F401

    assert callable(fast_decode_plan)


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode
def test_fast_plan_decode_warmup_uses_full_plan(dtype: torch.dtype) -> None:
    """On the first call fast_plan_decode must route through self.plan() and
    flip vllm_first_call to False on the wrapper object."""
    from unittest.mock import patch

    from vllm.v1.attention.backends.flashinfer import fast_plan_decode

    torch.set_default_device("cuda")
    set_random_seed(0)

    kv_lens = [128, 64]
    block_size = 16
    num_seqs = len(kv_lens)
    num_query_heads, num_kv_heads = 8, 2
    head_size = 128

    kv_indptr, kv_indices, kv_last_page_lens, _ = _make_paged_kv_metadata(
        kv_lens, block_size, NUM_BLOCKS
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = _make_cg_decode_wrapper(num_seqs, kv_indices.clone(), workspace)

    assert getattr(wrapper, "vllm_first_call", True) is True

    with patch.object(wrapper, "plan", wraps=wrapper.plan) as mock_plan:
        fast_plan_decode(
            wrapper,
            indptr_cpu=kv_indptr,
            indices=kv_indices,
            last_page_len_cpu=kv_last_page_lens,
            num_qo_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_size,
            page_size=block_size,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        mock_plan.assert_called_once()

    assert wrapper.vllm_first_call is False, (
        "vllm_first_call should be False after the first fast_plan_decode call"
    )


@pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode
def test_fast_plan_decode_matches_full_plan(
    kv_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
) -> None:
    """fast_plan_decode's cudagraph path (delegating to FlashInfer's
    fast_decode_plan) must produce attention output numerically identical to
    a standard plan() call.

    Both the warmup call (self.plan) and the subsequent fast call
    (fast_decode_plan) are verified against the same reference.
    """
    from vllm.v1.attention.backends.flashinfer import fast_plan_decode

    torch.set_default_device("cuda")
    set_random_seed(0)
    num_seqs = len(kv_lens)
    num_query_heads, num_kv_heads = num_heads

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
    key_value_cache = torch.randn(
        NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )

    kv_indptr, kv_indices, kv_last_page_lens, _ = _make_paged_kv_metadata(
        kv_lens, block_size, NUM_BLOCKS
    )

    # Reference output via the standard plan()
    workspace_ref = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    ref_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_ref, "NHD", use_tensor_cores=True
    )
    ref_wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        "NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    ref_output = ref_wrapper.run(query, key_value_cache)

    # CUDAGraph wrapper exercised through fast_plan_decode
    kv_indices_buf = kv_indices.clone()
    workspace_cg = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    cg_wrapper = _make_cg_decode_wrapper(num_seqs, kv_indices_buf, workspace_cg)

    plan_kwargs: dict = dict(
        indptr_cpu=kv_indptr,
        indices=kv_indices_buf,
        last_page_len_cpu=kv_last_page_lens,
        num_qo_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        page_size=block_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # First call – warmup path (routes through self.plan)
    fast_plan_decode(cg_wrapper, **plan_kwargs)
    warmup_output = cg_wrapper.run(query, key_value_cache)
    torch.testing.assert_close(warmup_output, ref_output, atol=1e-2, rtol=1e-2)

    # Second call – fast path (routes through fast_decode_plan from FlashInfer)
    fast_plan_decode(cg_wrapper, **plan_kwargs)
    fast_output = cg_wrapper.run(query, key_value_cache)
    torch.testing.assert_close(fast_output, ref_output, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@torch.inference_mode
def test_flashinfer_decode_with_paged_kv(
    kv_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    sliding_window: int | None,
) -> None:
    torch.set_default_device("cuda")
    set_random_seed(0)
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)

    key_value_cache = torch.randn(
        NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=True
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        "NONE",
        window_left=sliding_window - 1 if sliding_window is not None else -1,
        q_data_type=dtype,
        kv_data_type=dtype,
        logits_soft_cap=soft_cap,
    )

    output = wrapper.run(query, key_value_cache)

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * num_seqs,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap,
        sliding_window=sliding_window,
    )
    (
        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOWS)
@torch.inference_mode
def test_flashinfer_prefill_with_paged_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    sliding_window: int | None,
) -> None:
    torch.set_default_device("cuda")
    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_value_cache = torch.randn(
        NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    # Normalize the scale of the key and value caches to mitigate
    # numerical instability.
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    qo_indptr = [0]
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)
        qo_indptr.append(qo_indptr[-1] + query_lens[i])

    qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32)
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        window_left=sliding_window - 1 if sliding_window is not None else -1,
        q_data_type=dtype,
        kv_data_type=dtype,
        logits_soft_cap=soft_cap,
    )

    output = wrapper.run(
        query,
        key_value_cache,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap,
        sliding_window=sliding_window,
    )
    (
        torch.testing.assert_close(output, ref_output, atol=5e-2, rtol=1e-2),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize("seq_lens", [[(1, 132), (5, 18)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
def test_flashinfer_prefill_with_paged_fp8_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
) -> None:
    pytest.skip("TODO: fix the accuracy issue")
    torch.set_default_device("cuda")
    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    kv_cache_dtype = torch.float8_e4m3fn

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    NUM_BLOCKS_FP8 = 2048
    key_value_cache = torch.randn(
        NUM_BLOCKS_FP8, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )
    key_cache, value_cache = torch.chunk(key_value_cache, 2, dim=1)
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    k_scale = key_cache.amax().item() / 448.0
    v_scale = value_cache.amax().item() / 448.0

    kv_cache_fp8 = torch.cat([key_cache / k_scale, value_cache / v_scale], dim=1).to(
        kv_cache_dtype
    )

    assert kv_cache_fp8.shape == key_value_cache.shape
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS_FP8, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    qo_indptr = [0]
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)
        qo_indptr.append(qo_indptr[-1] + query_lens[i])

    qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32)
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        q_data_type=dtype,
        kv_data_type=kv_cache_dtype,
        logits_soft_cap=soft_cap,
    )

    output = wrapper.run(query, kv_cache_fp8, k_scale=k_scale, v_scale=v_scale)

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache.squeeze(1),
        value_cache=value_cache.squeeze(1),
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap,
    )
    del query
    del block_tables
    # verify prefill fp8
    (
        torch.testing.assert_close(output, ref_output, atol=5e-2, rtol=1e-2),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.skip(reason="TODO: fix the accuracy issue")
@torch.inference_mode
def test_flashinfer_decode_with_paged_fp8_kv(
    kv_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
) -> None:
    # test doesn't work for num_heads = (16,16)
    torch.set_default_device("cuda")
    set_random_seed(0)
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    use_tensor_cores = True
    kv_cache_dtype = torch.float8_e4m3fn

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
    NUM_BLOCKS_FP8 = 2048
    key_value_cache = torch.randn(
        NUM_BLOCKS_FP8, 2, block_size, num_kv_heads, head_size, dtype=dtype
    )
    key_cache, value_cache = torch.chunk(key_value_cache, 2, dim=1)
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    k_scale = key_cache.amax().item() / 448.0
    v_scale = value_cache.amax().item() / 448.0

    key_cache_fp8 = (key_cache / k_scale).to(kv_cache_dtype)
    value_cache_fp8 = (value_cache / v_scale).to(kv_cache_dtype)
    assert key_cache_fp8.shape[1] == 1 and value_cache_fp8.shape[1] == 1
    kv_cache_fp8 = torch.cat([key_cache_fp8, value_cache_fp8], dim=1)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS_FP8, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=use_tensor_cores
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        "NONE",
        q_data_type=dtype,
        kv_data_type=kv_cache_dtype,
        logits_soft_cap=soft_cap,
    )
    output = wrapper.run(query, kv_cache_fp8, k_scale=k_scale, v_scale=v_scale)
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * num_seqs,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap,
    )
    # Temporary fix: Increasing the tolerance. Seems like a flashinfer issue
    (
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode
def test_flashinfer_prefill_cudagraph_zero_rows_padding(
    dtype: torch.dtype,
) -> None:
    """Pin the zero-row CG padding contract used by the native spec-decode path.

    Trailing request slots carry duplicate qo_indptr / paged_kv_indptr entries
    and last_page_len == 0; query/output are sized to qo_indptr[-1]. Real-row
    output must be bit-identical to an unpadded non-CG reference.
    """
    torch.set_default_device("cuda")
    set_random_seed(0)

    real_bs = 3
    padded_bs = 8
    q_len = 5
    num_qo_heads, num_kv_heads = 32, 8
    head_dim = 128
    page_size = 16
    pages_per_req = 4

    real_pages = real_bs * pages_per_req
    num_total_pages = real_pages + 16

    kv_cache = torch.randn(
        num_total_pages, 2, page_size, num_kv_heads, head_dim, dtype=dtype
    )

    # Persistent buffers are sized for padded request count.
    qo_indptr_buf = torch.zeros(padded_bs + 1, dtype=torch.int32)
    paged_kv_indptr_buf = torch.zeros(padded_bs + 1, dtype=torch.int32)
    paged_kv_indices_buf = torch.zeros(padded_bs * pages_per_req, dtype=torch.int32)
    paged_kv_last_page_len_buf = torch.zeros(padded_bs, dtype=torch.int32)

    qo_indptr_cpu = torch.tensor(
        [0, 5, 10, 15, 15, 15, 15, 15, 15], dtype=torch.int32
    )
    paged_kv_indptr_cpu = torch.tensor(
        [0, 4, 8, 12, 12, 12, 12, 12, 12], dtype=torch.int32
    )
    paged_kv_last_page_len_cpu = torch.tensor(
        [16, 16, 16, 0, 0, 0, 0, 0], dtype=torch.int32
    )
    paged_kv_indices_cpu = torch.arange(real_bs * pages_per_req, dtype=torch.int32)

    # Query is sized to real token count.
    real_query = torch.randn(real_bs * q_len, num_qo_heads, head_dim, dtype=dtype)

    workspace_cg = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    cg_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_cg,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=qo_indptr_buf,
        paged_kv_indptr_buf=paged_kv_indptr_buf,
        paged_kv_indices_buf=paged_kv_indices_buf,
        paged_kv_last_page_len_buf=paged_kv_last_page_len_buf,
    )
    cg_wrapper.plan(
        qo_indptr=qo_indptr_cpu,
        paged_kv_indptr=paged_kv_indptr_cpu,
        paged_kv_indices=paged_kv_indices_cpu,
        paged_kv_last_page_len=paged_kv_last_page_len_cpu,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    cg_output = cg_wrapper.run(real_query, kv_cache)

    workspace_ref = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    ref_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_ref, "NHD")
    ref_wrapper.plan(
        qo_indptr=qo_indptr_cpu[: real_bs + 1].clone(),
        paged_kv_indptr=paged_kv_indptr_cpu[: real_bs + 1].clone(),
        paged_kv_indices=paged_kv_indices_cpu,
        paged_kv_last_page_len=paged_kv_last_page_len_cpu[:real_bs].clone(),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    ref_output = ref_wrapper.run(real_query, kv_cache)

    torch.testing.assert_close(cg_output, ref_output, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "query_lens, seq_lens, num_actual_tokens_override, "
    "expected_qo_indptr, expected_last_page_len",
    [
        ([5, 5, 0], [64, 72, 0], None, [0, 5, 10, 10], [16, 8, 0]),
        ([5, 5, 5], [64, 72, 80], None, [0, 5, 10, 15], [16, 8, 16]),
        ([5, 5, 0], [64, 72, 0], 40, [0, 5, 10, 10], [16, 8, 0]),
    ],
    ids=[
        "zero_row_padding",
        "no_padding",
        "padded_num_actual_tokens",
    ],
)
def test_spec_decode_routes_to_fispecdecode(
    query_lens,
    seq_lens,
    num_actual_tokens_override,
    expected_qo_indptr,
    expected_last_page_len,
):
    from vllm.v1.attention.backends.flashinfer import FISpecDecode

    builder = _make_flashinfer_spec_decode_builder()
    attn_metadata, fake_wrapper = _build_flashinfer_spec_decode_metadata(
        builder,
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        num_actual_tokens_override=num_actual_tokens_override,
    )

    assert isinstance(attn_metadata.decode, FISpecDecode), (
        f"expected FISpecDecode for query_lens={query_lens}, "
        f"got {type(attn_metadata.decode).__name__}"
    )
    assert attn_metadata.decode.wrapper is fake_wrapper

    assert fake_wrapper.requested_batch_size == len(query_lens)
    assert fake_wrapper.requested_use_cudagraph is True
    assert fake_wrapper.plan_kwargs is not None
    assert fake_wrapper.plan_kwargs["causal"] is True

    qo = fake_wrapper.plan_kwargs["qo_indptr"]
    assert qo.tolist() == expected_qo_indptr

    last_page_len = fake_wrapper.plan_kwargs["paged_kv_last_page_len"]
    assert last_page_len.tolist() == expected_last_page_len

    paged_kv_indptr = fake_wrapper.plan_kwargs["paged_kv_indptr"]
    if query_lens[-1] == 0:
        assert paged_kv_indptr[-1].item() == paged_kv_indptr[-2].item()

    # Prevent query[:40] with qo_indptr[-1] == 10.
    assert attn_metadata.num_decode_tokens == expected_qo_indptr[-1]


@pytest.mark.parametrize(
    "dcp_size, expected",
    [
        (1, "UNIFORM_BATCH"),
        (2, "UNIFORM_SINGLE_TOKEN_DECODE"),
    ],
)
def test_get_cudagraph_support_dcp_downgrade(dcp_size, expected):
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    vllm_config = create_vllm_config(
        model_name=SPEC_DECODE_MODEL,
        tensor_parallel_size=dcp_size,
        max_model_len=512,
        block_size=SPEC_DECODE_BLOCK_SIZE,
    )
    vllm_config.parallel_config.decode_context_parallel_size = dcp_size
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    support = FlashInferMetadataBuilder.get_cudagraph_support(
        vllm_config, kv_cache_spec
    )
    assert support is getattr(AttentionCGSupport, expected)
