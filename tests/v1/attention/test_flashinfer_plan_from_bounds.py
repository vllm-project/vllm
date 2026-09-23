# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer plans from the CPU bounds on seq_lens instead of copying seq_lens
from the device."""

import itertools
import types
import unittest.mock
from typing import Any

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FlashInfer backend requires a CUDA platform.", allow_module_level=True)

import flashinfer
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.flashinfer import (
    FlashInferMetadataBuilder,
    _PinnedPlanWorkspaces,
)
from vllm.v1.attention.backends.utils import CommonAttentionMetadata, PerLayerParameters

NUM_QO_HEADS, NUM_KV_HEADS, HEAD_SIZE = 8, 2, 128
NUM_SPEC = 3
BLOCK_SIZE = 16
# Six decodes, two draft verifications and a prefill chunk. The upper bounds
# count up to NUM_SPEC drafts that may still be rejected; rows 3 and 5 get a
# page too many. Lower bounds are NUM_SPEC below, except on the prefill.
QUERY_LENS = [1] * 6 + [1 + NUM_SPEC] * 2 + [17]
EXACT = [1328, 18, 463, 64, 65, 127, 300, 1000, 2000]
UPPER = [1328, 21, 466, 65, 67, 129, 303, 1003, 2000]
# GPU clock cycles to spin so that an event recorded afterwards is still pending.
PENDING_CYCLES = 200_000_000


@pytest.mark.parametrize("split", ["default", "disabled", "one_page_chunks"])
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("kind", ["prefill", "decode"])
def test_fa2_plan_from_upper_bound_matches_exact_plan(
    kind: str, block_size: int, split: str
) -> None:
    """fa2 planned from the upper bound and then given the exact last-page
    lengths attends like fa2 planned from the exact lengths."""
    torch.manual_seed(0)
    exact = [1328, 18, 463, 64, 65, 127]
    if kind == "prefill":
        # Several query tokens: the upper bound stays on the same page.
        qo_lens = [1, 4, 1, 3, 2, 1]
        upper = [n + min(NUM_SPEC, -n % block_size) for n in exact]
    else:
        # One query token: the upper bound may add one or two pages.
        qo_lens = [1] * len(exact)
        upper = [n + s for n, s in zip(exact, [0, 3, 6, 1, 5, 2 * block_size])]
    extra: dict[str, Any] = {}
    if split == "disabled":
        extra["disable_split_kv"] = True
    elif split == "one_page_chunks":
        extra["fixed_split_size"] = block_size
    pages = [cdiv(n, block_size) for n in upper]
    block_ids = torch.randperm(4096, dtype=torch.int32, device="cuda")
    rows = block_ids[: sum(pages)].split(pages)
    kv_cache = torch.randn(
        4096,
        2,
        block_size,
        NUM_KV_HEADS,
        HEAD_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    query = torch.randn(
        sum(qo_lens), NUM_QO_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    qo_indptr = torch.tensor([0, *itertools.accumulate(qo_lens)], dtype=torch.int32)

    def run(lens: list[int]) -> torch.Tensor:
        num_pages = [cdiv(n, block_size) for n in lens]
        indptr = torch.tensor([0, *itertools.accumulate(num_pages)], dtype=torch.int32)
        indices = torch.cat([row[:p] for row, p in zip(rows, num_pages)])
        last_page_len = torch.tensor(
            [n - (p - 1) * block_size for n, p in zip(lens, num_pages)],
            dtype=torch.int32,
        )
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        args = (NUM_QO_HEADS, NUM_KV_HEADS, HEAD_SIZE, block_size)
        dtypes = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        if kind == "prefill":
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace, "NHD", backend="fa2"
            )
            wrapper.plan(
                qo_indptr,
                indptr,
                indices,
                last_page_len,
                *args,
                causal=True,
                **dtypes,
                **extra,
            )
        else:
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace, "NHD", use_tensor_cores=True, backend="fa2"
            )
            wrapper.plan(
                indptr,
                indices,
                last_page_len,
                *args,
                pos_encoding_mode="NONE",
                **dtypes,
                **extra,
            )
        # What the builder writes after planning from the upper bound.
        exact_last_page_len = [n - (p - 1) * block_size for n, p in zip(exact, pages)]
        if lens is upper:
            wrapper._paged_kv_last_page_len_buf[: len(exact)].copy_(
                torch.tensor(exact_last_page_len, dtype=torch.int32)
            )
        return wrapper.run(query, kv_cache)

    output, expected = run(upper), run(exact)
    if split == "disabled":
        assert torch.equal(output, expected)
    else:
        # Surplus pages can form extra empty chunks, changing the summation order.
        torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)


class _Fa2PrefillWrapper(flashinfer.BatchPrefillWithPagedKVCacheWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**kwargs, "backend": "fa2"})


class _Fa2DecodeWrapper(flashinfer.BatchDecodeWithPagedKVCacheWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**kwargs, "backend": "fa2"})


@pytest.fixture(autouse=True)
def _fa2_wrappers(monkeypatch):
    """GPUs differ in the kernels FlashInfer picks; the builder tests need fa2."""
    module = "vllm.v1.attention.backends.flashinfer."
    monkeypatch.setattr(
        module + "BatchPrefillWithPagedKVCacheWrapper", _Fa2PrefillWrapper
    )
    monkeypatch.setattr(
        module + "BatchDecodeWithPagedKVCacheWrapper", _Fa2DecodeWrapper
    )


def _make_builder(vllm_config) -> FlashInferMetadataBuilder:
    # Keep every row on the FlashInfer wrappers, also where TRTLLM is the default.
    vllm_config.attention_config.use_trtllm_attention = False
    head_size = vllm_config.model_config.get_head_size()

    def per_layer_parameters(vllm_config, layer_names, impl_cls):
        params = PerLayerParameters(
            window_left=-1,
            logits_soft_cap=0.0,
            sm_scale=head_size**-0.5,
            has_sinks=False,
        )
        return {name: params for name in layer_names}

    with (
        set_current_vllm_config(vllm_config),
        unittest.mock.patch(
            "vllm.v1.attention.backends.flashinfer.get_per_layer_parameters",
            per_layer_parameters,
        ),
    ):
        return FlashInferMetadataBuilder(
            create_standard_kv_cache_spec(vllm_config),
            ["model.layers.0.self_attn.attn"],
            vllm_config,
            torch.device("cuda"),
        )


@pytest.mark.parametrize(
    "case, copies",
    [
        ("fa2", False),
        ("straddling_verification_row", True),
        ("not_fa2", True),
        ("not_fa2_exact_bounds", False),
        ("no_lower_bound", True),
    ],
)
def test_build_copies_seq_lens_only_when_the_bounds_do_not_suffice(
    case: str, copies: bool, monkeypatch
) -> None:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3-0.6B", block_size=BLOCK_SIZE, max_model_len=4096
    )
    builder = _make_builder(vllm_config)
    exact, upper = list(EXACT), list(UPPER)
    if case == "straddling_verification_row":
        exact[7], upper[7] = 1008, 1009
    lower = [n - NUM_SPEC for n in upper[:-1]] + upper[-1:]
    if case == "not_fa2_exact_bounds":
        upper = lower = exact
    cam = _mixed_metadata(
        exact, upper, None if case == "no_lower_bound" else lower, vllm_config
    )

    def build():
        with set_current_vllm_config(vllm_config):
            return builder.build(common_prefix_len=0, common_attn_metadata=cam)

    # Warm up the kernels, both wrappers and their pinned buffer rings.
    for _ in range(6):
        build()
    if case.startswith("not_fa2"):
        # As if the wrappers ran kernels that take the lengths from the plan.
        monkeypatch.setattr(
            "vllm.v1.attention.backends.flashinfer._reads_kv_lens_from_device",
            lambda wrapper: False,
        )
    torch.accelerator.synchronize()
    # Any read of seq_lens back from the device now raises.
    torch.cuda.set_sync_debug_mode("error")
    try:
        if copies:
            with pytest.raises(RuntimeError, match="synchroniz"):
                build()
            return
        attn_metadata = build()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.accelerator.synchronize()

    exact_gpu = torch.tensor(exact, dtype=torch.int32, device="cuda")
    num_decodes = QUERY_LENS.count(1)
    for wrapper, start, stop in (
        (attn_metadata.decode.wrapper, 0, num_decodes),
        (attn_metadata.prefill.wrapper, num_decodes, len(exact)),
    ):
        indptr = wrapper._paged_kv_indptr_buf[: stop - start + 1]
        last_page_len = wrapper._paged_kv_last_page_len_buf[: stop - start]
        lens = (indptr[1:] - indptr[:-1] - 1) * BLOCK_SIZE + last_page_len
        assert torch.equal(lens, exact_gpu[start:stop])
    if case == "fa2":
        indptr = builder.paged_kv_indptr.gpu[: len(exact) + 1]
        assert torch.any(indptr[1:] - indptr[:-1] > cdiv(exact_gpu, BLOCK_SIZE))


def _mixed_metadata(
    exact: list[int], upper: list[int], lower: list[int] | None, vllm_config
) -> CommonAttentionMetadata:
    """Decodes, verifications and a prefill (QUERY_LENS) with exact seq_lens on
    the device and the CPU bounds."""
    cam = create_common_attn_metadata(
        BatchSpec(seq_lens=upper, query_lens=QUERY_LENS),
        BLOCK_SIZE,
        torch.device("cuda"),
        max_block_idx=vllm_config.cache_config.num_gpu_blocks,
    )
    return cam.replace(
        seq_lens=torch.tensor(exact, dtype=torch.int32, device="cuda"),
        seq_lens_cpu_lower_bound=(
            None if lower is None else torch.tensor(lower, dtype=torch.int32)
        ),
    )


def test_seq_lens_bounds_check_does_not_synchronize() -> None:
    """With VLLM_DEBUG_SEQ_LENS_BOUNDS the builder asserts on the device that
    the exact seq_lens lie within the bounds it planned from, without reading
    anything back."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3-0.6B", block_size=BLOCK_SIZE, max_model_len=4096
    )
    builder = _make_builder(vllm_config)
    builder._check_seq_lens_bounds = True
    lower = [n - NUM_SPEC for n in UPPER[:-1]] + UPPER[-1:]
    cam = _mixed_metadata(list(EXACT), list(UPPER), lower, vllm_config)

    def build():
        with set_current_vllm_config(vllm_config):
            return builder.build(common_prefix_len=0, common_attn_metadata=cam)

    for _ in range(6):
        build()
    torch.accelerator.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        build()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    # The bounds hold: the device-side assertion stays quiet.
    torch.accelerator.synchronize()


def _decode_metadata(
    upper: list[int], lower: list[int], vllm_config
) -> CommonAttentionMetadata:
    cam = create_common_attn_metadata(
        BatchSpec(seq_lens=upper, query_lens=[1] * len(upper)),
        BLOCK_SIZE,
        torch.device("cuda"),
        max_block_idx=vllm_config.cache_config.num_gpu_blocks,
    )
    return cam.replace(seq_lens_cpu_lower_bound=torch.tensor(lower, dtype=torch.int32))


def _fake_wrapper(backend: str = "fa2", **missing: bool) -> types.SimpleNamespace:
    wrapper = types.SimpleNamespace(_backend=backend)
    for name in ("_pin_memory_int_workspace_buffer", "_paged_kv_last_page_len_buf"):
        if not missing.get(name):
            setattr(wrapper, name, torch.empty(0))
    return wrapper


@pytest.mark.parametrize(
    "case, expected",
    [
        ("fa2", "upper"),
        ("wrapper_not_created_yet", None),
        ("backend_not_picked_yet", None),
        ("fa3", None),
        ("fa3_exact_bounds", "exact"),
        ("no_pinned_buffer", None),
        ("no_pinned_buffer_exact_bounds", None),
        ("no_last_page_len_buffer", None),
        ("lower_above_upper", None),
    ],
)
def test_bounds_are_used_only_for_the_wrapper_that_can_take_them(
    case: str, expected: str | None
) -> None:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3-0.6B", block_size=BLOCK_SIZE, max_model_len=4096
    )
    builder = _make_builder(vllm_config)
    upper = [100, 203, 300]
    lower = [n - NUM_SPEC for n in upper]
    if case.endswith("exact_bounds"):
        lower = upper
    elif case == "lower_above_upper":
        lower[1] = upper[1] + 1
    wrapper = {
        "wrapper_not_created_yet": None,
        "backend_not_picked_yet": _fake_wrapper("auto"),
        "fa3": _fake_wrapper("fa3"),
        "fa3_exact_bounds": _fake_wrapper("fa3"),
        "no_pinned_buffer": _fake_wrapper(_pin_memory_int_workspace_buffer=True),
        "no_pinned_buffer_exact_bounds": _fake_wrapper(
            _pin_memory_int_workspace_buffer=True
        ),
        "no_last_page_len_buffer": _fake_wrapper(_paged_kv_last_page_len_buf=True),
    }.get(case, _fake_wrapper())

    from_bounds = builder._seq_lens_cpu_from_bounds(
        _decode_metadata(upper, lower, vllm_config),
        num_decodes=len(upper),
        decode_uses_trtllm=False,
        prefill_uses_trtllm=False,
        decode_wrapper=wrapper,
        prefill_wrapper=None,
    )

    if expected is None:
        assert from_bounds is None
    else:
        seq_lens_cpu, seq_lens_exact = from_bounds
        assert seq_lens_cpu.tolist() == upper
        assert seq_lens_exact == (expected == "exact")


def test_bounds_wait_for_the_first_plan_of_an_auto_wrapper() -> None:
    """A wrapper created with backend="auto" picks its kernels on its first
    plan(). bf16 tensor-core decode picks fa2 on every GPU."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3-0.6B", block_size=BLOCK_SIZE, max_model_len=4096
    )
    builder = _make_builder(vllm_config)
    upper = [100, 203, 300]
    cam = _decode_metadata(upper, [n - NUM_SPEC for n in upper], vllm_config)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        use_tensor_cores=True,
    )

    def from_bounds():
        return builder._seq_lens_cpu_from_bounds(
            cam, len(upper), False, False, decode_wrapper=wrapper, prefill_wrapper=None
        )

    assert from_bounds() is None
    pages = [cdiv(n, BLOCK_SIZE) for n in upper]
    wrapper.plan(
        torch.tensor([0, *itertools.accumulate(pages)], dtype=torch.int32),
        torch.arange(sum(pages), dtype=torch.int32, device="cuda"),
        torch.tensor([n - (p - 1) * BLOCK_SIZE for n, p in zip(upper, pages)]).int(),
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_SIZE,
        BLOCK_SIZE,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    assert from_bounds() is not None


def test_plan_without_a_pinned_buffer_is_called_directly() -> None:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3-0.6B", block_size=BLOCK_SIZE, max_model_len=4096
    )
    builder = _make_builder(vllm_config)
    calls: list[dict[str, int]] = []
    builder._plan(object(), lambda **kwargs: calls.append(kwargs), batch_size=3)
    assert calls == [{"batch_size": 3}]


class _Wrapper:
    pass


def test_pinned_plan_workspace_is_not_reused_while_its_copy_is_pending() -> None:
    workspaces = _PinnedPlanWorkspaces()
    wrapper = _Wrapper()
    own = torch.zeros(1024, dtype=torch.uint8, pin_memory=True)
    buf, event = workspaces.acquire(wrapper, own)
    assert buf is own
    torch.cuda._sleep(PENDING_CYCLES)
    event.record()

    second, second_event = workspaces.acquire(wrapper, own)
    assert second is not own
    assert second.is_pinned() and not second.any()
    second_event.record()
    torch.accelerator.synchronize()
    # Once the copies have run, both buffers are reused in turn.
    assert all(workspaces.acquire(wrapper, own)[0] is b for b in (own, second, own))


def test_pinned_plan_workspaces_add_at_most_max_extra_buffers(monkeypatch) -> None:
    monkeypatch.setattr(_PinnedPlanWorkspaces, "MAX_EXTRA_BUFFERS", 1)
    workspaces = _PinnedPlanWorkspaces()
    first, second = _Wrapper(), _Wrapper()
    own = [torch.zeros(1024, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
    for wrapper, pinned in zip((first, second), own):
        _, event = workspaces.acquire(wrapper, pinned)
        torch.cuda._sleep(PENDING_CYCLES)
        event.record()

    # The first wrapper gets the one extra buffer.
    extra, event = workspaces.acquire(first, own[0])
    assert extra is not own[0]
    event.record()
    # The second waits for its copy instead of adding a buffer.
    buf, event = workspaces.acquire(second, own[1])
    assert buf is own[1]
    event.record()
    torch.accelerator.synchronize()
