# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer plans from the CPU bounds on seq_lens instead of copying seq_lens
from the device."""

import itertools
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
from vllm.v1.attention.backends.utils import PerLayerParameters

NUM_QO_HEADS, NUM_KV_HEADS, HEAD_SIZE = 8, 2, 128
NUM_SPEC = 3
BLOCK_SIZE = 16
# Six decodes, two draft verifications and a prefill chunk. The upper bounds
# count up to NUM_SPEC drafts that may still be rejected; rows 3 and 5 get a
# page too many. Lower bounds are NUM_SPEC below, except on the prefill.
QUERY_LENS = [1] * 6 + [1 + NUM_SPEC] * 2 + [17]
EXACT = [1328, 18, 463, 64, 65, 127, 300, 1000, 2000]
UPPER = [1328, 21, 466, 65, 67, 129, 303, 1003, 2000]


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
    case: str, copies: bool
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
    cam = create_common_attn_metadata(
        BatchSpec(seq_lens=upper, query_lens=QUERY_LENS),
        BLOCK_SIZE,
        torch.device("cuda"),
        max_block_idx=vllm_config.cache_config.num_gpu_blocks,
    )
    cam = cam.replace(
        seq_lens=torch.tensor(exact, dtype=torch.int32, device="cuda"),
        seq_lens_cpu_lower_bound=(
            None if case == "no_lower_bound" else torch.tensor(lower, dtype=torch.int32)
        ),
    )

    def build():
        with set_current_vllm_config(vllm_config):
            return builder.build(common_prefix_len=0, common_attn_metadata=cam)

    # Warm up the kernels, both wrappers and their pinned buffer rings.
    for _ in range(6):
        build()
    if case.startswith("not_fa2"):
        builder._prefill_runs_fa2 = builder._decode_runs_fa2 = False
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


def test_pinned_plan_workspace_is_not_reused_while_its_copy_is_pending() -> None:
    class Wrapper:
        pass

    workspaces = _PinnedPlanWorkspaces()
    wrapper = Wrapper()
    own = torch.zeros(1024, dtype=torch.uint8, pin_memory=True)
    buf, event = workspaces.acquire(wrapper, own)
    assert buf is own
    # Queue enough GPU work that the event recorded after this plan is pending.
    x = torch.randn(4096, 4096, device="cuda")
    for _ in range(50):
        x = x @ x
        x = x / x.norm()
    event.record()
    if event.query():
        pytest.skip("the GPU finished the queued work before the next plan")

    second, second_event = workspaces.acquire(wrapper, own)
    assert second is not own
    assert second.is_pinned() and not second.any()
    second_event.record()
    torch.accelerator.synchronize()
    # Once the copies have run, both buffers are reused in turn.
    assert all(workspaces.acquire(wrapper, own)[0] is b for b in (own, second, own))
