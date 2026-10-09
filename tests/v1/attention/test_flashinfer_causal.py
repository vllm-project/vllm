# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for mixed-causal native FlashInfer planning."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.attention.backends.flashinfer_causal import (
    causal_group_indices,
    symmetric_window_mask,
)


@pytest.mark.parametrize(
    "flags", [[True, False, True], [False, True, False], [True] * 3, [False] * 3]
)
def test_partition_preserves_query_and_page_order(flags):
    # KV indptr intentionally starts at a nonzero offset, as sliced metadata
    # does; page gathering must index the full batch's page list.
    query_indptr = torch.tensor([0, 1, 4, 6], dtype=torch.int32)
    kv_indptr = torch.tensor([5, 7, 10, 11], dtype=torch.int32)
    groups = causal_group_indices(torch.tensor(flags), query_indptr, kv_indptr)
    all_tokens = []
    all_pages = []
    for mode, requests, tokens, pages in groups:
        assert requests.tolist() == [i for i, flag in enumerate(flags) if flag == mode]
        expected_tokens = []
        expected_pages = []
        for i in requests.tolist():
            expected_tokens.extend(range(query_indptr[i], query_indptr[i + 1]))
            expected_pages.extend(range(kv_indptr[i], kv_indptr[i + 1]))
        assert tokens.tolist() == expected_tokens
        assert pages.tolist() == expected_pages
        all_tokens.extend(tokens.tolist())
        all_pages.extend(pages.tolist())
    assert sorted(all_tokens) == list(range(6))
    assert sorted(all_pages) == list(range(5, 11))


@pytest.mark.parametrize("window_left", [0, 1, 2, 10])
def test_symmetric_window_mask_matches_absolute_positions(window_left):
    query_lens = [4, 2, 1]
    kv_lens = [4, 7, 9]
    actual = symmetric_window_mask(
        torch.tensor(query_lens),
        torch.tensor(kv_lens),
        window_left,
        torch.device("cpu"),
    )
    expected = [
        abs(query_pos - key_pos) <= window_left
        for query_len, kv_len in zip(query_lens, kv_lens)
        for query_pos in range(kv_len - query_len, kv_len)
        for key_pos in range(kv_len)
    ]
    assert actual.dtype == torch.bool
    assert actual.tolist() == expected


def test_bidirectional_window_excludes_distant_future():
    mask = symmetric_window_mask(
        torch.tensor([5]), torch.tensor([5]), 1, torch.device("cpu")
    ).reshape(5, 5)
    assert mask[0].tolist() == [True, True, False, False, False]
    assert mask[2].tolist() == [False, True, True, True, False]


@pytest.mark.parametrize("flags", [[True], [[True, False]], [1, 0]])
def test_invalid_causal_flags_rejected(flags):
    with pytest.raises(ValueError):
        causal_group_indices(
            torch.tensor(flags), torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])
        )


def test_group_plans_use_disjoint_pages_and_symmetric_window():
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    wrappers = {True: Mock(), False: Mock()}
    builder = SimpleNamespace(
        _get_prefill_wrapper=lambda causal: wrappers[causal],
        window_left=1,
        device=torch.device("cpu"),
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=512,
        vo_split=2,
        page_size=4,
        sm_scale=512**-0.5,
        logits_soft_cap=0.0,
        q_data_type_prefill=torch.bfloat16,
        kv_cache_dtype=torch.uint8,
        prefill_fixed_split_size=-1,
        disable_split_kv=True,
    )
    groups = FlashInferMetadataBuilder._plan_causal_groups(
        builder,
        torch.tensor([True, False, True]),
        torch.tensor([0, 1, 4, 6], dtype=torch.int32),
        torch.tensor([2, 4, 6, 8], dtype=torch.int32),
        torch.tensor([1, 2, 3], dtype=torch.int32),
        torch.tensor([90, 91, 10, 11, 20, 21, 30, 31], dtype=torch.int32),
        torch.tensor([5, 6, 7], dtype=torch.int32),
        torch.bfloat16,
    )
    causal_plan = wrappers[True].plan.call_args.kwargs
    noncausal_plan = wrappers[False].plan.call_args.kwargs
    assert causal_plan["qo_indptr"].tolist() == [0, 1, 3]
    assert causal_plan["paged_kv_indices"].tolist() == [10, 11, 30, 31]
    assert causal_plan["paged_kv_last_page_len"].tolist() == [1, 3]
    assert causal_plan["causal"] is True
    assert causal_plan["window_left"] == 1
    assert causal_plan["custom_mask"] is None
    assert noncausal_plan["qo_indptr"].tolist() == [0, 3]
    assert noncausal_plan["paged_kv_indices"].tolist() == [20, 21]
    assert noncausal_plan["causal"] is False
    assert noncausal_plan["window_left"] == -1
    assert noncausal_plan["custom_mask"].reshape(3, 6)[0].tolist() == [
        False,
        False,
        True,
        True,
        True,
        False,
    ]
    for plan in (causal_plan, noncausal_plan):
        assert plan["head_dim_qk"] == 512
        assert plan["head_dim_vo"] == 256
    assert groups[0].token_indices.tolist() == [0, 4, 5]
    assert groups[1].token_indices.tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    "unsupported",
    ["trtllm", "dcp", "sinks", "speculative", "cascade", "full_graph"],
)
def test_mixed_causal_rejects_unsupported_dispatch(unsupported):
    pytest.importorskip("flashinfer")
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    builder = SimpleNamespace(
        nvfp4_trtllm=unsupported == "trtllm",
        use_dcp=unsupported == "dcp",
        has_sinks=unsupported == "sinks",
        reorder_batch_threshold=2 if unsupported == "speculative" else 1,
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(
                has_full_cudagraphs=lambda: unsupported == "full_graph"
            )
        ),
    )
    metadata = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=3,
        causal=torch.tensor([True, False]),
    )
    with pytest.raises(NotImplementedError, match="Mixed causal FlashInfer"):
        FlashInferMetadataBuilder.build(
            builder, 16 if unsupported == "cascade" else 0, metadata
        )
