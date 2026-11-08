# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills


def apply_split_decodes_and_prefills(
    query_lens: list[int],
    decode_threshold: int,
    require_uniform: bool,
    padded_num_tokens: int | None = None,
    is_prefilling: list[bool] | None = None,
    treat_short_extends_as_decodes: bool = True,
):
    """Helper function to apply split_decodes_and_prefills and return
    the results."""
    device = torch.device("cpu")
    seq_lens = [10 * (i + 1) for i in range(len(query_lens))]
    common_metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=16,
        device=device,
    )

    if padded_num_tokens is not None:
        common_metadata.num_actual_tokens = padded_num_tokens
    if is_prefilling is not None:
        common_metadata.is_prefilling = torch.tensor(is_prefilling)

    return split_decodes_and_prefills(
        common_metadata,
        decode_threshold=decode_threshold,
        require_uniform=require_uniform,
        treat_short_extends_as_decodes=treat_short_extends_as_decodes,
    )


def test_split_decodes_and_prefills_nonuniform_all_ones():
    query_lens = [1, 1, 1]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 1, False)
    )
    assert num_decodes == 3
    assert num_prefills == 0
    assert num_decode_tokens == 3
    assert num_prefill_tokens == 0


def test_split_decodes_and_prefills_nonuniform_all_short_decodes():
    query_lens = [1, 2, 1, 3, 2, 1, 2]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, False)
    )
    assert num_decodes == 7
    assert num_prefills == 0
    assert num_decode_tokens == sum(query_lens)
    assert num_prefill_tokens == 0


def test_split_decodes_and_prefills_nonuniform_all_prefills():
    query_lens = [4, 5, 6, 7]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, False)
    )
    assert num_decodes == 0
    assert num_prefills == 4
    assert num_decode_tokens == 0
    assert num_prefill_tokens == sum(query_lens)


def test_split_decodes_and_prefills_nonuniform_mixed_batch():
    query_lens = [2, 1, 3, 4, 5, 6, 7, 8]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 4, False)
    )
    assert num_decodes == 4  # 2, 1, 3, 4 are all <= 4
    assert num_prefills == 4  # 5, 6, 7, 8 are all > 4
    assert num_decode_tokens == 10  # 2 + 1 + 3 + 4
    assert num_prefill_tokens == 26  # 5 + 6 + 7 + 8


def test_split_decodes_and_prefills_uniform_all_ones():
    query_lens = [1, 1, 1]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 1, True)
    )
    assert num_decodes == 3
    assert num_prefills == 0
    assert num_decode_tokens == 3
    assert num_prefill_tokens == 0


def test_split_decodes_and_prefills_uniform_short_extend():
    result = apply_split_decodes_and_prefills(
        [1, 1],
        decode_threshold=1,
        require_uniform=True,
        is_prefilling=[False, True],
        treat_short_extends_as_decodes=False,
    )
    assert result == (1, 1, 1, 1)


def test_split_decodes_and_prefills_uniform_all_short_decodes():
    query_lens = [2, 2, 1, 3, 2, 1, 2]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, True)
    )
    assert num_decodes == 2
    assert num_prefills == 5
    assert num_decode_tokens == 4
    assert num_prefill_tokens == (1 + 3 + 2 + 1 + 2)


def test_split_decodes_and_prefills_uniform_all_prefills():
    query_lens = [4, 5, 6, 7]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, True)
    )
    assert num_decodes == 0
    assert num_prefills == 4
    assert num_decode_tokens == 0
    assert num_prefill_tokens == sum(query_lens)


def test_split_decodes_and_prefills_uniform_mixed_batch_all_uniform_decodes():
    query_lens = [2, 2, 2, 4, 5, 6, 7, 8]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 4, True)
    )
    assert num_decodes == 3  # 2, 2, 2 are all <= 4 and uniform
    assert num_prefills == 5  # 4, 5, 6, 7, 8 are all > 4
    assert num_decode_tokens == 6  # 2 + 2 + 2
    assert num_prefill_tokens == 30  # 4 + 5 + 6 + 7 + 8


def test_split_decodes_and_prefills_uniform_mixed_batch_non_uniform_decodes():
    query_lens = [2, 1, 2, 4, 5, 6, 7, 8]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 4, True)
    )
    assert num_decodes == 1  # only the first 2 is taken as decode
    assert num_prefills == 7  # 1, 2, 4, 5, 6, 7, 8 are all > 4 or non-uniform
    assert num_decode_tokens == 2  # only the first 2
    assert num_prefill_tokens == (sum(query_lens) - 2)  # rest of the tokens


def test_split_decodes_and_prefills_uniform_padded_batch_all_same():
    """Uniform batch where all query lengths are identical with 0 length padded reqs."""
    # All query lengths are 2, with decode_threshold=3 (so 2 <= 3)
    # This triggers the padded uniform path at line 891
    query_lens = [2, 2, 2, 0]
    padded_num_tokens = 8
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, True, padded_num_tokens)
    )
    # With uniform batch, all requests are treated as decodes
    assert num_decodes == 4
    assert num_prefills == 0
    assert num_decode_tokens == padded_num_tokens
    assert num_prefill_tokens == 0
