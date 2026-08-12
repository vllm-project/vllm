# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.gemma4_mtp import Gemma4MTPMaskedEmbedder

HIDDEN_SIZE = 2
VOCAB_SIZE = 16
NUM_CENTROIDS = 4
TOP_K = 2

# Token scores, chosen so the ranking among selected candidates is
# unambiguous: 5 wins, then 2, then 6.
TOKEN_SCORES = {5: 100.0, 2: 50.0, 6: 25.0}

BEST_TOKEN = 5
SECOND_TOKEN = 2
THIRD_TOKEN = 6


def make_embedder() -> Gemma4MTPMaskedEmbedder:
    embedder = Gemma4MTPMaskedEmbedder(
        hidden_size=HIDDEN_SIZE,
        vocab_size=VOCAB_SIZE,
        num_centroids=NUM_CENTROIDS,
        centroid_intermediate_top_k=TOP_K,
    )
    # Identity ordering: centroid c owns tokens [4c, 4c+1, 4c+2, 4c+3].
    embedder.token_ordering.copy_(torch.arange(VOCAB_SIZE))
    # With hidden_states = [1, 0], centroid scores are the first weight
    # column, so centroids 0 and 1 are always selected -> candidate tokens
    # 0..7.
    with torch.no_grad():
        embedder.centroids.weight.zero_()
        embedder.centroids.weight[:, 0] = torch.tensor([10.0, 9.0, 1.0, 0.0])
    return embedder


def make_lm_head_weight() -> torch.Tensor:
    lm_head_weight = torch.zeros((VOCAB_SIZE, HIDDEN_SIZE))
    for token_id, score in TOKEN_SCORES.items():
        lm_head_weight[token_id, 0] = score
    return lm_head_weight


HIDDEN_STATES = torch.tensor([[1.0, 0.0]])


@pytest.mark.cpu_test
def test_sparse_argmax_without_suppression_returns_best_token():
    embedder = make_embedder()

    top = embedder.get_top_tokens(HIDDEN_STATES, make_lm_head_weight())

    assert top.tolist() == [BEST_TOKEN]


@pytest.mark.cpu_test
def test_sparse_argmax_skips_suppressed_token():
    """Regression test: the sparse path bypasses ``compute_logits``.

    ``Gemma4Proposer._greedy_sample`` calls ``get_top_tokens`` directly, so
    suppression applied in ``compute_logits`` never runs. Without masking
    here the drafter can emit a suppressed token.
    """
    embedder = make_embedder()
    embedder.set_suppressed_token_ids([BEST_TOKEN])

    top = embedder.get_top_tokens(HIDDEN_STATES, make_lm_head_weight())

    assert top.tolist() == [SECOND_TOKEN]


@pytest.mark.cpu_test
def test_sparse_argmax_skips_multiple_suppressed_tokens():
    embedder = make_embedder()
    embedder.set_suppressed_token_ids([BEST_TOKEN, SECOND_TOKEN])

    top = embedder.get_top_tokens(HIDDEN_STATES, make_lm_head_weight())

    assert top.tolist() == [THIRD_TOKEN]


@pytest.mark.cpu_test
@pytest.mark.parametrize("token_ids", [[], None])
def test_empty_suppression_is_a_noop(token_ids):
    embedder = make_embedder()
    embedder.set_suppressed_token_ids(token_ids)

    assert embedder.has_suppressed_tokens is False
    assert not embedder.suppress_mask.any()
    top = embedder.get_top_tokens(HIDDEN_STATES, make_lm_head_weight())
    assert top.tolist() == [BEST_TOKEN]


@pytest.mark.cpu_test
def test_suppress_mask_is_not_persistent():
    """The mask is config-derived, so it must stay out of ``state_dict``.

    A persistent buffer would surface as an unexpected key during weight
    loading.
    """
    embedder = make_embedder()
    embedder.set_suppressed_token_ids([BEST_TOKEN])

    assert "suppress_mask" not in embedder.state_dict()
    assert "token_ordering" in embedder.state_dict()


@pytest.mark.cpu_test
def test_suppression_survives_batched_hidden_states():
    embedder = make_embedder()
    embedder.set_suppressed_token_ids([BEST_TOKEN])

    hidden_states = HIDDEN_STATES.expand(4, HIDDEN_SIZE).contiguous()
    top = embedder.get_top_tokens(hidden_states, make_lm_head_weight())

    assert top.tolist() == [SECOND_TOKEN] * 4
