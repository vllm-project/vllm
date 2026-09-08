# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.uno import UnoProposer, uno_query_layout
from vllm.v1.spec_decode.uno_noise import fill_uno_noise
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID


@pytest.mark.parametrize("k", [1, 2, 5])
def test_layout_samples_every_query_and_adapts_only_noise(k):
    mask, sample_indices, mapping = uno_query_layout(
        2, k, 7, torch.device("cpu"), num_input_tokens=2 * k + 3
    )
    assert sample_indices.tolist() == list(range(2 * k))
    assert mask[: 2 * k].reshape(2, k).tolist() == [[False] + [True] * (k - 1)] * 2
    assert mapping[: 2 * k] == tuple([0] + [7] * (k - 1)) * 2
    assert not mask[2 * k :].any()
    assert mapping[2 * k :] == (0, 0, 0)


@pytest.mark.parametrize("step", [1, 116, 10**6, 10**20])
def test_noise_is_bounded_repeatable_and_preserves_seed_and_padding(step):
    mask, _, _ = uno_query_layout(2, 8, 7, torch.device("cpu"), 19)
    seeds = torch.tensor([11] * 8 + [23] * 8 + [0] * 3)
    original = torch.full((19,), 99, dtype=torch.int32)
    first, second = original.clone(), original.clone()
    rng_state = torch.get_rng_state().clone()
    fill_uno_noise(first, mask, seeds, step, 1, 31)
    fill_uno_noise(second, mask, seeds, step, 1, 31)
    assert torch.equal(first, second)
    assert torch.equal(first[~mask], original[~mask])
    assert torch.all((first[mask] >= 1) & (first[mask] < 31))
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_noise_changes_with_seed_and_step():
    mask = torch.ones(32, dtype=torch.bool)
    outputs = []
    for seed, step in [(11, 1), (12, 1), (11, 2)]:
        ids = torch.zeros(32, dtype=torch.int32)
        fill_uno_noise(ids, mask, torch.full_like(ids, seed), step, 1, 1000)
        outputs.append(ids)
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[0], outputs[2])


def test_noise_rejects_empty_range_before_modifying_inputs():
    ids = torch.tensor([17, 19])
    with pytest.raises(ValueError, match="nonempty"):
        fill_uno_noise(ids, torch.ones(2, dtype=torch.bool), ids, 1, 5, 5)
    assert ids.tolist() == [17, 19]


@pytest.fixture
def proposer():
    drafter = object.__new__(UnoProposer)
    drafter.device = torch.device("cpu")
    drafter.num_speculative_tokens = 3
    drafter.speculative_config = SimpleNamespace(
        num_speculative_tokens=3, uno_noise_seed=17, uno_mask_token_id=31
    )
    drafter.max_model_len = 32
    drafter.max_batch_size = 2
    drafter.block_size = 4
    drafter.uno_lora_id = 7
    drafter._step = 0
    drafter._lora_hook = None
    drafter._last_draft_probs = None
    drafter._pending_lora_map = ()
    drafter.input_ids = torch.zeros(6, dtype=torch.int32)
    drafter.positions = torch.zeros(6, dtype=torch.int64)
    drafter._slot_mapping_buffer = torch.full((6,), PADDING_SLOT_ID)
    drafter._draft_attn_layer_names = {"attention"}
    drafter.use_local_argmax_reduction = False
    drafter.use_heterogeneous_vocab = False
    drafter._enable_probabilistic_draft_probs = False
    drafter.use_fp64_gumbel = False
    drafter.vllm_config = Mock()
    return drafter


@pytest.fixture
def context():
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        seq_lens=torch.tensor([7, 16], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([7, 16], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=8,
        max_query_len=4,
        max_seq_len=16,
        block_table_tensor=torch.tensor(
            [[2, 4, 6, 8, 10, 12, 14, 16], [3, 5, 7, 9, 11, 13, 15, 17]],
            dtype=torch.int32,
        ),
        slot_mapping=torch.zeros(8, dtype=torch.int64),
    )


def _inputs(context, rejected=None):
    return dict(
        target_token_ids=torch.zeros(8, dtype=torch.int32),
        target_positions=torch.tensor([3, 4, 5, 6, 12, 13, 14, 15]),
        target_hidden_states=torch.zeros(8, 4),
        next_token_ids=torch.tensor([29, 30], dtype=torch.int32),
        token_indices_to_sample=None,
        num_rejected_tokens_gpu=rejected,
    )


@pytest.mark.parametrize(
    ("rejected", "positions", "slots", "seq_lens"),
    [
        (None, [7, 8, 9, 16, 17, 18], [19, 24, 25, 44, 45, 46], [10, 19]),
        ([1, 2], [6, 7, 8, 14, 15, 16], [18, 19, 24, 38, 39, 44], [9, 17]),
        ([3, 3], [4, 5, 6, 13, 14, 15], [16, 17, 18, 37, 38, 39], [7, 16]),
    ],
)
def test_queries_follow_accepted_prefix_and_cross_real_kv_blocks(
    proposer, context, rejected, positions, slots, seq_lens
):
    rejected_tensor = None if rejected is None else torch.tensor(rejected)
    num_tokens, sample_indices, metadata = proposer.set_inputs_first_pass(
        **_inputs(context, rejected_tensor), cad=context
    )
    assert num_tokens == 6
    assert sample_indices.tolist() == [0, 1, 2, 3, 4, 5]
    assert proposer.positions.tolist() == positions
    assert metadata.slot_mapping.tolist() == slots
    assert metadata.seq_lens.tolist() == seq_lens
    assert metadata.query_start_loc.tolist() == [0, 3, 6]
    assert metadata.query_start_loc_cpu.tolist() == [0, 3, 6]
    assert metadata.causal is True
    assert proposer.input_ids[[0, 3]].tolist() == [29, 30]
    assert proposer._pending_lora_map == (0, 7, 7, 0, 7, 7)
    assert torch.equal(context.seq_lens, torch.tensor([7, 16]))


def test_queries_do_not_write_null_blocks_or_positions_beyond_context_limit(
    proposer, context
):
    proposer.max_model_len = 18
    context.block_table_tensor[0, 2] = 0
    _, _, metadata = proposer.set_inputs_first_pass(**_inputs(context), cad=context)
    assert metadata.slot_mapping.tolist() == [
        19,
        PADDING_SLOT_ID,
        PADDING_SLOT_ID,
        44,
        45,
        PADDING_SLOT_ID,
    ]


@pytest.mark.parametrize("failure", [None, "activate", "forward"])
def test_draft_scope_restores_base_mapping_on_success_and_exceptions(proposer, failure):
    calls = []

    def hook(mapping):
        calls.append(mapping)
        if failure == "activate" and mapping is not None:
            raise RuntimeError("activation failed")

    proposer.set_lora_hook(hook)
    expected = pytest.raises(RuntimeError) if failure else nullcontext()
    with expected, proposer._draft_lora((0, 7, 7)):
        if failure == "forward":
            raise RuntimeError("forward failed")
    assert calls == [(0, 7, 7), None]


@pytest.fixture
def forward_model(proposer, monkeypatch):
    active_mapping = []
    proposer.set_lora_hook(lambda mapping: active_mapping.append(mapping))
    monkeypatch.setattr(
        "vllm.v1.spec_decode.uno.set_forward_context", lambda *args, **kw: nullcontext()
    )
    proposer.build_per_group_and_layer_attn_metadata = Mock(return_value=([], {}))

    class Model:
        def __call__(self, input_ids, positions, inputs_embeds):
            assert active_mapping[-1] == (0, 7, 7, 0, 7, 7)
            return torch.nn.functional.one_hot(input_ids.long(), num_classes=32).float()

        def compute_logits(self, hidden_states):
            assert active_mapping[-1] == (0, 7, 7, 0, 7, 7)
            return hidden_states

    proposer.model = Model()
    return active_mapping


def test_propose_samples_all_k_rows_and_restores_base_mapping(
    proposer, context, forward_model
):
    drafts = proposer.propose(
        3,
        **_inputs(context),
        common_attn_metadata=context,
        sampling_metadata=SimpleNamespace(all_greedy=True),
    )
    assert drafts.tolist() == proposer.input_ids.reshape(2, 3).tolist()
    assert drafts[:, 0].tolist() == [29, 30]
    assert forward_model[-1] is None
    assert proposer.take_last_draft_probs() is None
    assert proposer._pending_lora_map == ()


@pytest.mark.parametrize("failure", ["prepare", "forward", "sample"])
def test_failed_proposal_clears_old_probabilities_and_restores_adapter(
    proposer, context, forward_model, failure
):
    proposer._last_draft_probs = torch.ones(2, 3, 32)
    failing = Mock(side_effect=RuntimeError("draft failed"))
    if failure == "prepare":
        proposer.set_inputs_first_pass = failing
    elif failure == "forward":
        proposer.model = failing
    else:
        proposer.model.compute_logits = failing
    with pytest.raises(RuntimeError, match="draft failed"):
        proposer.propose(
            3,
            **_inputs(context),
            common_attn_metadata=context,
            sampling_metadata=SimpleNamespace(all_greedy=True),
        )
    assert not forward_model or forward_model[-1] is None
    assert proposer.take_last_draft_probs() is None
    assert proposer._pending_lora_map == ()


def test_zero_drafts_bypasses_model_and_clears_old_probabilities(proposer, context):
    proposer._last_draft_probs = torch.ones(2, 3, 32)
    proposer.model = Mock(side_effect=AssertionError("unexpected forward"))
    drafts = proposer.propose(
        0,
        **_inputs(context),
        common_attn_metadata=context,
        sampling_metadata=SimpleNamespace(all_greedy=True),
    )
    assert drafts.shape == (2, 0)
    assert proposer.take_last_draft_probs() is None
    proposer.model.assert_not_called()


@pytest.fixture
def sampling_metadata():
    return SamplingMetadata(
        temperature=torch.tensor([0.5, 2.0]),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.empty(0),
        presence_penalties=torch.empty(0),
        repetition_penalties=torch.empty(0),
        output_token_ids=[[], []],
        spec_token_ids=[[], []],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def test_native_dense_probabilities_remain_request_major(
    proposer, context, forward_model, sampling_metadata
):
    proposer._enable_probabilistic_draft_probs = True
    sampling = sampling_metadata
    drafts = proposer.propose(
        3, **_inputs(context), common_attn_metadata=context, sampling_metadata=sampling
    )
    probs = proposer.take_last_draft_probs()
    assert drafts.shape == (2, 3)
    assert probs.shape == (2, 3, 32)
    raw_logits = torch.nn.functional.one_hot(proposer.input_ids.long(), 32).float()
    expected = (
        raw_logits / sampling.temperature.repeat_interleave(3)[:, None]
    ).softmax(-1)
    torch.testing.assert_close(probs.reshape(6, 32), expected)
    torch.testing.assert_close(probs.sum(-1), torch.ones(2, 3))
    assert forward_model[-1] is None


def test_dummy_profile_samples_full_draft_batch_with_noise_only_adapter(
    proposer, forward_model, sampling_metadata
):
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(sampling_metadata=sampling_metadata)
    )
    proposer._enable_probabilistic_draft_probs = True
    sample = Mock(wraps=proposer._sample_from_logits)
    proposer._sample_from_logits = sample

    proposer.dummy_run(8, use_cudagraphs=True)

    sample.assert_called_once()
    logits, metadata = sample.call_args.args
    assert logits.shape == (6, 32)
    torch.testing.assert_close(metadata.temperature, torch.ones(6))
    torch.testing.assert_close(sampling_metadata.temperature, torch.tensor([0.5, 2.0]))
    assert forward_model == [(0, 7, 7, 0, 7, 7), None]
    assert proposer._slot_mapping_buffer.tolist() == [PADDING_SLOT_ID] * 6
    assert proposer.take_last_draft_probs() is None


def test_graph_capture_skips_duplicate_shared_model_forward(proposer):
    proposer.model = Mock(side_effect=AssertionError("unexpected forward"))
    hook = Mock(side_effect=AssertionError("unexpected adapter activation"))
    proposer.set_lora_hook(hook)

    proposer.dummy_run(6, is_graph_capturing=True)

    proposer.model.assert_not_called()
    hook.assert_not_called()
