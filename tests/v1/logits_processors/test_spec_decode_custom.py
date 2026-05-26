# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.sample import rejection_sampler
from vllm.v1.sample.logits_processor import (
    LogitsProcessor,
    LogitsProcessors,
    build_logitsprocs,
)
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (
    RejectionSampler,
    apply_sampling_constraints,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


class SpecDecodeCapableLogitsProcessor(LogitsProcessor):
    @classmethod
    def supports_spec_decode(cls) -> bool:
        return True

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool) -> None:
        del vllm_config, device, is_pin_memory
        self.spec_decode_calls: list[list[int]] = []

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + 1

    def apply_with_spec_decode(
        self,
        logits: torch.Tensor,
        num_draft_tokens: list[int],
    ) -> torch.Tensor:
        self.spec_decode_calls.append(num_draft_tokens)
        return logits + 1

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update):
        del batch_update


class SpecDecodeMissingApplyLogitsProcessor(LogitsProcessor):
    @classmethod
    def supports_spec_decode(cls) -> bool:
        return True

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool) -> None:
        del vllm_config, device, is_pin_memory

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + 1

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update):
        del batch_update


class ArgmaxInvariantMissingApplyLogitsProcessor(SpecDecodeMissingApplyLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return True


class ArgmaxInvariantSpecDecodeLogitsProcessor(SpecDecodeCapableLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return True

    def apply_with_spec_decode(
        self,
        logits: torch.Tensor,
        num_draft_tokens: list[int],
    ) -> torch.Tensor:
        self.spec_decode_calls.append(num_draft_tokens)
        logits[:, 0] = 5.0
        return logits


class UnsupportedSpecDecodeLogitsProcessor(SpecDecodeCapableLogitsProcessor):
    @classmethod
    def supports_spec_decode(cls) -> bool:
        return False


class DraftLogitsProcessor(SpecDecodeCapableLogitsProcessor):
    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool) -> None:
        super().__init__(vllm_config, device, is_pin_memory)
        self.draft_calls: list[list[int] | None] = []

    def apply_to_speculative_draft_logits(
        self,
        logits: torch.Tensor,
        num_draft_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        self.draft_calls.append(num_draft_tokens)
        logits[:, 3] = 100.0
        return logits


class ArgmaxInvariantDraftLogitsProcessor(DraftLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return True


class InvalidDraftTokenLogitsProcessor(DraftLogitsProcessor):
    def apply_to_speculative_draft_logits(
        self,
        logits: torch.Tensor,
        num_draft_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        self.draft_calls.append(num_draft_tokens)
        logits[:, 4] = 100.0
        return logits


class DummyDraftModel:
    def __init__(self):
        self.compute_logits_called = False
        self.get_top_tokens_called = False

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.compute_logits_called = True
        return hidden_states.clone()

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        del hidden_states
        self.get_top_tokens_called = True
        return torch.tensor([0, 0], dtype=torch.long)


class DummyVocabMapping:
    def __init__(self):
        self.constrain_calls = 0

    def constrain_draft_logits(self, logits: torch.Tensor) -> torch.Tensor:
        self.constrain_calls += 1
        logits[:, 4] = float("-inf")
        return logits

    def map_draft_to_target_ids(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        return draft_token_ids + 100


def _spec_decode_config():
    return SimpleNamespace(speculative_config=object())


def _sampling_metadata(
    logitsprocs: LogitsProcessors,
    *,
    all_greedy: bool = True,
    all_random: bool = False,
    temperature: torch.Tensor | None = None,
) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=temperature,
        all_greedy=all_greedy,
        all_random=all_random,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.tensor([]),
        presence_penalties=torch.tensor([]),
        repetition_penalties=torch.tensor([]),
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=logitsprocs,
    )


def _make_draft_proposer() -> SpecDecodeBaseProposer:
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.use_local_argmax_reduction = True
    proposer.use_heterogeneous_vocab = False
    proposer.vocab_mapping = None
    proposer._enable_probabilistic_draft_probs = False
    proposer.model = DummyDraftModel()
    return proposer


def test_build_logitsprocs_allows_opted_in_processors_with_spec_decode():
    processors = build_logitsprocs(
        _spec_decode_config(),
        torch.device("cpu"),
        is_pin_memory=False,
        is_pooling_model=False,
        custom_logitsprocs=[SpecDecodeCapableLogitsProcessor],
    )

    assert any(
        isinstance(processor, MinTokensLogitsProcessor)
        for processor in processors.non_argmax_invariant
    )
    assert any(
        isinstance(processor, SpecDecodeCapableLogitsProcessor)
        for processor in processors.non_argmax_invariant
    )


def test_build_logitsprocs_rejects_processors_without_spec_decode_opt_in():
    with pytest.raises(ValueError, match="supports_spec_decode"):
        build_logitsprocs(
            _spec_decode_config(),
            torch.device("cpu"),
            is_pin_memory=False,
            is_pooling_model=False,
            custom_logitsprocs=[UnsupportedSpecDecodeLogitsProcessor],
        )


@pytest.mark.parametrize(
    "processor_cls",
    [
        SpecDecodeMissingApplyLogitsProcessor,
        ArgmaxInvariantMissingApplyLogitsProcessor,
    ],
)
def test_build_logitsprocs_requires_spec_decode_apply_for_opted_in_processors(
    processor_cls,
):
    with pytest.raises(ValueError, match="apply_with_spec_decode"):
        build_logitsprocs(
            _spec_decode_config(),
            torch.device("cpu"),
            is_pin_memory=False,
            is_pooling_model=False,
            custom_logitsprocs=[processor_cls],
        )


def test_rejection_sampler_fails_loudly_for_missing_spec_decode_apply():
    metadata = _sampling_metadata(
        LogitsProcessors(
            [SpecDecodeMissingApplyLogitsProcessor(None, torch.device("cpu"), False)]
        )
    )
    spec_decode_metadata = SimpleNamespace(num_draft_tokens=[2])

    sampler = SimpleNamespace(logprobs_mode="raw_logprobs")

    with pytest.raises(NotImplementedError, match="apply_with_spec_decode"):
        RejectionSampler(sampler).apply_logits_processors(
            torch.zeros((2, 5)),
            metadata,
            spec_decode_metadata,
        )


def test_argmax_invariant_processors_apply_with_non_greedy_spec_decode(monkeypatch):
    def fake_expand_batch_to_tokens(
        x: torch.Tensor,
        cu_num_tokens: torch.Tensor,
        num_tokens: int,
        replace_from: int = 0,
        replace_to: int = 0,
    ) -> torch.Tensor:
        del num_tokens
        counts = torch.diff(torch.cat([cu_num_tokens.new_zeros(1), cu_num_tokens])).to(
            torch.long
        )
        expanded = torch.repeat_interleave(x, counts)
        if replace_from != replace_to:
            expanded = torch.where(
                expanded == replace_from,
                torch.full_like(expanded, replace_to),
                expanded,
            )
        return expanded

    monkeypatch.setattr(
        rejection_sampler, "expand_batch_to_tokens", fake_expand_batch_to_tokens
    )

    processor = ArgmaxInvariantSpecDecodeLogitsProcessor(
        None, torch.device("cpu"), False
    )
    metadata = _sampling_metadata(
        LogitsProcessors([processor]),
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0, 1.0]),
    )

    logits = torch.zeros((3, 5))
    logits[:, 4] = 10.0
    logits = apply_sampling_constraints(
        logits,
        num_draft_tokens=[1, 2],
        cu_num_draft_tokens=torch.tensor([1, 3], dtype=torch.int32),
        sampling_metadata=metadata,
    )

    assert processor.spec_decode_calls == [[1, 2]]
    assert torch.equal(logits[:, 0], torch.full((3,), 5.0))
    assert torch.equal(logits[:, 4], torch.full((3,), 10.0))


def test_argmax_invariant_processors_skip_all_greedy_spec_decode():
    processor = ArgmaxInvariantSpecDecodeLogitsProcessor(
        None, torch.device("cpu"), False
    )
    metadata = _sampling_metadata(LogitsProcessors([processor]))

    logits = torch.zeros((2, 5))
    logits[:, 4] = 10.0
    processed = apply_sampling_constraints(
        logits,
        num_draft_tokens=[2],
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int32),
        sampling_metadata=metadata,
    )

    assert processed is logits
    assert processor.spec_decode_calls == []
    assert torch.equal(logits[:, 0], torch.zeros(2))


def test_logitsprocs_caches_draft_logits_hooks_from_all_buckets():
    processor = DraftLogitsProcessor(None, torch.device("cpu"), False)
    argmax_processor = ArgmaxInvariantDraftLogitsProcessor(
        None, torch.device("cpu"), False
    )
    logitsprocs = LogitsProcessors(
        [
            processor,
            SpecDecodeCapableLogitsProcessor(None, torch.device("cpu"), False),
            argmax_processor,
        ]
    )

    assert len(logitsprocs.spec_decode_draft_logits_hooks) == 2

    logits = torch.zeros((1, 5))
    for hook in logitsprocs.spec_decode_draft_logits_hooks:
        hook(logits, [1])

    assert processor.draft_calls == [[1]]
    assert argmax_processor.draft_calls == [[1]]


def test_draft_logits_hook_disables_local_argmax_fast_path():
    processor = DraftLogitsProcessor(None, torch.device("cpu"), False)
    metadata = _sampling_metadata(LogitsProcessors([processor]))

    proposer = _make_draft_proposer()

    hidden_states = torch.zeros((2, 5))
    hidden_states[:, 0] = 10.0

    token_ids, draft_probs = proposer._sample_draft_tokens(
        hidden_states,
        metadata,
        num_draft_tokens=[1, 1],
    )

    assert draft_probs is None
    assert torch.equal(token_ids, torch.tensor([3, 3]))
    assert proposer.model.compute_logits_called
    assert not proposer.model.get_top_tokens_called
    assert processor.draft_calls == [[1, 1]]


def test_argmax_invariant_draft_logits_hook_disables_local_argmax_fast_path():
    processor = ArgmaxInvariantDraftLogitsProcessor(None, torch.device("cpu"), False)
    metadata = _sampling_metadata(LogitsProcessors([processor]))

    proposer = _make_draft_proposer()

    hidden_states = torch.zeros((2, 5))
    hidden_states[:, 0] = 10.0

    token_ids, draft_probs = proposer._sample_draft_tokens(
        hidden_states,
        metadata,
        num_draft_tokens=[1, 1],
    )

    assert draft_probs is None
    assert torch.equal(token_ids, torch.tensor([3, 3]))
    assert proposer.model.compute_logits_called
    assert not proposer.model.get_top_tokens_called
    assert processor.draft_calls == [[1, 1]]


def test_draft_logits_hook_preserves_heterogeneous_vocab_constraint():
    processor = InvalidDraftTokenLogitsProcessor(None, torch.device("cpu"), False)
    metadata = _sampling_metadata(LogitsProcessors([processor]))

    proposer = _make_draft_proposer()
    vocab_mapping = DummyVocabMapping()
    proposer.use_heterogeneous_vocab = True
    proposer.vocab_mapping = vocab_mapping

    hidden_states = torch.zeros((2, 5))
    hidden_states[:, 0] = 10.0

    token_ids, draft_probs = proposer._sample_draft_tokens(
        hidden_states,
        metadata,
        num_draft_tokens=[1, 1],
    )

    assert draft_probs is None
    assert torch.equal(token_ids, torch.tensor([100, 100]))
    assert processor.draft_calls == [[1, 1]]
    assert vocab_mapping.constrain_calls == 2
