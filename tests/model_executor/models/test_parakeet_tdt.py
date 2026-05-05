# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm.model_executor.models.parakeet_tdt import (
    ParakeetForTDT,
    ParakeetTDTForcedDecoderState,
    ParakeetTDTModel,
)


def test_parakeet_tdt_forced_tokens_follow_request_ids_after_reorder():
    state = ParakeetTDTForcedDecoderState(eos_token_id=99)
    state.set_sequences(["req-a", "req-b"], [[11, 12], [21, 22]])

    forced = state.get_forced_token_ids(
        request_ids=["req-b", "req-a"],
        request_indices=torch.tensor([0, 1], dtype=torch.long),
        positions=torch.tensor([1, 0], dtype=torch.long),
        device=torch.device("cpu"),
    )

    assert forced.tolist() == [22, 11]


def test_parakeet_tdt_forced_tokens_fall_back_to_eos_after_sequence_end():
    state = ParakeetTDTForcedDecoderState(eos_token_id=99)
    state.set_sequences(["req-a"], [[11]])

    forced = state.get_forced_token_ids(
        request_ids=["req-a"],
        request_indices=torch.tensor([0, 0], dtype=torch.long),
        positions=torch.tensor([0, 2], dtype=torch.long),
        device=torch.device("cpu"),
    )

    assert forced.tolist() == [11, 99]


def test_parakeet_tdt_forced_state_removes_finished_requests():
    state = ParakeetTDTForcedDecoderState(eos_token_id=99)
    state.set_sequences(["req-a"], [[11]])

    state.remove_sequences(["req-a"])
    forced = state.get_forced_token_ids(
        request_ids=["req-a"],
        request_indices=torch.tensor([0], dtype=torch.long),
        positions=torch.tensor([0], dtype=torch.long),
        device=torch.device("cpu"),
    )

    assert forced.tolist() == [99]


def test_parakeet_tdt_prepare_generation_step_uses_stable_forced_token_buffer():
    model = ParakeetForTDT.__new__(ParakeetForTDT)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=100, eos_token_id=99)
    model._forced_decoder_state = ParakeetTDTForcedDecoderState(eos_token_id=99)
    model.model = SimpleNamespace(
        greedy_decode=lambda encoder_output: [int(encoder_output.item()), 99]
    )

    prepared = model.prepare_generation_step(
        encoder_outputs=[torch.tensor(11), torch.tensor(21)],
        request_ids=["req-a", "req-b"],
        request_indices=torch.tensor([0, 1], dtype=torch.long),
        encoder_request_ids=["req-a", "req-b"],
        finished_request_ids=(),
        positions=torch.tensor([0, 0], dtype=torch.long),
        device=torch.device("cpu"),
    )
    forced_token_ids = prepared["forced_token_ids"]
    assert forced_token_ids.tolist() == [11, 21]

    first_data_ptr = forced_token_ids.data_ptr()
    prepared = model.prepare_generation_step(
        request_ids=["req-b", "req-a"],
        request_indices=torch.tensor([0, 1], dtype=torch.long),
        finished_request_ids=(),
        positions=torch.tensor([0, 0], dtype=torch.long),
        device=torch.device("cpu"),
    )
    forced_token_ids = prepared["forced_token_ids"]
    assert forced_token_ids.tolist() == [21, 11]
    assert forced_token_ids.data_ptr() == first_data_ptr


def test_parakeet_tdt_prepare_dummy_generation_step_uses_forced_token_buffer():
    model = ParakeetForTDT.__new__(ParakeetForTDT)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=100, eos_token_id=99)
    model._forced_decoder_state = ParakeetTDTForcedDecoderState(eos_token_id=99)
    model.model = SimpleNamespace(
        greedy_decode=lambda encoder_output: [int(encoder_output.item()), 99]
    )

    dummy_prepared = model.prepare_dummy_generation_step(
        num_tokens=2,
        device=torch.device("cpu"),
    )
    dummy_forced_token_ids = dummy_prepared["forced_token_ids"]
    assert dummy_forced_token_ids.tolist() == [99, 99]
    dummy_data_ptr = dummy_forced_token_ids.data_ptr()

    prepared = model.prepare_generation_step(
        encoder_outputs=[torch.tensor(11), torch.tensor(21)],
        request_ids=["req-a", "req-b"],
        request_indices=torch.tensor([0, 1], dtype=torch.long),
        encoder_request_ids=["req-a", "req-b"],
        finished_request_ids=(),
        positions=torch.tensor([0, 0], dtype=torch.long),
        device=torch.device("cpu"),
    )
    forced_token_ids = prepared["forced_token_ids"]

    assert forced_token_ids.tolist() == [11, 21]
    assert forced_token_ids.data_ptr() == dummy_data_ptr


def test_parakeet_tdt_forward_uses_prepared_forced_token_ids():
    model = ParakeetForTDT.__new__(ParakeetForTDT)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=100, eos_token_id=99)

    logits = ParakeetForTDT.forward(
        model,
        input_ids=torch.zeros(2, dtype=torch.long),
        positions=torch.tensor([0, 0], dtype=torch.long),
        forced_token_ids=torch.tensor([11, 21], dtype=torch.long),
    )
    assert logits.argmax(dim=-1).tolist() == [11, 21]


def test_parakeet_tdt_pads_variable_length_audio_features():
    model = ParakeetForTDT.__new__(ParakeetForTDT)
    nn.Module.__init__(model)

    input_features, attention_mask = model._parse_and_validate_audio_input(
        input_features=[torch.ones(2, 3), torch.full((4, 3), 2.0)],
        attention_mask=[
            torch.ones(2, dtype=torch.bool),
            torch.ones(4, dtype=torch.bool),
        ],
    )

    assert input_features.shape == (2, 4, 3)
    assert attention_mask.shape == (2, 4)
    assert input_features[0, 2:].tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert attention_mask[0].tolist() == [True, True, False, False]


def test_parakeet_tdt_greedy_decode_advances_blank_duration_zero():
    class FakeDecoder:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, token_id, state, device):
            del token_id, state, device
            self.calls += 1
            return torch.zeros(1, 4), None

    decoder = FakeDecoder()
    fake_model = SimpleNamespace(
        config=SimpleNamespace(
            vocab_size=3,
            blank_token_id=2,
            eos_token_id=1,
            durations=[0, 1],
            max_symbols_per_step=4,
        ),
        decoder=decoder,
        _joint_logits=lambda encoder_frame, pred_state: torch.tensor(
            [[0.0, 0.0, 1.0, 1.0, 0.0]], device=encoder_frame.device
        ),
    )

    token_ids = ParakeetTDTModel.greedy_decode(fake_model, torch.zeros(2, 4))

    assert token_ids == [1]
    assert decoder.calls == 2
