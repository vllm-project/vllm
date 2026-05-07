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


def test_parakeet_tdt_forward_uses_request_ids_for_forced_tokens():
    model = ParakeetForTDT.__new__(ParakeetForTDT)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=100, eos_token_id=99)
    model._forced_decoder_state = ParakeetTDTForcedDecoderState(eos_token_id=99)
    model.model = SimpleNamespace(
        greedy_decode=lambda encoder_output: [int(encoder_output.item()), 99]
    )

    logits = ParakeetForTDT.forward(
        model,
        input_ids=torch.zeros(2, dtype=torch.long),
        positions=torch.tensor([0, 0], dtype=torch.long),
        encoder_outputs=[torch.tensor(11), torch.tensor(21)],
        request_ids=["req-a", "req-b"],
        request_indices=torch.tensor([0, 1], dtype=torch.long),
        encoder_request_ids=["req-a", "req-b"],
        finished_request_ids=(),
    )
    assert logits.argmax(dim=-1).tolist() == [11, 21]

    logits = ParakeetForTDT.forward(
        model,
        input_ids=torch.zeros(2, dtype=torch.long),
        positions=torch.tensor([0, 0], dtype=torch.long),
        request_ids=["req-b", "req-a"],
        request_indices=torch.tensor([0, 1], dtype=torch.long),
        finished_request_ids=(),
    )
    assert logits.argmax(dim=-1).tolist() == [21, 11]


def test_parakeet_tdt_uses_request_id_generation_path():
    assert ParakeetForTDT.uses_request_ids_for_generation is True


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
        encoder_projector=lambda encoder_output: encoder_output,
        _joint_logits=lambda encoder_frame, pred_state: torch.tensor(
            [[0.0, 0.0, 1.0, 1.0, 0.0]], device=encoder_frame.device
        ),
    )

    token_ids = ParakeetTDTModel.greedy_decode(fake_model, torch.zeros(2, 4))

    assert token_ids == [1]
    assert decoder.calls == 2


def test_parakeet_tdt_greedy_decode_projects_encoder_once():
    class FakeDecoder:
        def predict(self, token_id, state, device):
            del token_id, state, device
            return torch.zeros(1, 4), None

    class FakeProjector:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, encoder_output):
            self.calls += 1
            return encoder_output

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                vocab_size=3,
                blank_token_id=2,
                eos_token_id=1,
                durations=[0, 1],
                max_symbols_per_step=4,
            )
            self.decoder = FakeDecoder()
            self.encoder_projector = FakeProjector()

        def joint(self, encoder_state, pred_state):
            del pred_state
            return torch.tensor(
                [[0.0, 0.0, 1.0, 0.0, 1.0]],
                device=encoder_state.device,
            )

        def _joint_logits(self, encoder_state, pred_state):
            return ParakeetTDTModel._joint_logits(self, encoder_state, pred_state)

    fake_model = FakeModel()

    ParakeetTDTModel.greedy_decode(fake_model, torch.zeros(2, 4))

    assert fake_model.encoder_projector.calls == 1


def test_parakeet_tdt_greedy_decode_does_not_skip_after_positive_duration():
    class FakeDecoder:
        def predict(self, token_id, state, device):
            del token_id, state, device
            return torch.zeros(1, 4), None

    def joint_logits(encoder_frame, pred_state):
        del pred_state
        token_id = int(encoder_frame[0, 0].item())
        logits = torch.full((1, 11), -1.0, device=encoder_frame.device)
        logits[0, token_id] = 1.0
        logits[0, 10] = 1.0
        return logits

    fake_model = SimpleNamespace(
        config=SimpleNamespace(
            vocab_size=10,
            blank_token_id=8,
            eos_token_id=9,
            durations=[1],
            max_symbols_per_step=1,
        ),
        decoder=FakeDecoder(),
        encoder_projector=lambda encoder_output: encoder_output,
        _joint_logits=joint_logits,
    )

    token_ids = ParakeetTDTModel.greedy_decode(
        fake_model,
        torch.tensor([[0.0], [1.0], [2.0]]),
    )

    assert token_ids == [0, 1, 2, 9]
