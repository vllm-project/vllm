from types import SimpleNamespace

import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def test_v2_runner_uses_model_state_sample_hidden_state_compaction():
    runner = object.__new__(GPUModelRunner)
    captured = {}

    class FakeModelState:
        def prepare_sample_hidden_states(self, hidden_states, input_batch):
            captured["input_logits_indices"] = input_batch.logits_indices.tolist()
            selected_rows = torch.tensor([0, 2], dtype=torch.int64)
            return hidden_states[selected_rows], selected_rows

    class FakeModel:
        def compute_logits(self, sample_hidden_states):
            captured["compute_shape"] = tuple(sample_hidden_states.shape)
            return torch.ones(sample_hidden_states.shape[0], 5)

    class FakeSampler:
        def __call__(self, logits, input_batch):
            captured["sampler_logits_shape"] = tuple(logits.shape)
            captured["logits_row_indices"] = input_batch.logits_row_indices.tolist()
            return SimpleNamespace(
                num_sampled=torch.tensor([0], dtype=torch.int32),
                num_rejected=torch.tensor([3], dtype=torch.int32),
            )

    runner.model_state = FakeModelState()
    runner.model = FakeModel()
    runner.sampler = FakeSampler()
    runner.rejection_sampler = None
    runner.structured_outputs_worker = None

    input_batch = SimpleNamespace(
        logits_indices=torch.arange(3, dtype=torch.int64),
        num_draft_tokens=1,
    )
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)

    output, num_sampled, num_rejected = GPUModelRunner.sample(
        runner,
        hidden_states,
        input_batch,
        grammar_output=None,
    )

    assert output.num_sampled.tolist() == [0]
    assert num_sampled.tolist() == [0]
    assert num_rejected.tolist() == [3]
    assert captured == {
        "input_logits_indices": [0, 1, 2],
        "compute_shape": (2, 4),
        "sampler_logits_shape": (2, 5),
        "logits_row_indices": [0, 2],
    }
    assert not hasattr(input_batch, "logits_row_indices")
