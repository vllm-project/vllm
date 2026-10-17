# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.interfaces import LocalArgmaxMixin
from vllm.v1.worker.gpu.spec_decode.speculator import _attach_reduced_draft_vocab


class _LMHead:
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        tp_size: int = 1,
        org_vocab_size: int | None = None,
    ):
        self.weight = weight
        self.tp_size = tp_size
        self.org_vocab_size = (
            org_vocab_size if org_vocab_size is not None else weight.shape[0]
        )


class _PlainDraftModel(torch.nn.Module, LocalArgmaxMixin):
    """A drafter whose get_top_tokens is the generic LocalArgmaxMixin
    default -- the shape most MTP heads and EAGLE drafters are in today."""

    def __init__(self, lm_head: _LMHead):
        super().__init__()
        self.lm_head = lm_head
        self.logits_processor = None  # unused by this test; get_top_tokens is replaced


class _SelfRestrictingDraftModel(torch.nn.Module):
    """A drafter that already has its OWN get_top_tokens (e.g. Gemma4's
    centroid-projection head) -- the generic patch must leave it alone."""

    def __init__(self, lm_head: _LMHead):
        super().__init__()
        self.lm_head = lm_head
        self.calls = 0

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.zeros(hidden_states.shape[0], dtype=torch.long)


def _speculator(model, *, vocab_path: str | None):
    return SimpleNamespace(model=model, vocab_size=model.lm_head.weight.shape[0])


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("VLLM_SPEC_DRAFT_VOCAB", raising=False)


def test_noop_when_env_var_unset(monkeypatch):
    weight = torch.randn(8, 4)
    model = _PlainDraftModel(_LMHead(weight))
    _attach_reduced_draft_vocab(_speculator(model, vocab_path=None))
    assert "get_top_tokens" not in vars(model)


def test_noop_when_model_has_its_own_restriction(monkeypatch, tmp_path):
    vocab_file = tmp_path / "ids.txt"
    vocab_file.write_text("0\n1\n2\n")
    monkeypatch.setenv("VLLM_SPEC_DRAFT_VOCAB", str(vocab_file))

    weight = torch.randn(8, 4)
    model = _SelfRestrictingDraftModel(_LMHead(weight))
    _attach_reduced_draft_vocab(_speculator(model, vocab_path=str(vocab_file)))

    # The model's own get_top_tokens must still be the one that runs.
    hidden_states = torch.zeros(2, 4)
    model.get_top_tokens(hidden_states)
    assert model.calls == 1


def test_noop_when_tp_size_is_not_one(monkeypatch, tmp_path):
    vocab_file = tmp_path / "ids.txt"
    vocab_file.write_text("0\n1\n2\n")
    monkeypatch.setenv("VLLM_SPEC_DRAFT_VOCAB", str(vocab_file))

    weight = torch.randn(8, 4)
    model = _PlainDraftModel(_LMHead(weight, tp_size=2))
    _attach_reduced_draft_vocab(_speculator(model, vocab_path=str(vocab_file)))
    assert "get_top_tokens" not in vars(model)


def test_attaches_and_remaps_to_true_vocab_ids(monkeypatch, tmp_path):
    """Kept ids 1, 4 and 6 out of an 8-row vocab. The reduced matmul's
    argmax lands on the LOCAL index of row 4 (built to score highest);
    get_top_tokens must return the TRUE id, 4 -- not the local index, 1."""
    torch.manual_seed(0)
    vocab_file = tmp_path / "ids.txt"
    vocab_file.write_text("6\n1\n4\n")  # written out of order on purpose
    monkeypatch.setenv("VLLM_SPEC_DRAFT_VOCAB", str(vocab_file))

    hidden_size = 4
    weight = torch.zeros(8, hidden_size)
    probe = torch.tensor([1.0, 0.0, 0.0, 0.0])
    weight[1] = -probe  # kept, local index 0 -- scores low
    weight[4] = probe * 5.0  # kept, local index 1 -- scores highest
    weight[6] = -probe  # kept, local index 2 -- scores low
    weight[2] = probe * 100.0  # NOT kept -- would win if the mask were missing

    original_weight = weight.clone()
    model = _PlainDraftModel(_LMHead(weight))
    speculator = _speculator(model, vocab_path=str(vocab_file))
    _attach_reduced_draft_vocab(speculator)

    assert "get_top_tokens" in vars(model)
    top = model.get_top_tokens(probe.unsqueeze(0))
    assert top.tolist() == [4]

    # The original weight tensor is untouched -- this is the property that
    # matters when an EAGLE drafter shares its lm_head.weight with the
    # target model (see eagle/utils.py's load_eagle_model).
    assert torch.equal(weight, original_weight)


def test_reduced_weight_never_aliases_the_shared_tensor(monkeypatch, tmp_path):
    vocab_file = tmp_path / "ids.txt"
    vocab_file.write_text("0\n2\n4\n")
    monkeypatch.setenv("VLLM_SPEC_DRAFT_VOCAB", str(vocab_file))

    weight = torch.randn(8, 4)
    model = _PlainDraftModel(_LMHead(weight))
    _attach_reduced_draft_vocab(_speculator(model, vocab_path=str(vocab_file)))

    cells = model.get_top_tokens.__func__.__closure__
    reduced_weight = next(
        c.cell_contents
        for c in cells
        if torch.is_tensor(c.cell_contents) and c.cell_contents.dim() == 2
    )
    assert reduced_weight.data_ptr() != weight.data_ptr()


def test_noop_without_a_plain_lm_head(monkeypatch, tmp_path):
    vocab_file = tmp_path / "ids.txt"
    vocab_file.write_text("0\n1\n")
    monkeypatch.setenv("VLLM_SPEC_DRAFT_VOCAB", str(vocab_file))

    class _NoLMHeadModel(torch.nn.Module, LocalArgmaxMixin):
        pass

    model = _NoLMHeadModel()
    speculator = SimpleNamespace(model=model, vocab_size=8)
    _attach_reduced_draft_vocab(speculator)
    assert "get_top_tokens" not in vars(model)
