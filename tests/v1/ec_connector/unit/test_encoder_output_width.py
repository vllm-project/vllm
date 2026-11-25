# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup measurement of the encoder output width EC producers size by."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.ec_connector.unit.utils import create_ec_vllm_config
from vllm.distributed.ec_transfer.ec_connector import encoder_output_width as eow

pytestmark = pytest.mark.cpu_test


class _DeepStackModel:
    """Emits each modality at its own width, as Qwen3-Omni does."""

    widths = {"image": 4 * 16, "audio": 16}

    def embed_multimodal(self, *, modality):
        return [torch.zeros(3, self.widths[modality])]


class _Budget:
    def __init__(self, *args, **kwargs):
        self.mm_max_toks_per_item = dict.fromkeys(_DeepStackModel.widths, 8)

    def get_dummy_encoder_profile_inputs(self, modality, max_items_per_batch):
        assert max_items_per_batch == 1
        return [(modality, None)]


def _worker(first_pp_rank=True, supports_mm=True):
    runner = SimpleNamespace(
        supports_mm_inputs=supports_mm,
        get_model=_DeepStackModel,
        device=torch.device("cpu"),
    )
    pp_group = SimpleNamespace(is_first_rank=first_pp_rank)
    return SimpleNamespace(model_runner=runner, vllm_config=None), pp_group


@pytest.fixture
def patched(monkeypatch):
    import vllm.distributed.parallel_state as ps
    import vllm.multimodal.encoder_budget as budget_mod
    import vllm.multimodal.utils as mm_utils

    monkeypatch.setattr(budget_mod, "MultiModalBudget", _Budget)
    monkeypatch.setattr(
        mm_utils,
        "group_and_batch_mm_kwargs",
        lambda items, **_: iter([(items[0][0], 1, {"modality": items[0][0]})]),
    )
    return lambda group: monkeypatch.setattr(ps, "get_pp_group", lambda: group)


def test_worker_reports_the_encoder_output_width_per_modality(patched):
    """The width comes from the tensor the encoder emits, not from config."""
    worker, group = _worker()
    patched(group)
    assert eow._measure_on_worker(worker) == {"image": 64, "audio": 16}


@pytest.mark.parametrize("first_pp_rank, supports_mm", [(False, True), (True, False)])
def test_ranks_without_an_encoder_report_nothing(patched, first_pp_rank, supports_mm):
    worker, group = _worker(first_pp_rank, supports_mm)
    patched(group)
    assert eow._measure_on_worker(worker) == {}


def test_producer_records_the_widths_its_encoder_ranks_agree_on():
    config = create_ec_vllm_config(ec_role="ec_producer", encoder_output_widths={})
    reports = [{"image": 64}, {}, {"image": 64}]
    eow.measure_encoder_output_widths(config, lambda fn: reports)
    assert config.ec_transfer_config.mm_encoder_output_widths == {"image": 64}
    assert eow.get_encoder_output_width(config, "image") == 64


@pytest.mark.parametrize(
    "reports, match",
    [
        ([{}, {}], "could not measure"),
        ([{"image": 64}, {"image": 16}], "disagree"),
    ],
    ids=["no-encoder", "disagree"],
)
def test_producer_refuses_to_start_without_a_trustworthy_width(reports, match):
    config = create_ec_vllm_config(ec_role="ec_producer", encoder_output_widths={})
    with pytest.raises(ValueError, match=match):
        eow.measure_encoder_output_widths(config, lambda fn: reports)
