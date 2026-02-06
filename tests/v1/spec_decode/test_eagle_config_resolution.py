# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.eagle import EagleProposer


def test_dflash_use_aux_hidden_state_resolution():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.method = "dflash"
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            eagle_config={"use_aux_hidden_state": True},
            dflash_config={"use_aux_hidden_state": False},
        )
    )
    assert proposer._get_eagle3_use_aux_hidden_state_from_config() is False


def test_eagle3_use_aux_hidden_state_resolution():
    proposer = EagleProposer.__new__(EagleProposer)
    proposer.method = "eagle3"
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(eagle_config={"use_aux_hidden_state": False})
    )
    assert proposer._get_eagle3_use_aux_hidden_state_from_config() is False
