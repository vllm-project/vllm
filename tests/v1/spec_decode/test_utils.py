# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.utils import first_slot_mapping_if_ubatched


def test_first_slot_mapping_if_ubatched_passthrough_dict():
    slot_mapping = {"layer.0": torch.zeros(4, dtype=torch.int64)}
    assert first_slot_mapping_if_ubatched(slot_mapping) is slot_mapping


def test_first_slot_mapping_if_ubatched_passthrough_none():
    assert first_slot_mapping_if_ubatched(None) is None


def test_first_slot_mapping_if_ubatched_takes_first_ubatch():
    # When the target model runs with DBO ubatching, the model runner
    # provides one slot-mapping dict per ubatch.
    ubatch0 = {"layer.0": torch.zeros(4, dtype=torch.int64)}
    ubatch1 = {"layer.0": torch.ones(4, dtype=torch.int64)}
    assert first_slot_mapping_if_ubatched([ubatch0, ubatch1]) is ubatch0


def test_first_slot_mapping_if_ubatched_empty_list():
    assert first_slot_mapping_if_ubatched([]) is None
