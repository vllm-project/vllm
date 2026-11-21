# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.layer_layout import (
    LayerObjectLayout,
    LayerSlot,
)


def test_pack_places_layers_back_to_back():
    layout = LayerObjectLayout.pack([("layer0", 64), ("layer1", 32), ("layer2", 64)])
    assert layout.slots == {
        "layer0": LayerSlot(0, 64),
        "layer1": LayerSlot(64, 32),
        "layer2": LayerSlot(96, 64),
    }
    assert list(layout.slots) == ["layer0", "layer1", "layer2"]
    assert layout.size == 160


@pytest.mark.parametrize(
    ("layer_sizes", "match"),
    [
        ([("layer0", 64), ("layer0", 64)], "twice"),
        ([("layer0", 0)], "non-positive"),
    ],
)
def test_pack_rejects_invalid_layers(layer_sizes, match):
    with pytest.raises(ValueError, match=match):
        LayerObjectLayout.pack(layer_sizes)
