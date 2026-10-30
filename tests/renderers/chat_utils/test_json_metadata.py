# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.exceptions import VLLMValidationError
from vllm.renderers.chat_utils.content_parser import _load_embeds_dict
from vllm.renderers.chat_utils.item_tracker import _parse_metadata_array


@pytest.mark.parametrize("values", [[1, 32, 48], [0.0, 0.12345678912345678]])
def test_json_metadata_preserves_numeric_precision(values):
    tensor = _parse_metadata_array("grid", values, {"grid"})
    assert tensor.tolist() == values
    assert tensor.dtype == (
        torch.float64 if isinstance(values[0], float) else torch.int64
    )


@pytest.mark.parametrize(
    "key,values",
    [
        ("image_embeds", [1, 2]),
        ("grid", [True]),
        ("grid", [float("nan")]),
        ("grid", [float("inf")]),
        ("grid", [[1, 2]]),
        ("grid", [2**64]),
    ],
)
def test_json_arrays_are_only_valid_numeric_metadata(key, values):
    with pytest.raises(VLLMValidationError):
        _parse_metadata_array(key, values, {"grid"})


@pytest.mark.asyncio
async def test_json_metadata_bypasses_tensor_deserialization():
    from unittest.mock import AsyncMock

    tensor = torch.ones(2, 3)
    fetch = AsyncMock(return_value=tensor)
    result = await _load_embeds_dict(
        {"image_embeds": "legacy-base64", "image_grid_thw": [1, 2, 2]}, fetch
    )
    fetch.assert_awaited_once_with("legacy-base64")
    assert result["image_embeds"] is tensor
    assert result["image_grid_thw"] == [1, 2, 2]
