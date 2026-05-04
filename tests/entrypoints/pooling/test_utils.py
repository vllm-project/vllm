# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.responses import ORJSONResponse

from vllm.entrypoints.pooling.utils import encode_pooling_output_float_or_ndarray


def _pooling_output(data):
    return SimpleNamespace(outputs=SimpleNamespace(data=data))


def test_encode_pooling_output_float_or_ndarray_returns_numpy_array():
    output = _pooling_output(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))

    encoded = encode_pooling_output_float_or_ndarray(output)

    assert isinstance(encoded, np.ndarray)
    np.testing.assert_allclose(encoded, [1.0, 2.0, 3.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        response = ORJSONResponse(content={"embedding": encoded})
    assert json.loads(response.body)["embedding"] == pytest.approx([1.0, 2.0, 3.0])


def test_encode_pooling_output_float_or_ndarray_falls_back_to_list():
    class DataWithUnsupportedNumpy:
        def is_contiguous(self):
            return True

        def numpy(self):
            raise TypeError("unsupported dtype")

        def tolist(self):
            return [1.0, 2.0, 3.0]

    output = _pooling_output(DataWithUnsupportedNumpy())

    assert encode_pooling_output_float_or_ndarray(output) == [1.0, 2.0, 3.0]
