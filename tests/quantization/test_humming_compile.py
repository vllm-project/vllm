# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

pytestmark = pytest.mark.skipif(
    not (has_humming() and current_platform.is_cuda()),
    reason="requires humming and CUDA",
)


def test_humming_input_quant_compiles_fullgraph():
    """Humming's forward helpers look up ``DeviceInfo`` on every call; vLLM
    traces them with ``fullgraph=True`` (vllm-project/humming#147)."""
    import vllm.utils.humming as hm

    weight_schema = hm.HummingWeightSchema(b_dtype="int4", weight_scale_group_size=128)
    input_schema = hm.HummingInputSchema(a_dtype="int8")
    from humming.transform import prepare_layer_config

    config = prepare_layer_config(
        shape_n=256,
        shape_k=256,
        weight_schema=weight_schema,
        input_schema=input_schema,
        torch_dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    x = torch.randn(16, 256, dtype=torch.bfloat16, device="cuda")

    def quant(t: torch.Tensor):
        return hm.may_process_input(config, t)

    expected = quant(x)
    actual = torch.compile(quant, fullgraph=True)(x)
    for a, e in zip(actual, expected):
        if e is None:
            assert a is None
        else:
            torch.testing.assert_close(a, e)
