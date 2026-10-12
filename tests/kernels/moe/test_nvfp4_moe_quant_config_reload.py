# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    make_nvfp4_moe_quant_config,
)


def test_input_gscale_storage_is_stable_across_rebuilds():
    """Rebuilding the quant config (as every reload does) must update the input
    global scales in the storage captured by the first config."""
    layer = torch.nn.Module()
    scale = torch.ones(4, 4, 4, dtype=torch.float8_e4m3fn)

    def build(input_scale):
        return make_nvfp4_moe_quant_config(
            backend=NvFp4MoeBackend.FLASHINFER_TRTLLM,
            w13_scale=scale,
            w2_scale=scale,
            w13_scale_2=torch.ones(4),
            w2_scale_2=torch.ones(4),
            a13_scale=torch.full((4,), input_scale),
            a2_scale=torch.full((4,), input_scale),
            layer=layer,
        )

    first = build(2.0)
    second = build(8.0)
    assert second.a1_gscale.data_ptr() == first.a1_gscale.data_ptr()
    assert second.a2_gscale.data_ptr() == first.a2_gscale.data_ptr()
    torch.testing.assert_close(first.a1_gscale, torch.full((4,), 1 / 8.0))
