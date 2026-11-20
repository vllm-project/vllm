# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MXFP8 producers that feed DeepSeek-V4.1's MXFP8 linears on gfx950
instead of a separate `mxfp8_e4m3_quantize` launch: the Mega-MHC seam
(`fused_wqa_wkv`, shared expert `gate_up_proj`) and the fused SiLU-and-mul
(shared expert `down_proj`). Each must be bit-identical to the unfused path.
"""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def _on_cdna4() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 4


pytestmark = pytest.mark.skipif(not _on_cdna4(), reason="MXFP8 on CDNA4")


def _assert_same_mxfp8(values, scales, ref_values, ref_scales):
    torch.testing.assert_close(
        values.view(torch.uint8), ref_values.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(scales, ref_scales.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [1, 17, 300, 5000])
@pytest.mark.parametrize("d", [576, 2304])
@pytest.mark.parametrize("swiglu_limit", [None, 10.0])
def test_silu_and_mul_mxfp8_matches_act_then_quant(num_tokens, d, swiglu_limit):
    from vllm.models.deepseek_v4.amd.silu_and_mul_mxfp8 import silu_and_mul_mxfp8

    set_random_seed(0)
    x = (torch.randn(num_tokens, 2 * d, device="cuda") * 4).to(torch.bfloat16)
    with set_current_vllm_config(VllmConfig()):
        act = (
            SiluAndMul() if swiglu_limit is None else SiluAndMulWithClamp(swiglu_limit)
        )
    ref_values, ref_scales = mxfp8_e4m3_quantize(act(x))
    values, scales = silu_and_mul_mxfp8(x, swiglu_limit)
    assert scales.shape == (num_tokens, d // 32)
    _assert_same_mxfp8(values, scales, ref_values, ref_scales)


@pytest.mark.parametrize("num_tokens", [1, 300, 4096])
@pytest.mark.parametrize("with_post", [False, True])
def test_mhc_pre_delayed_mxfp8_matches_mega_mhc_seam(num_tokens, with_post):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.mhc import (
        HAS_AITER_MHC_FUSED_POST_PRE_DELAYED_RMS_NORM,
        MHCPreDelayedOp,
    )

    if not HAS_AITER_MHC_FUSED_POST_PRE_DELAYED_RMS_NORM:
        pytest.skip("AITER fused mHC seam unavailable")
    pytest.importorskip("aiter.ops.mega_mhc")
    from vllm.models.deepseek_v4.amd.model import mxfp8_input
    from vllm.models.deepseek_v41.amd.mhc import mhc_pre_delayed_mxfp8

    set_random_seed(0)
    hidden, hc_mult = 5120, 4
    mix = (2 + hc_mult) * hc_mult
    residual = torch.randn(
        num_tokens, hc_mult, hidden, dtype=torch.bfloat16, device="cuda"
    )
    fn = torch.randn(mix, hc_mult * hidden, device="cuda") * 0.01
    args = (
        residual,
        fn,
        torch.rand(3, device="cuda"),
        torch.randn(mix, device="cuda") * 0.1,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
    )
    kwargs = dict(
        pre_mix=torch.softmax(torch.randn(num_tokens, hc_mult, device="cuda"), -1),
        norm_weight=(torch.rand(hidden, device="cuda") + 0.5).to(torch.bfloat16),
        norm_eps=1e-6,
        fn_packed=rocm_aiter_ops.mega_mhc_pack_fn(fn),
    )
    if with_post:
        kwargs.update(
            sublayer_out=torch.randn(
                num_tokens, hidden, dtype=torch.bfloat16, device="cuda"
            ),
            post_layer_mix=torch.rand(num_tokens, hc_mult, 1, device="cuda"),
            comb_res_mix=torch.rand(num_tokens, hc_mult, hc_mult, device="cuda"),
        )
    with set_current_vllm_config(VllmConfig()):
        pre_delayed = MHCPreDelayedOp()
    ref = pre_delayed(*args, **kwargs)
    out = mhc_pre_delayed_mxfp8(pre_delayed, *args, **kwargs)
    for a, b in zip(out, ref):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    qa = mxfp8_input(out[3])
    assert qa is not None and mxfp8_input(ref[3]) is None
    _assert_same_mxfp8(qa.data, qa.scale, *mxfp8_e4m3_quantize(ref[3]))
