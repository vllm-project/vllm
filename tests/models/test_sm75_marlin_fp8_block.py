import pytest
import torch

from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 7,
    reason="SM75 only",
)
def test_sm75_selects_marlin_block_fp8(default_vllm_config):
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
        weight_shape=(2048, 2048),
        input_dtype=torch.float16,
        out_dtype=torch.float16,
    )
    supported, _ = MarlinFP8ScaledMMLinearKernel.is_supported()
    assert supported is True
    can_implement, _ = MarlinFP8ScaledMMLinearKernel.can_implement(config)
    assert can_implement is True
    kernel = MarlinFP8ScaledMMLinearKernel(
        config, ["weight", "weight_scale", "input_scale", "input_scale_ub"]
    )
    assert isinstance(kernel, FP8ScaledMMLinearKernel)

    selected = init_fp8_linear_kernel(
        activation_quant_key=kFp8Dynamic128Sym,
        weight_quant_key=kFp8Static128BlockSym,
        input_dtype=torch.float16,
        out_dtype=torch.float16,
        weight_shape=(2048, 2048),
    )
    assert isinstance(selected, MarlinFP8ScaledMMLinearKernel)
