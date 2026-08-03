import pytest
import torch

import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as mxfp4_oracle
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Static


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7,
    reason="SM75 only",
)
def test_sm75_marlin_mxfp4_selected(default_vllm_config, monkeypatch):
    moe = FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=1024,
        intermediate_size=512,
        num_local_experts=8,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.float16,
        device="cuda",
        routing_method=RoutingMethodType.Default,
    )
    supported, reason = MarlinExperts.is_supported_config(
        MarlinExperts, moe, kMxfp4Static, None, FusedMoEActivationFormat.Standard
    )
    assert supported is True, reason

    # default_vllm_config has model_config=None; the real override would raise
    monkeypatch.setattr(mxfp4_oracle, "_user_moe_activation_override", lambda: None)
    backend, kcls = select_mxfp4_moe_backend(moe)
    assert backend == Mxfp4MoeBackend.MARLIN
    assert kcls is MarlinExperts
