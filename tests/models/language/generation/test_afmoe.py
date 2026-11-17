# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F

from tests.quantization.utils import is_quant_method_supported
from vllm.model_executor.models.afmoe import AfmoeMoE
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA model-loading regression"
)
@pytest.mark.parametrize(
    ("dtype", "quantization"),
    [("bfloat16", None), ("float16", None), ("float32", None), ("bfloat16", "fp8")],
)
def test_afmoe_loaded_precision(vllm_runner, dtype, quantization):
    if quantization is not None and not is_quant_method_supported(quantization):
        pytest.skip(f"{quantization} is unsupported on this GPU")

    def check_precision(model):
        model_dtype = getattr(torch, dtype)
        router_dtype = (
            torch.bfloat16
            if dtype == "bfloat16" and quantization is None
            else torch.float32
        )
        routers = 0
        for module in model.modules():
            if isinstance(module, AfmoeMoE):
                assert module.gate.weight.dtype == router_dtype
                assert module.expert_bias.dtype == torch.float32
                for tokens in (1, 17, 33):
                    inputs = torch.linspace(
                        -1,
                        1,
                        tokens * module.gate.weight.shape[1],
                        device=module.gate.weight.device,
                        dtype=model_dtype,
                    ).view(tokens, -1)
                    logits, _ = module.gate(inputs)
                    assert logits.dtype == router_dtype
                    assert torch.isfinite(logits).all()
                    if router_dtype == torch.bfloat16:
                        expected = F.linear(inputs, module.gate.weight)
                        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
                routers += 1
        assert routers == 54
        assert model.model.norm.weight.dtype == model_dtype

    with vllm_runner(
        "arcee-ai/Trinity-Nano-Preview",
        dtype=dtype,
        quantization=quantization,
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=1,
    ) as runner:
        runner.apply_model(check_precision)
        outputs = runner.generate_greedy(["The capital of France is"], max_tokens=8)
        assert len(outputs) == 1 and outputs[0][0]
