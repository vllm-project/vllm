# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.models.deepseek_v32.nvidia.model import DeepseekV32ForCausalLM


class _BackboneBuilt(Exception):
    pass


@pytest.mark.cpu_test
def test_quant_config_sees_fused_qkv_a_proj(monkeypatch: pytest.MonkeyPatch):
    """The quant config must know `fused_qkv_a_proj` when the backbone builds its
    layers, or checkpoints that leave `q_a_proj`/`kv_a_proj_with_mqa` unquantized
    (e.g. nvidia/DeepSeek-V3.2-NVFP4) get a quantized fused layer and fail to load.
    """

    def capture_mapping(*, vllm_config, prefix):
        raise _BackboneBuilt(dict(vllm_config.quant_config.packed_modules_mapping))

    monkeypatch.setattr(
        DeepseekV32ForCausalLM, "model_cls", staticmethod(capture_mapping)
    )
    quant_config = Fp8Config(is_checkpoint_fp8_serialized=True)
    configure_quant_config(quant_config, DeepseekV32ForCausalLM)
    hf_config = SimpleNamespace(
        model_type="deepseek_v32",
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config), quant_config=quant_config
    )

    with pytest.raises(_BackboneBuilt) as built:
        DeepseekV32ForCausalLM(vllm_config=vllm_config)

    mapping = built.value.args[0]
    assert mapping["fused_qkv_a_proj"] == ["q_a_proj", "kv_a_proj_with_mqa"]
    assert mapping["wk_weights_proj"] == ["wk", "weights_proj"]
