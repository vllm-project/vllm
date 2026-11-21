# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch.nn as nn

from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.models.deepseek_v4.nvidia import dspark as dsv4_dspark
from vllm.models.deepseek_v4.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM as V4Nvidia,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4ForCausalLM as TargetV4Nvidia,
)
from vllm.models.deepseek_v4.xpu.dspark import (
    DSparkDeepseekV4ForCausalLM as V4XPU,
)
from vllm.models.deepseek_v4.xpu.model import (
    DeepseekV4ForCausalLM as TargetV4XPU,
)
from vllm.models.deepseek_v32.nvidia.model import DeepseekV32ForCausalLM
from vllm.models.deepseek_v41.nvidia import dspark as dsv41_dspark
from vllm.models.deepseek_v41.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM as V41Nvidia,
)
from vllm.models.deepseek_v41.nvidia.model import (
    DeepseekV41LLMForCausalLM as TargetV41Nvidia,
)
from vllm.platforms import current_platform

PACKED_MAPPING_CASES = [
    pytest.param(V4Nvidia, TargetV4Nvidia, id="v4_nvidia"),
    pytest.param(V4XPU, TargetV4XPU, id="v4_xpu"),
    pytest.param(V41Nvidia, TargetV41Nvidia, id="v41_nvidia"),
]
CONTEXT_WKV_CASES = [
    pytest.param(dsv4_dspark, id="v4"),
    pytest.param(dsv41_dspark, id="v41"),
]

# Importing the ROCm platform on x86 CPU loads a second _C extension.
if current_platform.is_rocm():
    from vllm.models.deepseek_v4.amd import dspark as dsv4_amd_dspark
    from vllm.models.deepseek_v4.amd.dspark import (
        DSparkDeepseekV4ForCausalLM as V4AMD,
    )
    from vllm.models.deepseek_v4.amd.model import (
        DeepseekV4ForCausalLM as TargetV4AMD,
    )
    from vllm.models.deepseek_v41.amd import dspark as dsv41_amd_dspark
    from vllm.models.deepseek_v41.amd.dspark import (
        DSparkDeepseekV4ForCausalLM as V41AMD,
    )
    from vllm.models.deepseek_v41.amd.model import (
        DeepseekV41LLMForCausalLM as TargetV41AMD,
    )

    PACKED_MAPPING_CASES.extend(
        [
            pytest.param(V4AMD, TargetV4AMD, id="v4_amd"),
            pytest.param(V41AMD, TargetV41AMD, id="v41_amd"),
        ]
    )
    CONTEXT_WKV_CASES.extend(
        [
            pytest.param(dsv4_amd_dspark, id="v4_amd"),
            pytest.param(dsv41_amd_dspark, id="v41_amd"),
        ]
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(("model_cls", "target_cls"), PACKED_MAPPING_CASES)
def test_deepseek_dspark_packed_modules_mapping(model_cls, target_cls):
    """Verify that all DSpark draft model entry points reuse their target model's
    packed_modules_mapping so configure_quant_config() wires it into quant_config
    and is_layer_skipped() expands fused prefixes to their unfused shard names.
    Fixes #65618.
    """
    assert model_cls.packed_modules_mapping == target_cls.packed_modules_mapping

    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
        ignored_layers=[
            "model.layers.32.attn.wq_a",
            "model.layers.32.attn.wkv",
            "model.layers.32.ffn.shared_experts.w1",
            "model.layers.32.ffn.shared_experts.w3",
        ],
    )
    configure_quant_config(quant_config, model_cls)
    assert quant_config.packed_modules_mapping == target_cls.packed_modules_mapping

    for fused_prefix in (
        "model.layers.32.attn.fused_wqa_wkv",
        "model.layers.32.ffn.shared_experts.gate_up_proj",
    ):
        assert is_layer_skipped(
            prefix=fused_prefix,
            ignored_layers=quant_config.ignored_layers,
            fused_mapping=quant_config.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize("dspark_mod", CONTEXT_WKV_CASES)
@pytest.mark.parametrize("skip_wkv", [True, False])
def test_dspark_context_wkv_proj_respects_ignored_layers(
    monkeypatch: pytest.MonkeyPatch,
    dspark_mod,
    skip_wkv: bool,
):
    captured_quant_configs = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    def capture_merged_linear(*args, **kwargs):
        captured_quant_configs.append(kwargs.get("quant_config"))
        return DummyModule()

    for name in ("VocabParallelEmbedding", "HCHeadOp"):
        if hasattr(dspark_mod, name):
            monkeypatch.setattr(dspark_mod, name, DummyModule)
    monkeypatch.setattr(dspark_mod, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(dspark_mod, "RMSNorm", DummyModule)
    monkeypatch.setattr(dspark_mod, "DeepseekV4DecoderLayer", DummyModule)
    monkeypatch.setattr(dspark_mod, "MergedColumnParallelLinear", capture_merged_linear)
    monkeypatch.setattr(dspark_mod, "DSparkMarkovHead", DummyModule)
    monkeypatch.setattr(dspark_mod, "DSparkConfidenceHead", DummyModule)
    if hasattr(dspark_mod, "_use_sequence_parallel"):
        monkeypatch.setattr(dspark_mod, "_use_sequence_parallel", lambda _: False)
    monkeypatch.setattr(
        dspark_mod, "get_current_vllm_config", lambda: SimpleNamespace()
    )
    if hasattr(dspark_mod, "maybe_init_gemm_rs"):
        monkeypatch.setattr(dspark_mod, "maybe_init_gemm_rs", lambda *a, **kw: False)

    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
        ignored_layers=["model.layers.30.attn.wkv"] if skip_wkv else [],
    )
    configure_quant_config(quant_config, dspark_mod.DSparkDeepseekV4ForCausalLM)

    hf_config = SimpleNamespace(
        hidden_size=64,
        hc_mult=2,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        num_hidden_layers=30,
        dspark_target_layer_ids=[0, 1],
        n_mtp_layers=3,
        vocab_size=128,
        draft_vocab_size=128,
        index_topk=4,
        head_dim=16,
        dspark_markov_rank=8,
        enable_confidence_head=False,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_config)
        ),
        kernel_config=SimpleNamespace(moe_backend="auto"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        quant_config=quant_config,
    )

    dspark_mod.DSparkDeepseekV4Model(vllm_config=vllm_config, prefix="model")

    assert len(captured_quant_configs) == 1
    assert captured_quant_configs[0] is (None if skip_wkv else quant_config)


class _BackboneBuilt(Exception):
    pass


@pytest.mark.cpu_test
def test_deepseek_v32_quant_config_sees_fused_qkv_a_proj(
    monkeypatch: pytest.MonkeyPatch,
):
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
