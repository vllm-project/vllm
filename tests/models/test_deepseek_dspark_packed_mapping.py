# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

if not torch.cuda.is_available():
    torch.cuda.get_device_properties = lambda *args, **kwargs: SimpleNamespace(
        gcnArchName="gfx942"
    )

from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.models.deepseek_v4.amd.dspark import (
    DSparkDeepseekV4ForCausalLM as V4AMD,
)
from vllm.models.deepseek_v4.amd.model import (
    DeepseekV4ForCausalLM as TargetV4AMD,
)
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
from vllm.models.deepseek_v41.amd.dspark import (
    DSparkDeepseekV4ForCausalLM as V41AMD,
)
from vllm.models.deepseek_v41.amd.model import (
    DeepseekV41LLMForCausalLM as TargetV41AMD,
)
from vllm.models.deepseek_v41.nvidia import dspark as dsv41_dspark
from vllm.models.deepseek_v41.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM as V41Nvidia,
)
from vllm.models.deepseek_v41.nvidia.model import (
    DeepseekV41LLMForCausalLM as TargetV41Nvidia,
)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("model_cls", "target_cls"),
    [
        (V4AMD, TargetV4AMD),
        (V4Nvidia, TargetV4Nvidia),
        (V4XPU, TargetV4XPU),
        (V41AMD, TargetV41AMD),
        (V41Nvidia, TargetV41Nvidia),
    ],
    ids=["v4_amd", "v4_nvidia", "v4_xpu", "v41_amd", "v41_nvidia"],
)
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
@pytest.mark.parametrize("dspark_mod", [dsv4_dspark, dsv41_dspark], ids=["v4", "v41"])
@pytest.mark.parametrize("skip_wkv", [True, False])
def test_nvidia_dspark_context_wkv_proj_respects_ignored_layers(
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

    monkeypatch.setattr(dspark_mod, "VocabParallelEmbedding", DummyModule)
    monkeypatch.setattr(dspark_mod, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(dspark_mod, "RMSNorm", DummyModule)
    monkeypatch.setattr(dspark_mod, "DeepseekV4DecoderLayer", DummyModule)
    monkeypatch.setattr(
        dspark_mod, "MergedColumnParallelLinear", capture_merged_linear
    )
    monkeypatch.setattr(dspark_mod, "DSparkMarkovHead", DummyModule)
    monkeypatch.setattr(dspark_mod, "DSparkConfidenceHead", DummyModule)
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
