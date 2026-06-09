from types import SimpleNamespace

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.models.deepseek_v4 import quant_config as deepseek_v4_quant_config
from vllm.models.deepseek_v4.nvidia import model as deepseek_v4_model
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4MixtureOfExperts,
    DeepseekV4MoE,
)


def test_deepseek_v4_fused_moe_metadata_is_available_to_mixture():
    moe = object.__new__(DeepseekV4MoE)
    moe.n_routed_experts = 256
    moe.n_shared_experts = 1
    moe.experts = SimpleNamespace(
        logical_num_experts=256,
        global_num_experts=256,
        local_num_experts=128,
    )
    moe._sync_fused_moe_metadata()

    mixture = object.__new__(DeepseekV4MixtureOfExperts)
    mixture.extract_moe_parameters(moe)

    assert mixture.num_logical_experts == 256
    assert mixture.num_physical_experts == 256
    assert mixture.num_local_physical_experts == 128
    assert mixture.num_routed_experts == 256
    assert mixture.num_shared_experts == 1
    assert mixture.num_redundant_experts == 0


def test_deepseek_v4_fused_moe_metadata_handles_moe_runner_shape():
    moe = object.__new__(DeepseekV4MoE)
    moe.n_routed_experts = 256
    moe.n_shared_experts = 1
    moe.experts = SimpleNamespace(
        moe_config=SimpleNamespace(
            num_logical_experts=256,
            num_experts=256,
            num_local_experts=128,
        ),
        routed_experts=SimpleNamespace(
            global_num_experts=256,
            local_num_experts=128,
        ),
    )

    moe._sync_fused_moe_metadata()

    assert moe.n_logical_experts == 256
    assert moe.n_physical_experts == 256
    assert moe.n_local_physical_experts == 128
    assert moe.n_local_experts == 128
    assert moe.n_redundant_experts == 0


def test_deepseek_v4_fp4_quant_config_handles_routed_experts_after_moe_refactor(
    monkeypatch,
):
    class FakeMxfp4MoEMethod:
        def __init__(self, moe_config):
            self.moe_config = moe_config

    quant_config = deepseek_v4_quant_config.DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
    )
    layer = object.__new__(RoutedExperts)
    layer.moe_config = object()

    monkeypatch.setattr(
        deepseek_v4_quant_config,
        "Mxfp4MoEMethod",
        FakeMxfp4MoEMethod,
    )
    monkeypatch.setattr(
        deepseek_v4_quant_config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    expert_dtype="fp4",
                    quantization_config={},
                )
            )
        ),
    )

    method = quant_config.get_quant_method(layer, "model.layers.3.mlp.experts")

    assert isinstance(method, FakeMxfp4MoEMethod)
    assert method.moe_config is layer.moe_config


def test_deepseek_v4_fused_moe_init_exports_moe_metadata(monkeypatch):
    class FakeGate:
        def __init__(self, *args, **kwargs):
            self.e_score_correction_bias = None
            self.tid2eid = None

    class FakeFusedMoE:
        logical_num_experts = 256
        global_num_experts = 256
        local_num_experts = 128

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeMLP:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(deepseek_v4_model, "GateLinear", FakeGate)
    monkeypatch.setattr(deepseek_v4_model, "DeepseekV4MLP", FakeMLP)
    monkeypatch.setattr(deepseek_v4_model, "FusedMoE", FakeFusedMoE)
    monkeypatch.setattr(
        deepseek_v4_model,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        deepseek_v4_model,
        "get_tensor_model_parallel_rank",
        lambda: 1,
    )

    config = SimpleNamespace(
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=8,
        hidden_size=7168,
        moe_intermediate_size=2048,
        swiglu_limit=7.0,
        hidden_act="silu",
        norm_topk_prob=True,
        num_hash_layers=0,
        vocab_size=128000,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
        kernel_config=SimpleNamespace(moe_backend="auto"),
        parallel_config=SimpleNamespace(enable_expert_parallel=True),
    )

    moe = DeepseekV4MoE(vllm_config, prefix="model.layers.3.mlp")

    assert moe.n_logical_experts == 256
    assert moe.n_physical_experts == 256
    assert moe.n_local_physical_experts == 128
    assert moe.n_local_experts == 128
    assert moe.n_shared_experts == 1
    assert moe.n_redundant_experts == 0
