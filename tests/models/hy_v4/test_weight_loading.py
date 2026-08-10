# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-mapping guards for `vllm.models.hy_v4`.

The port drops the reference implementation's fused-MLA path, which changed the
indexer's ``wk``/``weights_proj`` merge from a conditional to an unconditional
``stacked_params_mapping`` entry and removed a ``packed_modules_mapping`` entry.
A mistake there does not raise: weights silently keep their initialization
sentinel. These tests therefore assert that a synthetic checkpoint in the
HY V4 naming scheme assigns a value to *every* model parameter.
"""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.configs.hy_v4 import HYV4Config

NUM_LAYERS = 3


def _hf_config(*, enable_ihc: bool, sparse: bool) -> HYV4Config:
    """A minimal HY V4 config whose MLA dims a real backend accepts."""
    config = HYV4Config(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=512,
        q_lora_rank=128,
        kv_lora_rank=512 if sparse else 256,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_routed_experts=8,
        n_shared_experts=1,
        moe_intermediate_size=64,
        num_experts_per_tok=2,
        index_topk=64,
        index_head_dim=128,
        index_n_heads=4,
        indexer_types=["full"] * NUM_LAYERS if sparse else None,
        layer_types=["sparse" if sparse else "full_attention"] * NUM_LAYERS,
        # Cover both a dense and a MoE decoder layer.
        mlp_layer_types=["dense"] + ["sparse"] * (NUM_LAYERS - 1),
        enable_ihc=enable_ihc,
        hc_mult=2,
        learnable_sink=True,
        gated_mla=True,
        gating_type="elementwise",
        swiglu_limit=10.0,
        enable_lm_head_fp32=False,
        num_nextn_predict_layers=1,
        torch_dtype="bfloat16",
    )
    config.architectures = ["HYV4ForCausalLM"]
    return config


def _model_config(tmp_path, hf_config: HYV4Config) -> ModelConfig:
    """Round-trip the config through disk to exercise the config registry."""
    path = str(tmp_path)
    hf_config.save_pretrained(path)
    model_config = ModelConfig(
        model=path,
        tokenizer=path,
        trust_remote_code=False,
        dtype="bfloat16",
        seed=0,
        skip_tokenizer_init=True,
        enforce_eager=True,
    )
    assert isinstance(model_config.hf_config, HYV4Config)
    return model_config


def _split_weights(
    name: str,
    shape: tuple[int, ...],
    hf_config: HYV4Config,
) -> list[tuple[str, torch.Tensor]]:
    """Invert one parameter name into the checkpoint tensors that feed it."""
    if name.endswith(".gate_up_proj.weight"):
        base = name[: -len(".gate_up_proj.weight")]
        half = shape[0] // 2
        return [
            (f"{base}.gate_proj.weight", torch.randn(half, *shape[1:])),
            (f"{base}.up_proj.weight", torch.randn(half, *shape[1:])),
        ]
    if name.endswith("indexer.wk_weights_proj.weight"):
        base = name[: -len(".wk_weights_proj.weight")]
        head_dim = hf_config.index_head_dim
        return [
            (f"{base}.wk.weight", torch.randn(head_dim, *shape[1:])),
            (
                f"{base}.weights_proj.weight",
                torch.randn(shape[0] - head_dim, *shape[1:]),
            ),
        ]
    if name.endswith("w13_weight"):
        base = name.split(".experts.")[0]
        inter = shape[1] // 2
        out = []
        for expert in range(hf_config.n_routed_experts):
            out.append(
                (
                    f"{base}.experts.{expert}.gate_proj.weight",
                    torch.randn(inter, shape[2]),
                )
            )
            out.append(
                (
                    f"{base}.experts.{expert}.up_proj.weight",
                    torch.randn(inter, shape[2]),
                )
            )
        return out
    if name.endswith("w2_weight"):
        base = name.split(".experts.")[0]
        return [
            (
                f"{base}.experts.{expert}.down_proj.weight",
                torch.randn(shape[1], shape[2]),
            )
            for expert in range(hf_config.n_routed_experts)
        ]
    if name.endswith(".expert_bias"):
        base = name[: -len(".expert_bias")]
        return [(f"{base}.gate.e_score_correction_bias", torch.randn(*shape))]
    if name.endswith(".hc_fn.weight") or name.endswith(".hc_head_fn.weight"):
        # The checkpoint stores the iHC gate projections without ".weight".
        return [(name[: -len(".weight")], torch.randn(*shape))]
    if name.endswith(".learnable_sink_param"):
        # The checkpoint holds every head; the loader narrows to the TP shard.
        return [(name, torch.randn(hf_config.num_attention_heads))]
    return [(name, torch.randn(*shape))]


def _assert_fully_loaded(model, checkpoint) -> None:
    expected = {name for name, _ in model.named_parameters()}
    loaded = model.load_weights(
        (name, tensor.to("cuda", torch.bfloat16)) for name, tensor in checkpoint
    )
    assert not expected - loaded, (
        f"parameters never loaded: {sorted(expected - loaded)}"
    )
    assert not loaded - expected, f"loaded unknown params: {sorted(loaded - expected)}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("enable_ihc", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_backbone_load_weights_covers_every_parameter(
    tmp_path, dist_init, enable_ihc: bool, sparse: bool
) -> None:
    from vllm.models.hy_v4 import HYV4ForCausalLM

    torch.set_default_dtype(torch.bfloat16)
    hf_config = _hf_config(enable_ihc=enable_ihc, sparse=sparse)
    vllm_config = VllmConfig(model_config=_model_config(tmp_path, hf_config))

    with set_current_vllm_config(vllm_config), torch.device("cuda"):
        model = HYV4ForCausalLM(vllm_config=vllm_config, prefix="")

    checkpoint: list[tuple[str, torch.Tensor]] = []
    for name, param in model.named_parameters():
        checkpoint.extend(_split_weights(name, tuple(param.shape), hf_config))

    with set_current_vllm_config(vllm_config):
        _assert_fully_loaded(model, checkpoint)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("sparse", [False, True])
def test_mtp_load_weights_covers_every_parameter(
    tmp_path, dist_init, sparse: bool
) -> None:
    from vllm.models.hy_v4 import HYV4MTP

    torch.set_default_dtype(torch.bfloat16)
    hf_config = _hf_config(enable_ihc=True, sparse=sparse)
    model_config = _model_config(tmp_path, hf_config)
    spec_config = SpeculativeConfig(
        method="hy_v4_mtp",
        model=model_config.model,
        num_speculative_tokens=1,
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        enforce_eager=True,
    )
    vllm_config = VllmConfig(model_config=model_config, speculative_config=spec_config)

    with set_current_vllm_config(vllm_config), torch.device("cuda"):
        mtp = HYV4MTP(vllm_config=vllm_config, prefix="")

    mtp_prefix = f"model.layers.{hf_config.num_hidden_layers}."
    checkpoint: list[tuple[str, torch.Tensor]] = []
    for name, param in mtp.named_parameters():
        if name == "model.embed_tokens.weight":
            ckpt_name = name
        else:
            assert name.startswith(mtp_prefix), name
            tail = name[len(mtp_prefix) :]
            if tail.startswith("shared_head.head."):
                ckpt_name = "lm_head.weight"
            else:
                ckpt_name = f"model.mtp_layers.0.{tail.removeprefix('mtp_block.')}"
        checkpoint.extend(_split_weights(ckpt_name, tuple(param.shape), hf_config))

    with set_current_vllm_config(vllm_config):
        _assert_fully_loaded(mtp, checkpoint)
