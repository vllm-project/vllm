# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online FP8 weights of the Muse Spark DSpark draft on ROCm.

`speculative_config.quantization="fp8_per_channel"` (`fc` and the decoder
linears) and `draft_lm_head_quantization="fp8"` (the draft's own lm_head and
Markov head): a tiny draft checkpoint loaded through the real loader against the
same checkpoint in BF16.
"""

import json

import pytest
import torch

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.online.lm_head import (
    lm_head_padding_size,
    lm_head_quant_method,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    pad_vocab_size,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

from .test_muse_spark_rocm_mono import (
    DSPARK_TARGETS,
    TINY_TEXT,
    _checkpoint_config,
    requires_gfx950,
)

# The module-scoped drafts own the distributed environment.
pytestmark = [pytest.mark.skip_global_cleanup, requires_gfx950]

F8 = current_platform.fp8_dtype()

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, RANK, VOCAB = 512, 8, 2, 64, 256, 512
# The checkpoint config of Muse-Spark-Open-DSpark at a tiny width.
TINY_DSPARK = {
    "architectures": ["MuseSparkDSparkModel"],
    "model_type": "muse_spark_dspark",
    "dtype": "bfloat16",
    "draft_position_mode": "rope",
    "ffn_dim_multiplier": 2.0,
    "head_dim": HEAD_DIM,
    "hidden_size": HIDDEN,
    "layer_types": ["sliding_attention"] * 2,
    "max_position_embeddings": 4096,
    "metap": {
        "base_width": 256.0,
        "m_emb": 1.0,
        "metap_mode": "sp",
        "router_has_one_inf_dim": True,
        "use_metap": True,
        "use_metap_output_multiplier_without_metap": False,
    },
    "moe": {"is_enabled": False},
    "norm_affine": True,
    "norm_zero_centered_gamma": True,
    "num_attention_heads": HEADS,
    "num_hidden_layers": 2,
    "num_key_value_heads": KV_HEADS,
    "output_soft_cap_temp": 20.0,
    "post_norm_affine": False,
    "residual_dtype": "fp32",
    "rms_post_norm_eps": 1e-08,
    "sliding_window": 2048,
    "global_attn_cfg": [2048],
    "use_attn_o_gate": True,
    "use_attn_o_norm": True,
    "use_ffn_learnable_scales": True,
    "use_residual_gate": True,
    "use_rope": True,
    "vocab_size": VOCAB,
    "draft_vocab_size": VOCAB,
    "mask_token_id": 3,
    "pad_token_id": 3,
    "target_hidden_size": TINY_TEXT["hidden_size"],
    "target_num_hidden_layers": TINY_TEXT["num_hidden_layers"],
    "num_target_layers": len(DSPARK_TARGETS),
    "target_layer_ids": DSPARK_TARGETS,
    "markov_rank": RANK,
    "markov_head_type": "vanilla",
    "enable_confidence_head": True,
    "confidence_head_with_markov": True,
    "sample_from_anchor": True,
    "block_size": 7,
    "dflash_config": {
        "causal": True,
        "mask_token_id": 3,
        "target_layer_ids": DSPARK_TARGETS,
        "use_aux_hidden_state": True,
    },
}
FP8_DRAFT = {"quantization": "fp8_per_channel", "draft_lm_head_quantization": "fp8"}


def _checkpoint_tensors() -> dict[str, torch.Tensor]:
    """The tensor names and shapes of the Muse-Spark-Open-DSpark checkpoint."""
    q, kv, ffn = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, 2 * HIDDEN
    shapes = {
        "fc.weight": (HIDDEN, TINY_TEXT["hidden_size"] * len(DSPARK_TARGETS)),
        "hidden_norm.weight": (HIDDEN,),
        "norm.weight": (HIDDEN,),
        "lm_head.weight": (VOCAB, HIDDEN),
        "markov_head.markov_w1.weight": (VOCAB, RANK),
        "markov_head.markov_w2.weight": (VOCAB, RANK),
        "confidence_head.proj.weight": (1, HIDDEN + RANK),
        "confidence_head.proj.bias": (1,),
    }
    for i in range(2):
        layer = f"layers.{i}"
        shapes |= {
            f"{layer}.add_attn.gate": (HIDDEN,),
            f"{layer}.add_ffn.gate": (HIDDEN,),
            f"{layer}.pre_ffn_norm.weight": (HIDDEN,),
            f"{layer}.post_ffn_norm.weight": (HIDDEN,),
            f"{layer}.self_attn.pre_qkv_proj_norm.weight": (HIDDEN,),
            f"{layer}.self_attn.q_proj.weight": (q, HIDDEN),
            f"{layer}.self_attn.k_proj.weight": (kv, HIDDEN),
            f"{layer}.self_attn.v_proj.weight": (kv, HIDDEN),
            f"{layer}.self_attn.o_gate_proj.weight": (q, HIDDEN),
            f"{layer}.self_attn.o_proj.weight": (HIDDEN, q),
            f"{layer}.feed_forward.gate_proj.weight": (ffn, HIDDEN),
            f"{layer}.feed_forward.up_proj.weight": (ffn, HIDDEN),
            f"{layer}.feed_forward.down_proj.weight": (HIDDEN, ffn),
        }
    gen = torch.Generator().manual_seed(0)
    tensors = {
        name: (torch.randn(shape, generator=gen) * 0.05).to(torch.bfloat16)
        for name, shape in shapes.items()
    }
    tensors["d2t"] = torch.zeros(VOCAB, dtype=torch.long)
    return tensors


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    from safetensors.torch import save_file

    root = tmp_path_factory.mktemp("muse_spark_draft")
    (root / "target").mkdir()
    (root / "target" / "config.json").write_text(
        json.dumps(_checkpoint_config(TINY_TEXT))
    )
    (root / "dspark").mkdir()
    (root / "dspark" / "config.json").write_text(json.dumps(TINY_DSPARK))
    save_file(_checkpoint_tensors(), str(root / "dspark" / "model.safetensors"))
    return root


def _vllm_config(checkpoints, **spec) -> VllmConfig:
    model_config = ModelConfig(
        model=str(checkpoints / "target"),
        tokenizer_mode="skip",
        dtype="bfloat16",
        max_model_len=4096,
        language_model_only=True,
        enforce_eager=True,
    )
    parallel_config = ParallelConfig(distributed_executor_backend="mp")
    return VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        speculative_config=SpeculativeConfig(
            model=str(checkpoints / "dspark"),
            method="dspark",
            num_speculative_tokens=7,
            attention_backend="ROCM_AITER_FA",
            target_model_config=model_config,
            target_parallel_config=parallel_config,
            **spec,
        ),
        cache_config=CacheConfig(block_size=16),
        attention_config=AttentionConfig(backend="ROCM_AITER_FA"),
    )


def _load_draft(checkpoints, **spec):
    """The draft as `load_dspark_model` builds and loads it."""
    from vllm.model_executor.model_loader import get_model
    from vllm.model_executor.models.utils import get_draft_quant_config

    vllm_config = _vllm_config(checkpoints, **spec)
    vllm_config.quant_config = get_draft_quant_config(vllm_config)
    draft_model_config = vllm_config.speculative_config.draft_model_config
    with set_current_vllm_config(vllm_config):
        return get_model(vllm_config=vllm_config, model_config=draft_model_config)


def _fp8_per_channel(w: torch.Tensor) -> torch.Tensor:
    """BF16 dequantization of the online `fp8_per_channel` weight of `w`."""
    fp8_max = torch.tensor(torch.finfo(F8).max, dtype=torch.float32, device=w.device)
    scale = (w.abs().amax(-1).float() / fp8_max).clamp_min(1.0 / (fp8_max * 512))
    q = (w.float() / scale[:, None]).clamp(-fp8_max, fp8_max).to(F8)
    return (q.float() * scale[:, None]).bfloat16()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.fixture(scope="module")
def drafts(checkpoints):
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    with pytest.MonkeyPatch.context() as env:
        env.setenv("VLLM_ROCM_USE_AITER", "1")
        from vllm._aiter_ops import rocm_aiter_ops

        rocm_aiter_ops.refresh_env_variables()
        device = torch.device("cuda", 0)
        torch.accelerator.set_device_index(device)
        with set_current_vllm_config(_vllm_config(checkpoints)):
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{checkpoints / 'dist_store'}",
                backend="nccl",
            )
            ensure_model_parallel_initialized(1, 1)
            init_workspace_manager(device)
            yield _load_draft(checkpoints), _load_draft(checkpoints, **FP8_DRAFT)
        cleanup_dist_env_and_memory()


def test_draft_lm_head_shard_is_16_aligned_at_tp8(
    tmp_path, dist_init, monkeypatch
) -> None:
    """At TP8, the FP8 lm_head the draft builds for the released draft's
    vocabulary (202048 rows) gives each rank a multiple of 16 rows."""
    import vllm.model_executor.layers.vocab_parallel_embedding as vocab_parallel
    from vllm.models.muse_spark.amd import dspark

    for module in (dspark, vocab_parallel):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 8)
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "config.json").write_text(
        json.dumps(_checkpoint_config(TINY_TEXT))
    )
    (tmp_path / "dspark").mkdir()
    (tmp_path / "dspark" / "config.json").write_text(
        json.dumps(TINY_DSPARK | {"vocab_size": 202048, "draft_vocab_size": 202048})
    )
    vllm_config = _vllm_config(tmp_path, **FP8_DRAFT)
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("meta"),
    ):
        lm_head = dspark.MuseSparkDSparkForCausalLM(vllm_config=vllm_config).lm_head
    assert isinstance(lm_head.quant_method, Fp8PtpcOnlineLinearMethod)
    assert lm_head.num_embeddings_per_partition % 16 == 0


def test_draft_loads_as_fp8_per_channel(drafts) -> None:
    """Every draft linear (including the fused QKV + O-gate, loaded per shard),
    the Markov head and the lm_head hold the `fp8_per_channel` weights of the
    BF16 checkpoint, whatever layout their FP8 kernel keeps."""
    from vllm.models.muse_spark.amd.dspark import _weight_rows

    bf16, fp8 = drafts
    ref_modules = dict(bf16.named_modules())
    heads = ["lm_head", "model.markov_head.markov_w2"]
    names = heads + [
        name
        for name, module in bf16.named_modules()
        if isinstance(module, LinearBase) and "confidence_head" not in name
    ]
    for name in names:
        layer, ref = fp8.get_submodule(name), ref_modules[name]
        assert isinstance(layer.quant_method, Fp8PtpcOnlineLinearMethod), name
        assert layer.weight.dtype == F8, name
        rows = ref.weight.shape[0]
        torch.testing.assert_close(
            _weight_rows(layer, 0, rows),
            _fp8_per_channel(ref.weight),
            rtol=2**-7,
            atol=0,
            msg=name,
        )
    for name in heads:
        quant_method = bf16.get_submodule(name).quant_method
        assert isinstance(quant_method, UnquantizedEmbeddingMethod)


# Relative error bounds against the BF16 draft: FP8 rounding of the weight only
# (W8A16), or of the weight and the per-token quantized activation (W8A8).
W8A16_REL, W8A8_REL = 0.03, 0.05


def test_fp8_draft_tracks_the_bf16_draft(drafts) -> None:
    """The context K/V projection, the base and Markov logits and `fc` of the FP8
    draft stay within FP8 rounding error of the BF16 draft."""
    bf16, fp8 = drafts
    gen = torch.Generator(device="cuda").manual_seed(1)
    num_ctx = 9
    # The stacked context K/V weight is rebuilt from the dequantized qkv rows.
    assert fp8.model._stacked_kv_weight.shape == bf16.model._stacked_kv_weight.shape
    assert fp8.model._stacked_kv_weight.dtype == torch.bfloat16
    assert _rel(fp8.model._stacked_kv_weight, bf16.model._stacked_kv_weight) < W8A16_REL
    context = torch.randn(num_ctx, HIDDEN, generator=gen, device="cuda")
    kv = [
        model.model._project_muse_context_kv(context.to(torch.bfloat16))
        for model in (bf16, fp8)
    ]
    assert _rel(kv[1], kv[0]) < W8A16_REL

    aux = torch.randn(
        num_ctx, TINY_TEXT["hidden_size"] * len(DSPARK_TARGETS), generator=gen,
        device="cuda", dtype=torch.float32,
    )  # fmt: skip
    combined = [model.combine_hidden_states(aux) for model in (bf16, fp8)]
    assert combined[1].dtype == torch.bfloat16
    assert _rel(combined[1], combined[0]) < W8A8_REL

    hidden = torch.randn(num_ctx, HIDDEN, generator=gen, device="cuda").bfloat16()
    logits = [model.compute_draft_logits(hidden) for model in (bf16, fp8)]
    assert _rel(logits[1], logits[0]) < W8A8_REL
    markov = torch.randn(num_ctx, RANK, generator=gen, device="cuda").bfloat16()
    bias = [
        model.model.markov_head.bias(markov, model.logits_processor)
        for model in (bf16, fp8)
    ]
    assert _rel(bias[1], bias[0]) < W8A8_REL


def test_fp8_lm_head_on_a_tp8_shard(drafts) -> None:
    """The FP8 lm_head runs on a TP8 shard of the released draft's vocabulary
    (202048 rows, 25256 per rank unpadded) and tracks the BF16 GEMM."""
    tp_size = 8
    rows = pad_vocab_size(202048, lm_head_padding_size(tp_size)) // tp_size
    with set_default_torch_dtype(torch.bfloat16):
        head = ParallelLMHead(
            rows, HIDDEN, disable_tp=True, quant_method=lm_head_quant_method("fp8")
        )
    gen = torch.Generator(device="cuda").manual_seed(2)
    weight = (torch.randn(rows, HIDDEN, generator=gen, device="cuda") * 0.05).bfloat16()
    head.weight = torch.nn.Parameter(weight, requires_grad=False)
    head.quant_method.process_weights_after_loading(head)
    hidden = torch.randn(9, HIDDEN, generator=gen, device="cuda").bfloat16()
    out = head.quant_method.apply(head, hidden)
    assert out.shape == (9, rows)
    assert _rel(out, hidden @ weight.t()) < W8A8_REL
