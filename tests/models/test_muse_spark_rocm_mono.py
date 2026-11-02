# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the Muse Spark monokernel (``--monokernel``).

The guards run on real vLLM configs and on a tiny real MuseSpark1x model
(Quark MXFP4 experts, ROCm AITER FA, a bf16 LBHNC KV cache). The FlyDSL
launches build only for the TEP8 checkpoint shapes, so the step tests run
them as the stock layer pieces they replace: from layer i's attention output
to layer i+1's q, o-gate and KV insert, reading and writing the device
memory at the pointers the host passes. A mono step must then match the
stock layer loop: hidden states, aux hidden states and KV caches.
"""

import dataclasses
import inspect
import json
from typing import Any

import pytest
import torch

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    CUDAGraphMode,
    LoRAConfig,
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
    WeightTransferConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.models.muse_spark.amd.model import norm_residual_then_norm
from vllm.models.muse_spark.amd.mono.decoder import (
    FIRST_LAYER,
    LAST_LAYER,
    MonoDims,
    MonoKernels,
    MuseSparkMono,
    bucket_of,
    built_buckets,
    collect_layer_weights,
    config_refusals,
    deployment_refusals,
    fused_attention_layers,
    kv_cache_refusal,
    layer_refusals,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.muse_spark import MuseSparkTextConfig

# The GPU tests share one tiny model and its single-rank parallel state.
pytestmark = pytest.mark.skip_global_cleanup

NUM_LAYERS = 62
ROWS_PER_REQUEST = 8

# The text config of Muse-Spark-Open-816B-A42B (the shape the kernels build).
CHECKPOINT_TEXT: dict[str, Any] = {
    "model_type": "muse_spark_text",
    "attention_bias": False,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 2,
    "final_logit_softcapping": 20.0,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 8192,
    "intermediate_size": 16384,
    "max_position_embeddings": 1048576,
    "moe_hidden_size": 4096,
    "moe_intermediate_size": 4096,
    "num_attention_heads": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": NUM_LAYERS,
    "num_key_value_heads": 16,
    "num_local_experts": 256,
    "num_shared_experts": 0,
    "output_multiplier": 0.17677669529663687,
    "post_norm_eps": 1e-08,
    "qk_scale_factor": 5.4730064863875,
    "residual_gate_temperature": 0.3,
    "rms_norm_eps": 1e-05,
    "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
    "routed_scaling_factor": 1.0,
    "sliding_window": 2048,
    "tie_word_embeddings": False,
    "vocab_size": 202048,
    # Full attention (and no RoPE) every 4th layer from layer 1.
    "layer_types": [
        "full_attention" if i % 4 == 1 else "sliding_attention"
        for i in range(NUM_LAYERS)
    ],
    "layer_rope_theta": [0 if i % 4 == 1 else 500000.0 for i in range(NUM_LAYERS)],
    "mlp_layer_types": ["sparse"] * NUM_LAYERS,
}
# One TP rank of the checkpoint's layout at TP = EP = 1.
TINY_TEXT = CHECKPOINT_TEXT | {
    "hidden_size": 512,
    "intermediate_size": 512,
    "moe_hidden_size": 256,
    "moe_intermediate_size": 256,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "num_local_experts": 8,
    "vocab_size": 512,
    "max_position_embeddings": 4096,
}
_MX_FP4 = {
    "dtype": "fp4",
    "qscheme": "per_group",
    "ch_axis": -1,
    "group_size": 32,
    "round_method": "half_even",
    "scale_type": "float",
    "scale_format": "e8m0",
    "scale_calculation_mode": "even",
    "observer_cls": "PerBlockMXObserver",
    "is_scale_quant": False,
}


def _quark_exclude(num_experts: int) -> list[str]:
    """The checkpoint's exclusions: everything but the experts of layers 1-60."""
    names = ["lm_head"]
    for i in range(NUM_LAYERS):
        layer = f"model.language_model.layers.{i}"
        names += [f"{layer}.self_attn.{p}_proj" for p in ("q", "k", "v", "o", "gate")]
        names += [f"{layer}.mlp.{p}" for p in ("gate", "pre_expert_proj")]
        names.append(f"{layer}.mlp.post_expert_proj")
        if i in (0, NUM_LAYERS - 1):
            names += [
                f"{layer}.mlp.experts.{e}.{p}_proj"
                for e in range(num_experts)
                for p in ("gate", "up", "down")
            ]
    return names


def _checkpoint_config(text: dict[str, Any]) -> dict[str, Any]:
    return {
        "architectures": ["MuseSparkForConditionalGeneration"],
        "model_type": "muse_spark",
        "dtype": "bfloat16",
        "image_token_id": 500,
        "audio_token_id": 501,
        "video_token_id": 502,
        "projector_hidden_act": "gelu",
        "projector_hidden_size": 64,
        "out_hidden_size": 256,
        "text_config": text,
        "vision_config": {
            "model_type": "muse_spark_vision",
            "hidden_act": "gelu",
            "hidden_size": 64,
            "intermediate_size": 128,
            "layer_norm_eps": 1e-05,
            "layer_types": ["full_attention"],
            "max_position_embeddings": 64,
            "merge_size": 2,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "patch_size": 14,
            "patch_temporal": 2,
            "pos_emb_height": 4,
            "pos_emb_width": 4,
            "rope_parameters": {"rope_theta": 10000, "rope_type": "axial"},
        },
        "audio_config": {
            "model_type": "muse_spark_audio",
            "conv_kernel_size": 7,
            "depths": [1],
            "hidden_act": "gelu",
            "hidden_sizes": [64],
            "mlp_ratio": 4.0,
            "num_hidden_layers": 1,
            "out_hidden_size": text["hidden_size"],
            "patch_sizes": [16],
            "rms_norm_eps": 1e-06,
        },
        "quantization_config": {
            "quant_method": "quark",
            "global_quant_config": {
                "input_tensors": _MX_FP4 | {"is_dynamic": True},
                "weight": _MX_FP4 | {"is_dynamic": False},
                "output_tensors": None,
                "bias": None,
            },
            "exclude": _quark_exclude(text["num_local_experts"]),
            "layer_quant_config": {},
            "layer_type_quant_config": {},
            "kv_cache_quant_config": {},
            "export": {"kv_cache_group": [], "pack_method": "reorder"},
        },
    }


# The target_layer_ids of Muse-Spark-Open-DSpark.
DSPARK_TARGETS = [13, 29, 45, 57, 61]
DSPARK = {
    "model_type": "muse_spark_dspark",
    "num_hidden_layers": 2,
    "moe": {"is_enabled": False},
    "global_attn_cfg": [2048],
    "layer_types": ["sliding_attention"] * 2,
    "target_layer_ids": DSPARK_TARGETS,
    "num_target_layers": len(DSPARK_TARGETS),
    "markov_rank": 8,
    "markov_head_type": "vanilla",
    "dflash_config": {"causal": True, "target_layer_ids": DSPARK_TARGETS},
}


def _vllm_config(
    target, draft=None, tp: int = 1, cache_dtype: str = "auto", **parallel
) -> VllmConfig:
    model_config = ModelConfig(
        model=str(target),
        tokenizer_mode="skip",
        dtype="bfloat16",
        max_model_len=4096,
        language_model_only=True,
        enforce_eager=True,
    )
    # Config-only: "mp" skips the executor's local GPU count check.
    parallel_config = ParallelConfig(
        tensor_parallel_size=tp,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        **parallel,
    )
    speculative_config = None
    if draft is not None:
        speculative_config = SpeculativeConfig(
            model=str(draft),
            method="dspark",
            num_speculative_tokens=ROWS_PER_REQUEST - 1,
            target_model_config=model_config,
            target_parallel_config=parallel_config,
        )
    return VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        speculative_config=speculative_config,
        cache_config=CacheConfig(block_size=16, cache_dtype=cache_dtype),
        attention_config=AttentionConfig(backend="ROCM_AITER_FA"),
    )


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("muse_spark")
    for name, config in (("tiny", _checkpoint_config(TINY_TEXT)), ("dspark", DSPARK)):
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
    return root


# ---------------------------------------------------------------- host guards


def test_buckets_pad_rows_up_to_the_next_build() -> None:
    assert [bucket_of(r) for r in (8, 16, 24, 32, 40, 64, 72, 96, 104, 128)] == [
        8, 16, 32, 32, 64, 64, 96, 96, 128, 128,
    ]  # fmt: skip
    assert [bucket_of(r) for r in (0, 4, 12, 136)] == [None] * 4
    assert built_buckets(1) == [8]
    assert built_buckets(3) == [8, 16, 32]
    assert built_buckets(16) == built_buckets(128) == [8, 16, 32, 64, 96, 128]


def test_kv_cache_guard_accepts_only_dense_bf16_lbhnc() -> None:
    lbhnc = torch.empty(10, 2, 16, 128, dtype=torch.bfloat16, device="meta")
    assert kv_cache_refusal(lbhnc, 2) is None
    lhbnc = torch.empty(2, 10, 16, 128, dtype=torch.bfloat16, device="meta")
    assert kv_cache_refusal(lhbnc.transpose(0, 1), 2).startswith("kv:stride")
    fp8 = torch.empty(10, 2, 16, 128, dtype=torch.float8_e4m3fn, device="meta")
    assert kv_cache_refusal(fp8, 2).startswith("kv:torch.float8")
    block32 = torch.empty(10, 2, 32, 128, dtype=torch.bfloat16, device="meta")
    assert kv_cache_refusal(block32, 2).startswith("kv:shape")
    assert kv_cache_refusal(torch.empty(0, dtype=torch.bfloat16), 2) == "kv:unbound"


def test_config_guard_accepts_only_the_checkpoint_shape() -> None:
    assert config_refusals(MuseSparkTextConfig(**CHECKPOINT_TEXT)) == []
    assert config_refusals(MuseSparkTextConfig(**TINY_TEXT)) == [
        "config.hidden_size=512 (kernel: 8192)",
        "config.moe_hidden_size=256 (kernel: 4096)",
        "config.moe_intermediate_size=256 (kernel: 4096)",
        "config.num_local_experts=8 (kernel: 256)",
        "config.num_attention_heads=16 (kernel: 128)",
        "config.num_key_value_heads=2 (kernel: 16)",
    ]


def _moe_parallel(use_ep: bool = True, ep_size: int = 8, ep_rank: int = 3):
    return FusedMoEParallelConfig(
        tp_size=1 if use_ep else ep_size,
        pcp_size=1,
        dp_size=1,
        ep_size=ep_size if use_ep else 1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=ep_rank if use_ep else 0,
        sp_size=1,
        use_ep=use_ep,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )


def test_deployment_guard_accepts_only_tep8_with_dspark(checkpoints) -> None:
    tiny, dspark = checkpoints / "tiny", checkpoints / "dspark"
    tep8 = _vllm_config(tiny, dspark, tp=8)
    assert deployment_refusals(tep8, 3, 8, _moe_parallel()) == []

    tp4 = _vllm_config(tiny, dspark, tp=4, expert_placement_strategy="round_robin")
    tp4.parallel_config.enable_expert_parallel = False
    assert deployment_refusals(tp4, 3, 4, _moe_parallel(use_ep=False)) == [
        "tensor_parallel_size=4 (kernel: 8)",
        "expert parallelism off (needs --enable-expert-parallel)",
        "expert placement round_robin",
        "MoE layers not expert parallel",
    ]

    no_draft = _vllm_config(tiny, tp=8, cache_dtype="fp8")
    no_draft.lora_config = LoRAConfig()
    no_draft.model_config.enable_sleep_mode = True
    no_draft.weight_transfer_config = WeightTransferConfig()
    assert deployment_refusals(no_draft, 3, 8, _moe_parallel()) == [
        "needs speculative decoding with 7 draft tokens",
        "LoRA on",
        "sleep mode on",
        "weight transfer on",
        "kv_cache_dtype=fp8 (kernel: bf16)",
    ]


# ------------------------------------------------- the tiny model on one GPU


def _on_gfx950() -> bool:
    if not (current_platform.is_rocm() and torch.cuda.is_available()):
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


requires_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="MuseSpark1x on ROCm needs AITER's gfx950 kernels"
)


@dataclasses.dataclass
class TinyModel:
    lm: Any
    vllm_config: VllmConfig
    dims: MonoDims
    refusals_before_finalize: list[str]
    kv_specs: dict[str, Any]

    @property
    def model(self):
        return self.lm.model

    @property
    def layers(self):
        return self.lm.model.layers


@pytest.fixture(scope="module")
def tiny_model(checkpoints):
    from vllm.config.load import LoadConfig
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.muse_spark.amd.multimodal import MuseSparkForConditionalGeneration
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
    from vllm.v1.worker.workspace import init_workspace_manager

    with pytest.MonkeyPatch.context() as env:
        env.setenv("VLLM_ROCM_USE_AITER", "1")
        env.setenv("VLLM_KV_CACHE_LAYOUT", "LBHNC")
        from vllm._aiter_ops import rocm_aiter_ops

        rocm_aiter_ops.refresh_env_variables()
        device = torch.device("cuda", 0)
        torch.accelerator.set_device_index(device)
        vllm_config = _vllm_config(checkpoints / "tiny", checkpoints / "dspark")
        store = checkpoints / "dist_store"
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{store}",
                backend="nccl",
            )
            ensure_model_parallel_initialized(1, 1)
            init_workspace_manager(device)
            with set_default_torch_dtype(torch.bfloat16), device:
                lm = MuseSparkForConditionalGeneration(vllm_config=vllm_config)
            model_config = vllm_config.model_config
            text_config = model_config.hf_text_config
            dims = MonoDims.from_config(text_config, 1)
            before = layer_refusals(lm.model, dims)
            DummyModelLoader(LoadConfig(load_format="dummy")).load_weights(
                lm, model_config
            )
            lm.load_weights([])
            process_weights_after_loading(lm, model_config, device)
            kv = dict(
                block_size=16,
                num_kv_heads=dims.local_kv_heads,
                head_size=dims.head_dim,
                dtype=torch.bfloat16,
            )
            specs = {
                "full": FullAttentionSpec(**kv),
                "swa": SlidingWindowSpec(**kv, sliding_window=2048),
            }
            yield TinyModel(lm, vllm_config, dims, before, specs)
        cleanup_dist_env_and_memory()


@requires_gfx950
def test_layer_guards_follow_the_weight_finalize(tiny_model) -> None:
    """Refused until load_weights and the quant finalize produced the kernel
    layouts (folded router, materialized gates, shuffled experts)."""
    before = tiny_model.refusals_before_finalize
    for reason in (
        "layer 1: add_attn coefficients not materialized",
        "layer 1: router multiplier not folded",
        "layer 1: expert weights not shuffled",
    ):
        assert reason in before
    assert layer_refusals(tiny_model.model, tiny_model.dims) == []


@requires_gfx950
def test_weight_views_alias_the_rank_slices(tiny_model) -> None:
    """The launches read vLLM's parameters in place; only the norm
    multipliers are copies. Rank 1 of 2 reads the second row half."""
    dims = dataclasses.replace(tiny_model.dims, world_size=2)
    layer, nxt = tiny_model.layers[1], tiny_model.layers[2]
    w = collect_layer_weights(tiny_model.layers, 1, 1, dims)
    moe = layer.feed_forward
    experts = moe.experts.routed_experts

    assert w.w_r.shape == (dims.num_experts // 2, dims.hidden_size)
    for view, base, row0 in (
        (w.w_r, moe.router.weight, dims.num_experts // 2),
        (w.w_pre, moe.pooler.weight, dims.pooled_size // 2),
        (w.w_pp, moe.unpooler.weight, dims.hidden_size // 2),
        (w.w_o, layer.self_attn.o_proj.weight, 0),
        (w.w13, experts.w13_weight, 0),
        (w.w13s, experts.w13_weight_scale, 0),
        (w.w2, experts.w2_weight, 0),
        (w.w2s, experts.w2_weight_scale, 0),
        (w.bias, moe.expert_bias, 0),
        (w.alpha_a, layer.add_attn.alpha, 0),
        (w.beta_a, layer.add_attn.beta, 0),
        (w.alpha, layer.add_ffn.alpha, 0),
        (w.beta, layer.add_ffn.beta, 0),
        (w.w_qkvg, nxt.self_attn.qkv_proj.weight, 0),
    ):
        assert view.is_contiguous()
        offset = row0 * base.stride(0) * base.element_size()
        assert view.data_ptr() == base.data_ptr() + offset
    for copy, norm in (
        (w.w_pffn, layer.pre_ffn_norm),
        (w.w_pren, moe.pre_expert_norm),
        (w.w_post, moe.post_expert_norm),
        (w.w_pff, layer.post_ffn_norm),
        (w.w_in, nxt.self_attn.pre_qkv_proj_norm),
    ):
        assert torch.equal(copy, (norm.weight.float() + norm.gain_center).bfloat16())
    # Layer 2 is a sliding-window layer (RoPE), layer 5 a full one (none).
    assert w.next_rope
    assert not collect_layer_weights(tiny_model.layers, 4, 0, dims).next_rope


@requires_gfx950
def test_fused_attention_layers_follow_the_window(tiny_model) -> None:
    layers = tiny_model.layers
    fused = fused_attention_layers(layers, 4096, 1 << 20)
    assert sorted(fused) == list(range(FIRST_LAYER + 1, LAST_LAYER + 1))
    assert [i for i, t in fused.items() if t == "full"] == list(range(5, 61, 4))
    assert fused_attention_layers(layers, (1 << 20) + 1, 1 << 20) == {}


# ------------------------------------------- the launches, as the stock pieces

# The launch signatures of the FlyDSL sources, up to the peer pointers.
_K3_HEAD = (
    "attn_o attn_g stream_pre w_o alpha_a beta_a w_pffn w_r bias w_pre w_pren "
    "w13 w13s w2 w2s w_post w_pp w_pff alpha beta w_in w_qkvg stream_out xn_out "
    "part_out q_out g_out k_cache v_cache slots cos_sin pos scratch epoch health tl"
)
_HA_HEAD = (
    "attn_o attn_g stream_in w_o alpha_a beta_a w_pffn w_r bias w_pre w_pren "
    "h_out ids_out sc_out scratch epoch health tl"
)
_HM_HEAD = "h ids sc w13 w13s w2 w2s w_post part_out scratch epoch health tl"
_HB_HEAD = (
    "part w_pp w_pff alpha beta w_in w_qkvg stream_out sfull_out q_out g_out "
    "k_cache v_cache slots cos_sin pos scratch epoch health tl"
)
_REPLICAS = " w13r w13sr w2r w2sr eptab"
_DESC_WORDS = 10


def _params(head: str, world_size: int, tail: str) -> list[str]:
    """A launch's parameters in order."""
    return head.split() + [f"peer{r}" for r in range(world_size)] + tail.split()


def _device_view(ptr: int, shape, dtype: torch.dtype, strides=None) -> torch.Tensor:
    """The device memory at ``ptr`` as a tensor, as a launch reads it."""
    strides = strides or torch.empty(shape, device="meta").stride()
    extent = 1 + sum((n - 1) * s for n, s in zip(shape, strides))
    nbytes = extent * torch.empty((), dtype=dtype).element_size()

    class _Memory:
        __cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }

    raw = torch.as_tensor(_Memory(), device="cuda")
    return raw.view(dtype).as_strided(shape, strides)


class StockLaunches:
    """The FlyDSL monokernel sources, with each launch running the stock layer
    pieces it replaces on the memory at its arguments' pointers."""

    def __init__(self, model, dims: MonoDims, bnd: int) -> None:
        self.model, self.dims, self.bnd = model, dims, bnd
        self.calls: list[tuple[str, dict]] = []
        self.pending: dict[int, dict] = {}
        self.kernels = MonoKernels(
            *(_Kernel(self, name) for name in ("k3", "k5", "k5m")), _AttnFrag()
        )

    def launcher(self, name: str, params: list[str], run):
        signature = inspect.Signature(
            [inspect.Parameter(p, inspect.Parameter.POSITIONAL_ONLY) for p in params]
            + [inspect.Parameter("stream", inspect.Parameter.KEYWORD_ONLY)]
        )

        def launch(*args, **kwargs):
            bound = dict(signature.bind(*args, **kwargs).arguments)
            self.calls.append((name, bound))
            run(bound)

        return launch

    def _attention(
        self, i: int, attn_o: int, rows: int, nreq: int, fused: bool
    ) -> torch.Tensor:
        """Layer i's attention output: at ``attn_o``, or, for a fused build,
        run here from the descriptor row at ``attn_o``."""
        layer = self.model.layers[i]
        d = self.dims
        if not fused:
            return _device_view(attn_o, (rows, d.q_rows), torch.bfloat16)
        words = _device_view(attn_o, (_DESC_WORDS,), torch.int64).tolist()
        md, _, kv_cache, _ = _attention_context(layer)
        assert words[1:5] == [
            kv_cache.data_ptr(),
            kv_cache.stride(0),
            md.block_table.data_ptr(),
            md.seq_lens.data_ptr(),
        ]
        assert words[8] == nreq
        q = _device_view(words[0], (rows, d.q_rows), torch.bfloat16)
        out = _device_view(words[5], (rows, d.q_rows), torch.bfloat16)
        out.copy_(layer.self_attn.attn(q, None, None))
        return out

    def run_layer(self, rows: int, a: dict, b: dict) -> None:
        """Layer i from its attention output (launch ``a``) to layer i+1's q,
        o-gate and stream (launch ``b``)."""
        i, nreq = a["layer"], rows // ROWS_PER_REQUEST
        d, layer, nxt = self.dims, self.model.layers[i], self.model.layers[i + 1]
        o = self._attention(i, a["attn_o"], rows, nreq, a["fused"])
        g = _device_view(
            a["attn_g"], (rows, d.q_rows), torch.bfloat16, (a["g_stride"], 1)
        )
        stream = _device_view(a["stream"], (rows, d.hidden_size), torch.float32)
        positions = _device_view(b["pos"], (rows,), torch.int64)
        _, _, kv_cache, slots = _attention_context(nxt)
        assert b["k_cache"] == kv_cache.data_ptr()
        assert b["v_cache"] == kv_cache.stride(0)
        assert b["slots"] == slots.data_ptr()
        rope = nxt.self_attn.rotary_emb
        if rope is not None:
            assert b["cos_sin"] == rope.cos_sin_cache.data_ptr()

        y = layer.after_attention(stream, layer.self_attn.back(o, g))
        attn_input, stream_next = norm_residual_then_norm(
            y, nxt.self_attn.pre_qkv_proj_norm, out_dtype=nxt.default_dtype
        )
        qkv, _ = nxt.self_attn.qkv_proj(attn_input)
        q, _, _, g_next = nxt.self_attn._qk_norm_rope_kv_insert(qkv, positions)
        streams = ["stream_out"] + (["sfull_out"] if b.get("gather") else [])
        for names, t in ((streams, stream_next), (["q_out"], q), (["g_out"], g_next)):
            for name in names:
                _device_view(b[name], t.shape, t.dtype).copy_(t)


@dataclasses.dataclass(frozen=True)
class _Key:
    S: int
    W: int
    next_rope: bool = True
    attn: str = ""
    bnd: int = 0


class _Kernel:
    """One FlyDSL source module: its layouts and builds."""

    BLOCKS = 4
    HEALTH_WORDS = 8 + 8 * BLOCKS
    TL_POINTS = 4

    def __init__(self, owner: StockLaunches, name: str) -> None:
        self.owner, self.name = owner, name

    def peer_layout(self, S, W, bnd=0):
        return {}, 4096

    def scratch_layout(self, S, W=1, bnd=0):
        return {}, 4096

    def build_vllm(self, S, W, next_rope=True, attn=""):
        return _Key(S, W, next_rope, attn, self.owner.bnd if self.name == "k5" else 0)

    def build(self, key: _Key):
        o, w = self.owner, key.W
        if self.name == "k3":
            params = _params(_K3_HEAD, w, "layer rank g_stride" + _REPLICAS)
            return o.launcher("k3", params, lambda a: _run_k3(o, key, a))
        if self.name == "k5m":
            params = _params(_HM_HEAD, w, "layer rank nrows" + _REPLICAS)
            return o.launcher("hm", params, lambda a: None)
        ha = _params(_HA_HEAD, w, "layer rank g_stride s_full nrows")
        hb = _params(_HB_HEAD, w, "layer rank so_stride so_off gather nrows")

        def run_ha(a):
            o.pending[a["layer"]] = a | {"fused": bool(key.attn)}

        return (
            o.launcher("ha", ha, run_ha),
            o.launcher("hb", hb, lambda a: _run_hb(o, key, a)),
        )


class _AttnFrag:
    TABLE_ROWS = 64
    DESC_WORDS = _DESC_WORDS
    VLLM_MAX_KEYS = 1 << 20

    @staticmethod
    def part_bytes() -> int:
        return 4096

    @staticmethod
    def vllm_desc(q, kv_cache, block_table, seq_lens, out, part, nreq, scale):
        return [
            q.data_ptr(),
            kv_cache.data_ptr(),
            kv_cache.stride(0),
            block_table.data_ptr(),
            seq_lens.data_ptr(),
            out.data_ptr(),
            part.data_ptr(),
            block_table.stride(0),
            nreq,
            0,
        ]


def _run_k3(o: StockLaunches, key: _Key, a: dict) -> None:
    o.run_layer(key.S, a | {"stream": a["stream_pre"], "fused": bool(key.attn)}, a)


def _run_hb(o: StockLaunches, key: _Key, b: dict) -> None:
    a = o.pending.pop(b["layer"])
    assert a["nrows"] == b["nrows"]
    o.run_layer(b["nrows"], a | {"stream": a["stream_in"]}, b)


def _attention_context(layer):
    return get_attention_context(layer.self_attn.attn.layer_name)


class Step:
    """One verify step's real attention metadata and KV caches."""

    def __init__(self, tiny: TinyModel, query_lens: list[int], seed: int) -> None:
        from tests.v1.attention.utils import (
            BatchSpec,
            create_common_attn_metadata,
            dense_kv_cache_views,
        )
        from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend
        from vllm.v1.kv_cache_layout import KVCacheLayout

        device = torch.device("cuda", 0)
        generator = torch.Generator().manual_seed(seed)
        seq_lens = [
            int(torch.randint(40, 300, (), generator=generator)) + q for q in query_lens
        ]
        self.common = create_common_attn_metadata(
            BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
            16,
            device,
            arange_block_indices=True,
        )
        num_blocks = int(self.common.block_table_tensor.max()) + 1
        builder_cls = AiterFlashAttentionBackend.get_builder_cls()
        self.names = [layer.self_attn.attn.layer_name for layer in tiny.layers]
        self.metadata: dict[str, Any] = {}
        self.kv_caches: dict[str, torch.Tensor] = {}
        for kind, spec in tiny.kv_specs.items():
            names = [
                n
                for n, layer in zip(self.names, tiny.layers)
                if (layer.self_attn.attn.impl.sliding_window == (2047, 0))
                == (kind == "swa")
            ]
            builder = builder_cls(spec, names, tiny.vllm_config, device)
            metadata = builder.build(0, self.common)
            for name in names:
                raw = torch.zeros(
                    num_blocks * spec.page_size_bytes, dtype=torch.int8, device=device
                )
                (view,) = dense_kv_cache_views(
                    raw, spec, num_blocks, 1, KVCacheLayout.LBHNC
                )
                self.metadata[name] = metadata
                self.kv_caches[name] = view
        self.tokens = sum(query_lens)
        self.input_ids = torch.randint(
            0, TINY_TEXT["vocab_size"], (self.tokens,), generator=generator
        ).to(device)
        self.positions = torch.cat(
            [torch.arange(s - q, s) for s, q in zip(seq_lens, query_lens)]
        ).to(device)

    def run(self, tiny: TinyModel, mode=CUDAGraphMode.NONE):
        """MuseSpark1xModel.forward on zeroed KV caches; returns the hidden
        states, the aux hidden states and the KV caches after the step."""
        from vllm.forward_context import set_forward_context
        from vllm.v1.worker.utils import bind_kv_cache

        for cache in self.kv_caches.values():
            cache.zero_()
        context = tiny.vllm_config.compilation_config.static_forward_context
        bind_kv_cache(self.kv_caches, context, [])
        slots = dict.fromkeys(self.names, self.common.slot_mapping)
        with (
            set_current_vllm_config(tiny.vllm_config),
            set_forward_context(
                self.metadata,
                tiny.vllm_config,
                num_tokens=self.tokens,
                cudagraph_runtime_mode=mode,
                slot_mapping=slots,
            ),
        ):
            out = tiny.model(self.input_ids, self.positions, None)
        hidden, aux = out if isinstance(out, tuple) else (out, [])
        kv = {name: cache.clone() for name, cache in self.kv_caches.items()}
        return hidden.clone(), [a.clone() for a in aux], kv


def _run_mono(
    tiny: TinyModel, step: Step, bnd: int, buckets: list[int], mode=CUDAGraphMode.NONE
):
    """The step with the monokernel attached, on the stock launches."""
    from vllm.distributed import get_tp_group

    launches = StockLaunches(tiny.model, tiny.dims, bnd)
    tiny.model.mono = MuseSparkMono(
        tiny.model,
        launches.kernels,
        rank=0,
        group=get_tp_group().cpu_group,
        device=torch.device("cuda", 0),
        dims=tiny.dims,
        buckets=buckets,
        fused_attention=fused_attention_layers(
            tiny.layers, 4096, _AttnFrag.VLLM_MAX_KEYS
        ),
    )
    try:
        return launches, step.run(tiny, mode)
    finally:
        tiny.model.mono.close()
        tiny.model.mono = None


def _assert_same_step(stock, mono) -> None:
    (hidden_s, aux_s, kv_s), (hidden_m, aux_m, kv_m) = stock, mono
    torch.testing.assert_close(hidden_m, hidden_s, rtol=0, atol=0)
    for name in kv_s:
        torch.testing.assert_close(kv_m[name], kv_s[name], rtol=0, atol=0)
    assert len(aux_m) == len(aux_s)
    # The stock loop blends an aux state on its own, the mono path takes the
    # blend the next layer's input norm makes: fp32 rounding apart.
    for a_m, a_s in zip(aux_m, aux_s):
        torch.testing.assert_close(a_m, a_s, rtol=1e-5, atol=1e-6)


@requires_gfx950
@pytest.mark.parametrize(
    "aux_layers",
    # vLLM's aux ids (L = the output of layer L - 1) for the target_layer_ids of
    # Muse-Spark-Open-DSpark, of Muse-Spark-Open-DFlash2, and of the embedding
    # and every stock / mono seam (0, 1, 59, 60, 61).
    [(14, 30, 46, 58, 62), (2, 17, 31, 45, 60), (1, 2, 60, 61, 62)],
    ids=["dspark", "dflash2", "seams"],
)
@pytest.mark.parametrize(
    ("num_reqs", "bnd"),
    [(1, 0), (2, 0), (3, 0), (3, 3), (4, 3), (5, 0), (7, 0)],
)
def test_mono_step_matches_the_stock_loop(
    tiny_model, aux_layers, num_reqs, bnd
) -> None:
    """8 / 16 rows: K3 with fused attention; 24-32 rows: HA + HM + HB on the
    32-row build (fused attention, replicated (bnd 3) or token-owner (bnd 0)
    stream); 40-56 rows: the 64-row build with vLLM's attention op between
    the launches. Same hidden states, aux states and KV caches as stock."""
    tiny_model.lm.set_aux_hidden_state_layers(aux_layers)
    step = Step(tiny_model, [ROWS_PER_REQUEST] * num_reqs, seed=num_reqs)
    stock = step.run(tiny_model)
    launches, mono = _run_mono(tiny_model, step, bnd, [8, 16, 32, 64])

    kernels = {name for name, _ in launches.calls}
    assert kernels == ({"k3"} if num_reqs <= 2 else {"ha", "hm", "hb"})
    assert [a["layer"] for name, a in launches.calls if name in ("k3", "hb")] == list(
        range(FIRST_LAYER, LAST_LAYER + 1)
    )
    _assert_same_step(stock, mono)


@requires_gfx950
@pytest.mark.parametrize(
    ("query_lens", "mode"),
    [
        ([20], CUDAGraphMode.NONE),  # prefill
        ([8, 20], CUDAGraphMode.NONE),  # mixed
        ([4, 4], CUDAGraphMode.NONE),  # shorter verify
        ([8] * 17, CUDAGraphMode.NONE),  # 136 rows: above the largest build
        ([8], CUDAGraphMode.PIECEWISE),
    ],
    ids=["prefill", "mixed", "query-len-4", "136-rows", "piecewise"],
)
def test_other_steps_run_stock(tiny_model, query_lens, mode) -> None:
    tiny_model.lm.set_aux_hidden_state_layers((14, 30, 46, 58, 62))
    step = Step(tiny_model, query_lens, seed=len(query_lens))
    stock = step.run(tiny_model, mode)
    launches, mono = _run_mono(tiny_model, step, 0, [8, 16, 32, 64, 96, 128], mode)
    assert launches.calls == []
    _assert_same_step(stock, mono)
