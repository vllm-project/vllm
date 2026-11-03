# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the Muse Spark DSpark draft-layer kernels (``--monokernel``).

The guards run on the DSpark checkpoint config. The FlyDSL build (kernels/draft.py)
exists only for the TP8 checkpoint shape, so the step tests run a tiny real
DSpark draft model on one GPU with the launch replaced by the stock draft
layers: the replacement reads the launch table, the attention descriptors and
the input / output pointers the host passes, checks them against the model's
tensors, and runs the stock layer loop on that memory. A draft step on the
host path must then match the stock draft model: hidden states and the draft
KV caches, bit for bit; every other step runs stock.
"""

import dataclasses
import inspect
import json
from typing import Any

import pytest
import torch

from tests.models.test_muse_spark_rocm_mono import (
    TINY_TEXT,
    _checkpoint_config,
    _device_view,
    _vllm_config,
    requires_gfx950,
)
from vllm.config import CUDAGraphMode, set_current_vllm_config
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.models.muse_spark.amd.mono.drafter import (
    DRAFT_BUILDS,
    DSparkDrafter,
    draft_config_refusals,
    draft_dims,
    draft_layer_refusals,
    effective_norm_weight_fp32,
)
from vllm.transformers_utils.configs.muse_spark import MuseSparkDSparkConfig

pytestmark = pytest.mark.skip_global_cleanup

QUERY_ROWS = 7
TARGETS = [13, 29, 45, 57, 61]

# config.json of Muse-Spark-Open-DSpark (the shape the draft kernel builds).
DSPARK_CHECKPOINT: dict[str, Any] = {
    "model_type": "muse_spark_dspark",
    "architectures": ["MuseSparkDSparkModel"],
    "attention_bias": False,
    "bos_token_id": 200000,
    "eos_token_id": [200001, 200008],
    "pad_token_id": 200018,
    "draft_position_mode": "rope",
    "ffn_dim_multiplier": 2.0,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 8192,
    "intermediate_size": 16384,
    "intermediate_size_mlp": 16384,
    "layer_types": ["sliding_attention"] * 5,
    "max_position_embeddings": 1048576,
    "metap": {
        "base_width": 256.0,
        "m_emb": 1.0,
        "metap_mode": "sp",
        "router_has_one_inf_dim": True,
        "use_metap": True,
        "use_metap_output_multiplier_without_metap": False,
    },
    "non_linearity": "swiglu",
    "norm_affine": True,
    "norm_zero_centered_gamma": True,
    "num_attention_heads": 128,
    "num_hidden_layers": 5,
    "num_key_value_heads": 16,
    "output_norm_gain_center_type": "zero",
    "output_soft_cap_temp": 20.0,
    "post_norm_affine": False,
    "post_norm_gain_center_type": "one",
    "residual_dtype": "fp32",
    "rms_norm_eps": 1e-05,
    "rms_post_norm_eps": 1e-08,
    "rope_theta": 500000.0,
    "rope_use_fp32_in_outer_product": True,
    "sliding_window": 2048,
    "tie_word_embeddings": False,
    "use_attn_o_gate": True,
    "use_attn_o_norm": True,
    "use_ffn_learnable_scales": True,
    "use_qk_norm": True,
    "use_residual_gate": True,
    "use_rope": True,
    "use_sliding_window": True,
    "use_sqrt_attn_scaling": False,
    "vocab_size": 202048,
    "moe": {"is_enabled": False},
    "global_attn_cfg": [2048],
    "every_n_layers_nope": None,
    "num_target_layers": 5,
    "target_hidden_size": 8192,
    "target_num_hidden_layers": 62,
    "target_layer_ids": TARGETS,
    "draft_vocab_size": 202048,
    "mask_token_id": 200018,
    "markov_rank": 512,
    "markov_head_type": "vanilla",
    "enable_confidence_head": True,
    "confidence_head_with_markov": True,
    "sample_from_anchor": True,
    "block_size": 7,
    "sliding_window_non_causal": False,
    "full_attention_causal": False,
    "use_aux_hidden_state": True,
    "dflash_config": {
        "causal": True,
        "mask_token_id": 200018,
        "target_layer_ids": TARGETS,
        "use_aux_hidden_state": True,
    },
    "dtype": "bfloat16",
}
# One TP rank of the checkpoint's draft layout at TP = 1, on the tiny target.
TINY_DSPARK = DSPARK_CHECKPOINT | {
    "hidden_size": 512,
    "intermediate_size": 1024,
    "intermediate_size_mlp": 1024,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "vocab_size": 512,
    "draft_vocab_size": 512,
    "mask_token_id": 510,
    "pad_token_id": 510,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "markov_rank": 16,
    "target_hidden_size": 512,
    "max_position_embeddings": 4096,
    "dflash_config": DSPARK_CHECKPOINT["dflash_config"] | {"mask_token_id": 510},
}


def test_draft_builds_cover_one_and_two_requests() -> None:
    assert DRAFT_BUILDS == {7: 8, 14: 16}


def test_config_guard_accepts_only_the_checkpoint_shape() -> None:
    assert draft_config_refusals(MuseSparkDSparkConfig(**DSPARK_CHECKPOINT)) == []
    tiny = draft_config_refusals(MuseSparkDSparkConfig(**TINY_DSPARK))
    assert tiny == [
        "draft config.hidden_size=512 (kernel: 8192)",
        "draft config.intermediate_size_mlp=1024 (kernel: 16384)",
        "draft config.num_attention_heads=16 (kernel: 128)",
        "draft config.num_key_value_heads=2 (kernel: 16)",
    ]
    nope = DSPARK_CHECKPOINT | {"use_qk_norm": False, "residual_dtype": "bf16"}
    assert draft_config_refusals(MuseSparkDSparkConfig(**nope)) == [
        "draft config.use_qk_norm=False",
        "draft residual dtype not fp32",
    ]


def test_draft_kernel_layouts_fit_the_build() -> None:
    """CPU only: the launch table and the regions of kernels/draft.py."""
    pytest.importorskip("flydsl")
    from vllm.models.muse_spark.amd.mono.kernels import draft

    assert draft.T_G < draft.TAB_ROWS and draft.G_PEER + 8 <= draft.TAB_WORDS
    for S in (8, 16):
        for layout in (draft.scratch_layout(S), draft.peer_layout(S)):
            regions, total = layout
            ends = sorted((o, o + n) for o, n in regions.values())
            assert all(a[1] <= b[0] for a, b in zip(ends, ends[1:]))
            assert ends[-1][1] <= total


# ------------------------------------------------- the tiny draft on one GPU


@dataclasses.dataclass
class TinyDraft:
    lm: Any
    vllm_config: Any
    spec: Any

    @property
    def model(self):
        return self.lm.model

    @property
    def names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]


@pytest.fixture(scope="module")
def tiny_draft(tmp_path_factory):
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.models.muse_spark.common.dspark import MuseSparkDSparkForCausalLM
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.kv_cache_interface import SlidingWindowSpec
    from vllm.v1.worker.workspace import init_workspace_manager

    root = tmp_path_factory.mktemp("muse_spark_dspark")
    for name, config in (
        ("tiny", _checkpoint_config(TINY_TEXT)),
        ("dspark", TINY_DSPARK),
    ):
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
    with pytest.MonkeyPatch.context() as env:
        env.setenv("VLLM_ROCM_USE_AITER", "1")
        env.setenv("VLLM_KV_CACHE_LAYOUT", "LBHNC")
        from vllm._aiter_ops import rocm_aiter_ops

        rocm_aiter_ops.refresh_env_variables()
        device = torch.device("cuda", 0)
        torch.accelerator.set_device_index(device)
        vllm_config = _vllm_config(root / "tiny", root / "dspark")
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{root / 'dist_store'}",
                backend="nccl",
            )
            ensure_model_parallel_initialized(1, 1)
            init_workspace_manager(device)
            with set_default_torch_dtype(torch.bfloat16), device:
                lm = MuseSparkDSparkForCausalLM(vllm_config=vllm_config)
            generator = torch.Generator(device=device).manual_seed(0)
            with torch.no_grad():
                for name, p in lm.named_parameters():
                    if p.dtype.is_floating_point:
                        scale = 0.5 if p.dim() == 1 else 0.04
                        p.copy_(
                            torch.randn(p.shape, generator=generator, device=device)
                            * scale
                        )
            lm.process_weights_after_loading()
            assert lm.mono is None  # --monokernel off in this config
            spec = SlidingWindowSpec(
                block_size=16,
                num_kv_heads=TINY_DSPARK["num_key_value_heads"],
                head_size=64,
                dtype=torch.bfloat16,
                sliding_window=2048,
            )
            yield TinyDraft(lm, vllm_config, spec)
        cleanup_dist_env_and_memory()


@requires_gfx950
def test_layer_guards_accept_the_tiny_draft(tiny_draft) -> None:
    dims = draft_dims(tiny_draft.lm.config, 1)
    assert draft_layer_refusals(tiny_draft.model, dims) == []


class DraftStep:
    """One draft step: the draft group's real attention metadata, KV caches with
    random context keys, and the query rows of DSpark (7 a request)."""

    def __init__(self, tiny: TinyDraft, query_lens: list[int], seed: int) -> None:
        from tests.v1.attention.utils import (
            BatchSpec,
            create_common_attn_metadata,
            dense_kv_cache_views,
        )
        from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend
        from vllm.v1.kv_cache_layout import KVCacheLayout

        device = torch.device("cuda", 0)
        generator = torch.Generator().manual_seed(seed)
        num_reqs = len(query_lens)
        # contexts below and past the 2048-key window
        seq_lens = [
            int(torch.randint(40, 2600, (), generator=generator)) + q
            for q in query_lens
        ]
        self.common = create_common_attn_metadata(
            BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
            16,
            device,
            arange_block_indices=True,
        )
        self.positions = torch.cat(
            [torch.arange(s - q, s) for s, q in zip(seq_lens, query_lens)]
        ).to(device)
        # the draft group's slots of the query rows (vLLM: block_table[pos // 16])
        req = torch.repeat_interleave(
            torch.arange(num_reqs, device=device),
            torch.tensor(query_lens, device=device),
        )
        bt = self.common.block_table_tensor
        self.common.slot_mapping.copy_(
            bt[req, self.positions // 16].long() * 16 + self.positions % 16
        )
        num_blocks = int(bt.max()) + 1
        builder = AiterFlashAttentionBackend.get_builder_cls()(
            tiny.spec, tiny.names, tiny.vllm_config, device
        )
        self.metadata = builder.build(0, self.common)
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.kv_init: dict[str, torch.Tensor] = {}
        for name in tiny.names:
            raw = torch.zeros(
                num_blocks * tiny.spec.page_size_bytes, dtype=torch.int8, device=device
            )
            (view,) = dense_kv_cache_views(
                raw, tiny.spec, num_blocks, 1, KVCacheLayout.LBHNC
            )
            self.kv_caches[name] = view
            self.kv_init[name] = torch.randn(view.shape, generator=generator).to(
                device, torch.bfloat16
            )
        self.tokens = sum(query_lens)
        self.input_ids = torch.randint(
            0, TINY_DSPARK["vocab_size"], (self.tokens,), generator=generator
        ).to(device)

    def run(self, tiny: TinyDraft, mode=CUDAGraphMode.NONE):
        """The draft model's forward on the initial KV caches; returns the hidden
        states and the KV caches after the step."""
        from vllm.forward_context import set_forward_context
        from vllm.v1.worker.utils import bind_kv_cache

        for name, cache in self.kv_caches.items():
            cache.copy_(self.kv_init[name])
        context = tiny.vllm_config.compilation_config.static_forward_context
        bind_kv_cache(self.kv_caches, context, [])
        with (
            set_current_vllm_config(tiny.vllm_config),
            set_forward_context(
                dict.fromkeys(tiny.names, self.metadata),
                tiny.vllm_config,
                num_tokens=self.tokens,
                cudagraph_runtime_mode=mode,
                slot_mapping=dict.fromkeys(tiny.names, self.common.slot_mapping),
            ),
        ):
            hidden = tiny.lm(self.input_ids, self.positions, None)
        kv = {name: cache.clone() for name, cache in self.kv_caches.items()}
        return hidden.clone(), kv


_DRAFT_PARAMS = "x_in out wtab dtab pos slots scratch epoch health rank"
_DESC_WORDS = 10


class _StockDraftKernel:
    """kernels/draft.py with the launch replaced by the stock draft layers, run on
    the memory at the launch's pointers after checking them against the model."""

    TAB_WORDS, LAYERS = 16, 5
    TAB_ROWS = LAYERS + 2
    T_BND, T_LAYER, T_G, G_Q, G_PEER = 0, 4, LAYERS + 1, 0, 4
    HEALTH_WORDS = 8 + 8 * 4

    def __init__(self, tiny: TinyDraft) -> None:
        self.tiny = tiny
        self.calls: list[dict] = []

    def peer_layout(self, S, world=8):
        return {}, 4096

    def scratch_layout(self, S, world=8):
        return {}, 4096

    def build_vllm(self, S, W):
        return S

    def build(self, S):
        signature = inspect.Signature(
            [
                inspect.Parameter(p, inspect.Parameter.POSITIONAL_ONLY)
                for p in _DRAFT_PARAMS.split()
            ]
            + [inspect.Parameter("stream", inspect.Parameter.KEYWORD_ONLY)]
        )

        def launch(*args, **kwargs):
            bound = dict(signature.bind(*args, **kwargs).arguments)
            self.calls.append(bound | {"S": S})
            self._run(S, bound)

        return launch

    def _run(self, S: int, a: dict) -> None:
        model, names = self.tiny.model, self.tiny.names
        layers = model.layers
        rows = QUERY_ROWS * (S // 8)
        h = model.config.hidden_size
        tab = _device_view(a["wtab"], (self.TAB_ROWS, self.TAB_WORDS), torch.int64)
        tab = tab.tolist()
        desc = _device_view(a["dtab"], (64, _DESC_WORDS), torch.int64).tolist()

        def f32(ptr: int) -> torch.Tensor:
            return _device_view(ptr, (h,), torch.float32)

        # the boundary rows: the embeddings' row (ones, zeros), the FFN boundaries
        assert torch.equal(f32(tab[0][0]), torch.ones(h, device="cuda"))
        assert torch.equal(f32(tab[0][1]), torch.zeros(h, device="cuda"))
        for i in range(self.LAYERS + 1):
            nxt = (
                model.norm
                if i == self.LAYERS
                else layers[i].self_attn.pre_qkv_proj_norm
            )
            assert torch.equal(f32(tab[i][3]), effective_norm_weight_fp32(nxt))
            if i:
                prev = layers[i - 1]
                assert tab[i][0:2] == [
                    prev.add_ffn.alpha.data_ptr(),
                    prev.add_ffn.beta.data_ptr(),
                ]
                assert torch.equal(
                    f32(tab[i][2]), effective_norm_weight_fp32(prev.post_ffn_norm)
                )
        md, _, _, slots = get_attention_context(names[0])
        assert a["slots"] == slots.data_ptr()
        q = tab[self.T_G][self.G_Q]
        for i, layer in enumerate(layers):
            attn, mlp = layer.self_attn, layer.feed_forward
            assert tab[i][4:8] + tab[i][9:11] == [
                attn.qkv_proj.weight.data_ptr(),
                attn.o_proj.weight.data_ptr(),
                layer.add_attn.alpha.data_ptr(),
                layer.add_attn.beta.data_ptr(),
                mlp.gate_up_proj.weight.data_ptr(),
                mlp.down_proj.weight.data_ptr(),
            ]
            assert torch.equal(
                f32(tab[i][8]), effective_norm_weight_fp32(layer.pre_ffn_norm)
            )
            _, _, kv_cache, _ = get_attention_context(names[i])
            assert desc[i][:5] == [
                q,
                kv_cache.data_ptr(),
                kv_cache.stride(0),
                md.block_table.data_ptr(),
                md.seq_lens.data_ptr(),
            ]
            assert desc[i][8] == rows // QUERY_ROWS
        assert (
            tab[self.T_G][self.G_Q + 2]
            == layers[0].self_attn.rotary_emb.cos_sin_cache.data_ptr()
        )
        # the stock layer loop on the launch's input, positions and output
        x = _device_view(a["x_in"], (rows, h), torch.bfloat16)
        positions = _device_view(a["pos"], (rows,), torch.int64)
        out = _device_view(a["out"], (rows, h), torch.bfloat16)
        out.copy_(model.forward(None, positions, x))
        _device_view(a["epoch"], (1,), torch.int32).add_(1)


def _run_drafter(tiny: TinyDraft, step: DraftStep, **kw):
    """The step with the draft kernels attached, on the stock launch."""
    from vllm.distributed import get_tp_group

    kernel = _StockDraftKernel(tiny)
    from vllm.models.muse_spark.amd.mono.kernels import attn_frag

    tiny.lm.mono = DSparkDrafter(
        tiny.model,
        {"draft": kernel, "attn_frag": attn_frag},
        rank=0,
        group=get_tp_group().cpu_group,
        device=torch.device("cuda", 0),
        dims=draft_dims(tiny.lm.config, 1),
    )
    try:
        return kernel, step.run(tiny, **kw)
    finally:
        tiny.lm.mono.close()
        tiny.lm.mono = None


def _assert_same_step(stock, mono) -> None:
    (hidden_s, kv_s), (hidden_m, kv_m) = stock, mono
    torch.testing.assert_close(hidden_m, hidden_s, rtol=0, atol=0)
    for name in kv_s:
        torch.testing.assert_close(kv_m[name], kv_s[name], rtol=0, atol=0)


@requires_gfx950
@pytest.mark.parametrize("num_reqs", [1, 2])
def test_draft_step_matches_the_stock_layers(tiny_draft, num_reqs) -> None:
    """7 / 14 rows: one launch (S = 8 / 16) a step, the same draft hidden states
    and draft KV caches as the stock draft layers; the KV insert of the 7 query
    rows lands in the draft group's slots."""
    pytest.importorskip("flydsl")
    step = DraftStep(tiny_draft, [QUERY_ROWS] * num_reqs, seed=num_reqs)
    stock = step.run(tiny_draft)
    for name in tiny_draft.names:  # the step wrote the query rows' keys
        assert not torch.equal(stock[1][name], step.kv_init[name])
    kernel, mono = _run_drafter(tiny_draft, step)
    assert [c["S"] for c in kernel.calls] == [8 * num_reqs]
    _assert_same_step(stock, mono)


@requires_gfx950
def test_second_step_reuses_the_descriptor_table(tiny_draft) -> None:
    pytest.importorskip("flydsl")
    from vllm.distributed import get_tp_group
    from vllm.models.muse_spark.amd.mono.kernels import attn_frag

    step = DraftStep(tiny_draft, [QUERY_ROWS], seed=11)
    stock = step.run(tiny_draft)
    kernel = _StockDraftKernel(tiny_draft)
    tiny_draft.lm.mono = DSparkDrafter(
        tiny_draft.model,
        {"draft": kernel, "attn_frag": attn_frag},
        rank=0,
        group=get_tp_group().cpu_group,
        device=torch.device("cuda", 0),
        dims=draft_dims(tiny_draft.lm.config, 1),
    )
    try:
        first, second = step.run(tiny_draft), step.run(tiny_draft)
        b = tiny_draft.lm.mono.builds[8]
        assert len(b.fa.tables) == 1
        assert kernel.calls[0]["dtab"] == kernel.calls[1]["dtab"]
        assert kernel.calls[0]["wtab"] == b.table.data_ptr()
        assert int(b.epoch[0]) == 2
    finally:
        tiny_draft.lm.mono.close()
        tiny_draft.lm.mono = None
    _assert_same_step(stock, first)
    _assert_same_step(stock, second)


@requires_gfx950
@pytest.mark.parametrize(
    ("query_lens", "mode"),
    [
        ([QUERY_ROWS] * 3, CUDAGraphMode.NONE),  # 21 rows: no build
        ([QUERY_ROWS], CUDAGraphMode.PIECEWISE),
        ([5, 2], CUDAGraphMode.NONE),  # 7 rows, not 7 a request
    ],
    ids=["21-rows", "piecewise", "uneven"],
)
def test_other_draft_steps_run_stock(tiny_draft, query_lens, mode) -> None:
    pytest.importorskip("flydsl")
    step = DraftStep(tiny_draft, query_lens, seed=len(query_lens))
    stock = step.run(tiny_draft, mode=mode)
    kernel, mono = _run_drafter(tiny_draft, step, mode=mode)
    assert kernel.calls == []
    _assert_same_step(stock, mono)
