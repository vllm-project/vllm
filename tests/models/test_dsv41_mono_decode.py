# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 mono decode layer runs only where its kernels are exact:
decode-only steps of at most MAX_ROWS rows (MAX_ROWS_GFX942 on gfx942) with
causal SWA windows, eager or in a FULL graph. A PIECEWISE capture is replayed
for mixed batches, so it must never record the mono launches. Layers outside
the whole-layer kernels run the FFN launch on the same steps, from wo_b's
unreduced output."""

import sys
from collections import defaultdict
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.models.deepseek_v41.amd import mono_decode as md

M = 6


class _Runner:
    def __init__(self):
        self.calls = []
        self.draft = None

    def forward(self, *args, **kwargs):
        self.calls.append((args, {**kwargs, "draft_runner": self.draft}))
        return ("mono outputs",)

    def first_pre_mix(self, rows):
        return torch.zeros(rows, 4)

    def ffn(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return ("ffn outputs",)


def _layer(comp="comp", indexer=None):
    cache = lambda: torch.zeros(4, 128 * md.RECORD, dtype=torch.uint8)  # noqa: E731
    attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(prefix="swa", kv_cache=cache()),
        compressed_cache_prefix=comp,
        topk_indices_buffer=torch.zeros(64, 512, dtype=torch.int32),
        compress_ratio=1,
        indexer=indexer,
        _compressed_kv_cache=cache,
    )
    return SimpleNamespace(attn=attn, engram=None, hc_mult=4)


def _swa(**overrides):
    fields = dict(
        num_prefills=0,
        num_decodes=1,
        num_decode_tokens=M,
        decode_swa_width=md.SWA_WIDTH,
        decode_swa_indices=torch.zeros(64, 1, md.SWA_WIDTH, dtype=torch.int32),
        decode_swa_lens=torch.zeros(64, dtype=torch.int32),
        token_to_req_indices=torch.zeros(64, dtype=torch.int32),
        slot_mapping=torch.zeros(64, dtype=torch.int64),
        block_size=32,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _inputs(rows=M):
    return dict(
        x=torch.zeros(rows, 5120, dtype=torch.bfloat16),
        positions=torch.zeros(rows, dtype=torch.int64),
        residual=torch.zeros(rows, 4, 5120, dtype=torch.bfloat16),
        post_mix=torch.zeros(rows, 4, 1),
        res_mix=torch.zeros(rows, 4, 4),
        pre_mix=torch.zeros(rows, 4),
    )


@pytest.fixture
def run(monkeypatch):
    """Call the mono path for one layer under a forward context built from
    ``mode`` and ``metadata``; returns (result, runner calls)."""

    def _run(
        mode=CUDAGraphMode.FULL,
        metadata=None,
        ffn_only=False,
        entry=None,
        layer=None,
        max_rows=md.MAX_ROWS,
        mono=None,
        **inputs,
    ):
        # The row limit depends on the GPU (MAX_ROWS_GFX942 on gfx942), so the
        # test sets it and does not ask the GPU of the machine it runs on.
        monkeypatch.setattr(md, "_max_rows", lambda: max_rows)
        if metadata is None:
            metadata = {
                "swa": _swa(),
                "comp": SimpleNamespace(block_size=128, block_table=None),
            }
        fc = SimpleNamespace(cudagraph_runtime_mode=mode, attn_metadata=metadata)
        runner = _Runner()
        monkeypatch.setattr(md, "is_forward_context_available", lambda: True)
        monkeypatch.setattr(md, "get_forward_context", lambda: fc)

        def pick(device, draft=None):
            runner.draft = draft
            return runner

        monkeypatch.setattr(md, "_mono_runner", pick)
        monkeypatch.setattr(md.MonoDecodeLayer, "weights", lambda self, layer: None)
        args = {**_inputs(), **inputs}
        mono = mono or md.MonoDecodeLayer(ffn_only)
        if (entry or ("ffn" if ffn_only else "layer")) == "ffn":
            # x stands for wo_b's unreduced output; the first layer's attention
            # has no compressed cache
            part, _ = args.pop("x"), args.pop("positions")
            return mono.ffn(_layer(comp=None), part, **args), runner.calls
        return mono(layer or _layer(), **args), runner.calls

    return _run


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_decode_step_runs_mono(run, mode):
    out, calls = run(mode)
    assert out == ("mono outputs",) and len(calls) == 1
    args, kwargs = calls[0]
    # the caches reach the kernels as [blocks, block, record] bytes
    assert args[8].shape == (4, 32, md.RECORD) and args[8].stride() == (
        128 * md.RECORD,
        md.RECORD,
        1,
    )
    assert kwargs["comp_cache"].shape == (4, 128, md.RECORD)
    assert kwargs["fold"] and "select" not in kwargs


def test_first_layer_runs_mono_from_the_embedding(run):
    """The first layer's residual is the embedding broadcast to the four
    streams, which K1 takes as it is (no owed post to fold)."""
    out, calls = run(residual=None, post_mix=None, res_mix=None, pre_mix=None)
    assert out == ("mono outputs",) and len(calls) == 1
    args, kwargs = calls[0]
    assert args[2].shape == (M, 4, 5120) and not kwargs["fold"]


DRAFT_WIDTH = 192  # the DSpark draft's non-causal SWA rows: window 128 + 5 drafts


def _draft(**overrides):
    """A DSpark draft layer's mono path, and the metadata its step carries."""
    layer = _layer(comp=None)
    layer.attn.compress_ratio = 0
    swa = _swa(
        decode_swa_width=DRAFT_WIDTH,
        decode_swa_indices=torch.zeros(64, 1, DRAFT_WIDTH, dtype=torch.int32),
    )
    mono = md.MonoDecodeLayer(draft=True, experts=128, topk=3, swa_width=DRAFT_WIDTH)
    return dict(layer=layer, mono=mono, metadata={"swa": swa}, **overrides)


def test_draft_layer_runs_mono_on_its_own_runner(run):
    out, calls = run(**_draft())
    assert out == ("mono outputs",) and len(calls) == 1
    assert calls[0][0][9].stride(0) == DRAFT_WIDTH  # the non-causal SWA rows
    assert calls[0][1]["draft_runner"] is not None  # not the target's runner


def test_draft_first_layer_takes_the_broadcast_embedding(run):
    """The draft hands its first layer the embedding already repeated to the
    hc copies: K1 takes it as the settled residual."""
    x = torch.zeros(M, 4, 5120, dtype=torch.bfloat16)
    out, calls = run(
        **_draft(residual=None, post_mix=None, res_mix=None, pre_mix=None, x=x)
    )
    assert out == ("mono outputs",)
    args, kwargs = calls[0]
    assert args[1].shape == (M, 5120) and args[2] is x and not kwargs["fold"]


def test_draft_and_target_layers_take_only_their_own_swa_rows(run):
    causal = {"swa": _swa(), "comp": SimpleNamespace(block_size=128)}
    out, calls = run(**{**_draft(), "metadata": causal})
    assert out is None and not calls
    target = {"swa": _draft()["metadata"]["swa"], "comp": causal["comp"]}
    out, calls = run(metadata=target)
    assert out is None and not calls


def test_index_source_selects_between_the_kernels(run):
    """An index source hands the runner its compressor and indexer to run
    between K1 and K2."""
    out, calls = run(layer=_layer(indexer=object()))
    assert out == ("mono outputs",) and callable(calls[0][1]["select"])


def test_padded_decode_step_runs_mono(run):
    """A FULL graph pads the step to its capture size: the kernels take it and
    skip the pad rows (their slot is -1)."""
    swa = _swa(num_decode_tokens=M - 2)
    comp = SimpleNamespace(block_size=128, block_table=None)
    out, calls = run(metadata={"swa": swa, "comp": comp})
    assert out == ("mono outputs",) and len(calls) == 1


@pytest.mark.parametrize(
    "case",
    [
        dict(mode=CUDAGraphMode.PIECEWISE),
        dict(
            metadata={
                "swa": _swa(num_prefills=1),
                "comp": SimpleNamespace(block_size=128),
            }
        ),
        dict(
            metadata={
                "swa": _swa(decode_swa_width=256),
                "comp": SimpleNamespace(block_size=128),
            }
        ),
        dict(
            metadata={
                "swa": _swa(num_decode_tokens=M + 1),
                "comp": SimpleNamespace(block_size=128),
            }
        ),
        dict(metadata={}),  # a profile / dummy run: no attention metadata
        dict(**_inputs(md.MAX_ROWS + 6)),
        # a later pipeline stage's first layer: its stream arrives without
        # the mixes
        dict(residual=None, x=torch.zeros(M, 4, 5120, dtype=torch.bfloat16)),
    ],
    ids=[
        "piecewise",
        "prefill",
        "noncausal-window",
        "tokens-past-rows",
        "no-metadata",
        "rows",
        "stage-input",
    ],
)
def test_other_steps_take_the_original_path(run, case):
    out, calls = run(**case)
    assert out is None and not calls


def _linear():
    """A dense linear whose weights vLLM processes at load."""
    return SimpleNamespace(
        weight=object(),
        weight_scale=object(),
        quant_method=SimpleNamespace(process_weights_after_loading=lambda layer: None),
    )


# The routed experts and picks a token of the target's MoE layers and of the
# DSpark draft's 3 layers (dspark_n_routed_experts, dspark_num_experts_per_tok).
TARGET_ROUTING = (384, 6)
DRAFT_ROUTING = (128, 3)


def _decoder_layer(engram=None, routing=TARGET_ROUTING, **attn):
    attn = (
        dict(
            layer_id=21,
            compressor=None,
            indexer=None,
            compress_ratio=1,
            kv_mxfp8=False,
            kv_cache_dtype="fp8_ds_mla",
        )
        | {name: _linear() for name in md.ATTN_LINEARS_942}
        | attn
    )
    backend = Mxfp4MoeBackend.AITER_MXFP4_BF16
    # On gfx942, create wraps process_weights_after_loading to copy the experts.
    quant_method = SimpleNamespace(
        mxfp4_backend=backend, process_weights_after_loading=lambda layer: None
    )
    ffn = SimpleNamespace(
        experts=SimpleNamespace(
            routed_experts=SimpleNamespace(quant_method=quant_method)
        ),
        shared_experts=SimpleNamespace(gate_up_proj=_linear(), down_proj=_linear()),
        gate=SimpleNamespace(tid2eid=None),
        n_routed_experts=routing[0],
        n_activated_experts=routing[1],
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        scoring_func="sqrtsoftplus",
        renormalize=True,
    )
    return SimpleNamespace(
        attn=SimpleNamespace(**attn),
        ffn=ffn,
        engram=engram,
        use_sequence_parallel=False,
        fuse_seam_norm=True,
    )


def _indexer():
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41RocmMxfp4Indexer

    return object.__new__(DeepseekV41RocmMxfp4Indexer)


@pytest.mark.parametrize(
    "layer, path",
    [
        (_decoder_layer(), "whole"),
        (_decoder_layer(compress_ratio=0), "whole"),
        (_decoder_layer(engram=object()), "whole"),
        (_decoder_layer(compressor=object(), indexer=_indexer()), "whole"),
        (_decoder_layer(indexer=_indexer()), "whole"),
        (_decoder_layer(compressor=object()), "ffn"),
        (_decoder_layer(indexer=object()), "ffn"),
        (_decoder_layer(kv_cache_dtype="auto"), "ffn"),
        (_decoder_layer(layer_id=40), None),
    ],
    ids=[
        "standard",
        "window",
        "engram",
        "kv-source",
        "index-source",
        "compressor-only",
        "other-indexer",
        "kv",
        "draft",
    ],
)
def test_layers_take_whole_or_ffn_launch(monkeypatch, layer, path):
    """The whole-layer kernels serve every backbone layer of the checkpoint (an
    index source's compressor and indexer run between them); a layer outside
    them keeps vLLM's attention and runs its FFN half as one launch."""
    _deployment(monkeypatch)
    mono = md.MonoDecodeLayer.create(layer, _config())
    got = None if mono is None else "ffn" if mono.ffn_only else "whole"
    assert got == path


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_decode_step_runs_ffn_launch(run, mode):
    out, calls = run(mode, ffn_only=True, metadata={"swa": _swa()})
    assert out == ("ffn outputs",) and len(calls) == 1
    assert calls[0][0][1].shape == (M, 5120)  # wo_b's unreduced output
    assert calls[0][1] == {"topk": TARGET_ROUTING[1]}


def test_ffn_layer_never_runs_whole(run):
    out, calls = run(ffn_only=True, entry="layer")
    assert out is None and not calls


@pytest.mark.parametrize(
    "case",
    [
        dict(mode=CUDAGraphMode.PIECEWISE),
        dict(metadata={"swa": _swa(num_prefills=1)}),
        dict(metadata={"swa": _swa(num_decode_tokens=M + 1)}),
        dict(metadata={}),
        dict(**_inputs(md.MAX_ROWS + 6)),
    ],
    ids=["piecewise", "prefill", "tokens-past-rows", "no-metadata", "rows"],
)
def test_other_steps_reduce_wo_b_themselves(run, case):
    """None: the layer all-reduces wo_b's partial and runs the FFN as before."""
    case.setdefault("metadata", {"swa": _swa()})
    out, calls = run(ffn_only=True, **case)
    assert out is None and not calls


def _config(monokernel=True, **parallel):
    parallel = (
        dict(enable_expert_parallel=False, enable_eplb=False, data_parallel_size=1)
        | parallel
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=40,
                dspark_n_routed_experts=DRAFT_ROUTING[0],
                dspark_num_experts_per_tok=DRAFT_ROUTING[1],
            )
        ),
        parallel_config=SimpleNamespace(**parallel),
        kernel_config=SimpleNamespace(monokernel=monokernel),
    )


def _deployment(monkeypatch, cdna=4, cus=256, tp=2, target=None):
    """A deployment the kernels take (the runner module stubbed: no FlyDSL).
    ``target`` is the architecture FlyDSL compiles for, by default the GPU's."""
    import vllm.platforms.rocm as rocm

    monkeypatch.delenv("VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4", raising=False)
    monkeypatch.setattr(rocm, "get_cdna_version", lambda: cdna)
    monkeypatch.setattr(md, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(md, "_compute_units", lambda: cus)
    gpu = "gfx942" if cdna == 3 else "gfx950"
    monkeypatch.setattr(md, "_gpu_arch", lambda: gpu)
    arch = ModuleType("vllm.models.deepseek_v41.amd.mono.common.arch")
    arch.target_arch = lambda: target or gpu  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, arch.__name__, arch)
    # On gfx942, create builds the top-k extension with hipcc. These tests do
    # not need it, so they skip the build.
    from vllm.models.deepseek_v41.amd import ops
    from vllm.models.deepseek_v41.amd.ops import topk

    monkeypatch.setattr(topk, "build", lambda: True)
    # create records --monokernel for the gfx942 helper ops. The test puts the
    # recorded value back afterwards, so that no later test sees this one's.
    monkeypatch.setattr(ops, "_monokernel", ops._monokernel)
    name = "vllm.models.deepseek_v41.amd.mono.runner"
    runner = ModuleType(name)
    runner.BLOCKS = 256  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, runner)


def test_off_without_the_flag(monkeypatch):
    """No --monokernel: no mono layer, whatever the deployment."""
    _deployment(monkeypatch)
    assert (
        md.MonoDecodeLayer.create(_decoder_layer(), _config(monokernel=False)) is None
    )


@pytest.mark.parametrize("on", [True, False])
def test_create_records_the_flag_for_the_gfx942_helpers(monkeypatch, on):
    """The gfx942 decode helpers (amd/ops) run in forward passes, where vLLM's
    current config is not set, so create records --monokernel for them."""
    from vllm.models.deepseek_v41.amd import ops

    _deployment(monkeypatch, cdna=3, tp=4)
    md.MonoDecodeLayer.create(_decoder_layer(), _config(monokernel=on))
    assert ops._monokernel is on


@pytest.mark.parametrize("cdna", [2, 5])
def test_opt_in_outside_the_kernels_cdna_raises(monkeypatch, cdna):
    _deployment(monkeypatch, cdna=cdna)
    with pytest.raises(ValueError, match="needs CDNA3/4"):
        md.MonoDecodeLayer.create(_decoder_layer(), _config())


@pytest.mark.parametrize(
    "cdna, tp, target", [(3, 4, "gfx950"), (4, 2, "gfx942")], ids=["gfx942", "gfx950"]
)
def test_opt_in_with_another_arch_for_flydsl_raises(monkeypatch, cdna, tp, target):
    """An ARCH environment variable that names another architecture than the
    GPU's would make FlyDSL build kernels that do not match the weight copies,
    so the opt-in stops the server."""
    _deployment(monkeypatch, cdna=cdna, tp=tp, target=target)
    with pytest.raises(ValueError, match=f"makes it compile for {target}"):
        md.MonoDecodeLayer.create(_decoder_layer(), _config())


def _other_backend():
    layer = _decoder_layer()
    experts = layer.ffn.experts.routed_experts
    experts.quant_method.mxfp4_backend = Mxfp4MoeBackend.TRITON
    return layer


@pytest.mark.parametrize(
    "case, match",
    [
        (dict(parallel=dict(enable_expert_parallel=True)), "no expert"),
        (dict(parallel=dict(data_parallel_size=2)), "no expert"),
        (dict(parallel=dict(enable_eplb=True)), "no expert"),
        (dict(a4w4=True), "A8W4"),
        (dict(layer=_other_backend), "AITER_MXFP4_BF16"),
        (dict(cus=128), "256 compute units"),
    ],
    ids=["expert-parallel", "data-parallel", "eplb", "a4w4", "moe-backend", "cus"],
)
def test_opt_in_outside_the_kernels_moe_raises(monkeypatch, case, match):
    """The kernels read all 384 experts on each rank in AITER's A8W4 layout, with
    all 256 CTAs resident: any other deployment is an error, not garbage."""
    _deployment(monkeypatch, cus=case.get("cus", 256))
    if case.get("a4w4"):
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4", "1")
    layer = case.get("layer", _decoder_layer)()
    with pytest.raises(ValueError, match=match):
        md.MonoDecodeLayer.create(layer, _config(**case.get("parallel", {})))


@pytest.mark.parametrize("rows, runs", [(md.MAX_ROWS_GFX942, True), (48, False)])
def test_gfx942_decode_steps_of_up_to_42_rows(run, rows, runs):
    """On gfx942 the kernels take steps of up to MAX_ROWS_GFX942 rows. A step of
    48 rows, which gfx950 takes, runs vLLM's path there."""
    swa = _swa(num_decode_tokens=rows)
    comp = SimpleNamespace(block_size=128, block_table=None)
    out, calls = run(
        metadata={"swa": swa, "comp": comp},
        max_rows=md.MAX_ROWS_GFX942,
        **_inputs(rows),
    )
    assert (out == ("mono outputs",)) == runs and len(calls) == int(runs)


def _kind(mono):
    if mono is None:
        return None
    return f"ffn top {mono.topk}" if mono.ffn_only else "whole"


def _indexer_942():
    from vllm.models.deepseek_v41.attention import DeepseekV4Indexer

    return object.__new__(DeepseekV4Indexer)


@pytest.mark.parametrize(
    "layer, kind",
    [
        (_decoder_layer(), "whole"),
        (_decoder_layer(compress_ratio=0), "whole"),
        (_decoder_layer(compress_ratio=0, engram=object()), "whole"),
        (_decoder_layer(engram=object()), "whole"),
        (_decoder_layer(compressor=object(), indexer=_indexer_942()), "whole"),
        (_decoder_layer(indexer=_indexer_942()), "whole"),
        (_decoder_layer(compressor=object()), "ffn top 6"),
        (_decoder_layer(indexer=object()), "ffn top 6"),
        (_decoder_layer(kv_cache_dtype="auto"), "ffn top 6"),
        (_decoder_layer(layer_id=40, routing=DRAFT_ROUTING), "ffn top 3"),
        (_decoder_layer(layer_id=40), None),
    ],
    ids=[
        "standard",
        "first",
        "window-engram",
        "engram",
        "kv-source",
        "index-source",
        "compressor-only",
        "other-indexer",
        "kv",
        "draft",
        "draft-target-routing",
    ],
)
def test_gfx942_layers(monkeypatch, layer, kind):
    """On gfx942 every backbone layer runs whole, as on gfx950. Its index
    layers have vLLM's FP8 indexer, because the MXFP4 indexer cache needs
    CDNA4. The DSpark draft layers run the FFN launch with the draft's routing
    of 128 experts and top 3."""
    _deployment(monkeypatch, cdna=3, tp=4)
    assert _kind(md.MonoDecodeLayer.create(layer, _config())) == kind


def test_gfx942_tp2_runs_vllm_path(monkeypatch):
    """The gfx942 kernels are built for TP4 only. At TP2 every layer runs
    vLLM's path, and that is not an error."""
    _deployment(monkeypatch, cdna=3, tp=2)
    assert md.MonoDecodeLayer.create(_decoder_layer(), _config()) is None


def test_gfx942_takes_any_moe_backend(monkeypatch):
    """The gfx942 kernels read their own copies of the experts, made before
    vLLM's MoE backend converts them, so a backend other than AITER_MXFP4_BF16
    is not an error there."""
    _deployment(monkeypatch, cdna=3, tp=4)
    assert _kind(md.MonoDecodeLayer.create(_other_backend(), _config())) == "whole"


@pytest.mark.parametrize(
    "make, copied",
    [
        (_decoder_layer, ("gate_up_proj", "down_proj", *md.ATTN_LINEARS_942)),
        (
            lambda: _decoder_layer(layer_id=40, routing=DRAFT_ROUTING),
            ("gate_up_proj", "down_proj"),
        ),
    ],
    ids=["standard", "draft"],
)
def test_gfx942_copies_dense_linears_at_load(monkeypatch, make, copied):
    """On gfx942 the kernels' copies of the dense linears are made when vLLM
    processes each linear's weights at load, so that vLLM counts them as model
    memory when it sizes the KV cache. A draft layer runs vLLM's attention, so
    its attention linears get no copies."""
    _deployment(monkeypatch, cdna=3, tp=4)
    weights942 = ModuleType("vllm.models.deepseek_v41.amd.mono.weights942")
    weights942.linear_copy = lambda w, s: ("copy", w, s)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, weights942.__name__, weights942)
    layer = make()
    assert md.MonoDecodeLayer.create(layer, _config()) is not None
    sh = layer.ffn.shared_experts
    linears = {"gate_up_proj": sh.gate_up_proj, "down_proj": sh.down_proj}
    linears |= {name: getattr(layer.attn, name) for name in md.ATTN_LINEARS_942}
    for linear in linears.values():
        linear.quant_method.process_weights_after_loading(linear)
    made = {name for name, linear in linears.items() if hasattr(linear, "mono942")}
    assert made == set(copied)
    for name in copied:
        linear = linears[name]
        assert linear.mono942 == ("copy", linear.weight, linear.weight_scale)


def test_gfx942_reload_writes_the_copies_in_place(monkeypatch):
    """A reload of the weights (reload_weights, or a weight update in RL) runs
    process_weights_after_loading again. The kernels and captured CUDA graphs
    read the first copies by their addresses, so the copies of the reload are
    written into the first copies instead of replacing them."""
    _deployment(monkeypatch, cdna=3, tp=4)
    weights942 = ModuleType("vllm.models.deepseek_v41.amd.mono.weights942")
    weights942.linear_copy = lambda w, s: (w.clone(), s.clone())  # type: ignore[attr-defined]
    weights942.moe_copies = lambda w13, w13_s, w2, w2_s, inter: tuple(  # type: ignore[attr-defined]
        t.clone() for t in (w13, w13_s, w2, w2_s)
    )
    monkeypatch.setitem(sys.modules, weights942.__name__, weights942)
    layer = _decoder_layer()
    assert md.MonoDecodeLayer.create(layer, _config()) is not None
    experts, linear = layer.ffn.experts.routed_experts, layer.attn.wo_a

    def load(value):
        u8 = dict(dtype=torch.uint8)
        for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
            setattr(experts, name, torch.full((2, 4), value, **u8))
        linear.weight = torch.full((4, 8), value, **u8)
        linear.weight_scale = torch.full((4, 1), value, **u8)
        for module in (experts, linear):
            module.quant_method.process_weights_after_loading(module)

    load(1)
    first = (experts.mono942, linear.mono942)
    load(2)
    assert experts.mono942 is first[0] and linear.mono942 is first[1]
    assert all(bool((t == 2).all()) for t in (*first[0], *first[1]))


# With --monokernel on gfx942 the indexer layer and the DSpark draft sampling
# run gfx942 launches from the model folder. vLLM's ROCm indexer op and its
# DSpark speculator loop stay as they are, and they run every step that the
# gfx942 code does not take.


@pytest.mark.parametrize(
    "case, gfx942_path",
    [
        ("decode", True),
        ("profiling", False),
        ("prefill", False),
        ("mixed", False),
        ("no rows", False),
        ("writes its own K", False),
        ("without the flag", False),
    ],
)
def test_gfx942_indexer_runs_vllm_op_off_decode_steps(monkeypatch, case, gfx942_path):
    """Only a step of only decode rows, whose K cache the model already wrote,
    takes the gfx942 decode path. The profiling run, prefill and mixed steps,
    a layer that writes its own K cache and a process without --monokernel on
    gfx942 call vLLM's op with the same arguments."""
    from vllm.models.deepseek_v41.amd.ops import sparse_indexer as si

    prefills, decodes = {"prefill": (1, 0), "mixed": (1, 1), "no rows": (0, 0)}.get(
        case, (0, 1)
    )
    meta = SimpleNamespace(num_prefills=prefills, num_decodes=decodes)
    attn_metadata = None if case == "profiling" else {"idx": meta}
    monkeypatch.setattr(si, "DeepseekV32IndexerMetadata", SimpleNamespace)
    monkeypatch.setattr(
        si, "get_forward_context", lambda: SimpleNamespace(attn_metadata=attn_metadata)
    )
    monkeypatch.setattr(si, "enabled", lambda: case != "without the flag")
    calls = []

    def vllm_op(*args, **kwargs):
        calls.append(("vllm", args, kwargs))
        return "vllm"

    def decode_step(*args):
        calls.append(("gfx942", args, {}))
        return "gfx942"

    monkeypatch.setattr(si, "rocm_aiter_sparse_attn_indexer", vllm_op)
    monkeypatch.setattr(si, "_decode_step", decode_step)
    buffer = torch.zeros(6, 512, dtype=torch.int32)
    args = (torch.zeros(6, 4), "idx", torch.zeros(1), torch.zeros(1), None)
    args += (torch.zeros(1), 128, "ue8m0", 512, 128, 1024, 1024, buffer)
    kwargs = dict(
        skip_k_cache_insert=case != "writes its own K",
        compress_ratio=4,
        candidate_blocks=None,
        candidate_block_size=0,
        candidate_write=False,
    )
    out = si.gfx942_sparse_attn_indexer(*args, **kwargs)
    assert out == ("gfx942" if gfx942_path else "vllm")
    assert len(calls) == 1
    if not gfx942_path:
        assert calls[0][1] == args and calls[0][2] == kwargs


@pytest.mark.parametrize(
    "monokernel, gfx942, takes_gfx942_indexer",
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_rocm_attention_takes_the_gfx942_indexer_with_the_flag(
    monkeypatch, monokernel, gfx942, takes_gfx942_indexer
):
    """The attention builds its indexer before the mono decode layer records
    the flag for ops.enabled(), so it reads --monokernel from the config of
    the model that is being built. Without the flag, and on gfx950, the layer
    keeps vLLM's indexer."""
    import vllm.platforms.rocm as rocm_platform
    from vllm.models.deepseek_v41.amd import rocm
    from vllm.models.deepseek_v41.amd.ops.sparse_indexer import (
        Gfx942SparseAttnIndexer,
    )
    from vllm.models.deepseek_v41.attention import DeepseekV4Indexer

    config = SimpleNamespace(kernel_config=SimpleNamespace(monokernel=monokernel))
    monkeypatch.setattr(rocm, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(rocm_platform, "on_gfx942", lambda: gfx942)
    attn_cls = rocm.DeepseekV41ROCMAiterMLAAttention
    indexer_cls = attn_cls._indexer_cls(
        attn_cls.__new__(attn_cls), SimpleNamespace(rocm_mxfp4=False)
    )
    expected = (
        rocm.DeepseekV41Gfx942Indexer if takes_gfx942_indexer else DeepseekV4Indexer
    )
    assert indexer_cls is expected
    assert rocm.DeepseekV41Gfx942Indexer.attn_cls is Gfx942SparseAttnIndexer


@pytest.mark.parametrize("monokernel", [True, False])
@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx90a"])
def test_swa_decode_meta_runs_on_gfx942_and_gfx950_with_the_flag(
    monkeypatch, arch, monokernel
):
    """swa_decode_meta's kernel is plain Triton, so with --monokernel a decode
    batch gets its SWA metadata from the one launch on gfx950 as on gfx942.
    Without the flag, or on another arch, the builder keeps vLLM's build."""
    import vllm.platforms.rocm as rocm_platform
    from vllm.models.deepseek_v41.amd import ops, rocm
    from vllm.models.deepseek_v41.amd.ops import swa_decode_meta

    monkeypatch.setattr(ops, "current_platform", SimpleNamespace(is_rocm=lambda: True))
    monkeypatch.setattr(rocm_platform, "on_gfx942", lambda: arch == "gfx942")
    monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: arch == "gfx950")
    monkeypatch.setattr(ops, "_monokernel", ops._monokernel)
    ops.record_monokernel(_config(monokernel=monokernel))
    launches = []
    monkeypatch.setattr(
        swa_decode_meta, "launch", lambda *args, noncausal: launches.append(noncausal)
    )
    ints = lambda *shape: torch.zeros(shape, dtype=torch.int32)  # noqa: E731
    builder = SimpleNamespace(
        decode_threshold=M,
        window_size=md.SWA_WIDTH,
        block_size=128,
        decode_swa_indices=ints(64, 1, md.SWA_WIDTH),
        decode_swa_lens=ints(64),
        is_valid_token=torch.zeros(64, dtype=torch.bool),
        token_to_req_indices=ints(64),
        no_replay_start=ints(2),
        decode_swa_ragged_indices_buffer=ints(64 * md.SWA_WIDTH),
        decode_swa_ragged_indptr_buffer=ints(65),
        build_tile_scheduler=lambda rows: defaultdict(lambda: None),
        _build_deepseek_v4_metadata=lambda *args: {},
    )
    query_start_loc = torch.tensor([0, M], dtype=torch.int32)
    cam = SimpleNamespace(
        causal=True,
        max_query_len=M,
        num_reqs=1,
        num_actual_tokens=M,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=torch.tensor([1000], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([1000], dtype=torch.int32),
        slot_mapping=torch.arange(M),
        block_table_tensor=ints(1, 8),
        token_to_req_indices=lambda buffer: buffer[:M],
    )
    out = swa_decode_meta.build(
        builder, 0, cam, False, None, rocm.DeepseekV41ROCMAiterSparseSWAMetadata
    )
    # No later test may see this test's cached gate.
    ops.record_monokernel(_config(monokernel=False))
    takes_it = monokernel and arch != "gfx90a"
    assert launches == ([False] if takes_it else [])
    assert (out is not None) is takes_it


def test_dspark_speculator_asks_the_draft_model_first():
    """A draft model that samples a step itself returns True, and vLLM's
    loop then does not run. The stub speculator has none of the loop's
    buffers, so the loop would fail here."""
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    calls = []

    def sample_sequential(s, n, h):
        calls.append((s, n, h))
        return True

    spec = DSparkSpeculator.__new__(DSparkSpeculator)
    spec._draft_topk = None
    spec.model = SimpleNamespace(sample_sequential=sample_sequential)
    hidden = torch.zeros(4, 8)
    DSparkSpeculator._sample_sequential(spec, 2, hidden)
    assert calls == [(spec, 2, hidden)]


@pytest.mark.parametrize("taken", [True, False])
def test_dspark_draft_model_samples_with_fused_markov(monkeypatch, taken):
    """The DSv4.1 draft model answers the speculator with fused_markov, which
    returns False for a step that the fused launches do not take."""
    from vllm.models.deepseek_v41.amd import dspark
    from vllm.models.deepseek_v41.amd.ops import fused_markov

    calls = []

    def sample_sequential(s, n, h):
        calls.append((s, n, h))
        return taken

    monkeypatch.setattr(fused_markov, "sample_sequential", sample_sequential)
    spec, hidden = object(), torch.zeros(4, 8)
    model_cls = dspark.DSparkDeepseekV4ForCausalLM
    assert model_cls.sample_sequential(None, spec, 2, hidden) is taken
    assert calls == [(spec, 2, hidden)]
