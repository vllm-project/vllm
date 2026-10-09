# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 mono decode layer runs only where its kernels are exact:
decode-only steps of at most MAX_ROWS rows with causal SWA windows, eager or in
a FULL graph. A PIECEWISE capture is replayed for mixed batches, so it must
never record the mono launches. Layers outside the whole-layer kernels run the
FFN launch on the same steps, from wo_b's unreduced output."""

import sys
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

    def forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return ("mono outputs",)

    def ffn(self, *args):
        self.calls.append((args, {}))
        return ("ffn outputs",)


def _layer(comp="comp"):
    cache = lambda: torch.zeros(4, 128 * md.RECORD, dtype=torch.uint8)  # noqa: E731
    attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(prefix="swa", kv_cache=cache()),
        compressed_cache_prefix=comp,
        topk_indices_buffer=torch.zeros(64, 512, dtype=torch.int32),
        compress_ratio=1,
        _compressed_kv_cache=cache,
    )
    return SimpleNamespace(attn=attn)


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
        mode=CUDAGraphMode.FULL, metadata=None, ffn_only=False, entry=None, **inputs
    ):
        if metadata is None:
            metadata = {
                "swa": _swa(),
                "comp": SimpleNamespace(block_size=128, block_table=None),
            }
        fc = SimpleNamespace(cudagraph_runtime_mode=mode, attn_metadata=metadata)
        runner = _Runner()
        monkeypatch.setattr(md, "is_forward_context_available", lambda: True)
        monkeypatch.setattr(md, "get_forward_context", lambda: fc)
        monkeypatch.setattr(md, "_mono_runner", lambda device: runner)
        monkeypatch.setattr(md.MonoDecodeLayer, "weights", lambda self, layer: None)
        args = {**_inputs(), **inputs}
        mono = md.MonoDecodeLayer(ffn_only)
        if (entry or ("ffn" if ffn_only else "layer")) == "ffn":
            # x stands for wo_b's unreduced output; the first layer's attention
            # has no compressed cache
            part, _ = args.pop("x"), args.pop("positions")
            return mono.ffn(_layer(comp=None), part, **args), runner.calls
        return mono(_layer(), **args), runner.calls

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
        dict(residual=None),  # the first layer's seam broadcasts the embedding
    ],
    ids=[
        "piecewise",
        "prefill",
        "noncausal-window",
        "tokens-past-rows",
        "no-metadata",
        "rows",
        "first-layer",
    ],
)
def test_other_steps_take_the_original_path(run, case):
    out, calls = run(**case)
    assert out is None and not calls


def _decoder_layer(engram=None, **attn):
    attn = (
        dict(
            layer_id=21,
            compressor=None,
            indexer=None,
            compress_ratio=1,
            kv_mxfp8=False,
            kv_cache_dtype="fp8_ds_mla",
        )
        | attn
    )
    backend = Mxfp4MoeBackend.AITER_MXFP4_BF16
    quant_method = SimpleNamespace(mxfp4_backend=backend)
    ffn = SimpleNamespace(
        experts=SimpleNamespace(
            routed_experts=SimpleNamespace(quant_method=quant_method)
        ),
        shared_experts=object(),
        gate=SimpleNamespace(tid2eid=None),
        n_routed_experts=384,
        n_activated_experts=6,
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


@pytest.mark.parametrize(
    "layer, path",
    [
        (_decoder_layer(), "whole"),
        (_decoder_layer(compress_ratio=0), "ffn"),
        (_decoder_layer(engram=object()), "ffn"),
        (_decoder_layer(compressor=object(), indexer=object()), "ffn"),
        (_decoder_layer(indexer=object()), "ffn"),
        (_decoder_layer(kv_cache_dtype="auto"), "ffn"),
        (_decoder_layer(layer_id=40), None),
    ],
    ids=["standard", "first", "engram", "kv-source", "index-source", "kv", "draft"],
)
def test_layers_take_whole_or_ffn_launch(monkeypatch, layer, path):
    """The whole-layer kernels serve the standard layers; every other backbone
    layer keeps vLLM's attention and runs its FFN half as one launch."""
    _deployment(monkeypatch)
    mono = md.MonoDecodeLayer.create(layer, _config())
    got = None if mono is None else "ffn" if mono.ffn_only else "whole"
    assert got == path


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_decode_step_runs_ffn_launch(run, mode):
    out, calls = run(mode, ffn_only=True, metadata={"swa": _swa()})
    assert out == ("ffn outputs",) and len(calls) == 1
    assert calls[0][0][1].shape == (M, 5120)  # wo_b's unreduced output


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


def _config(**parallel):
    parallel = (
        dict(enable_expert_parallel=False, enable_eplb=False, data_parallel_size=1)
        | parallel
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=40)),
        parallel_config=SimpleNamespace(**parallel),
    )


def _deployment(monkeypatch, cdna=4, cus=256):
    """A deployment the kernels take (the runner module stubbed: no FlyDSL)."""
    import vllm.platforms.rocm as rocm

    monkeypatch.setenv("VLLM_ROCM_MONO_DECODE", "1")
    monkeypatch.delenv("VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4", raising=False)
    monkeypatch.setattr(rocm, "get_cdna_version", lambda: cdna)
    monkeypatch.setattr(md, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(md, "_compute_units", lambda: cus)
    name = "vllm.models.deepseek_v41.amd.mono.runner"
    runner = ModuleType(name)
    runner.BLOCKS = 256  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, runner)


@pytest.mark.parametrize("cdna", [3, 5])
def test_opt_in_outside_the_kernels_cdna_raises(monkeypatch, cdna):
    _deployment(monkeypatch, cdna=cdna)
    with pytest.raises(ValueError, match="needs CDNA4"):
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
