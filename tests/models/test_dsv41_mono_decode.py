# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's MonoSpec against the conformance kit, plus its own choices."""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from tests.models.common.mono.kit import MonoConformance, forward
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.models.common.mono.runtime import attach, kernel_for_layer
from vllm.models.common.mono.spec import MonoUnsupported
from vllm.models.deepseek_v41.amd import mono_spec as ms

SPECS_PATH = ms.SPECS_PATH
LAYERS = 6
HIDDEN = 16
BLOCKS = 256


class _StubRunner:
    def __init__(self, world, rank, group, device):
        self.world = world
        self.built: list = []
        self.epoch = torch.zeros(1, dtype=torch.int32)

    def scratch(self, width):
        pass

    def kernels(self, *args, **kwargs):
        self.built.append((args, kwargs))

    def ffn_kernel(self, width):
        self.built.append(width)

    def first_pre_mix(self, rows):
        return torch.zeros(rows, 4)

    def forward(self, *args, **kwargs):
        return ("whole", args, kwargs)

    def ffn(self, *args):
        return ("ffn", args)


def _stub_modules(monkeypatch) -> None:
    """The FlyDSL runner package, as a spec-shaped stub."""
    runner: Any = ModuleType("vllm.models.deepseek_v41.amd.mono.runner")
    runner.BLOCKS = BLOCKS
    runner.MAX_TOKENS = 64
    runner.DSV41MonoLayer = _StubRunner
    runner.MonoLayerWeights = _weights_stub
    runner.AttnWeights = _weights_stub
    plan: Any = ModuleType("vllm.models.deepseek_v41.amd.mono.attention.plan")
    plan.Dims = lambda world: world
    for name, module in (
        ("vllm.models.deepseek_v41.amd.mono.runner", runner),
        ("vllm.models.deepseek_v41.amd.mono.attention.plan", plan),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def _weights_stub(**fields) -> Any:
    return SimpleNamespace(check=lambda *args: None, **fields)


def _bytes(*shape) -> torch.Tensor:
    return torch.zeros(*shape, dtype=torch.uint8)


def _proj(out=HIDDEN, inp=HIDDEN) -> Any:
    return SimpleNamespace(
        weight=torch.nn.Parameter(torch.zeros(out, inp)),
        weight_scale=_bytes(out, 4),
        reduce_results=True,
    )


def _norm() -> Any:
    return SimpleNamespace(weight=torch.nn.Parameter(torch.ones(HIDDEN)))


def _moe(**overrides) -> Any:
    experts = SimpleNamespace(
        w13_weight=_bytes(4, HIDDEN),
        w13_weight_scale=_bytes(4, 4),
        w2_weight=_bytes(4, HIDDEN),
        w2_weight_scale=_bytes(4, 4),
        quant_method=SimpleNamespace(mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_BF16),
    )
    fields = dict(
        experts=SimpleNamespace(routed_experts=experts),
        shared_experts=SimpleNamespace(gate_up_proj=_proj(), down_proj=_proj()),
        gate=SimpleNamespace(
            weight=_proj().weight, e_score_correction_bias=None, tid2eid=None
        ),
        n_routed_experts=384,
        n_activated_experts=6,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        scoring_func="sqrtsoftplus",
        renormalize=True,
    )
    return SimpleNamespace(**(fields | overrides))


def _attn(index: int, **overrides) -> Any:
    fields = dict(
        layer_id=index,
        compressor=None,
        indexer=None,
        compress_ratio=1,
        kv_mxfp8=False,
        kv_cache_dtype="fp8_ds_mla",
        swa_cache_layer=SimpleNamespace(prefix=f"layers.{index}.swa", kv_cache=None),
        compressed_cache_prefix=f"layers.{index}.comp",
        topk_indices_buffer=torch.zeros(64, 8, dtype=torch.int32),
        fused_wqa_wkv=_proj(),
        q_norm=_norm(),
        kv_norm=_norm(),
        wq_b=_proj(),
        wo_a=_proj(),
        wo_b=_proj(),
        attn_sink=None,
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.zeros(8, 8)),
        _compressed_kv_cache=lambda: make_cache(),
    )
    return SimpleNamespace(**(fields | overrides))


class _Layer(torch.nn.Module):
    def __init__(self, index: int, **overrides):
        super().__init__()
        self.attn = _attn(index, **overrides.pop("attn", {}))
        self.ffn = _moe(**overrides.pop("ffn", {}))
        self.engram = None
        self.use_sequence_parallel = False
        self.fuse_seam_norm = True
        self.hc_mult = 4
        self.hc_attn_fn = self.hc_ffn_fn = None
        self.hc_attn_scale = self.hc_ffn_scale = 1.0
        self.hc_attn_base = self.hc_ffn_base = 0.0
        self.attn_norm = _norm()
        self.ffn_norm = _norm()
        self.mhc_post = lambda x, residual, post, res: residual
        for key, value in overrides.items():
            setattr(self, key, value)
        self.weight = torch.nn.Parameter(torch.zeros(HIDDEN))


class _Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.model_config.hf_config
        self.vllm_config = config
        self.layers = torch.nn.ModuleList(_Layer(i) for i in range(LAYERS))
        self.start_layer = 0
        self.end_layer = LAYERS
        self.mono = None


def make_cache(blocks: int = 4, block: int = 32) -> torch.Tensor:
    return _bytes(blocks, block * ms.RECORD)


def make_config(**overrides) -> Any:
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(monokernel=True),
        model_config=SimpleNamespace(
            enable_sleep_mode=False,
            hf_config=SimpleNamespace(
                num_hidden_layers=LAYERS, hidden_size=HIDDEN, index_topk=8
            ),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
            cudagraph_capture_sizes=list(range(1, ms.MAX_ROWS + 1)),
            static_forward_context={},
        ),
        lora_config=None,
        kv_transfer_config=None,
        speculative_config=None,
        weight_transfer_config=None,
        use_v2_model_runner=True,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _swa(**overrides) -> Any:
    fields = dict(
        num_prefills=0,
        num_decodes=1,
        num_decode_tokens=1,
        decode_swa_width=ms.SWA_WIDTH,
        decode_swa_indices=torch.zeros(8, 1, ms.SWA_WIDTH, dtype=torch.int32),
        decode_swa_lens=torch.zeros(8, dtype=torch.int32),
        token_to_req_indices=torch.zeros(8, dtype=torch.int32),
        slot_mapping=torch.zeros(8, dtype=torch.int64),
        block_size=32,
    )
    return SimpleNamespace(**(fields | overrides))


def _metadata(model, **overrides) -> dict:
    meta: dict = {}
    for layer in model.layers:
        meta[layer.attn.swa_cache_layer.prefix] = _swa(**overrides)
        meta[layer.attn.compressed_cache_prefix] = SimpleNamespace(
            block_size=32, block_table=torch.zeros(8, 4, dtype=torch.int32)
        )
    return meta


class TestDsv41Spec(MonoConformance):
    specs_path = SPECS_PATH

    @pytest.fixture(autouse=True)
    def _stubs(self, monkeypatch):
        _gpu(monkeypatch)

    def make_config(self, **overrides):
        return make_config(**overrides)

    def make_model(self, config):
        model = _Model(config)
        sfc = config.compilation_config.static_forward_context
        for layer in model.layers:
            layer.attn.swa_cache_layer.kv_cache = make_cache()
            sfc[layer.attn.swa_cache_layer.prefix] = layer.attn.swa_cache_layer
            sfc[layer.attn.compressed_cache_prefix] = SimpleNamespace(
                kv_cache=make_cache()
            )
        return model

    def decode_metadata(self, config, model):
        return _metadata(model)

    def skipped_metadata(self, config, model):
        return _metadata(model, num_prefills=1)

    def bad_caches(self, config, model):
        return {"swa": torch.zeros(4, dtype=torch.bfloat16)}

    def _run_layers(self, mono, model) -> None:
        positions = self._positions(mono)
        for i, layer in enumerate(model.layers):
            kernel_for_layer(mono, i, positions)


def _rocm_platform() -> ModuleType:
    """The real ROCm platform module, whose arch probe needs a GPU to answer."""
    if "vllm.platforms.rocm" not in sys.modules:
        probe = torch.cuda.get_device_properties
        torch.cuda.get_device_properties = lambda *_, **__: SimpleNamespace(
            gcnArchName="gfx950"
        )
        try:
            importlib.import_module("vllm.platforms.rocm")
        finally:
            torch.cuda.get_device_properties = probe
    return sys.modules["vllm.platforms.rocm"]


def _gpu(monkeypatch, cdna: int = 4, cus: int = BLOCKS) -> None:
    _stub_modules(monkeypatch)
    monkeypatch.setattr(ms, "_compute_units", lambda: cus)
    monkeypatch.setattr(_rocm_platform(), "get_cdna_version", lambda: cdna)


def _attach(monkeypatch, config=None, model=None, **layer_overrides):
    _gpu(monkeypatch)
    config = config or make_config()
    model = model if model is not None else TestDsv41Spec().make_model(config)
    for layer in model.layers:
        for key, value in layer_overrides.items():
            setattr(layer.attn, key, value)
    return attach(model, config, SPECS_PATH), model


def _indexer():
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41RocmMxfp4Indexer

    return object.__new__(DeepseekV41RocmMxfp4Indexer)


@pytest.mark.parametrize(
    "overrides, ffn_only",
    [
        (dict, False),
        (lambda: dict(compress_ratio=0), False),
        (lambda: dict(compressor=object(), indexer=_indexer()), False),
        (lambda: dict(indexer=_indexer()), False),
        (lambda: dict(compressor=object()), True),
        (lambda: dict(indexer=object()), True),
        (lambda: dict(kv_cache_dtype="auto"), True),
        (lambda: dict(kv_mxfp8=True), True),
        (lambda: dict(compress_ratio=3), True),
    ],
    ids=[
        "standard",
        "window",
        "kv-source",
        "index-source",
        "compressor-only",
        "other-indexer",
        "kv",
        "mxfp8",
        "ratio",
    ],
)
def test_layers_take_whole_or_ffn_launch(monkeypatch, overrides, ffn_only):
    """The whole-layer kernels serve every backbone layer of the checkpoint (an
    index source's compressor and indexer run between them); a layer outside
    them keeps vLLM's attention and runs its FFN half as one launch, from
    wo_b's unreduced output."""
    mono, model = _attach(monkeypatch, **overrides())
    kernels = {k.ffn_only for k in mono.kernels.values()}
    assert kernels == {ffn_only}
    assert all(layer.attn.wo_b.reduce_results is not ffn_only for layer in model.layers)


@pytest.mark.parametrize(
    "skip",
    [
        dict(use_sequence_parallel=True),
        dict(fuse_seam_norm=False),
    ],
    ids=["sequence-parallel", "unfused-seam"],
)
def test_skipped_layers_keep_vllm_path(monkeypatch, skip):
    config = make_config()
    model = TestDsv41Spec().make_model(config)
    for layer in model.layers:
        for key, value in skip.items():
            setattr(layer, key, value)
    mono, _ = _attach(monkeypatch, config=config, model=model)
    assert mono is None


def test_moe_routing_outside_the_kernels_keeps_vllm_path(monkeypatch):
    config = make_config()
    model = TestDsv41Spec().make_model(config)
    for layer in model.layers:
        layer.ffn.n_routed_experts = 256
    mono, _ = _attach(monkeypatch, config=config, model=model)
    assert mono is None


def _step(mono, model, metadata, width=1):
    with forward(mono.vllm_config, CUDAGraphMode.FULL, metadata):
        return mono.skip_reason(torch.zeros(width, dtype=torch.long))


@pytest.mark.parametrize(
    "case, why",
    [
        (dict(num_prefills=1), "not a decode-only step"),
        (dict(num_decodes=0), "not a decode-only step"),
        (dict(num_decode_tokens=4), "more decode tokens than rows"),
        (dict(decode_swa_width=256), "SWA width"),
        (dict(decode_swa_lens=None), "SWA decode metadata missing"),
    ],
    ids=["prefill", "no-decode", "tokens-past-rows", "noncausal-window", "no-swa"],
)
def test_other_steps_take_vllm_path(monkeypatch, case, why):
    mono, model = _attach(monkeypatch)
    assert why in _step(mono, model, _metadata(model, **case))


def test_decode_step_takes_the_kernels(monkeypatch):
    mono, model = _attach(monkeypatch)
    assert _step(mono, model, _metadata(model)) is None


def test_profile_run_takes_vllm_path(monkeypatch):
    mono, model = _attach(monkeypatch)
    assert _step(mono, model, None) == "no attention metadata"


def test_rows_past_the_kernels_take_vllm_path(monkeypatch):
    mono, model = _attach(monkeypatch)
    assert _step(mono, model, _metadata(model), width=ms.MAX_ROWS + 1) == "width"


def test_widths_follow_the_capture_sizes(monkeypatch):
    config = make_config()
    config.compilation_config.cudagraph_capture_sizes = [1, 2, 8, 64, 512]
    assert ms._widths(config) == (1, 2, 8)


def test_widths_default_to_every_row(monkeypatch):
    config = make_config()
    config.compilation_config.cudagraph_capture_sizes = [512]
    assert ms._widths(config) == tuple(range(1, ms.MAX_ROWS + 1))


@pytest.mark.parametrize("tp", [1, 3, 8])
def test_tensor_parallel_outside_the_kernels_refuses(tp):
    config = make_config()
    config.parallel_config.tensor_parallel_size = tp
    assert any("tensor parallel" in r for r in ms._refuse_config(config))


def test_a4w4_moe_refuses(monkeypatch):
    monkeypatch.setattr("vllm.envs.VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4", True)
    assert any("A8W4" in r for r in ms._refuse_config(make_config()))


@pytest.mark.parametrize("cdna", [3, 5])
def test_other_isa_refuses(monkeypatch, cdna):
    config = make_config()
    model = TestDsv41Spec().make_model(config)
    _gpu(monkeypatch, cdna=cdna)
    with pytest.raises(MonoUnsupported, match="CDNA4"):
        attach(model, config, SPECS_PATH)


def test_too_few_compute_units_refuses(monkeypatch):
    config = make_config()
    model = TestDsv41Spec().make_model(config)
    _gpu(monkeypatch, cus=BLOCKS // 2)
    with pytest.raises(MonoUnsupported, match="compute units"):
        attach(model, config, SPECS_PATH)


def test_other_moe_backend_refuses(monkeypatch):
    config = make_config()
    model = TestDsv41Spec().make_model(config)
    for layer in model.layers:
        experts = layer.ffn.experts.routed_experts
        experts.quant_method.mxfp4_backend = Mxfp4MoeBackend.TRITON
    _gpu(monkeypatch)
    with pytest.raises(MonoUnsupported, match="AITER_MXFP4_BF16"):
        attach(model, config, SPECS_PATH)


def test_off_without_the_flag(monkeypatch):
    config = make_config()
    config.kernel_config.monokernel = False
    mono, _ = _attach(monkeypatch, config=config)
    assert mono is None


def test_whole_launch_runs_the_layer(monkeypatch):
    mono, model = _attach(monkeypatch)
    positions = torch.zeros(1, dtype=torch.long)
    with forward(mono.vllm_config, CUDAGraphMode.FULL, _metadata(model)):
        kernel = kernel_for_layer(mono, 1, positions)
        out = kernel.launch(
            torch.zeros(1, HIDDEN),
            positions,
            torch.zeros(1, 4, HIDDEN),
            torch.zeros(1, 4, 1),
            torch.zeros(1, 4, 4),
            torch.zeros(1, 4),
        )
    assert out[0] == "whole"
    records = [a for a in out[1] if torch.is_tensor(a) and a.shape[-1:] == (ms.RECORD,)]
    assert len(records) == 1
    assert out[2]["comp_cache"].shape[-1] == ms.RECORD
    assert out[2]["fold"]


def test_first_layer_takes_the_embedding(monkeypatch):
    """Layer 0's residual is the embedding broadcast to the four streams, which
    K1 takes as it is (no owed post to fold)."""
    mono, model = _attach(monkeypatch)
    positions = torch.zeros(1, dtype=torch.long)
    with forward(mono.vllm_config, CUDAGraphMode.FULL, _metadata(model)):
        kernel = kernel_for_layer(mono, 0, positions)
        out = kernel.launch(torch.zeros(1, HIDDEN), positions, None, None, None, None)
    assert out[0] == "whole" and not out[2]["fold"]
    assert out[1][2].shape == (1, 4, HIDDEN)


def test_ffn_launch_takes_wo_b_unreduced(monkeypatch):
    mono, model = _attach(monkeypatch, compressor=object())
    positions = torch.zeros(1, dtype=torch.long)
    with forward(mono.vllm_config, CUDAGraphMode.FULL, _metadata(model)):
        kernel = kernel_for_layer(mono, 1, positions)
        assert kernel.ffn_only
        out = kernel.launch(
            torch.zeros(1, HIDDEN),
            torch.zeros(1, 4, HIDDEN),
            torch.zeros(1, 4, 1),
            torch.zeros(1, 4, 4),
            torch.zeros(1, 4),
        )
    assert out[0] == "ffn"


def test_launch_declines_a_layout_it_cannot_read(monkeypatch):
    """A later pipeline stage's first layer gets a stream without the mixes:
    ``skip_step`` cannot see that, so the launch declines it."""
    mono, model = _attach(monkeypatch)
    positions = torch.zeros(1, dtype=torch.long)
    with forward(mono.vllm_config, CUDAGraphMode.FULL, _metadata(model)):
        kernel = kernel_for_layer(mono, 1, positions)
        out = kernel.launch(
            torch.zeros(1, 4, HIDDEN), positions, None, None, None, None
        )
    assert out is None
