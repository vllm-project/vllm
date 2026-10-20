# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's DecodeSpec against the conformance kit, plus its own choices."""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from tests.models.common.specialized_decode.kit import DecodeConformance, forward
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.models.common.specialized_decode.dispatch import SpecializedDecode
from vllm.models.common.specialized_decode.guards import select_decode_spec
from vllm.models.common.specialized_decode.spec import DecodeUnsupported, Step
from vllm.models.deepseek_v41.amd.specialized_decode import spec as sd

SPECS = (sd.DECODE_SPEC,)
LAYERS = 6
HIDDEN = 16
BLOCKS = 256


class _StubRunner:
    def __init__(self, world, rank, group, device, **routing):
        self.world = world
        self.routing = routing
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


_MONO = "vllm.models.deepseek_v41.amd.mono"


def _stub_modules(monkeypatch) -> None:
    """The FlyDSL runner and arch modules, as spec-shaped stubs."""
    runner: Any = ModuleType(f"{_MONO}.runner")
    runner.BLOCKS = BLOCKS
    runner.MAX_TOKENS = 64
    runner.DSV41MonoLayer = _StubRunner
    runner.MonoLayerWeights = _weights_stub
    runner.AttnWeights = _weights_stub
    plan: Any = ModuleType(f"{_MONO}.attention.plan")
    plan.Dims = lambda world: world
    arch: Any = ModuleType(f"{_MONO}.common.arch")
    arch.target_arch = lambda: "gfx950"
    for name, module in (
        (f"{_MONO}.runner", runner),
        (f"{_MONO}.attention.plan", plan),
        (f"{_MONO}.common.arch", arch),
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
        self.taken: list = []

    def forward(self, x, positions, input_ids, *rest, decode_ffn=None):
        self.taken.append(("layer", self.attn.layer_id, decode_ffn is not None))
        return _stream()


def _stream() -> tuple[torch.Tensor, ...]:
    """What a decoder layer returns: the stream and its three mixes."""
    return (
        torch.zeros(1, HIDDEN),
        torch.zeros(1, 4, HIDDEN),
        torch.zeros(1, 4, 1),
        torch.zeros(1, 4, 4),
        torch.zeros(1, 4),
    )


class _Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.model_config.hf_config
        self.vllm_config = config
        self.layers = torch.nn.ModuleList(_Layer(i) for i in range(LAYERS))
        self.start_layer = 0
        self.end_layer = LAYERS
        self.decoder_replay_layers = None

    def prologue(self, input_ids, positions, intermediate_tensors, embeds, lookback):
        return embeds, input_ids, None, None, None, positions.shape[0]

    def after_layer(
        self, idx, hidden, residual, post_mix, res_mix, tokens, aux, replay=True
    ):
        return None

    def epilogue(self, hidden, *rest, replay=True):
        return hidden


def make_cache(blocks: int = 4, block: int = 32) -> torch.Tensor:
    return _bytes(blocks, block * sd.RECORD)


def make_config(**overrides) -> Any:
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(specialized_decode=True),
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
            use_ubatching=False,
            prefill_context_parallel_size=1,
        ),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
            cudagraph_capture_sizes=list(range(1, sd.MAX_ROWS + 1)),
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
        decode_swa_width=sd.SWA_WIDTH,
        decode_swa_indices=torch.zeros(8, 1, sd.SWA_WIDTH, dtype=torch.int32),
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


class TestDsv41Spec(DecodeConformance):
    specs = SPECS
    specs_module = "vllm.models.deepseek_v41.amd.specialized_decode.spec"

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

    def bad_caches(self, config, model):
        return {"swa": torch.zeros(4, dtype=torch.bfloat16)}


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


@pytest.fixture(autouse=True)
def _unpin_row_cap():
    """The row cap is cached per process, and these tests move the GPU under
    it."""
    sd.max_rows.cache_clear()
    yield
    sd.max_rows.cache_clear()


def _gpu(monkeypatch, cdna: int = 4, cus: int = BLOCKS, target: str = "") -> None:
    arch = "gfx942" if cdna == 3 else "gfx950"
    _stub_modules(monkeypatch)
    # The compute-unit probe reads the current accelerator, which a CPU run has
    # none of, and the arch probes read the GPU and FlyDSL's ARCH setting.
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(sd.current_platform, "num_compute_units", lambda _: cus)
    monkeypatch.setattr(_rocm_platform(), "get_cdna_version", lambda: cdna)
    monkeypatch.setattr(sd, "_gpu_arch", lambda: arch)
    monkeypatch.setattr(
        sys.modules[f"{_MONO}.common.arch"], "target_arch", lambda: target or arch
    )


def _build(monkeypatch, config=None, model=None, **layer_overrides):
    _gpu(monkeypatch)
    kit = TestDsv41Spec()
    config = config or make_config()
    model = model if model is not None else kit.make_model(config)
    for layer in model.layers:
        for key, value in layer_overrides.items():
            setattr(layer.attn, key, value)
    return kit.make_decode(config, model), model


def _plan(decode, layer_id: int):
    return decode.kernel.plans[layer_id]


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
    decode, model = _build(monkeypatch, **overrides())
    assert {p.ffn_only for p in decode.kernel.plans.values()} == {ffn_only}
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
    decode, _ = _build(monkeypatch, config=config, model=model)
    assert not decode.kernel.plans


def test_moe_routing_outside_the_kernels_keeps_vllm_path(monkeypatch):
    config = make_config()
    model = TestDsv41Spec().make_model(config)
    for layer in model.layers:
        layer.ffn.n_routed_experts = 256
    decode, _ = _build(monkeypatch, config=config, model=model)
    assert not decode.kernel.plans


def _step(decode, metadata, width=1):
    with forward(decode.vllm_config, CUDAGraphMode.FULL, metadata):
        return sd._skip_step(decode.vllm_config, decode.kernel, width)


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
    decode, model = _build(monkeypatch)
    assert why in _step(decode, _metadata(model, **case))


def test_decode_step_takes_the_kernels(monkeypatch):
    decode, model = _build(monkeypatch)
    assert _step(decode, _metadata(model)) is None


def test_profile_run_takes_vllm_path(monkeypatch):
    decode, model = _build(monkeypatch)
    assert _step(decode, None) == "no attention metadata"


def test_widths_follow_the_capture_sizes(monkeypatch):
    _gpu(monkeypatch)
    config = make_config()
    config.compilation_config.cudagraph_capture_sizes = [1, 2, 8, 64, 512]
    assert sd._widths(config) == (1, 2, 8)


@pytest.mark.parametrize("cdna,rows", [(4, sd.MAX_ROWS), (3, sd.MAX_ROWS_GFX942)])
def test_widths_default_to_every_row_the_gpu_takes(monkeypatch, cdna, rows):
    _gpu(monkeypatch, cdna=cdna)
    config = make_config()
    config.compilation_config.cudagraph_capture_sizes = [512]
    assert sd._widths(config) == tuple(range(1, rows + 1))


@pytest.mark.parametrize("tp", [1, 3, 8])
def test_tensor_parallel_outside_the_kernels_refuses(tp):
    config = make_config()
    config.parallel_config.tensor_parallel_size = tp
    assert any("tensor parallel" in r for r in sd._refuse_config(config))


def test_a4w4_moe_refuses(monkeypatch):
    monkeypatch.setattr("vllm.envs.VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4", True)
    assert any("A8W4" in r for r in sd._refuse_config(make_config()))


@pytest.mark.parametrize("cdna", [2, 5])
def test_other_isa_refuses(monkeypatch, cdna):
    config = make_config()
    model = TestDsv41Spec().make_model(config)
    _gpu(monkeypatch, cdna=cdna)
    with pytest.raises(DecodeUnsupported, match="CDNA3/4"):
        SpecializedDecode(model, config, select_decode_spec(config, SPECS))


def test_arch_env_against_the_gpu_refuses(monkeypatch):
    config = make_config()
    model = TestDsv41Spec().make_model(config)
    _gpu(monkeypatch, target="gfx942")
    with pytest.raises(DecodeUnsupported, match="ARCH"):
        SpecializedDecode(model, config, select_decode_spec(config, SPECS))


def test_too_few_compute_units_refuses(monkeypatch):
    config = make_config()
    model = TestDsv41Spec().make_model(config)
    _gpu(monkeypatch, cus=BLOCKS // 2)
    with pytest.raises(DecodeUnsupported, match="compute units"):
        SpecializedDecode(model, config, select_decode_spec(config, SPECS))


def test_other_moe_backend_refuses(monkeypatch):
    config = make_config()
    model = TestDsv41Spec().make_model(config)
    for layer in model.layers:
        experts = layer.ffn.experts.routed_experts
        experts.quant_method.mxfp4_backend = Mxfp4MoeBackend.TRITON
    _gpu(monkeypatch)
    with pytest.raises(DecodeUnsupported, match="AITER_MXFP4_BF16"):
        SpecializedDecode(model, config, select_decode_spec(config, SPECS))


def test_off_without_the_flag(monkeypatch):
    _gpu(monkeypatch)
    config = make_config()
    config.kernel_config.specialized_decode = False
    assert select_decode_spec(config, SPECS) is None


def test_whole_launch_runs_the_layer(monkeypatch):
    decode, model = _build(monkeypatch)
    positions = torch.zeros(1, dtype=torch.long)
    with forward(decode.vllm_config, CUDAGraphMode.FULL, _metadata(model)):
        out = decode.kernel.launch_whole(
            _plan(decode, 1),
            torch.zeros(1, HIDDEN),
            positions,
            torch.zeros(1, 4, HIDDEN),
            torch.zeros(1, 4, 1),
            torch.zeros(1, 4, 4),
            torch.zeros(1, 4),
        )
    assert out[0] == "whole"
    records = [a for a in out[1] if torch.is_tensor(a) and a.shape[-1:] == (sd.RECORD,)]
    assert len(records) == 1
    assert out[2]["comp_cache"].shape[-1] == sd.RECORD
    assert out[2]["fold"]


def test_first_layer_takes_the_embedding(monkeypatch):
    """Layer 0's residual is the embedding broadcast to the four streams, which
    K1 takes as it is (no owed post to fold)."""
    decode, model = _build(monkeypatch)
    positions = torch.zeros(1, dtype=torch.long)
    with forward(decode.vllm_config, CUDAGraphMode.FULL, _metadata(model)):
        out = decode.kernel.launch_whole(
            _plan(decode, 0), torch.zeros(1, HIDDEN), positions, None, None, None, None
        )
    assert out[0] == "whole" and not out[2]["fold"]
    assert out[1][2].shape == (1, 4, HIDDEN)


def test_ffn_launch_takes_wo_b_unreduced(monkeypatch):
    decode, model = _build(monkeypatch, compressor=object())
    with forward(decode.vllm_config, CUDAGraphMode.FULL, _metadata(model)):
        plan = _plan(decode, 1)
        assert plan.ffn_only
        out = decode.kernel.launch_ffn(
            plan,
            torch.zeros(1, HIDDEN),
            torch.zeros(1, 4, HIDDEN),
            torch.zeros(1, 4, 1),
            torch.zeros(1, 4, 4),
            torch.zeros(1, 4),
        )
    assert out[0] == "ffn"


def test_run_takes_every_layer_the_way_its_launch_allows(monkeypatch):
    """One loop over the backbone, from the model's prologue to its epilogue:
    a whole-layer launch replaces the layer, a declined one and a layer without
    a launch fall through to it, and an FFN-only launch rides into it."""
    config = make_config()
    model = TestDsv41Spec().make_model(config)
    model.layers[2].attn.compressor = object()
    model.layers[4].use_sequence_parallel = True
    decode, _ = _build(monkeypatch, config=config, model=model)

    def launch_whole(plan, *args, **kwargs):
        layer_id = plan.layer.attn.layer_id
        model.layers[layer_id].taken.append(("launch", layer_id))
        return None if layer_id == 1 else _stream()

    decode.kernel.launch_whole = launch_whole

    step = Step(width=1, kernel_width=1, positions=torch.zeros(1, dtype=torch.long))
    out = decode.kernel.run(step, None, torch.zeros(1, HIDDEN))

    assert out.shape == (1, HIDDEN)
    assert [layer.taken for layer in model.layers] == [
        [("launch", 0)],
        [("launch", 1), ("layer", 1, False)],
        [("layer", 2, True)],
        [("launch", 3)],
        [("layer", 4, False)],
        [("launch", 5)],
    ]


DRAFT_LAYERS = 2
DRAFT_EXPERTS = 128
DRAFT_TOPK = 3
DRAFT_TOKENS = 5


class _DraftModel(torch.nn.Module):
    """The DSpark draft head's backbone: its own layers, routing and stream."""

    def __init__(self, config):
        super().__init__()
        self.config = config.model_config.hf_config
        self.vllm_config = config
        self.layers = torch.nn.ModuleList(
            _Layer(
                LAYERS + i,
                attn=dict(compress_ratio=0),
                ffn=dict(
                    n_routed_experts=DRAFT_EXPERTS, n_activated_experts=DRAFT_TOPK
                ),
            )
            for i in range(DRAFT_LAYERS)
        )
        self.hc_mult = 4

    def prologue(self, input_ids, positions, inputs_embeds):
        return inputs_embeds, input_ids, positions.shape[0]

    def epilogue(self, hidden, *rest):
        return hidden


def make_draft_config(**overrides) -> Any:
    config = make_config(**overrides)
    config.speculative_config = SimpleNamespace(
        num_speculative_tokens=DRAFT_TOKENS, method="dspark"
    )
    hf = config.model_config.hf_config
    hf.sliding_window = sd.SWA_WIDTH
    hf.dspark_n_routed_experts = DRAFT_EXPERTS
    hf.dspark_num_experts_per_tok = DRAFT_TOPK
    return config


def _build_draft(monkeypatch, config=None, model=None):
    _gpu(monkeypatch)
    config = config or make_draft_config()
    model = model if model is not None else _DraftModel(config)
    sfc = config.compilation_config.static_forward_context
    for layer in model.layers:
        layer.attn.swa_cache_layer.kv_cache = make_cache()
        sfc[layer.attn.swa_cache_layer.prefix] = layer.attn.swa_cache_layer
    spec = select_decode_spec(config, sd.DRAFT_DECODE_SPEC)
    return SpecializedDecode(model, config, spec), model


def _draft_metadata(model, **overrides) -> dict:
    width = sd.draft_swa_width(model.vllm_config)
    fields = dict(
        decode_swa_width=width,
        decode_swa_indices=torch.zeros(8, 1, width, dtype=torch.int32),
    )
    return {
        layer.attn.swa_cache_layer.prefix: _swa(**(fields | overrides))
        for layer in model.layers
    }


def test_draft_rows_span_the_window_and_the_drafts():
    """The draft's SWA rows are a block's non-causal window, the causal window
    plus the drafts, padded; not a decode token's."""
    config = make_draft_config()
    width = sd.draft_swa_width(config)
    assert width >= sd.SWA_WIDTH + DRAFT_TOKENS
    assert width != sd.SWA_WIDTH


def test_draft_layers_take_the_drafts_own_routing_and_rows(monkeypatch):
    """Every draft layer takes a launch, from a runner built for the draft's
    128 experts, top-3 and non-causal rows, not the backbone's."""
    decode, model = _build_draft(monkeypatch)
    kernel = decode.kernel
    assert set(kernel.plans) == {layer.attn.layer_id for layer in model.layers}
    assert kernel.runner.routing == dict(
        experts=DRAFT_EXPERTS,
        topk=DRAFT_TOPK,
        ug_groups=sd.DRAFT_UG_GROUPS,
        swa_width=sd.draft_swa_width(model.vllm_config),
    )
    # The draft reads no compressed keys, so it binds only its SWA caches.
    assert kernel.cache_layers == tuple(
        sorted(layer.attn.swa_cache_layer.prefix for layer in model.layers)
    )
    decode.bind_caches()


def test_draft_layer_with_compressed_keys_keeps_vllm_path(monkeypatch):
    """The draft has no FFN-only launch: a layer the kernels cannot take whole
    runs as vLLM's."""
    config = make_draft_config()
    model = _DraftModel(config)
    for layer in model.layers:
        layer.attn.compressor = object()
    decode, _ = _build_draft(monkeypatch, config=config, model=model)
    assert not decode.kernel.plans


def test_draft_first_layer_takes_the_broadcast_embedding(monkeypatch):
    """The draft's first layer is handed the embedding already broadcast to the
    hc copies, so the launch takes it as the residual and the first copy as the
    stream."""
    decode, model = _build_draft(monkeypatch)
    positions = torch.zeros(1, dtype=torch.long)
    embeds = torch.zeros(1, 4, HIDDEN)
    with forward(decode.vllm_config, CUDAGraphMode.FULL, _draft_metadata(model)):
        out = decode.kernel.launch_whole(
            _plan(decode, LAYERS), embeds, positions, None, None, None, None
        )
    assert out[0] == "whole" and not out[2]["fold"]
    assert out[1][1].shape == (1, HIDDEN)
    assert out[1][2].shape == (1, 4, HIDDEN)


def test_draft_and_target_take_only_their_own_rows(monkeypatch):
    """Each refuses the other's SWA width, so a draft step cannot run on the
    backbone's kernels or the other way round."""
    draft, draft_model = _build_draft(monkeypatch)
    target, target_model = _build(monkeypatch)

    def skip(decode, metadata):
        with forward(decode.vllm_config, CUDAGraphMode.FULL, metadata):
            return decode.spec.skip_step(decode.vllm_config, decode.kernel, 1)

    assert skip(draft, _draft_metadata(draft_model)) is None
    assert skip(target, _metadata(target_model)) is None
    assert "SWA width" in skip(
        draft, _draft_metadata(draft_model, decode_swa_width=sd.SWA_WIDTH)
    )
    assert "SWA width" in skip(
        target,
        _metadata(
            target_model, decode_swa_width=sd.draft_swa_width(draft_model.vllm_config)
        ),
    )


def test_draft_run_takes_every_layer_the_way_its_launch_allows(monkeypatch):
    decode, model = _build_draft(monkeypatch)

    def launch_whole(plan, *args, **kwargs):
        layer_id = plan.layer.attn.layer_id
        model.layers[layer_id - LAYERS].taken.append(("launch", layer_id))
        return None if layer_id == LAYERS else _stream()

    decode.kernel.launch_whole = launch_whole

    step = Step(width=1, kernel_width=1, positions=torch.zeros(1, dtype=torch.long))
    out = decode.kernel.run(step, None, torch.zeros(1, 4, HIDDEN))

    assert out.shape == (1, HIDDEN)
    assert [layer.taken for layer in model.layers] == [
        [("launch", LAYERS), ("layer", LAYERS, False)],
        [("launch", LAYERS + 1)],
    ]


def test_launch_declines_a_layout_it_cannot_read(monkeypatch):
    """A later pipeline stage's first layer gets a stream without the mixes:
    ``skip_step`` cannot see that, so the launch declines it."""
    decode, model = _build(monkeypatch)
    positions = torch.zeros(1, dtype=torch.long)
    with forward(decode.vllm_config, CUDAGraphMode.FULL, _metadata(model)):
        out = decode.kernel.launch_whole(
            _plan(decode, 1),
            torch.zeros(1, 4, HIDDEN),
            positions,
            None,
            None,
            None,
            None,
        )
    assert out is None
