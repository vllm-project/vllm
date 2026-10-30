# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode library against the reference spec: which variant a
config gets, what the build and the cache binding hand the kernel, and which
steps reach it."""

import importlib
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.models.common.specialized_decode import reference
from tests.models.common.specialized_decode.kit import forward
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.models.common.specialized_decode.dispatch import SpecializedDecode
from vllm.models.common.specialized_decode.guards import (
    _features_in_use,
    _refuse_config,
    _select_spec,
    select_decode_spec,
)
from vllm.models.common.specialized_decode.spec import (
    DecodeUnsupported,
    Feature,
    Step,
)

SPECS = reference.DECODE_SPECS


@pytest.fixture
def config():
    return reference.make_config()


@pytest.fixture
def model(config):
    return reference.make_model(config)


def make_decode(config, model, bind: bool = True):
    """This rank's dispatch, built the way a model builds it."""
    spec = select_decode_spec(config, SPECS)
    assert spec is not None
    decode = SpecializedDecode(model, config, spec)
    if bind:
        decode.bind_caches()
    return decode


@pytest.fixture
def decode(config, model):
    return make_decode(config, model)


def positions(width: int) -> torch.Tensor:
    return torch.zeros(width, dtype=torch.long)


# Selection


def test_off_without_the_flag(config):
    config.kernel_config.specialized_decode = None
    sizes = list(config.compilation_config.cudagraph_capture_sizes)
    assert select_decode_spec(config, SPECS) is None
    assert config.compilation_config.cudagraph_capture_sizes == sizes


def test_the_first_servable_variant_wins(config):
    config.scheduler_config.max_num_seqs = 4
    spec, refusals = _select_spec(SPECS, config)
    assert spec is reference.REFERENCE
    assert refusals == {"reference-wide": ["needs max_num_seqs >= 8"]}

    config.scheduler_config.max_num_seqs = 8
    spec, refusals = _select_spec(SPECS, config)
    assert spec is reference.REFERENCE_WIDE
    assert refusals == {}


def test_every_variant_refused_names_every_variant(config):
    config.model_config.dtype = torch.float16
    with pytest.raises(DecodeUnsupported) as excinfo:
        select_decode_spec(config, SPECS)
    message = str(excinfo.value)
    for spec in SPECS:
        assert spec.name in message
    assert message.count("needs bfloat16") == len(SPECS)


def test_refusals_report_every_reason(config):
    config.parallel_config.enable_eplb = True
    config.use_v2_model_runner = False
    config.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    config.model_config.dtype = torch.float16
    assert _refuse_config(reference.REFERENCE, config) == [
        "eplb is not supported",
        "needs Model Runner V2",
        "CUDA graph mode PIECEWISE: decode needs FULL graphs",
        f"dtype {torch.float16}: needs bfloat16",
    ]


IN_USE = [
    (
        Feature.PIPELINE_PARALLEL,
        lambda c: setattr(c.parallel_config, "pipeline_parallel_size", 2),
    ),
    (
        Feature.DATA_PARALLEL,
        lambda c: setattr(c.parallel_config, "data_parallel_size", 2),
    ),
    (
        Feature.EXPERT_PARALLEL,
        lambda c: setattr(c.parallel_config, "enable_expert_parallel", True),
    ),
    (Feature.EPLB, lambda c: setattr(c.parallel_config, "enable_eplb", True)),
    (
        Feature.DECODE_CONTEXT_PARALLEL,
        lambda c: setattr(c.parallel_config, "decode_context_parallel_size", 2),
    ),
    (Feature.UBATCHING, lambda c: setattr(c.parallel_config, "use_ubatching", True)),
    (
        Feature.PREFILL_CONTEXT_PARALLEL,
        lambda c: setattr(c.parallel_config, "prefill_context_parallel_size", 2),
    ),
    (Feature.LORA, lambda c: setattr(c, "lora_config", object())),
    (Feature.KV_CONNECTOR, lambda c: setattr(c, "kv_transfer_config", object())),
    (
        Feature.ROUTED_EXPERT_CAPTURE,
        lambda c: setattr(c.aux_output_config, "enable_return_routed_experts", True),
    ),
    (
        Feature.SPECULATIVE_DECODING,
        lambda c: setattr(c, "speculative_config", object()),
    ),
    (Feature.SLEEP_MODE, lambda c: setattr(c.model_config, "enable_sleep_mode", True)),
    (
        Feature.CPU_WEIGHT_OFFLOAD,
        lambda c: setattr(c.offload_config.uva, "cpu_offload_gb", 1),
    ),
    (Feature.WEIGHT_UPDATES, lambda c: setattr(c, "weight_transfer_config", object())),
]


def test_every_feature_can_be_turned_on():
    assert {feature for feature, _ in IN_USE} == set(Feature)


@pytest.mark.parametrize(
    "feature, turn_on", IN_USE, ids=[feature.value for feature, _ in IN_USE]
)
def test_a_feature_the_spec_does_not_list_refuses(config, feature, turn_on):
    turn_on(config)
    assert _features_in_use(config) == {feature}
    assert _refuse_config(reference.REFERENCE, config) == [
        f"{feature.value} is not supported"
    ]
    serving = replace(reference.REFERENCE, supports=frozenset({feature}))
    assert _refuse_config(serving, config) == []


def test_the_reference_config_reads_like_a_real_one():
    """The fake config the tests pass has every field ``_features_in_use``
    reads off a real ``VllmConfig``, so a new feature cannot drift past it."""
    real = VllmConfig()
    real.model_config = SimpleNamespace(enable_sleep_mode=False)
    assert _features_in_use(real) == _features_in_use(reference.make_config()) == set()


def test_capture_sizes_become_the_widths(config):
    config.compilation_config.cudagraph_capture_sizes = [1, 3, 4, 16, 32]
    assert select_decode_spec(config, SPECS) is reference.REFERENCE
    assert config.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 16, 32]


def test_a_spec_can_keep_the_engines_capture_sizes(config):
    """A spec that pads serves whatever sizes the deployment captures, so it
    opts out of having them rewritten."""
    spec = replace(reference.REFERENCE, set_capture_sizes=False)
    config.compilation_config.cudagraph_capture_sizes = [1, 3, 4, 16, 32]

    assert select_decode_spec(config, spec) is spec

    assert config.compilation_config.cudagraph_capture_sizes == [1, 3, 4, 16, 32]


# Build


def test_build_takes_the_loaded_model_and_every_width(config, model, decode):
    assert decode.widths == reference.widths(config)
    assert decode.kernel.model is model
    assert decode.kernel.widths == decode.widths


def test_the_loaded_model_can_refuse(config):
    spec = select_decode_spec(config, SPECS)
    with pytest.raises(DecodeUnsupported, match="not a ReferenceModel"):
        SpecializedDecode(torch.nn.Linear(2, 2), config, spec)


def test_a_torch_compiled_model_is_refused(config):
    spec = select_decode_spec(config, SPECS)
    compiled = object.__new__(TorchCompileWithNoGuardsWrapper)
    with pytest.raises(DecodeUnsupported, match="support_torch_compile"):
        SpecializedDecode(compiled, config, spec)


def test_warmup_compiles_every_key_the_kernel_declares(config, model):
    registry = JitWarmupRegistry(config)
    with registry.activate():
        decode = make_decode(config, model, bind=False)
    assert len(registry) == 1
    registry.warmup()
    assert decode.kernel.compiled == set(decode.kernel.warmup_keys())


def test_a_kernel_without_keys_registers_no_warmup(config, model):
    spec = replace(
        select_decode_spec(config, SPECS),
        build=lambda c, m, w: SimpleNamespace(
            cache_layers=(), warmup_keys=lambda: [], bind_caches=lambda caches: None
        ),
    )
    registry = JitWarmupRegistry(config)
    with registry.activate():
        SpecializedDecode(model, config, spec)
    assert len(registry) == 0


# Cache binding


def test_binding_hands_the_kernel_its_cache_layers(config, model, decode):
    assert set(decode.kernel.bound) == set(decode.kernel.cache_layers)

    rebound = reference.set_caches(config, model)
    decode.bind_caches()
    assert set(decode.kernel.bound) == set(decode.kernel.cache_layers)
    assert all(decode.kernel.bound[name] is rebound[name] for name in rebound)


def test_caches_the_spec_refuses_never_reach_the_kernel(config, model, decode):
    before = decode.kernel.bound
    reference.set_caches(config, model, dtype=torch.float16)
    with pytest.raises(DecodeUnsupported, match="needs torch.bfloat16"):
        decode.bind_caches()
    assert decode.kernel.bound is before


# The per-step decision


def test_a_served_decode_step_takes_the_kernel(config, decode):
    rows = positions(min(decode.widths))
    with forward(config, CUDAGraphMode.FULL, {"prefill": False}):
        step = decode.plan(rows)
    assert isinstance(step, Step)
    assert step.width == step.kernel_width == min(decode.widths)
    assert step.positions is rows


def test_a_spec_that_pads_runs_a_short_step_at_the_next_width(config, model):
    """``pad_steps`` serves a width the spec was not built for by launching at
    the next one up, which is the width the kernel is handed."""
    spec = replace(reference.REFERENCE, pad_steps=True)
    decode = SpecializedDecode(model, config, spec)
    decode.bind_caches()
    assert decode.widths == reference.widths(config)

    with forward(config, CUDAGraphMode.FULL, {"prefill": False}):
        step = decode.plan(positions(3))
        over = decode.plan(positions(9))

    assert step is not None and step.width == 3 and step.kernel_width == 4
    assert over is None


def test_without_padding_an_unserved_width_takes_vllms_path(config, decode):
    """The default is exact: a width the spec was not built for falls back,
    rather than silently running on a wider launch."""
    with forward(config, CUDAGraphMode.FULL, {"prefill": False}):
        assert decode.plan(positions(3)) is None


@pytest.mark.parametrize(
    "case", ["pipeline stage", "width", "piecewise", "unbound caches", "skip_step"]
)
def test_steps_the_library_sends_to_vllms_path(config, model, case):
    decode = make_decode(config, model, bind=case != "unbound caches")
    rows = positions(max(decode.widths) + 1 if case == "width" else min(decode.widths))
    mode = CUDAGraphMode.PIECEWISE if case == "piecewise" else CUDAGraphMode.FULL
    stage = object() if case == "pipeline stage" else None
    with forward(config, mode, {"prefill": case == "skip_step"}):
        assert decode.plan(rows, stage) is None


def test_a_step_without_a_forward_context_takes_vllms_path(decode):
    assert decode.plan(positions(min(decode.widths))) is None


def test_run_takes_the_planned_step_to_the_kernel(config, model, decode):
    width = min(decode.widths)
    rows = positions(width)
    embeds = torch.zeros(width, model.config.hidden_size)
    with forward(config, CUDAGraphMode.FULL, {"prefill": False}):
        step = decode.plan(rows)
        hidden = decode.run(step, None, embeds)
    assert decode.kernel.launched == [width]
    assert torch.equal(hidden, model(None, rows, None, embeds))


def test_the_contract_imports_without_a_vendor_runtime(monkeypatch):
    """The library loads where neither FlyDSL nor aiter is installed, so a
    model's kernel code must stay out of its import graph."""
    for name in list(sys.modules):
        if name.split(".")[0] in ("flydsl", "aiter") or "specialized_decode" in name:
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name in ("flydsl", "aiter"):
        monkeypatch.setitem(sys.modules, name, None)

    module = importlib.import_module("vllm.models.common.specialized_decode")

    assert module.DecodeSpec is not None
    assert not [n for n in sys.modules if n.startswith(("flydsl.", "aiter."))]
