# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp's specialized decode on CPU: the `decode_backend` setting, the
deployment requirements (with the platform and the TP group's all-reduce
mocked) and the per-step gate that picks the steps the executor runs."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from vllm.config import CompilationMode, CUDAGraphMode
from vllm.models.qwen4_exp.common.qsa import Qwen4ExpQSAMetadata
from vllm.models.qwen4_exp.common.qsa_indexer import QSAIndexerMetadata
from vllm.models.qwen4_exp.nvidia import specialized_decode
from vllm.models.qwen4_exp.nvidia.specialized_decode import (
    DecodeConfig,
    max_decode_rows,
    specialized_decode_config,
    unsupported,
)
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime import decode
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.decode import (
    PureDecodeGate,
    SpecializedDecode,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.short_conv_attn import PleShortConvAttentionMetadata


def _kv_transfer(
    connector: str = "OffloadingConnector",
    module_path: str | None = None,
    spec_name: str | None = None,
) -> SimpleNamespace:
    """The fields of a KV transfer config the deployment check reads."""
    extra = {} if spec_name is None else {"spec_name": spec_name}
    return SimpleNamespace(
        kv_connector=connector,
        kv_connector_module_path=module_path,
        kv_connector_extra_config=extra,
    )


def _vllm_config(**overrides: object) -> SimpleNamespace:
    """A supported deployment: Qwen3.8-Flash-Next at TP4 with expert
    parallelism, MTP with three draft tokens, FULL_AND_PIECEWISE graphs, Model
    Runner V2. `overrides` sets dotted paths, e.g.
    `{"parallel_config.pipeline_parallel_size": 2}`.
    """
    config = SimpleNamespace(
        additional_config={"decode_backend": "latency_specialized"},
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            use_ubatching=False,
            enable_expert_parallel=True,
            enable_hc_sp=False,
            use_sequence_parallel_moe=False,
        ),
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=3,
            use_local_argmax_reduction=True,
            draft_lm_head_quantization=None,
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
        model_config=SimpleNamespace(
            enable_sleep_mode=False,
            hf_text_config=SimpleNamespace(
                hc_count=4, hidden_size=2560, hc_lowrank=320
            ),
        ),
        kernel_config=SimpleNamespace(moe_backend="auto"),
        weight_transfer_config=None,
        lora_config=None,
        kv_transfer_config=None,
    )
    for path, value in overrides.items():
        target = config
        *parents, name = path.split(".")
        for parent in parents:
            target = getattr(target, parent)
        setattr(target, name, value)
    return config


def _platform(monkeypatch: pytest.MonkeyPatch, capability: int) -> None:
    """Report a GPU of `capability` (103 is GB300) to the requirement check."""
    platform = SimpleNamespace(is_device_capability=lambda wanted: wanted == capability)
    monkeypatch.setattr(specialized_decode, "current_platform", platform)


@pytest.fixture
def on_gb300(monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, 103)


@pytest.fixture
def on_b200(monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, 100)


@pytest.fixture(autouse=True)
def tp_group(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """The TP group's MIN all-reduce, recorded; `other_ranks` is the flag
    the other ranks contribute (1: they support the specialized decode)."""
    state = SimpleNamespace(other_ranks=1, calls=[])

    def all_reduce(tensor: torch.Tensor, op: dist.ReduceOp, group: object) -> None:
        state.calls.append((tensor.tolist(), op, group))
        tensor.clamp_(max=state.other_ranks)

    monkeypatch.setattr(
        specialized_decode,
        "dist",
        SimpleNamespace(all_reduce=all_reduce, ReduceOp=dist.ReduceOp),
    )
    monkeypatch.setattr(
        specialized_decode,
        "get_tp_group",
        lambda: SimpleNamespace(cpu_group="tp-cpu-group"),
    )
    monkeypatch.setattr(
        specialized_decode, "envs", SimpleNamespace(VLLM_BATCH_INVARIANT=False)
    )
    return state


@pytest.mark.parametrize("extra", [None, {}, {"decode_backend": "auto"}])
def test_auto_keeps_the_stock_decode(extra, on_b200, tp_group):
    """Without the setting nothing is checked, even on an unsupported
    deployment, and no collective runs."""
    config = _vllm_config(additional_config=extra, speculative_config=None)
    assert specialized_decode_config(config) is None
    assert not tp_group.calls


def test_an_unknown_backend_is_refused(on_gb300, tp_group):
    config = _vllm_config(additional_config={"decode_backend": "fast"})
    with pytest.raises(ValueError, match="auto/latency_specialized"):
        specialized_decode_config(config)
    assert not tp_group.calls


@pytest.mark.parametrize(
    "tp, expert_parallel, cudagraph_mode, max_rows",
    [
        (1, False, CUDAGraphMode.FULL_AND_PIECEWISE, 256),
        (4, False, CUDAGraphMode.FULL_DECODE_ONLY, 256),
        (4, True, CUDAGraphMode.FULL, 256),
    ],
)
def test_supported_deployments(
    tp, expert_parallel, cudagraph_mode, max_rows, on_gb300, tp_group
):
    config = _vllm_config(
        **{
            "parallel_config.tensor_parallel_size": tp,
            "parallel_config.enable_expert_parallel": expert_parallel,
            "compilation_config.cudagraph_mode": cudagraph_mode,
        }
    )
    assert unsupported(config) == []
    assert specialized_decode_config(config) == DecodeConfig(
        tokens_per_request=4, max_rows=max_rows
    )
    assert tp_group.calls == [([1], dist.ReduceOp.MIN, "tp-cpu-group")]


@pytest.mark.parametrize("local_argmax", [False, True])
def test_the_draft_token_choice_needs_local_argmax_reduction(
    local_argmax, on_gb300, tp_group, monkeypatch
):
    """Without `use_local_argmax_reduction` the specialized decode still
    runs, and says once that the draft passes keep the stock token choice."""
    lines = []
    monkeypatch.setattr(
        specialized_decode,
        "logger",
        SimpleNamespace(info_once=lambda message, *args: lines.append(message)),
    )
    config = _vllm_config(
        **{"speculative_config.use_local_argmax_reduction": local_argmax}
    )
    assert specialized_decode_config(config) == DecodeConfig(
        tokens_per_request=4, max_rows=256
    )
    fallback = [line for line in lines if "use_local_argmax_reduction" in line]
    assert len(fallback) == (0 if local_argmax else 1)


@pytest.mark.parametrize("steps", [0, 1, 50])
def test_metadata_check_steps(steps, on_gb300, tp_group):
    """`decode_metadata_check_steps` sets the checked replays of every
    uniform decode shape; unset, none are checked."""
    extra = {"decode_backend": "latency_specialized"}
    assert specialized_decode_config(_vllm_config(additional_config=extra)) == (
        DecodeConfig(tokens_per_request=4, max_rows=256)
    )
    extra["decode_metadata_check_steps"] = steps
    config = specialized_decode_config(_vllm_config(additional_config=extra))
    assert config == DecodeConfig(
        tokens_per_request=4, max_rows=256, metadata_check_steps=steps
    )


@pytest.mark.parametrize("steps", [-1, True, 1.5, "50", None])
def test_metadata_check_steps_must_be_a_non_negative_integer(steps, on_gb300, tp_group):
    extra = {
        "decode_backend": "latency_specialized",
        "decode_metadata_check_steps": steps,
    }
    with pytest.raises(ValueError, match="non-negative integer"):
        specialized_decode_config(_vllm_config(additional_config=extra))
    assert not tp_group.calls


def test_metadata_check_steps_need_the_specialized_decode(on_gb300, tp_group):
    """The check runs only in the specialized decode, so asking for it without
    the specialized decode raises rather than checking nothing."""
    config = _vllm_config(additional_config={"decode_metadata_check_steps": 50})
    with pytest.raises(ValueError, match="latency_specialized"):
        specialized_decode_config(config)
    config = _vllm_config(additional_config={"decode_metadata_check_steps": 0})
    assert specialized_decode_config(config) is None
    assert not tp_group.calls


@pytest.mark.parametrize("overlap", [True, False])
def test_metadata_overlap(overlap, on_gb300, tp_group):
    """`decode_metadata_overlap` turns the overlapped metadata replay on or
    off; unset, it is on."""
    extra = {"decode_backend": "latency_specialized"}
    assert specialized_decode_config(_vllm_config(additional_config=extra)) == (
        DecodeConfig(tokens_per_request=4, max_rows=256, metadata_overlap=True)
    )
    extra["decode_metadata_overlap"] = overlap
    config = specialized_decode_config(_vllm_config(additional_config=extra))
    assert config == DecodeConfig(
        tokens_per_request=4, max_rows=256, metadata_overlap=overlap
    )


@pytest.mark.parametrize("overlap", [0, 1, "false", None])
def test_metadata_overlap_must_be_a_boolean(overlap, on_gb300, tp_group):
    extra = {
        "decode_backend": "latency_specialized",
        "decode_metadata_overlap": overlap,
    }
    with pytest.raises(ValueError, match="true or false"):
        specialized_decode_config(_vllm_config(additional_config=extra))
    assert not tp_group.calls


def test_metadata_overlap_off_needs_the_specialized_decode(on_gb300, tp_group):
    """Turning the overlap off without the specialized decode raises rather
    than doing nothing; leaving it on does not."""
    config = _vllm_config(additional_config={"decode_metadata_overlap": False})
    with pytest.raises(ValueError, match="latency_specialized"):
        specialized_decode_config(config)
    config = _vllm_config(additional_config={"decode_metadata_overlap": True})
    assert specialized_decode_config(config) is None
    assert not tp_group.calls


def test_draft_head_fp8_prefilter(on_gb300, tp_group):
    """`draft_head_fp8_prefilter` asks for the FP8-prefiltered draft head;
    unset, there is none."""
    extra = {"decode_backend": "latency_specialized"}
    tp1 = {"parallel_config.tensor_parallel_size": 1}
    assert specialized_decode_config(
        _vllm_config(additional_config=extra, **tp1)
    ) == DecodeConfig(tokens_per_request=4, max_rows=256)
    extra["draft_head_fp8_prefilter"] = True
    assert specialized_decode_config(
        _vllm_config(additional_config=extra, **tp1)
    ) == DecodeConfig(tokens_per_request=4, max_rows=256, draft_head_fp8=True)


@pytest.mark.parametrize(
    "overrides, missing",
    [
        ({"parallel_config.tensor_parallel_size": 4}, "tensor parallel size 1"),
        (
            {"speculative_config.use_local_argmax_reduction": False},
            "use_local_argmax_reduction",
        ),
        (
            {"speculative_config.draft_lm_head_quantization": "fp8"},
            "draft_lm_head_quantization",
        ),
    ],
)
def test_draft_head_fp8_prefilter_requirements(overrides, missing, on_gb300, tp_group):
    """The draft head's requirements are read from the config alone, so every
    rank decides alike; one missing raises at startup."""
    config = _vllm_config(
        additional_config={
            "decode_backend": "latency_specialized",
            "draft_head_fp8_prefilter": True,
        },
        **{"parallel_config.tensor_parallel_size": 1, **overrides},
    )
    with pytest.raises(ValueError, match=missing):
        specialized_decode_config(config)


@pytest.mark.parametrize("value", [1, "true", None])
def test_draft_head_fp8_prefilter_must_be_a_boolean(value, on_gb300, tp_group):
    extra = {"decode_backend": "latency_specialized", "draft_head_fp8_prefilter": value}
    with pytest.raises(ValueError, match="true or false"):
        specialized_decode_config(_vllm_config(additional_config=extra))
    assert not tp_group.calls


def test_draft_head_fp8_prefilter_needs_the_specialized_decode(on_gb300, tp_group):
    config = _vllm_config(additional_config={"draft_head_fp8_prefilter": True})
    with pytest.raises(ValueError, match="latency_specialized"):
        specialized_decode_config(config)
    config = _vllm_config(additional_config={"draft_head_fp8_prefilter": False})
    assert specialized_decode_config(config) is None


@pytest.mark.parametrize(
    "tp, tokens_per_request, rows",
    [(1, 4, 256), (4, 4, 256), (1, 2, 128), (4, 1, 64)],
)
def test_max_decode_rows(tp, tokens_per_request, rows):
    """64 requests at TP1 and TP4, of MTP draft tokens plus one rows."""
    assert max_decode_rows(tp, tokens_per_request) == rows


@pytest.mark.parametrize(
    "overrides, missing",
    [
        ({"parallel_config.tensor_parallel_size": 2}, "tensor parallel size 1 or 4"),
        ({"parallel_config.tensor_parallel_size": 8}, "tensor parallel size 1 or 4"),
        ({"parallel_config.pipeline_parallel_size": 2}, "pipeline"),
        ({"parallel_config.data_parallel_size": 2}, "data"),
        ({"parallel_config.decode_context_parallel_size": 2}, "context"),
        ({"parallel_config.prefill_context_parallel_size": 2}, "context"),
        ({"parallel_config.use_ubatching": True}, "micro-batching"),
        ({"speculative_config": None}, "MTP speculative decoding"),
        ({"speculative_config.method": "eagle3"}, "MTP speculative decoding"),
        (
            {"compilation_config.cudagraph_mode": CUDAGraphMode.PIECEWISE},
            "FULL CUDA graphs for decode batches",
        ),
        (
            {"compilation_config.cudagraph_mode": CUDAGraphMode.NONE},
            "FULL CUDA graphs for decode batches",
        ),
        (
            {"compilation_config.mode": CompilationMode.STOCK_TORCH_COMPILE},
            "torch.compile",
        ),
        ({"model_config.enable_sleep_mode": True}, "sleep mode"),
        ({"model_config.hf_text_config.hidden_size": 4096}, "Qwen3.8-Flash-Next"),
        ({"model_config.hf_text_config.hc_lowrank": 256}, "Qwen3.8-Flash-Next"),
        ({"weight_transfer_config": object()}, "runtime weight updates"),
        ({"lora_config": object()}, "LoRA"),
        ({"kv_transfer_config": _kv_transfer("NixlConnector")}, "KV transfer"),
        (
            {"kv_transfer_config": _kv_transfer(module_path="my.module")},
            "KV transfer",
        ),
        (
            {"kv_transfer_config": _kv_transfer(spec_name="MySpec")},
            "KV transfer",
        ),
        ({"parallel_config.enable_hc_sp": True}, "hyper-connection"),
        ({"parallel_config.use_sequence_parallel_moe": True}, "MoE sequence"),
        ({"kernel_config.moe_backend": "deep_gemm_mega_moe"}, "MegaMoE"),
    ],
)
def test_each_requirement_is_named(overrides, missing, on_gb300):
    reasons = unsupported(_vllm_config(**overrides))
    assert len(reasons) == 1 and missing in reasons[0]


@pytest.mark.parametrize("spec_name", [None, "TieringOffloadingSpec"])
def test_native_cpu_offload_is_allowed(spec_name, on_gb300):
    """`--kv-offloading-size` (vLLM's in-tree `OffloadingConnector`) moves
    whole blocks between steps, so it is the one KV connector allowed."""
    offload = _kv_transfer(spec_name=spec_name)
    assert not unsupported(_vllm_config(kv_transfer_config=offload))


def test_batch_invariance_is_refused(monkeypatch, on_gb300):
    monkeypatch.setattr(
        specialized_decode, "envs", SimpleNamespace(VLLM_BATCH_INVARIANT=True)
    )
    reasons = unsupported(_vllm_config())
    assert len(reasons) == 1 and "batch invariance" in reasons[0]


def test_every_missing_requirement_is_reported(on_b200, tp_group):
    """Every rank raises with every requirement it misses, after the MIN
    all-reduce tells the others."""
    config = _vllm_config(
        lora_config=object(),
        speculative_config=None,
        **{"parallel_config.tensor_parallel_size": 2},
    )
    with pytest.raises(ValueError) as error:
        specialized_decode_config(config)
    for reason in ("GB300", "LoRA", "tensor parallel size", "MTP"):
        assert reason in str(error.value)
    assert tp_group.calls == [([0], dist.ReduceOp.MIN, "tp-cpu-group")]


def test_another_rank_missing_a_requirement_refuses_on_every_rank(on_gb300, tp_group):
    tp_group.other_ranks = 0
    with pytest.raises(ValueError, match="another tensor parallel rank"):
        specialized_decode_config(_vllm_config())


def _metadata(cls: type, **fields: object) -> object:
    """A `cls` metadata object with only `fields` set: the gate reads host
    fields only."""
    metadata: object = object.__new__(cls)
    for name, value in fields.items():
        setattr(metadata, name, value)
    return metadata


def _decode_metadata() -> dict[str, object]:
    """The metadata of a pure decode step of three draft tokens a request."""
    return {
        "gdn": _metadata(GDNAttentionMetadata, num_prefills=0),
        "ple": _metadata(PleShortConvAttentionMetadata, num_prefills=0),
        "raw_key_cache": _metadata(QSAIndexerMetadata, num_prefills=0),
        "compressed_key_cache": _metadata(QSAIndexerMetadata, num_prefills=0),
        "qsa": _metadata(Qwen4ExpQSAMetadata, max_query_len=4),
    }


GATE = PureDecodeGate(
    prefill_counting_layers=("gdn", "ple", "raw_key_cache", "compressed_key_cache"),
    qsa_main_layers=("qsa",),
    max_decode_query_len=4,
)


def test_gate_takes_a_pure_decode():
    assert GATE.is_pure_decode(_decode_metadata())


@pytest.mark.parametrize(
    "layer, field, value",
    [
        ("gdn", "num_prefills", 1),
        ("ple", "num_prefills", 1),
        ("raw_key_cache", "num_prefills", 2),
        ("compressed_key_cache", "num_prefills", 1),
        ("qsa", "max_query_len", 5),
    ],
)
def test_gate_refuses_any_group_with_a_prefill(layer, field, value):
    metadata = _decode_metadata()
    setattr(metadata[layer], field, value)
    assert not GATE.is_pure_decode(metadata)


def test_gate_refuses_missing_or_foreign_metadata():
    metadata = _decode_metadata()
    del metadata["ple"]
    assert not GATE.is_pure_decode(metadata)
    metadata = _decode_metadata()
    metadata["gdn"] = _metadata(Qwen4ExpQSAMetadata, max_query_len=1)
    assert not GATE.is_pure_decode(metadata)


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A FULL CUDA graph capture (forward context mode NONE) of a pure decode
    step; tests change the context or the capture state."""
    state = SimpleNamespace(
        context=SimpleNamespace(
            attn_metadata=_decode_metadata(),
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ),
        capturing=True,
    )
    monkeypatch.setattr(decode, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(decode, "get_forward_context", lambda: state.context)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    return state


def _executor() -> SpecializedDecode:
    """An executor with the gate above and up to 64 rows (TP4), over no
    backbone: plan() reads only the gate, the config, the GDN layers'
    records protocol and the replayed metadata (none here)."""
    executor = object.__new__(SpecializedDecode)
    executor.config = DecodeConfig(tokens_per_request=4, max_rows=64)
    executor.gate = GATE
    executor.gdn = SimpleNamespace(records=None)
    executor.metadata = None
    return executor


@pytest.mark.parametrize("shape", [(64,), (4,), (3, 16)])
def test_plan_takes_full_captures_of_pure_decode_steps(shape, capture):
    """Text positions and MRoPE positions, whose last axis is the rows."""
    step = _executor().plan(torch.zeros(shape, dtype=torch.int64), None)
    assert step is not None and step.rows == shape[-1]


def test_plan_runs_other_steps_stock(capture):
    executor = _executor()
    positions = torch.zeros(16, dtype=torch.int64)
    assert executor.plan(torch.zeros(65, dtype=torch.int64), None) is None
    assert executor.plan(torch.zeros(0, dtype=torch.int64), None) is None
    assert executor.plan(positions, SimpleNamespace()) is None
    capture.capturing = False
    assert executor.plan(positions, None) is None
    capture.capturing = True
    capture.context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
    assert executor.plan(positions, None) is None
    capture.context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    capture.context.attn_metadata["gdn"].num_prefills = 1
    assert executor.plan(positions, None) is None


def test_plan_tells_the_gdn_records_which_full_captures_run_stock(capture):
    """With the GDN records protocol, a FULL capture that plan() refuses is
    a FULL graph of that size without the records core; eager runs and
    PIECEWISE captures are not."""
    executor = _executor()
    finished: list[int] = []
    executor.gdn = SimpleNamespace(
        records=SimpleNamespace(finish_capture=finished.append)
    )
    assert executor.plan(torch.zeros(65, dtype=torch.int64), None) is None
    capture.context.attn_metadata["gdn"].num_prefills = 1
    assert executor.plan(torch.zeros(16, dtype=torch.int64), None) is None
    capture.context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
    assert executor.plan(torch.zeros(16, dtype=torch.int64), None) is None
    capture.capturing = False
    capture.context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    assert executor.plan(torch.zeros(16, dtype=torch.int64), None) is None
    assert finished == [65, 16]
