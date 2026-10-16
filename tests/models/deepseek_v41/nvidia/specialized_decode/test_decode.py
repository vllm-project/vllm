# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 NVIDIA specialized decode's opt-in and its per-step
gates, on CPU.

``decode_backend=latency_specialized`` is resolved once per TP rank: an unsupported
deployment raises on every rank, naming every missing requirement. The
executor issues cross-rank collectives, so whether a forward runs it, and
which of its paths a step takes, must be a function of host state alone:
nothing here has a GPU, so a gate that read device state would fail. Only
FULL-graph captures of pure decode steps run the executor; with the default
``auto`` the model and its model state are the stock ones.

About 40 seconds on the CPU platform; on a CUDA platform the two-rank test
needs two GPUs."""

import inspect
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import regex as re
import torch

import vllm.envs as envs
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.models.deepseek_v41.nvidia import model as nvidia_model
from vllm.models.deepseek_v41.nvidia import specialized_decode
from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
from vllm.models.deepseek_v41.nvidia.specialized_decode.attention.candidate_topk import (  # noqa: E501
    supports_cooperative_candidates,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.attention.indexer import (
    row_shard_bounds,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.attention.sparse_mla import (
    sparse_index_key,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.step import (
    deferred_finalize,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.runtime.decode import (
    SpecializedDecode,
    step_gates,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.runtime.metadata_graphs import (
    AttnMetadataCudaGraphs,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.runtime.model_state import (
    DeepseekV41SpecializedDecodeModelState,
)
from vllm.models.deepseek_v41.nvidia.vl_model import DeepseekV41ForCausalLM
from vllm.v1.attention.backends.registry import AttentionBackendEnum

LATENCY_SPECIALIZED = {"decode_backend": "latency_specialized"}

# ------------------------------------------------------------ the opt-in


def _config(**overrides) -> SimpleNamespace:
    """The validated deployment: GB300 TP4, DSpark, FlashInfer sparse MLA."""
    fields = dict(
        additional_config=LATENCY_SPECIALIZED,
        use_v2_model_runner=True,
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        use_ubatching=False,
        spec_method="dspark",
        enable_adaptive_verification=False,
        backend=AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV41,
        cache_dtype="fp8",
        indexer_kv_dtype="mxfp4",
        indexer_sparse_logits=True,
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        compilation_mode=CompilationMode.VLLM_COMPILE,
        enable_sleep_mode=False,
        lora_config=None,
        kv_transfer_config=None,
        enable_flashinfer_autotune=None,
        enable_jit_warmup=True,
    )
    fields.update(overrides)
    f = SimpleNamespace(**fields)
    spec = (
        None
        if f.spec_method is None
        else SimpleNamespace(
            method=f.spec_method,
            enable_adaptive_verification=f.enable_adaptive_verification,
        )
    )
    return SimpleNamespace(
        additional_config=f.additional_config,
        kernel_config=SimpleNamespace(
            enable_flashinfer_autotune=f.enable_flashinfer_autotune,
            enable_jit_warmup=f.enable_jit_warmup,
        ),
        use_v2_model_runner=f.use_v2_model_runner,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=f.tensor_parallel_size,
            pipeline_parallel_size=f.pipeline_parallel_size,
            data_parallel_size=f.data_parallel_size,
            enable_expert_parallel=f.enable_expert_parallel,
            decode_context_parallel_size=f.decode_context_parallel_size,
            prefill_context_parallel_size=f.prefill_context_parallel_size,
            use_ubatching=f.use_ubatching,
        ),
        speculative_config=spec,
        attention_config=SimpleNamespace(
            backend=f.backend,
            indexer_kv_dtype=f.indexer_kv_dtype,
            indexer_sparse_logits=f.indexer_sparse_logits,
        ),
        cache_config=SimpleNamespace(cache_dtype=f.cache_dtype),
        compilation_config=SimpleNamespace(
            cudagraph_mode=f.cudagraph_mode, mode=f.compilation_mode
        ),
        model_config=SimpleNamespace(enable_sleep_mode=f.enable_sleep_mode),
        lora_config=f.lora_config,
        kv_transfer_config=f.kv_transfer_config,
    )


def _deployment(m: pytest.MonkeyPatch, capability: int = 103, mhc_ok=True) -> None:
    """The hardware facts the opt-in checks: GPU, DeepGEMM and multicast."""
    m.setattr(
        specialized_decode,
        "current_platform",
        SimpleNamespace(is_device_capability=lambda c: c == capability),
    )
    m.setattr(specialized_decode, "supports_mhc_overlap", lambda config: mhc_ok)
    m.setattr(specialized_decode, "supports_mhc_all_reduce", lambda config: mhc_ok)


def test_flashinfer_version_is_the_pinned_one():
    """The executor's FlashInfer is the one vLLM pins: a FlashInfer bump must
    recheck it here rather than turn its paths off with a warning."""
    pins = Path(__file__).parents[5] / "requirements" / "cuda.txt"
    pinned = re.search(r"^flashinfer-python==(\S+)", pins.read_text(), re.MULTILINE)
    assert pinned is not None
    assert pinned.group(1) == specialized_decode.FLASHINFER_VERSION


@pytest.mark.parametrize("extra", [{}, {"decode_backend": "auto"}, None])
def test_auto_resolves_to_nothing(monkeypatch, extra):
    """The default: nothing checked and no collective (no process group
    here), on any deployment."""
    _deployment(monkeypatch, capability=90, mhc_ok=False)
    config = _config(additional_config=extra, tensor_parallel_size=1)
    assert specialized_decode.specialized_decode_config(config) is None


def test_unknown_backend_raises():
    config = _config(additional_config={"decode_backend": "fast"})
    with pytest.raises(ValueError, match="auto/latency_specialized"):
        specialized_decode.specialized_decode_config(config)


@pytest.mark.parametrize(
    "mode", [CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.FULL_DECODE_ONLY]
)
@pytest.mark.parametrize("autotune", [None, True, False])
def test_validated_deployment_resolves(monkeypatch, dist_init, mode, autotune):
    _deployment(monkeypatch)
    config = _config(cudagraph_mode=mode, enable_flashinfer_autotune=autotune)
    want = specialized_decode.DecodeConfig(
        autotune=autotune is not False, jit_warmup=True
    )
    assert specialized_decode.specialized_decode_config(config) == want


_REFUSALS: dict[str, tuple[dict[str, object], str]] = {
    "gpu-sm100": (dict(capability=100), "GB300 (SM103)"),
    "gpu-sm120": (dict(capability=120), "GB300 (SM103)"),
    "model-runner-v1": (dict(use_v2_model_runner=False), "Model Runner V2"),
    "tp2": (dict(tensor_parallel_size=2), "tensor parallel size 4"),
    "tp8": (dict(tensor_parallel_size=8), "tensor parallel size 4"),
    "pp": (dict(pipeline_parallel_size=2), "tensor parallel size 4"),
    "dp": (dict(data_parallel_size=2), "tensor parallel size 4"),
    "ep": (dict(enable_expert_parallel=True), "tensor parallel size 4"),
    "dcp": (dict(decode_context_parallel_size=2), "tensor parallel size 4"),
    "pcp": (dict(prefill_context_parallel_size=2), "tensor parallel size 4"),
    "ubatching": (dict(use_ubatching=True), "tensor parallel size 4"),
    "no-spec": (dict(spec_method=None), "DSpark"),
    "mtp": (dict(spec_method="mtp"), "DSpark"),
    "adaptive": (dict(enable_adaptive_verification=True), "DSpark"),
    "flashmla": (
        dict(backend=AttentionBackendEnum.FLASHMLA_SPARSE_DSV41),
        "FLASHINFER_MLA_SPARSE_DSV41",
    ),
    "bf16-kv": (dict(cache_dtype="auto"), "fp8 KV cache"),
    "fp8-indexer": (dict(indexer_kv_dtype="fp8"), "MXFP4 indexer"),
    "dense-logits": (dict(indexer_sparse_logits=False), "sparse logits"),
    "eager": (dict(cudagraph_mode=CUDAGraphMode.NONE), "FULL CUDA graphs"),
    "piecewise": (dict(cudagraph_mode=CUDAGraphMode.PIECEWISE), "FULL CUDA graphs"),
    "full-mixed": (dict(cudagraph_mode=CUDAGraphMode.FULL), "FULL CUDA graphs"),
    "torch-compile": (
        dict(compilation_mode=CompilationMode.STOCK_TORCH_COMPILE),
        "no torch.compile",
    ),
    "sleep-mode": (dict(enable_sleep_mode=True), "no sleep mode"),
    "lora": (dict(lora_config=object()), "no LoRA"),
    "kv-transfer": (dict(kv_transfer_config=object()), "no LoRA, KV transfer"),
    "batch-invariant": (dict(batch_invariant=True), "batch invariance"),
    "no-multicast": (dict(mhc_ok=False), "NVLink multicast"),
}


@pytest.mark.parametrize("case", list(_REFUSALS))
def test_unsupported_deployment_raises(monkeypatch, dist_init, case):
    overrides, reason = _REFUSALS[case]
    overrides = dict(overrides)
    _deployment(
        monkeypatch,
        capability=overrides.pop("capability", 103),
        mhc_ok=overrides.pop("mhc_ok", True),
    )
    if overrides.pop("batch_invariant", False):
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with pytest.raises(ValueError, match=re.escape(reason)):
        specialized_decode.specialized_decode_config(_config(**overrides))


def test_every_missing_requirement_is_named(monkeypatch, dist_init):
    _deployment(monkeypatch, capability=100)
    config = _config(
        tensor_parallel_size=8,
        spec_method=None,
        enable_sleep_mode=True,
        lora_config=object(),
    )
    with pytest.raises(ValueError) as error:
        specialized_decode.specialized_decode_config(config)
    message = str(error.value)
    for reason in ("GB300", "tensor parallel size 4", "DSpark", "sleep", "no LoRA"):
        assert reason in message
    assert "Model Runner V2" not in message


def _lockstep_rank(rank: int, world_size: int, init_file: str) -> None:
    """Rank 1 lacks multicast; every rank must raise."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    with set_current_vllm_config(VllmConfig()), pytest.MonkeyPatch.context() as m:
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"file://{init_file}",
            local_rank=rank,
            backend="gloo",
        )
        initialize_model_parallel(world_size, 1)
        try:
            _deployment(m, mhc_ok=rank != 1)
            with pytest.raises(ValueError) as error:
                specialized_decode.specialized_decode_config(_config())
            expected = "NVLink multicast" if rank == 1 else "another tensor parallel"
            assert expected in str(error.value)
        finally:
            cleanup_dist_env_and_memory()


def test_ranks_refuse_in_lockstep():
    """One rank's missing requirement fails every rank, not only its own."""
    with tempfile.NamedTemporaryFile() as init_file:
        torch.multiprocessing.spawn(
            _lockstep_rank, args=(2, init_file.name), nprocs=2, join=True
        )


# --------------------------------------------------- which forwards run it

SWA_PREFIXES = ["layers.0.attn.swa", "layers.1.attn.swa"]


def _executor(candidates=None) -> SpecializedDecode:
    """An executor with only the state ``plan`` reads."""
    executor = object.__new__(SpecializedDecode)
    executor.candidates = candidates
    executor._swa_prefixes = SWA_PREFIXES
    return executor


def _decode_metadata(num_prefills: int = 0) -> dict[str, SimpleNamespace]:
    return {
        prefix: SimpleNamespace(num_prefills=num_prefills if i else 0)
        for i, prefix in enumerate(SWA_PREFIXES)
    }


@contextmanager
def _forward(monkeypatch, capturing: bool, mode: CUDAGraphMode, metadata):
    """A forward in ``mode``, captured into a CUDA graph or eager."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    context = ForwardContext({}, metadata, {}, cudagraph_runtime_mode=mode)
    with override_forward_context(context):
        yield


_PLANS = {
    # name: (rows, capturing, mode, metadata, intermediate tensors, runs)
    "full-decode": (6, True, CUDAGraphMode.FULL, _decode_metadata(), False, True),
    # Model Runner V2 captures FULL graphs with the forward context's mode NONE.
    "full-capture": (6, True, CUDAGraphMode.NONE, _decode_metadata(), False, True),
    "full-1-row": (1, True, CUDAGraphMode.FULL, _decode_metadata(), False, True),
    "full-384-rows": (384, True, CUDAGraphMode.FULL, _decode_metadata(), False, True),
    "full-385-rows": (385, True, CUDAGraphMode.FULL, _decode_metadata(), False, False),
    "no-rows": (0, True, CUDAGraphMode.FULL, _decode_metadata(), False, False),
    "piecewise": (6, True, CUDAGraphMode.PIECEWISE, _decode_metadata(), False, False),
    "eager": (6, False, CUDAGraphMode.NONE, _decode_metadata(), False, False),
    "eager-full-warm-up": (
        6,
        False,
        CUDAGraphMode.FULL,
        _decode_metadata(),
        False,
        False,
    ),
    "prefill": (6, True, CUDAGraphMode.FULL, _decode_metadata(1), False, False),
    "missing-group": (
        6,
        True,
        CUDAGraphMode.FULL,
        {SWA_PREFIXES[0]: SimpleNamespace(num_prefills=0)},
        False,
        False,
    ),
    "micro-batches": (6, True, CUDAGraphMode.FULL, [_decode_metadata()], False, False),
    "intermediate-tensors": (
        6,
        True,
        CUDAGraphMode.FULL,
        _decode_metadata(),
        True,
        False,
    ),
}


@pytest.mark.parametrize("case", list(_PLANS))
def test_only_full_graph_decode_captures_run_the_executor(monkeypatch, case):
    rows, capturing, mode, metadata, intermediate, runs = _PLANS[case]
    warmed = []
    executor = _executor(SimpleNamespace(warm=lambda: warmed.append(True)))
    with _forward(monkeypatch, capturing, mode, metadata):
        step = executor.plan(torch.empty(rows), object() if intermediate else None)
    if runs:
        assert step is not None and step.rows == rows and step.executor is executor
    else:
        assert step is None
    # The eager run before each capture warms the candidate overlap.
    assert warmed == ([True] if not capturing and 0 < rows <= 384 else [])


def test_no_forward_context_runs_stock(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert _executor().plan(torch.empty(6), None) is None


ROWS = (1, 6, 8, 9, 16, 17, 48, 49, 384)


@pytest.mark.parametrize("rows", ROWS)
def test_step_gates(rows):
    """Which paths a step of ``rows`` rows takes.

    - the next mHC coefficients and aux means on the side stream: <= 48 rows;
      inline after the fused all-reduce + mHC beyond;
    - low-batch GEMMs (routed experts, MXFP8 projections, side projections,
      publishing wo_b): <= 8 rows;
    - candidate overlap and Engram WKV ahead: <= 16 rows."""
    executor = object()
    step = step_gates(rows, executor)
    assert step.rows == rows and step.executor is executor
    assert step.side_stream is (rows <= 48)
    assert step.low_batch is (rows <= 8)
    assert step.overlap_candidates is (rows <= 16)
    assert step.engram_prefetch is (rows <= 16)
    assert step.overlap_inputs is (rows <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD)


@pytest.mark.parametrize(
    "rows, requests, world, slice_rows",
    [
        (48, 8, 4, 12),  # 8 requests x 6 DSpark rows
        (192, 32, 4, 48),
        (64, 32, 4, 16),  # 2 rows a request
        (96, 16, 2, 48),
        (6, 1, 4, None),  # one request: replicated
        (42, 7, 4, None),  # below 8 requests
        (54, 9, 4, None),  # 9 requests do not split over 4 ranks
        (50, 8, 4, None),  # varying rows a request
        (48, 8, 1, None),
        (60, 60, 4, None),  # odd slices: misaligned int32 indices
        (72, 12, 4, 18),
    ],
)
def test_row_shard_bounds(rows, requests, world, slice_rows):
    """Whole requests a rank; the slices tile the rows in rank order."""
    bounds = [row_shard_bounds(rows, requests, world, rank) for rank in range(world)]
    if slice_rows is None:
        assert bounds == [None] * world
        return
    assert bounds == [(r * slice_rows, (r + 1) * slice_rows) for r in range(world)]
    assert slice_rows * requests % rows == 0


@pytest.mark.parametrize(
    "rows, blocks, k, supported",
    [
        (6, 131072, 2048, True),
        (64, 4096, 512, True),
        (12, 8192, 1024, True),
        (0, 4096, 512, False),
        (65, 4096, 512, False),
        (6, 4096, 256, False),
        (6, 131074, 2048, False),
    ],
)
def test_cooperative_candidates_gate(rows, blocks, k, supported):
    assert supports_cooperative_candidates(rows, blocks, k) is supported


def _input_batch(query_lens, **overrides) -> SimpleNamespace:
    fields = dict(
        has_prefill=False,
        max_query_len=None,
        num_scheduled_tokens=np.array(query_lens + [7], dtype=np.int32),
        num_reqs=len(query_lens),
        num_reqs_after_padding=4,
        num_tokens=sum(query_lens),
        num_tokens_after_padding=24,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "batch, mode, key",
    [
        (_input_batch([6, 6, 6]), CUDAGraphMode.FULL, (3, 4, 18, 24)),
        (_input_batch([1]), CUDAGraphMode.FULL, (1, 4, 1, 24)),
        (_input_batch([6, 6, 6]), CUDAGraphMode.PIECEWISE, None),
        (_input_batch([6, 6, 6]), CUDAGraphMode.NONE, None),
        (_input_batch([6, 6, 6], has_prefill=True), CUDAGraphMode.FULL, None),
        # Adaptive verification splits the drafts on the device.
        (_input_batch([6, 6, 6], max_query_len=6), CUDAGraphMode.FULL, None),
        (_input_batch([6, 2, 6]), CUDAGraphMode.FULL, None),
    ],
    ids=["decode", "one-row", "piecewise", "eager", "prefill", "adaptive", "ragged"],
)
def test_metadata_graph_key(batch, mode, key):
    """Only FULL uniform-decode steps record their metadata build, keyed by
    every host value the build reads."""
    assert AttnMetadataCudaGraphs.key(batch, mode) == key


def test_sparse_index_keys():
    """Compressed layers share a table only with layers of the same compressed
    cache, index source and ratio; SWA-only layers share one."""
    swa = SimpleNamespace(compress_ratio=0)
    compressed = dict(compressed_cache_prefix="layers.2.attn.c", compress_ratio=4)
    assert sparse_index_key(swa) == "swa_only"
    a = sparse_index_key(SimpleNamespace(index_source_layer_id=2, **compressed))
    b = sparse_index_key(SimpleNamespace(index_source_layer_id=3, **compressed))
    assert a == "layers.2.attn.c:2:4" and a != b


@pytest.mark.parametrize("raises", [False, True])
def test_deferred_finalize_restores_the_layer_setting(raises):
    """The stock forward reads the layer's own deferral after the step."""
    config = SimpleNamespace(
        _defer_moe_finalize=False, defer_moe_finalize_max_num_tokens=16
    )
    expect = pytest.raises(RuntimeError) if raises else nullcontext()
    with expect, deferred_finalize(config, 384):
        assert config._defer_moe_finalize
        assert config.defer_moe_finalize_max_num_tokens == 384
        if raises:
            raise RuntimeError
    assert not config._defer_moe_finalize
    assert config.defer_moe_finalize_max_num_tokens == 16


# ------------------------------------------------- stock model when auto


class _Backbone(torch.nn.Module):
    """Records the ``decode_step`` the model passes to its backbone."""

    def forward(self, *args, decode_step=None, **kwargs):
        self.decode_step = decode_step
        return torch.empty(0)


def _llm(decode_config, decode=None) -> nvidia_model.DeepseekV41LLMForCausalLM:
    llm = object.__new__(nvidia_model.DeepseekV41LLMForCausalLM)
    torch.nn.Module.__init__(llm)
    llm.specialized_decode_config = decode_config
    llm.specialized_decode = decode
    llm.model = _Backbone()
    return llm


@pytest.mark.parametrize("on", [False, True])
def test_model_state_class(on):
    llm = _llm(specialized_decode.DecodeConfig(True, True) if on else None)
    want = DeepseekV41SpecializedDecodeModelState if on else DeepseekV41ModelState
    assert llm.get_model_state_cls() is want
    vl = object.__new__(DeepseekV41ForCausalLM)
    torch.nn.Module.__init__(vl)
    vl.language_model = llm
    assert vl.get_model_state_cls() is want


def test_forward_passes_the_planned_step():
    """Auto: no step; on: the executor's plan, which may be None."""
    llm = _llm(None)
    llm(torch.empty(6), torch.empty(6))
    assert llm.model.decode_step is None
    plans = []

    def plan(*args):
        plans.append(args)
        return "step"

    llm = _llm(specialized_decode.DecodeConfig(True, True), SimpleNamespace(plan=plan))
    positions = torch.empty(6)
    llm(torch.empty(6), positions)
    assert llm.model.decode_step == "step"
    assert len(plans) == 1 and plans[0][0] is positions and plans[0][1] is None


def test_stock_seams_default_to_the_stock_behavior(monkeypatch):
    """The backbone takes no step unless given one; the model state's
    metadata build is ``DefaultModelState.prepare_attn`` and its replay-start
    upload a plain copy."""
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    forward = inspect.signature(nvidia_model.DeepseekV4Model.forward)
    assert forward.parameters["decode_step"].default is None

    calls = []

    def prepare_attn(self, *args, **kwargs):
        calls.append((args, kwargs))
        return {"a": 1}

    monkeypatch.setattr(DefaultModelState, "prepare_attn", prepare_attn)
    state = object.__new__(DeepseekV41ModelState)
    assert state._build_attn_metadata(1, 2, for_capture=True) == {"a": 1}
    assert calls == [((1, 2), {"for_capture": True})]

    state._replay_start = torch.full((8,), -1, dtype=torch.int32)
    state._replay_start_staging = SimpleNamespace(
        copy_to_gpu=lambda values, out: out.copy_(torch.from_numpy(values))
    )
    out = state._upload_replay_start(np.array([3, 0, 5], dtype=np.int32), 3)
    assert out.tolist() == [3, 0, 5] and state._replay_start[3].item() == -1


def test_zero_replay_starts_upload_once():
    """Decode steps' replay starts are zeros the device already holds."""
    uploads = []

    def copy_to_gpu(values, out):
        uploads.append(len(values))
        return out.copy_(torch.from_numpy(values))

    state = object.__new__(DeepseekV41SpecializedDecodeModelState)
    state._replay_start = torch.full((8,), -1, dtype=torch.int32)
    state._replay_start_staging = SimpleNamespace(copy_to_gpu=copy_to_gpu)
    state._zero_replay_starts = 0
    for starts in ([0] * 4, [0] * 3, [0] * 4, [0] * 6, [0, 2, 0], [0] * 2):
        values = np.array(starts, dtype=np.int32)
        out = state._upload_replay_start(values, len(starts))
        assert out.tolist() == starts
    assert uploads == [4, 6, 3, 2]
