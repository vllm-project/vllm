# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MTP draft front of Qwen4Exp's specialized decode (`DraftFront`): the
fused front norm (CuTe DSL) against the stock `GemmaRMSNorm` pair, the whole
front against the stock front on full weights, CUDA graph replay, warm-up,
the per-pass gate, the replicated projection weights and a front whose
weights were never bound. On the executor's CPU stand-ins
(`test_executor.py`), a pass that takes the front runs the draft layer and
the final mixer with the read kernels and the boundary kernel, and computes
the stock layer's result; a MoE block that no longer passes the boundary
kernel's check at capture raises."""

import contextlib
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.platforms import current_platform

from .test_executor import HIDDEN as STAND_IN_HIDDEN
from .test_executor import (
    StandInBoundaryKernel,
    StandInHyperConnection,
    _layer,
    _without_reads,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(90),
    reason="the specialized decode's draft front kernel requires SM90+",
)

if current_platform.is_cuda():
    from vllm.models.qwen4_exp.common.qsa import Qwen4ExpQSAMetadata
    from vllm.models.qwen4_exp.common.qsa_indexer import QSAIndexerMetadata
    from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMultiTokenPredictor
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention
    from vllm.models.qwen4_exp.nvidia.specialized_decode import DecodeConfig, mtp
    from vllm.models.qwen4_exp.nvidia.specialized_decode.mtp import (
        front as front_module,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.mtp import front_norm
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime import (
        decode,
        draft_front,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.boundary import (
        Boundaries,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.decode import (
        PureDecodeGate,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.draft_front import (
        DraftFront,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.gemm import (
        PlannedGemms,
        draft_front_gemm_plans,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.qsa import QsaRows

H = 2560
STREAMS = 4
EPS = 1e-6


def _norms(seed: int) -> tuple[GemmaRMSNorm, GemmaRMSNorm]:
    """The draft's two pre-projection norms with random BF16 weights."""
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        embed_norm = GemmaRMSNorm(H, eps=EPS).to(torch.bfloat16)
        hidden_norm = GemmaRMSNorm(STREAMS * H, eps=EPS).to(torch.bfloat16)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for norm in (embed_norm, hidden_norm):
            norm.weight.copy_(
                0.2 * torch.randn(norm.weight.shape, generator=generator, device="cuda")
            )
    return embed_norm, hidden_norm


def _inputs(rows: int, seed: int, pad: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Embeddings and streams of `rows` tokens; `pad` extra columns make
    row-strided views."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    embedded = torch.randn(rows, H + pad, generator=generator, device="cuda")
    hidden = 3.0 * torch.randn(
        rows, STREAMS * H + pad, generator=generator, device="cuda"
    )
    return (
        embedded.to(torch.bfloat16)[:, :H],
        hidden.to(torch.bfloat16)[:, : STREAMS * H],
    )


def _stock(embed_norm, hidden_norm, embedded, hidden):
    """The stock norms (`GemmaRMSNorm` decomposed in PyTorch)."""
    with set_current_vllm_config(VllmConfig()):
        return embed_norm.forward_native(embedded), hidden_norm.forward_native(hidden)


def _ulps(a: torch.Tensor, b: torch.Tensor) -> int:
    """The largest difference of two BF16 tensors in units in the last place."""
    ai = a.contiguous().view(torch.int16).to(torch.int32)
    bi = b.contiguous().view(torch.int16).to(torch.int32)
    ai = torch.where(ai < 0, -32768 - ai, ai)
    bi = torch.where(bi < 0, -32768 - bi, bi)
    return int((ai - bi).abs().max())


def _check_norm(got: torch.Tensor, ref: torch.Tensor) -> None:
    """Stock's arithmetic except the sum order: at most one BF16 unit in the
    last place apart, almost always bitwise."""
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert _ulps(got, ref) <= 1
    same = (got.view(torch.int16) == ref.contiguous().view(torch.int16)).float().mean()
    assert same >= 0.99
    cos = F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
    assert cos >= 0.99999


@pytest.mark.parametrize("rows", [1, 4, 16, 64, 256])
def test_front_norm_matches_the_stock_norms(rows):
    embed_norm, hidden_norm = _norms(rows)
    embedded, hidden = _inputs(rows, rows + 1)
    got_e, got_h = front_norm(
        embedded, hidden, embed_norm.weight, hidden_norm.weight, EPS, EPS
    )
    ref_e, ref_h = _stock(embed_norm, hidden_norm, embedded, hidden)
    _check_norm(got_e, ref_e)
    _check_norm(got_h, ref_h)


def test_front_norm_reads_row_strided_inputs():
    embed_norm, hidden_norm = _norms(7)
    embedded, hidden = _inputs(5, 8, pad=64)
    assert not embedded.is_contiguous() and not hidden.is_contiguous()
    got_e, got_h = front_norm(
        embedded, hidden, embed_norm.weight, hidden_norm.weight, EPS, EPS
    )
    ref_e, ref_h = _stock(embed_norm, hidden_norm, embedded, hidden)
    _check_norm(got_e, ref_e)
    _check_norm(got_h, ref_h)


def test_front_norm_rows_are_independent():
    """A row's bits do not depend on the other rows of the call."""
    embed_norm, hidden_norm = _norms(3)
    embedded, hidden = _inputs(16, 4)
    all_e, all_h = front_norm(
        embedded, hidden, embed_norm.weight, hidden_norm.weight, EPS, EPS
    )
    one_e, one_h = front_norm(
        embedded[5:6], hidden[5:6], embed_norm.weight, hidden_norm.weight, EPS, EPS
    )
    assert torch.equal(all_e[5:6], one_e) and torch.equal(all_h[5:6], one_h)


def test_front_norm_supports_only_its_layouts():
    embedded, hidden = _inputs(4, 1)
    assert mtp.front_norm_supports(embedded, hidden)
    assert not mtp.front_norm_supports(embedded[:3], hidden)
    assert not mtp.front_norm_supports(embedded.float(), hidden)
    assert not mtp.front_norm_supports(embedded[:, 8:], hidden[:, 32:])
    big_e, big_h = _inputs(mtp.MAX_ROWS + 1, 2)
    assert not mtp.front_norm_supports(big_e, big_h)


def _front(seed: int, max_rows: int = 64) -> "DraftFront":
    """A draft front over random norms and full projection weights, without
    a draft model (`run` reads only these)."""
    front = object.__new__(DraftFront)
    front.config = DecodeConfig(tokens_per_request=4, max_rows=max_rows)
    front.embed_norm, front.hidden_norm = _norms(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed + 100)
    front.weights = {
        name: (0.02 * torch.randn(H, H, generator=generator, device="cuda")).to(
            torch.bfloat16
        )
        for name in ("fc_embedding", "fc_hidden")
    }
    front.replicas = {}
    front.gemm_plans = draft_front_gemm_plans()
    front.layer_gemms = PlannedGemms([])
    front.layer_qsa = QsaRows([])
    return front


@pytest.mark.parametrize("rows", [1, 4, 16, 64, 256])
def test_the_front_matches_the_stock_front(rows):
    front = _front(rows, max_rows=256)
    embedded, hidden = _inputs(rows, rows + 2)
    projected, streams = front.run(embedded, hidden)
    ref_e, ref_h = _stock(front.embed_norm, front.hidden_norm, embedded, hidden)
    ref_projected = F.linear(ref_e, front.weights["fc_embedding"])
    ref_streams = F.linear(
        ref_h.view(rows * STREAMS, H), front.weights["fc_hidden"]
    ).view(rows, STREAMS * H)
    assert projected.shape == (rows, H) and streams.shape == (rows, STREAMS * H)
    for got, ref in ((projected, ref_projected), (streams, ref_streams)):
        cos = F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
        assert cos >= 0.99999
        assert (got.float() - ref.float()).abs().max() <= 0.02 * ref.float().abs().max()


def test_the_front_replays_bitwise_and_compiles_nothing_in_a_capture():
    front = _front(11)
    front.warm_up()
    misses = front_module._compiled.cache_info().misses
    embedded, hidden = _inputs(4, 12)
    eager = front.run(embedded, hidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = front.run(embedded, hidden)
    for _ in range(2):
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(captured[0], eager[0]) and torch.equal(captured[1], eager[1])
    assert front_module._compiled.cache_info().misses == misses


def _metadata(cls: type, **fields: object) -> object:
    """A `cls` metadata object with only `fields` set."""
    metadata: object = object.__new__(cls)
    for name, value in fields.items():
        setattr(metadata, name, value)
    return metadata


def _decode_metadata(max_query_len: int = 4) -> dict[str, object]:
    """The draft layer's metadata in a pure decode step (three draft tokens)."""
    return {
        "raw": _metadata(QSAIndexerMetadata, num_prefills=0),
        "compressed": _metadata(QSAIndexerMetadata, num_prefills=0),
        "qsa": _metadata(Qwen4ExpQSAMetadata, max_query_len=max_query_len),
    }


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
    monkeypatch.setattr(draft_front, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(decode, "get_forward_context", lambda: state.context)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    return state


def _gated_front() -> "DraftFront":
    """A served draft front with the draft layer's gate, up to 64 rows."""
    front = object.__new__(DraftFront)
    front.config = DecodeConfig(tokens_per_request=4, max_rows=64)
    front.refusal = None
    front.weights = {name: torch.empty(0) for name in draft_front.PROJECTIONS}
    front.gate = PureDecodeGate(
        prefill_counting_layers=("raw", "compressed"),
        qsa_main_layers=("qsa",),
        max_decode_query_len=4,
    )
    front.layer_qsa = QsaRows([])
    return front


@pytest.mark.parametrize("rows, max_query_len", [(1, 1), (16, 1), (4, 4), (64, 4)])
def test_takes_full_captures_of_draft_decode_passes(rows, max_query_len, capture):
    """The first pass (four rows a request) and the later ones (one)."""
    capture.context.attn_metadata = _decode_metadata(max_query_len)
    assert _gated_front().takes(*_inputs(rows, 0))


def test_other_passes_keep_the_stock_front(capture):
    front = _gated_front()
    inputs = _inputs(4, 0)
    assert not front.takes(*_inputs(65, 0))
    capture.capturing = False
    assert not front.takes(*inputs)
    capture.capturing = True
    capture.context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
    assert not front.takes(*inputs)
    capture.context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    capture.context.attn_metadata["raw"].num_prefills = 1
    assert not front.takes(*inputs)
    capture.context.attn_metadata = _decode_metadata(max_query_len=5)
    assert not front.takes(*inputs)
    capture.context.attn_metadata = _decode_metadata()
    del capture.context.attn_metadata["compressed"]
    assert not front.takes(*inputs)
    capture.context.attn_metadata = _decode_metadata()
    embedded, hidden = _inputs(4, 0, pad=8)
    assert not front.takes(embedded[:, 4:], hidden)
    assert front.takes(*inputs)
    front.refusal = "a refused deployment"
    assert not front.takes(*inputs)
    front.refusal, front.weights = None, {}
    assert not front.takes(*inputs)


def test_replicas_take_the_whole_checkpoint_tensor():
    """The wrapped loader copies the whole tensor, then runs the stock loader;
    a missing replica refuses the front at the end of loading."""
    front = object.__new__(DraftFront)
    front.replicas, front.loaded, front.refusal = {}, set(), draft_front.UNBOUND
    calls: list[tuple] = []
    layers = {}
    for name in ("fc_embedding", "fc_hidden"):
        shard = torch.nn.Parameter(
            torch.zeros(H // 4, H, dtype=torch.bfloat16, device="cuda"),
            requires_grad=False,
        )
        shard.weight_loader = lambda param, weight, *args, _n=name: calls.append(
            (_n, tuple(weight.shape), args)
        )
        layers[name] = SimpleNamespace(weight=shard)
        front._replicate(name, layers[name])
    whole = torch.randn(H, H, device="cuda").to(torch.bfloat16)
    layers["fc_embedding"].weight.weight_loader(layers["fc_embedding"].weight, whole)
    assert torch.equal(front.replicas["fc_embedding"], whole)
    assert front.loaded == {"fc_embedding"}
    # A tensor that is not the whole weight is the stock loader's alone.
    layers["fc_hidden"].weight.weight_loader(layers["fc_hidden"].weight, whole[:8], 0)
    assert front.loaded == {"fc_embedding"}
    assert calls == [("fc_embedding", (H, H), ()), ("fc_hidden", (8, H), (0,))]
    front.finish_loading()
    assert front.refusal == "full checkpoint tensors of fc_hidden"
    assert front.weights == {} and front.replicas == {}
    # A refused front's loaders still load the shards, without replicas.
    layers["fc_embedding"].weight.weight_loader(layers["fc_embedding"].weight, whole)
    assert calls[-1] == ("fc_embedding", (H, H), ())


class _Projection:
    """The parts of a draft projection layer that the front and the stock
    front use: an unquantized TP4 shard with a loader, called as the gathered
    layer (the whole weight)."""

    def __init__(self, seed: int) -> None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        self.whole = (0.02 * torch.randn(H, H, generator=generator, device="cuda")).to(
            torch.bfloat16
        )
        self.quant_method = UnquantizedLinearMethod()
        self.weight = torch.nn.Parameter(
            self.whole[: H // 4].clone(), requires_grad=False
        )
        self.weight.weight_loader = lambda param, weight, *args: None
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return F.linear(x, self.whole)


class _Reached(Exception):
    """Raised by the draft layer: the front's outputs reached it."""


def _draft_layer(attention: nn.Module) -> nn.Module:
    """A draft layer holding a stand-in `attention` (built without
    `nn.Module.__init__`) without registering it as a submodule."""
    layer = nn.Module()
    layer.__dict__["self_attn"] = attention
    return layer


class _RecordingGemms(PlannedGemms):
    """`PlannedGemms` without layers that records whether its block is
    open."""

    def __init__(self) -> None:
        super().__init__([])
        self.open = False

    @contextlib.contextmanager
    def active(self) -> Iterator[None]:
        self.open = True
        try:
            yield
        finally:
            self.open = False


class _RecordingQsa(QsaRows):
    """`QsaRows` without layers that records the rows of its open blocks."""

    def __init__(self) -> None:
        super().__init__([])
        self.rows: list[int] = []

    @contextlib.contextmanager
    def active(self, rows: int) -> Iterator[None]:
        self.rows.append(rows)
        yield


@pytest.mark.parametrize("taken", [True, False])
def test_the_draft_layer_takes_the_gemm_plans_only_with_the_front(capture, taken):
    """A pass that takes the front runs the draft layer with its GEMM plans
    and its QSA layers' choice for the pass's rows; a pass that runs the stock
    front runs the layer as built."""
    front = _gated_front()
    gemms = _RecordingGemms()
    front.layer_gemms = gemms
    qsa = _RecordingQsa()
    front.layer_qsa = qsa

    def run(embedded: torch.Tensor, hidden: torch.Tensor):
        return embedded[:, :H], hidden

    front.run = run  # type: ignore[method-assign]
    capture.capturing = taken
    embed_norm, hidden_norm = _norms(31)
    seen: list[bool] = []

    def layer(**kwargs: object) -> None:
        seen.append(gemms.open)
        raise _Reached(kwargs)

    predictor = SimpleNamespace(
        hc_count=STREAMS,
        hidden_size=H,
        pre_fc_norm_embedding=embed_norm,
        pre_fc_norm_hidden=hidden_norm,
        fc_embedding=_Projection(32),
        fc_hidden=_Projection(33),
        draft_front=front,
        layers=[layer],
        hyper_connection_mixer=None,
        num_mtp_layers=1,
        use_hc_sequence_parallel=False,
    )
    embedded, hidden = _inputs(4, 34)
    with set_current_vllm_config(VllmConfig()), pytest.raises(_Reached):
        Qwen4ExpMultiTokenPredictor.forward(
            predictor,
            None,
            torch.zeros(4, dtype=torch.long, device="cuda"),
            hidden_states=hidden,
            inputs_embeds=embedded,
        )
    assert seen == [taken]
    assert qsa.rows == ([4] if taken else [])
    assert predictor.fc_embedding.calls == (0 if taken else 1)
    assert not front.layer_gemms.open


def _stand_in_pass(rows: int, sites: str):
    """A draft layer and final mixer of the executor's CPU stand-ins, the
    draft front running them with the stand-in boundary kernel for `sites`
    (as `test_executor.SITES`), and a pass's projected embedding, streams
    and positions."""
    log: list[str] = []
    generator = torch.Generator().manual_seed(40)
    layer = _layer(0, "qwen_sparse_attention", False, log, generator)
    final_mixer = StandInHyperConnection("final_hc", log, generator, use_combine=False)
    front = object.__new__(DraftFront)
    front.kernel_read_rows = 64
    front.boundaries = Boundaries(
        StandInBoundaryKernel(), sites == "tp", [None if sites == "none" else layer.mlp]
    )
    front.layer_gemms = _RecordingGemms()
    front.layer_qsa = _RecordingQsa()
    projected = torch.randn(rows, STAND_IN_HIDDEN, generator=generator)
    streams = torch.randn(rows, STREAMS * STAND_IN_HIDDEN, generator=generator)
    front.run = lambda embedded, hidden: (projected, streams)  # type: ignore[method-assign]
    predictor = SimpleNamespace(
        layers=[layer], hyper_connection_mixer=final_mixer, num_mtp_layers=1
    )
    return front, predictor, log, (projected, streams, torch.arange(rows))


@pytest.mark.parametrize("rows", [4, 64, 65])
@pytest.mark.parametrize("sites", ["none", "tp", "local"])
def test_the_draft_layer_runs_as_the_executor_runs_a_target_layer(
    rows, sites, monkeypatch
):
    """The stock layer's and final mixer's result, with the boundary kernel
    at the reads after the blocks it reduces, the read kernels for the other
    reads of a pass of up to 64 rows (the first read combining the projected
    embedding without injection logits), and the attention block's output
    projection restored."""
    front, predictor, log, (projected, streams, positions) = _stand_in_pass(rows, sites)
    layer, final_mixer = predictor.layers[0], predictor.hyper_connection_mixer
    hidden_states, block_output, injection = layer(
        hidden_states=streams,
        prev_block_output=projected,
        prev_injection=None,
        positions=positions,
        input_ids=None,
        query_start_loc=None,
        ngram_context=None,
    )
    expected_multi, expected_sample, _ = final_mixer.combine_and_mix(
        hidden_states, block_output, injection
    )
    expected_log = list(log)
    log.clear()
    reads: list[tuple[str, bool]] = []
    projects: list[str] = []

    def kernel_read(hc, hidden_states, block_output, injection):
        reads.append((hc.name, injection is None))
        return decode.stock_read(hc, hidden_states, block_output, injection)

    def kernel_project(hc, xn):
        projects.append(hc.name)
        return decode.stock_project(hc, xn)

    monkeypatch.setattr(draft_front, "hc_read", kernel_read)
    monkeypatch.setattr(draft_front, "hc_project", kernel_project)
    monkeypatch.setattr(draft_front, "moe_unsupported", lambda mlp: "")

    sample, multi = front.run_pass(
        predictor, torch.empty(rows, 1), streams, positions, spec_step_idx=2
    )

    assert torch.equal(sample, expected_sample)
    assert torch.equal(multi, expected_multi)
    names = {id(hc.weight): hc.name for hc in (layer.mlp_hyper_connection, final_mixer)}
    boundary_reads = [(names[id(w)], kind) for w, kind in front.boundaries.kernel.calls]
    assert (
        boundary_reads
        == {
            "none": [],
            "tp": [("0.mlp_hc", "attention"), ("final_hc", "moe")],
            "local": [("final_hc", "moe")],
        }[sites]
    )
    served = {name for name, _ in boundary_reads}
    assert log == _without_reads(expected_log, served)
    order = ["0.attn_hc", "0.mlp_hc", "final_hc"]
    if rows <= 64:
        # Only the first read goes without injection logits.
        assert reads == [
            (name, name == "0.attn_hc") for name in order if name not in served
        ]
        assert projects == [name for name in order if name in served]
    else:
        assert reads == [] and projects == []
    attention = layer.self_attn
    assert attention.reduced == [True, sites != "tp"]
    assert attention.projection.reduce_results
    assert front.layer_qsa.rows == [rows] and not front.layer_gemms.open


def test_a_draft_moe_block_that_no_longer_passes_raises(monkeypatch):
    """The draft boundaries are built before the quantization methods process
    the weights, so `_run_layer` checks the MoE block again at capture: a
    block that no longer passes would be reduced twice."""
    front, predictor, _, inputs = _stand_in_pass(4, "tp")
    monkeypatch.setattr(draft_front, "moe_unsupported", lambda mlp: "a sigmoid gate")
    with pytest.raises(RuntimeError, match="no longer has a sigmoid gate"):
        front.run_pass(predictor, torch.empty(4, 1), inputs[1], inputs[2], 0)
    assert front.boundaries.kernel.calls == []


def test_a_front_whose_weights_were_never_bound_keeps_the_stock_front(
    capture, monkeypatch
):
    """Loaders that skip the draft model's load_weights (dummy weights, the
    weight cache, sharded state) never call finish_loading: the front refuses,
    and a pure decode capture's pass runs the stock front."""
    monkeypatch.setattr(draft_front, "get_tensor_model_parallel_world_size", lambda: 4)
    logged: list[tuple] = []
    monkeypatch.setattr(
        draft_front.logger, "info_once", lambda msg, *args: logged.append(args)
    )
    embed_norm, hidden_norm = _norms(21)
    indexer = SimpleNamespace(
        raw_key_cache=SimpleNamespace(prefix="raw"),
        compressed_key_cache=SimpleNamespace(prefix="compressed"),
    )
    attention = _metadata(Qwen4ExpQSAAttention, indexer=indexer, layer_name="qsa")
    predictor = SimpleNamespace(
        hc_count=STREAMS,
        hidden_size=H,
        pre_fc_norm_embedding=embed_norm,
        pre_fc_norm_hidden=hidden_norm,
        fc_embedding=_Projection(22),
        fc_hidden=_Projection(23),
        layers=[_draft_layer(attention)],
        hyper_connection_mixer=None,
        num_mtp_layers=1,
        use_hc_sequence_parallel=False,
    )
    front = DraftFront(predictor, DecodeConfig(tokens_per_request=4, max_rows=64))
    assert front.refusal == draft_front.UNBOUND
    assert set(front.replicas) == set(draft_front.PROJECTIONS) and not front.weights
    embedded, hidden = _inputs(4, 24)
    assert not front.takes(embedded, hidden)
    assert logged == [(draft_front.UNBOUND,)]

    def layer(**kwargs: object) -> None:
        raise _Reached(kwargs)

    predictor.draft_front, predictor.layers = front, [layer]
    with set_current_vllm_config(VllmConfig()), pytest.raises(_Reached) as reached:
        Qwen4ExpMultiTokenPredictor.forward(
            predictor,
            None,
            torch.zeros(4, dtype=torch.long, device="cuda"),
            hidden_states=hidden,
            inputs_embeds=embedded,
        )
    assert predictor.fc_embedding.calls == predictor.fc_hidden.calls == 1
    out = reached.value.args[0]
    assert out["prev_block_output"].shape == (4, H)
    assert out["hidden_states"].shape == (4, STREAMS * H)
    # Loading that never saw the whole tensors refuses and frees the replicas.
    front.finish_loading()
    assert front.refusal == "full checkpoint tensors of fc_embedding and fc_hidden"
    assert front.replicas == {} and front.weights == {}
