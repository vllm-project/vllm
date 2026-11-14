# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDN decode core of Qwen4Exp's specialized decode (`gdn_decode_core`,
CuTe DSL) against the stock core (`causal_conv1d_update` followed by
`fused_gdn_decode_post_conv_mtp`) on vLLM's caches: every written recurrent
state and the convolution state bit for bit and the outputs within one BF16
unit (`_assert_outputs_match`), for every schedule,
with mixed accepted counts, a padded request, requests of fewer rows than the
verification and requests without rows (a FULL graph's batch padding); its
CUDA graph replay; and the executor's GDN layer (`GdnLayers`: the `ba`
projection on a side stream joined before the core) against the stock layer,
eagerly and replayed in a CUDA graph, with its capture-time check of the
fork and join on the captured graph, which a capture without the join
fails."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

if TYPE_CHECKING:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(100),
    reason="the specialized decode's GDN core requires SM100+",
)

if current_platform.is_cuda():
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.gdn import (
        Schedule,
        default_schedule,
        gdn_decode_core,
        warm_up_gdn_decode_core,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.gdn.proj import (
        gdn_projection,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.gdn import (
        CORE_KERNEL,
        PROJECTION_KERNEL,
        GdnLayers,
        capture_dependencies,
        captured_kernel,
        require_dependencies,
    )

D = 128
Q = 4  # rows per request: three MTP draft tokens plus one
EPS = 1e-6
HIDDEN = 256
SCHEDULES = [
    None,
    (3, 8, 12),
    (3, 4, 12),
    (1, 2, 8),
    (1, 2, 4),
    (1, 1, 8, 2),
    (1, 1, 4),
]


class Caches:
    """Random weights, caches and step metadata of one GDN layer.

    `lengths` gives each request's rows (default `Q`); `empty` requests
    without rows and with null slots follow, as a FULL CUDA graph pads its
    batch. Without `lengths`, the last of several requests has null slots.
    """

    def __init__(
        self,
        key_heads: int,
        requests: int,
        seed: int,
        lengths: list[int] | None = None,
        empty: int = 0,
    ) -> None:
        gen = torch.Generator(device="cuda").manual_seed(seed)

        def randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
            return torch.randn(*shape, generator=gen, device="cuda") * scale

        self.kh, self.hv = key_heads, 3 * key_heads
        self.requests = requests + empty
        conv_dim = (2 * key_heads + self.hv) * D
        sizes = [Q] * requests if lengths is None else lengths
        rows, slots = sum(sizes), Q * requests + 1
        self.qkvz = randn(rows, conv_dim + self.hv * D).bfloat16()
        self.ba = randn(rows, 2 * self.hv, scale=2.0).bfloat16()
        self.conv_weight = randn(conv_dim, 4, scale=0.5).bfloat16()
        self.a_log = torch.log(
            torch.rand(self.hv, generator=gen, device="cuda") * 15 + 1
        )
        self.dt_bias = randn(self.hv, scale=0.5).bfloat16()
        self.norm_weight = (1 + randn(D, scale=0.1)).bfloat16()
        # vLLM's default (SD) convolution layout, [slots, q + 2, conv_dim].
        self.conv = randn(slots, Q + 2, conv_dim).bfloat16()
        self.state = randn(slots, self.hv, D, D, scale=0.05)
        order = torch.randperm(slots - 1, generator=torch.Generator().manual_seed(seed))
        self.slots = torch.cat(
            [(order + 1).view(requests, Q).int(), torch.zeros(empty, Q).int()]
        ).cuda()
        self.accepted = (
            torch.cat([torch.arange(requests) % Q + 1, torch.ones(empty)]).int().cuda()
        )
        ends = torch.tensor(sizes + [0] * empty).cumsum(0)
        self.cu = torch.cat([torch.zeros(1), ends]).int().cuda()
        if lengths is None and requests > 1:
            self.slots[requests - 1] = 0  # a padded request with rows

    def conv_view(self) -> torch.Tensor:
        return self.conv.transpose(-1, -2)

    def metadata(self) -> GDNAttentionMetadata:
        """The step's GDN metadata, as a FULL-graph decode step carries it."""
        rows = self.qkvz.shape[0]
        return GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=self.requests,
            num_spec_decode_tokens=rows,
            num_actual_tokens=rows,
            spec_query_start_loc=self.cu,
            spec_state_indices_tensor=self.slots,
            num_accepted_tokens=self.accepted,
        )

    def stock(self) -> torch.Tensor:
        """The stock core as `QwenGatedDeltaNetAttention` runs it."""
        rows = self.qkvz.shape[0]
        out = torch.zeros(rows, self.hv, D, dtype=torch.bfloat16, device="cuda")
        mixed_qkv, gate = self.qkvz.clone().split(
            [(2 * self.kh + self.hv) * D, self.hv * D], dim=-1
        )
        b, a = self.ba.chunk(2, dim=-1)
        n = self.requests
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            self.conv_view(),
            self.conv_weight,
            None,
            "silu",
            conv_state_indices=self.slots[:, 0],
            num_accepted_tokens=self.accepted,
            query_start_loc=self.cu,
            max_query_len=Q,
            validate_data=False,
        )
        ops.fused_gdn_decode_post_conv_mtp(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=self.a_log,
            dt_bias=self.dt_bias,
            state_indices=self.slots[:n],
            cu_seqlens=self.cu,
            num_accepted_tokens=self.accepted,
            state=self.state,
            output_gate=gate.reshape(rows, -1, D),
            norm_weight=self.norm_weight,
            out=out,
            scale=D**-0.5,
            norm_eps=EPS,
            output_gate_activation="sigmoid",
        )
        return out.flatten(-2)

    def fused(self, schedule, out: torch.Tensor | None = None) -> torch.Tensor:
        if out is None:
            out = torch.zeros(
                self.qkvz.shape[0], self.hv * D, dtype=torch.bfloat16, device="cuda"
            )
        return gdn_decode_core(
            self.qkvz,
            self.ba,
            self.conv_weight,
            self.conv_view(),
            self.state,
            self.a_log,
            self.dt_bias,
            self.norm_weight,
            self.slots,
            self.accepted,
            self.cu,
            self.requests,
            out,
            EPS,
            None if schedule is None else Schedule(*schedule),
        )


def _assert_outputs_match(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Outputs within one BF16 unit of the stock core's. At small batches the
    stock kernel splits each head's values over two or four CTAs and adds
    their sums of squares across the cluster, so its norm's scale can differ
    from the core's in the last bit."""
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=2**-7, atol=2**-20
    )


def _assert_matches_the_stock_core(caches: Caches, schedule) -> None:
    conv0, state0 = caches.conv.clone(), caches.state.clone()
    expected = caches.stock()
    expected_conv, expected_state = caches.conv.clone(), caches.state.clone()
    caches.conv.copy_(conv0)
    caches.state.copy_(state0)

    actual = caches.fused(schedule)

    _assert_outputs_match(actual, expected)
    assert torch.equal(caches.conv, expected_conv)
    assert torch.equal(caches.state, expected_state)
    assert not torch.equal(caches.state, state0)


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("key_heads,requests", [(4, 1), (4, 5), (16, 3)])
def test_core_matches_the_stock_core(key_heads, requests, schedule):
    _assert_matches_the_stock_core(Caches(key_heads, requests, seed=requests), schedule)


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_core_matches_the_stock_core_with_short_and_empty_requests(schedule):
    caches = Caches(4, 5, seed=11, lengths=[4, 2, 1, 3, 4], empty=3)
    _assert_matches_the_stock_core(caches, schedule)


@pytest.mark.parametrize("schedule", [(3, 4, 12), (1, 1, 8, 2)])
def test_core_replays_bitwise_in_a_cuda_graph(schedule):
    caches = Caches(4, 4, seed=7)
    conv0, state0 = caches.conv.clone(), caches.state.clone()
    eager = caches.fused(schedule).clone()
    eager_caches = caches.conv.clone(), caches.state.clone()

    out = torch.zeros_like(eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        caches.fused(schedule, out)
    for _ in range(2):
        caches.conv.copy_(conv0)
        caches.state.copy_(state0)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(out, eager)
        assert torch.equal(caches.conv, eager_caches[0])
        assert torch.equal(caches.state, eager_caches[1])


def test_warm_up_runs_every_schedule():
    schedules = [Schedule(3, 4, 12), Schedule(1, 2, 8), Schedule(1, 1, 8, 2)]
    warm_up_gdn_decode_core(4, Q, schedules, torch.device("cuda"))


class Linear:
    """A BF16 projection as `GdnLayers` uses a layer's: its weight, and a
    call returning `(output, None)`; `reduce_results` False, as a
    tensor-parallel output projection is within `attention_partial`."""

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight
        self.bias = None
        self.reduce_results = False

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return F.linear(x, self.weight), None


class GdnModule:
    """The parts of `QwenGatedDeltaNetAttention` that `GdnLayers` uses,
    over `Caches`: BF16 projections, and the stock forward of a speculative
    decode step (projections, zero-filled core output, stock core, output
    projection)."""

    def __init__(self, caches: Caches) -> None:
        gen = torch.Generator(device="cuda").manual_seed(3)

        def weight(rows: int, columns: int) -> torch.Tensor:
            return (
                torch.randn(rows, columns, generator=gen, device="cuda") * 0.05
            ).bfloat16()

        self.caches = caches
        self.w_qkvz = weight(caches.qkvz.shape[1], HIDDEN)
        self.w_ba = weight(caches.ba.shape[1], HIDDEN)
        self.w_out = weight(HIDDEN, caches.hv * D)
        self.num_k_heads, self.tp_size = caches.kh, 1
        self.value_dim = caches.hv * D
        self.kv_cache = [caches.conv, caches.state]
        self.conv1d = SimpleNamespace(weight=caches.conv_weight.unsqueeze(1))
        self.A_log, self.dt_bias = caches.a_log, caches.dt_bias
        self.norm = SimpleNamespace(weight=caches.norm_weight)
        self.layer_norm_epsilon = EPS
        self.in_proj_qkvz = Linear(self.w_qkvz)
        self.in_proj_ba = Linear(self.w_ba)
        self.out_proj = Linear(self.w_out)

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.caches.qkvz = self.in_proj_qkvz(hidden_states)[0]
        self.caches.ba = self.in_proj_ba(hidden_states)[0]
        return self.out_proj(self.caches.stock())[0]


@pytest.mark.parametrize(
    "key_heads,lengths,empty", [(4, [4], 0), (4, [4, 2, 4, 1], 2), (16, [4] * 3, 0)]
)
def test_gdn_layers_match_the_stock_layer(key_heads, lengths, empty):
    caches = Caches(key_heads, len(lengths), seed=5, lengths=lengths, empty=empty)
    attn = GdnModule(caches)
    metadata = caches.metadata()
    # The step's rows: the requests', then the padding rows of the empty ones.
    real = int(caches.cu[-1])
    hidden_states = torch.randn(real + Q * empty, HIDDEN, device="cuda").bfloat16()
    gdn = GdnLayers(
        [cast("QwenGatedDeltaNetAttention", attn)],
        Q,
        16,
        gdn_decode_core,
        default_schedule,
        torch.cuda.Stream(),
    )
    assert gdn.uses_core(metadata)
    conv0, state0 = caches.conv.clone(), caches.state.clone()
    expected = attn(hidden_states)
    expected_caches = caches.conv.clone(), caches.state.clone()

    def run_from_the_start() -> torch.Tensor:
        caches.conv.copy_(conv0)
        caches.state.copy_(state0)
        return gdn(attn, hidden_states, metadata)

    # The core does not write the padding rows (stock zero-fills them): the
    # requests' rows match (states bit for bit, outputs within the last bit of
    # the core's, `_assert_outputs_match`), the padding rows stay finite.
    out = run_from_the_start()
    assert _relative_error(out[:real], expected[:real]) < 2**-7
    assert torch.isfinite(out[real:]).all()
    assert torch.equal(caches.conv, expected_caches[0])
    assert torch.equal(caches.state, expected_caches[1])

    # The capture runs GdnLayers' check of the fork and join on the graph.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run_from_the_start()
    for _ in range(2):
        caches.conv.copy_(conv0)
        caches.state.copy_(state0)
        graph.replay()
        torch.accelerator.synchronize()
        assert _relative_error(out[:real], expected[:real]) < 2**-7
        assert torch.isfinite(out[real:]).all()
        assert torch.equal(caches.conv, expected_caches[0])
        assert torch.equal(caches.state, expected_caches[1])


def test_the_capture_check_catches_a_missing_fork_wait_or_join():
    x, y = torch.zeros(8, device="cuda"), torch.zeros(8, device="cuda")
    side, unforked = torch.cuda.Stream(), torch.cuda.Stream()
    start, done = torch.cuda.Event(), torch.cuda.Event()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        main = torch.cuda.current_stream()
        x.add_(1)
        start.record(main)
        fork = capture_dependencies(main)
        assert fork
        # A stream that did not wait for the fork does not follow it.
        with pytest.raises(RuntimeError, match="does not wait for the fork"):
            require_dependencies(unforked, fork, "the side kernel", "the fork")
        with torch.cuda.stream(side):
            side.wait_event(start)
            require_dependencies(side, fork, "the side kernel", "the fork")
            y.add_(2)
            side_nodes = capture_dependencies(side)
            done.record(side)
        x.add_(3)
        # Before the join, main's next operation does not follow the side's.
        with pytest.raises(RuntimeError, match="does not wait for the side"):
            require_dependencies(main, side_nodes, "the core", "the side")
        main.wait_event(done)
        require_dependencies(main, side_nodes, "the core", "the side")
        x.add_(y)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(x, torch.full_like(x, 6.0))


def _staged_reference(attn: GdnModule, x: torch.Tensor):
    """The staged layer composed by hand from its kernels: the projection of
    qkvz and ba, the core into zero-filled rows, the output projection."""
    caches = attn.caches
    caches.qkvz, caches.ba = gdn_projection(x, [attn.w_qkvz, attn.w_ba])
    core_out = torch.zeros(
        x.shape[0], caches.hv * D, dtype=torch.bfloat16, device="cuda"
    )
    caches.fused(None, core_out)
    return gdn_projection(core_out, [attn.w_out])[0]


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (
        (actual.float() - expected.float()).abs().max() / expected.float().abs().max()
    ).item()


@pytest.mark.parametrize("lengths,empty", [([4], 0), ([4] * 4, 0), ([4, 2], 1)])
def test_staged_gdn_layers_match_their_kernels_and_the_stock_layer(lengths, empty):
    """Steps of up to 16 rows run the staged-weight projections: bit for bit
    the kernels composed by hand, within BF16 rounding of the stock layer,
    and the same when replayed from a capture, which runs the capture checks
    of the node order."""
    caches = Caches(4, len(lengths), seed=13, lengths=lengths, empty=empty)
    attn = GdnModule(caches)
    metadata = caches.metadata()
    real = int(caches.cu[-1])
    rows = real + Q * empty
    hidden_states = torch.randn(rows, HIDDEN, device="cuda").bfloat16()
    gdn = GdnLayers(
        [cast("QwenGatedDeltaNetAttention", attn)], Q, 4, gdn_decode_core,
        default_schedule, torch.cuda.Stream(), projection=gdn_projection,
    )  # fmt: skip
    assert gdn.staged(rows)
    gdn.warm_up()
    conv0, state0 = caches.conv.clone(), caches.state.clone()

    def restore() -> None:
        caches.conv.copy_(conv0)
        caches.state.copy_(state0)

    stock = attn(hidden_states)[:real].clone()
    stock_caches = caches.conv.clone(), caches.state.clone()
    restore()
    reference = _staged_reference(attn, hidden_states).clone()
    reference_caches = caches.conv.clone(), caches.state.clone()
    restore()

    out = gdn(attn, hidden_states, metadata)
    assert torch.equal(out[:real], reference[:real])
    assert torch.equal(caches.conv, reference_caches[0])
    assert torch.equal(caches.state, reference_caches[1])
    assert _relative_error(out[:real], stock) < 2**-6
    assert _relative_error(caches.state, stock_caches[1]) < 2**-6
    assert torch.isfinite(out).all()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        restore()
        out = gdn(attn, hidden_states, metadata)
    for _ in range(2):
        restore()
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(out[:real], reference[:real])
        assert torch.equal(caches.conv, reference_caches[0])
        assert torch.equal(caches.state, reference_caches[1])


def test_the_staged_capture_check_catches_a_kernel_in_between_or_a_join():
    caches = Caches(4, 1, seed=17)
    attn = GdnModule(caches)
    x = torch.randn(Q, HIDDEN, device="cuda").bfloat16()
    core_out = torch.zeros(Q, caches.hv * D, dtype=torch.bfloat16, device="cuda")
    # Compile and load every kernel before the capture.
    gdn_projection(x, [attn.w_qkvz, attn.w_ba])
    gdn_projection(core_out, [attn.w_out])
    caches.fused(None)
    y = torch.zeros(8, device="cuda")
    side = torch.cuda.Stream()
    start, done = torch.cuda.Event(), torch.cuda.Event()

    def project() -> int:
        gdn_projection(x, [attn.w_qkvz, attn.w_ba])
        return captured_kernel(main, PROJECTION_KERNEL, None, "the projection", "")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        main = torch.cuda.current_stream()
        # The order the staged layer captures passes.
        projection = project()
        caches.fused(None, core_out)
        core = captured_kernel(main, CORE_KERNEL, projection, "the core", "it")
        gdn_projection(core_out, [attn.w_out])
        captured_kernel(main, PROJECTION_KERNEL, core, "the out projection", "it")
        # Another kernel is not the projection.
        y.add_(1)
        with pytest.raises(RuntimeError, match="is not the one kernel"):
            captured_kernel(main, PROJECTION_KERNEL, None, "the projection", "")
        # A kernel between the projection and the core.
        projection = project()
        y.add_(1)
        caches.fused(None, core_out)
        with pytest.raises(RuntimeError, match="does not follow it alone"):
            captured_kernel(main, CORE_KERNEL, projection, "the core", "it")
        # A side stream joined before the core.
        projection = project()
        start.record(main)
        with torch.cuda.stream(side):
            side.wait_event(start)
            y.add_(1)
            done.record(side)
        main.wait_event(done)
        caches.fused(None, core_out)
        with pytest.raises(RuntimeError, match="does not follow it alone"):
            captured_kernel(main, CORE_KERNEL, projection, "the core", "it")
