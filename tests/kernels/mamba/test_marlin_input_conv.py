# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare fused projection/conv with the existing public kernels on GB10."""

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.model_executor.layers.mamba.ops.marlin_input_conv import _first_cache_indices
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_nvfp4_like,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

M, N, K = 8, 18560, 4096
CONV_START, CONV_END = 8192, 18432
CHANNELS, HISTORY = 10240, 10

BATCHED_SHAPES = [(b, t) for b in range(2, 5) for t in range(1, 9)]


@pytest.fixture(scope="module")
def operands():
    if not current_platform.is_cuda() or not torch.cuda.is_available():
        pytest.skip("requires NVIDIA CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor, properties.multi_processor_count) != (
        12,
        1,
        48,
    ):
        pytest.skip("requires the supported SM121/48SM device")
    if not hasattr(torch.ops._C, "marlin_mamba_input_conv"):
        pytest.skip("requires vLLM build with Marlin input-conv support")

    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(123)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.1
        weight = torch.randn(N, K, dtype=torch.bfloat16, device=device) / 64
        weight_ref, packed, scales, global_scale = rand_marlin_weight_nvfp4_like(
            weight, 16
        )
        del weight, weight_ref
        taps = torch.randn(CHANNELS, 4, dtype=torch.bfloat16, device=device) * 0.1
        bias = torch.randn(CHANNELS, dtype=torch.bfloat16, device=device) * 0.1
        pristine = (
            torch.randn(5, HISTORY + 1, CHANNELS, device=device, dtype=torch.bfloat16)
            * 0.1
        )
        # Keep the original B1 seed/weights unchanged; only extra rows are new.
        batch_input = torch.cat(
            (x, torch.randn(24, K, dtype=x.dtype, device=device) * 0.1)
        )
    return dict(
        input=x,
        batch_input=batch_input,
        weight=packed,
        scales=scales,
        global_scale=global_scale,
        workspace=marlin_make_workspace_new(device),
        taps=taps,
        bias=bias,
        pristine=pristine,
    )


def _arguments(operands, page, accepted, gap):
    # Keep the entire backing, including inter-page padding, observable.
    backing = operands["pristine"][:, : HISTORY + gap].clone()
    history = backing[:, :HISTORY]
    projected = torch.full((M, N), 91, dtype=torch.bfloat16, device=history.device)
    args = {
        name: value
        for name, value in operands.items()
        if name not in ("pristine", "batch_input")
    }
    args.update(
        projected=projected,
        history=history,
        # Public speculative metadata has eight columns; only column0 owns
        # the physical ReplaySSM page. Poison other columns to detect misuse.
        cache_indices=_first_cache_indices(
            torch.tensor(
                [[page] + [-999] * 7], dtype=torch.int32, device=history.device
            )
        ),
        accepted_tokens=torch.tensor(
            [accepted], dtype=torch.int32, device=history.device
        ),
        pad_slot_id=-1,
        null_block_id=0,
    )
    return args, backing


def _reference(args, backing):
    expected_backing = backing.clone()
    expected = ops.marlin_gemm(
        a=args["input"],
        c=None,
        b_q_weight=args["weight"],
        b_bias=None,
        b_scales=args["scales"],
        global_scale=args["global_scale"],
        b_zeros=None,
        workspace=args["workspace"],
        b_q_type=scalar_types.float4_e2m1f,
        size_m=args["input"].shape[0],
        size_n=N,
        size_k=K,
        use_atomic_add=False,
        use_fp32_reduce=True,
    )
    causal_conv1d_update(
        x=expected[:, CONV_START:CONV_END].T.unsqueeze(0),
        conv_state=expected_backing[:, : args["history"].shape[1]].transpose(1, 2),
        weight=args["taps"],
        bias=args["bias"],
        activation="silu",
        conv_state_indices=args["cache_indices"],
        num_accepted_tokens=args["accepted_tokens"],
    )
    return expected, expected_backing


def _assert_nrmse(actual, expected):
    actual, expected = actual.float(), expected.float()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    rms_error = (actual - expected).square().mean().sqrt()
    rms_reference = expected.square().mean().sqrt()
    # Full-K accumulation can associate FP32 additions differently from Marlin.
    assert (rms_error / rms_reference.clamp_min(1e-12)).item() <= 0.001


def _assert_bytes_equal(actual, expected):
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_mamba_output_uses_stock_marlin_kernel_and_graph(operands):
    """The Mamba-owned launch retains stock packing and exact graph replay."""
    device = operands["input"].device
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(711)
        x = torch.randn(8, 8192, device=device, dtype=torch.bfloat16) * 0.1
        weight = torch.randn(4096, 8192, device=device, dtype=x.dtype) / 64
        _reference_weight, packed, scales, global_scale = rand_marlin_weight_nvfp4_like(
            weight, 16
        )
    workspace = operands["workspace"]
    actual = ops.mamba_marlin_output(x, packed, scales, global_scale, workspace)
    stock = ops.marlin_gemm(
        x,
        None,
        packed,
        None,
        scales,
        global_scale,
        None,
        workspace,
        scalar_types.float4_e2m1f,
        8,
        4096,
        8192,
        use_atomic_add=False,
        use_fp32_reduce=True,
    )
    _assert_nrmse(actual, stock)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ops.mamba_marlin_output(x, packed, scales, global_scale, workspace)
    for _ in range(3):
        x.add_(0.01)
        graph.replay()
        eager = ops.mamba_marlin_output(x, packed, scales, global_scale, workspace)
        _assert_bytes_equal(captured, eager)
    assert torch.count_nonzero(workspace) == 0
    with pytest.raises(RuntimeError, match="shape/contiguity"):
        ops.mamba_marlin_output(x[:7], packed, scales, global_scale, workspace)


def _batched_arguments(operands, batches, tokens, *, regular=False, inactive=False):
    args, _ = _arguments(operands, 1, 1, 1)
    history_len = 3 if regular else HISTORY
    backing = operands["pristine"][:, : history_len + 1].clone()
    args["history"] = backing[:, :history_len]
    rows = batches * tokens
    args["input"] = operands["batch_input"][:rows].clone()
    args["projected"] = torch.full(
        (rows, N), 91, dtype=torch.bfloat16, device=backing.device
    )
    slots = [4, 1, 3, 2][:batches]
    accepted = [1, 4, 7, 2][:batches]
    if inactive:
        slots, accepted = [4, -1, 0, 2], [1, -999, -999, 8]
    columns = torch.tensor(
        [[slot] + [-999] * 7 for slot in slots],
        device=backing.device,
        dtype=torch.int32,
    )
    args["cache_indices"] = _first_cache_indices(columns, batches)
    assert args["cache_indices"].stride(0) == 8
    args["accepted_tokens"] = (
        None
        if regular
        else torch.tensor(accepted, device=backing.device, dtype=torch.int32)
    )
    return args, backing, slots


def _batched_reference(args, backing, slots):
    """Use the existing public Marlin and conv kernels, without copied math."""
    expected_backing = backing.clone()
    rows, batches = args["input"].shape[0], len(slots)
    tokens = rows // batches
    expected = ops.marlin_gemm(
        args["input"],
        None,
        args["weight"],
        None,
        args["scales"],
        args["global_scale"],
        None,
        args["workspace"],
        scalar_types.float4_e2m1f,
        rows,
        N,
        K,
        use_atomic_add=False,
        use_fp32_reduce=True,
    )
    for batch, slot in enumerate(slots):
        block = expected[batch * tokens : (batch + 1) * tokens]
        if slot <= 0:
            block[:, CONV_START:CONV_END].fill_(91)
            continue
        causal_conv1d_update(
            x=block[:, CONV_START:CONV_END].T.unsqueeze(0),
            conv_state=expected_backing[:, : args["history"].shape[1]].transpose(1, 2),
            weight=args["taps"],
            bias=args["bias"],
            activation="silu",
            conv_state_indices=args["cache_indices"][batch : batch + 1],
            num_accepted_tokens=(
                args["accepted_tokens"][batch : batch + 1]
                if args["accepted_tokens"] is not None
                else None
            ),
        )
    return expected, expected_backing


def _assert_batched_projection(actual, expected):
    for start, end in (
        (0, CONV_START),
        (CONV_START, 16384),
        (16384, 17408),
        (17408, CONV_END),
        (CONV_END, N),
    ):
        _assert_nrmse(actual[:, start:end], expected[:, start:end])


@pytest.mark.parametrize("batches,tokens", BATCHED_SHAPES)
def test_uniform_batched_projection_conv(operands, batches, tokens):
    """Each request uses its strided page and acceptance, in one whole M tile."""
    assert hasattr(torch.ops._C.marlin_mamba_input_conv, "batched")
    args, backing, slots = _batched_arguments(operands, batches, tokens)
    before = backing.clone()
    expected, expected_backing = _batched_reference(args, backing, slots)
    assert ops.marlin_mamba_input_conv_batched(**args) is None
    _assert_batched_projection(args["projected"], expected)
    for slot in range(backing.shape[0]):
        if slot in slots:
            _assert_nrmse(
                backing[slot, : tokens + 2], expected_backing[slot, : tokens + 2]
            )
        else:
            _assert_bytes_equal(backing[slot], before[slot])
    _assert_bytes_equal(backing[:, tokens + 2 :], before[:, tokens + 2 :])


@pytest.mark.parametrize(
    "regular,tokens", [(True, 1), (False, 2), (False, 5), (False, 8)]
)
def test_batched_regular_or_inactive_requests(operands, regular, tokens):
    """None acceptance is regular T1/H3; null/pad skip poisoned acceptance."""
    args, backing, slots = _batched_arguments(
        operands, 4, tokens, regular=regular, inactive=not regular
    )
    before = backing.clone()
    expected, expected_backing = _batched_reference(args, backing, slots)
    ops.marlin_mamba_input_conv_batched(**args)
    _assert_batched_projection(args["projected"], expected)
    _assert_nrmse(backing, expected_backing)
    for slot in range(backing.shape[0]):
        if slot not in slots or slot == 0:
            _assert_bytes_equal(backing[slot], before[slot])
    if not regular:
        for batch in (1, 2):
            _assert_bytes_equal(
                args["projected"][
                    batch * tokens : (batch + 1) * tokens, CONV_START:CONV_END
                ],
                torch.full_like(expected[:tokens, CONV_START:CONV_END], 91),
            )


@pytest.mark.parametrize("invalid", ["nonuniform", "zero_stride", "alias"])
def test_batched_rejects_invalid_metadata_before_mutation(operands, invalid):
    args, backing, _ = _batched_arguments(operands, 2, 4)
    if invalid == "nonuniform":
        args["input"] = args["input"][:7].clone()
        args["projected"] = args["projected"][:7].clone()
    elif invalid == "zero_stride":
        args["cache_indices"] = args["cache_indices"][:1].expand(2)
    else:
        # The strided index span includes the column padding between requests.
        storage = torch.zeros((4, 16), dtype=torch.int32, device=backing.device)
        args["cache_indices"] = storage[:2, 0]
        args["workspace"] = storage.view(-1)
    before, projected_before = backing.clone(), args["projected"].clone()
    with pytest.raises(RuntimeError, match="uniform|shape|overlaps"):
        ops.marlin_mamba_input_conv_batched(**args)
    _assert_bytes_equal(backing, before)
    _assert_bytes_equal(args["projected"], projected_before)


def _assert_result(args, backing, expected, expected_backing, before, page):
    for start, end in (
        (0, CONV_START),
        (CONV_START, 16384),
        (16384, 17408),
        (17408, CONV_END),
        (CONV_END, N),
    ):
        _assert_nrmse(args["projected"][:, start:end], expected[:, start:end])
    history_len = args["history"].shape[1]
    _assert_nrmse(backing[page, :history_len], expected_backing[page, :history_len])
    for other_page in range(backing.shape[0]):
        if other_page != page:
            _assert_bytes_equal(backing[other_page], before[other_page])
    _assert_bytes_equal(backing[:, history_len:], before[:, history_len:])


@pytest.mark.parametrize("accepted", [1, 4, 8])
@pytest.mark.parametrize("page,gap", [(1, 0), (4, 1)])
def test_projection_conv_matches_separate_kernels(operands, accepted, page, gap):
    """Read the selected acceptance history and update only its SD cache page."""
    args, backing = _arguments(operands, page, accepted, gap)
    before = backing.clone()
    expected, expected_backing = _reference(args, backing)
    assert ops.marlin_mamba_input_conv(**args) is None
    _assert_result(args, backing, expected, expected_backing, before, page)


@pytest.mark.parametrize("tokens", [1, 3, 7, 8])
def test_cuda_graph_replay_reads_current_metadata(operands, tokens):
    """Replay uses caller buffers and current page/acceptance metadata."""
    args, backing = _arguments(operands, 1, 1, 1)
    args["input"] = args["input"][:tokens].clone()
    args["projected"] = args["projected"][:tokens].clone()
    before = backing.clone()
    stream = torch.cuda.Stream(device=backing.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ops.marlin_mamba_input_conv(**args)
    torch.cuda.current_stream().wait_stream(stream)
    backing.copy_(before)
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        ops.marlin_mamba_input_conv(**args)

    for page, accepted in ((1, 1), (4, 8)):
        backing.copy_(before)
        args["cache_indices"].fill_(page)
        args["accepted_tokens"].fill_(accepted)
        args["projected"].fill_(91)
        expected, expected_backing = _reference(args, backing)
        graph.replay()
        _assert_result(args, backing, expected, expected_backing, before, page)


@pytest.mark.parametrize("page", [-1, 0])
@pytest.mark.parametrize("tokens", range(1, 9))
def test_inactive_page_preserves_history_and_dead_conv_slots(operands, page, tokens):
    """Padding/null pages must return before reading invalid acceptance data."""
    args, backing = _arguments(operands, page, -999, 1)
    args["input"] = args["input"][:tokens].clone()
    args["projected"] = args["projected"][:tokens].clone()
    before = backing.clone()
    expected = ops.marlin_gemm(
        args["input"],
        None,
        args["weight"],
        None,
        args["scales"],
        args["global_scale"],
        None,
        args["workspace"],
        scalar_types.float4_e2m1f,
        tokens,
        N,
        K,
        use_atomic_add=False,
        use_fp32_reduce=True,
    )
    ops.marlin_mamba_input_conv(**args)
    _assert_bytes_equal(backing, before)
    _assert_bytes_equal(
        args["projected"][:, CONV_START:CONV_END],
        torch.full_like(args["projected"][:, CONV_START:CONV_END], 91),
    )
    _assert_nrmse(args["projected"][:, :CONV_START], expected[:, :CONV_START])
    _assert_nrmse(args["projected"][:, CONV_END:], expected[:, CONV_END:])


@pytest.mark.parametrize("alias", ["history", "input", "workspace"])
def test_rejects_overlapping_operands_before_mutation(operands, alias):
    args, backing = _arguments(operands, 1, 4, 1)
    if alias == "history":
        args["projected"] = args["history"].as_strided((M, N), (N, 1))
    elif alias == "input":
        args["input"] = args["projected"].view(-1)[: M * K].view(M, K)
    else:
        args["workspace"] = args["weight"].view(-1)[:48]
    before = backing.clone()
    projected_before = args["projected"].clone()
    with pytest.raises(RuntimeError, match="disjoint|overlaps"):
        ops.marlin_mamba_input_conv(**args)
    _assert_bytes_equal(backing, before)
    _assert_bytes_equal(args["projected"], projected_before)


@pytest.mark.parametrize("invalid", ["dtype", "batch", "history_layout"])
def test_rejects_unsupported_geometry_before_mutation(operands, invalid):
    args, backing = _arguments(operands, 1, 4, 1)
    if invalid == "dtype":
        args["input"] = args["input"].to(torch.float16)
        match = "device/dtype mismatch"
    elif invalid == "batch":
        args["input"] = args["input"].repeat(2, 1)
        match = r"requires 1\.\.8 actual rows"
    else:
        args["history"] = args["history"].transpose(1, 2)
        match = "SD history view"
    before = backing.clone()
    projected_before = args["projected"].clone()
    with pytest.raises(RuntimeError, match=match):
        ops.marlin_mamba_input_conv(**args)
    _assert_bytes_equal(backing, before)
    _assert_bytes_equal(args["projected"], projected_before)


def test_mutation_schema_and_fake_tensor(operands):
    args, _ = _arguments(operands, 4, 4, 1)
    opcheck(
        torch.ops._C.marlin_mamba_input_conv.default,
        (),
        kwargs=args,
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.parametrize("tokens", range(1, 9))
@pytest.mark.parametrize("configured", ["actual", "max"])
@pytest.mark.parametrize("previous", ["first", "capacity"])
def test_short_query_preserves_physical_history(operands, tokens, configured, previous):
    """Actual short tokens may follow a full prior draft; suffix/gaps stay exact."""
    history_len = tokens + 2 if configured == "actual" else HISTORY
    accepted = 1 if previous == "first" else history_len - 2
    backing = operands["pristine"][:, : history_len + 1].clone()
    before = backing.clone()
    args = {k: v for k, v in operands.items() if k not in ("pristine", "batch_input")}
    args.update(
        input=operands["input"][:tokens].clone(),
        projected=torch.full(
            (tokens, N), 91, dtype=torch.bfloat16, device=backing.device
        ),
        history=backing[:, :history_len],
        cache_indices=torch.tensor([1], device=backing.device, dtype=torch.int32),
        accepted_tokens=torch.tensor(
            [accepted], device=backing.device, dtype=torch.int32
        ),
        pad_slot_id=-1,
        null_block_id=0,
    )
    expected, expected_backing = _reference(args, backing)
    ops.marlin_mamba_input_conv(**args)
    torch.cuda.synchronize()
    _assert_result(args, backing, expected, expected_backing, before, 1)
    _assert_bytes_equal(backing[1, tokens + 2 :], before[1, tokens + 2 :])


def test_regular_one_token_accepts_none(operands):
    """The real non-speculative convolution has H3 and no acceptance tensor."""
    backing = operands["pristine"][:, :4].clone()
    before = backing.clone()
    args = {k: v for k, v in operands.items() if k not in ("pristine", "batch_input")}
    args.update(
        input=operands["input"][:1].clone(),
        projected=torch.full((1, N), 91, dtype=torch.bfloat16, device=backing.device),
        history=backing[:, :3],
        cache_indices=torch.tensor([4], device=backing.device, dtype=torch.int32),
        accepted_tokens=None,
        pad_slot_id=-1,
        null_block_id=0,
    )
    expected, expected_backing = _reference(args, backing)
    ops.marlin_mamba_input_conv(**args)
    torch.cuda.synchronize()
    _assert_result(args, backing, expected, expected_backing, before, 4)
    opcheck(
        torch.ops._C.marlin_mamba_input_conv.default,
        (),
        kwargs=args,
        test_utils=("test_schema", "test_faketensor"),
    )
