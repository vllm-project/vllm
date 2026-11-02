# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused routed LoRA for one linear output slice.

Tokens are grouped into plans: tokens that use the same resident adapter on
every route. One cooperative launch handles a whole slice. Each plan tile is
owned by a team of WORKERS blocks that split the shrink over the hidden size,
accumulate FP32 partials in scratch, meet at a per-tile counter, then expand
disjoint output columns, summing all routes before a single output update.

The launch is cooperative so every block is resident; a team's spin-wait can
therefore always complete. A wait that exceeds its bound traps rather than
continuing with partial sums. The last block to leave a tile clears its scratch
rows and counter, so no memset is needed between calls.
"""

import ctypes
import functools
from typing import NamedTuple

import torch

from vllm.lora.ops.triton_ops.kernel_utils import mm_k
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import current_stream, direct_register_custom_op

_MAX_SPINS = 1 << 22
MAX_ROUTED_LORA_RANK = 128
_CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95


@triton.jit
def _trap(value):
    tl.inline_asm_elementwise(
        "mov.u32 $0, $1; trap;",
        "=r,r",
        [value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _plan_tiles(Starts, num_plans, BM: tl.constexpr, PLAN_BLOCK: tl.constexpr):
    plans = tl.arange(0, PLAN_BLOCK)
    in_range = plans < num_plans
    begins = tl.load(Starts + plans, mask=in_range, other=0)
    ends = tl.load(Starts + plans + 1, mask=in_range, other=0)
    return plans, (ends - begins + BM - 1) // BM


@triton.jit
def _tile_tokens(
    Sorted, Starts, num_plans, item, BM: tl.constexpr, PLAN_BLOCK: tl.constexpr
):
    """Map a compact tile index to its plan and token rows."""
    plans, plan_tiles = _plan_tiles(Starts, num_plans, BM, PLAN_BLOCK)
    tile_ends = tl.cumsum(plan_tiles, 0)
    plan = tl.sum((tile_ends <= item).to(tl.int32), 0)
    token_tile = item - tl.sum(tl.where(plans == plan, tile_ends - plan_tiles, 0), 0)
    start = tl.load(Starts + plan)
    m_len = tl.minimum(BM, tl.load(Starts + plan + 1) - start - token_tile * BM)
    rows = tl.arange(0, BM)
    # Padding rows repeat real rows; only real rows are written.
    tokens = tl.load(Sorted + start + token_tile * BM + rows % m_len).to(tl.int64)
    return plan.to(tl.int64), tokens, rows < m_len


@triton.jit
def _shrink(
    X,
    A,
    Plans,
    Sorted,
    Starts,
    Z,
    Counters,
    num_plans,
    item,
    worker,
    stride_x,
    H: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
    BR: tl.constexpr,
    WORKERS: tl.constexpr,
    PLAN_BLOCK: tl.constexpr,
):
    plan, tokens, valid = _tile_tokens(Sorted, Starts, num_plans, item, BM, PLAN_BLOCK)
    ranks = tl.arange(0, BR)
    z_ptrs = Z + tokens[:, None] * (K * R) + ranks[None, :]
    z_mask = valid[:, None] & (ranks[None, :] < R)
    hidden = worker * BH + tl.arange(0, BH)
    for route in tl.static_range(K):
        adapter = tl.load(Plans + plan * K + route).to(tl.int64)
        if (adapter >= 0) & (adapter < E):
            partial = mm_k(
                X + tokens[:, None] * stride_x + hidden[None, :],
                A + (adapter * R + (ranks % R)[None, :]) * H + hidden[:, None],
                1,
                1,
                hidden,
                H,
                BM,
                BR,
                BH,
                H % (BH * WORKERS) == 0,
                WORKERS,
                False,
                A.dtype.element_ty,
                False,
                worker * BH,
            )
            tl.atomic_add(z_ptrs + route * R, partial, mask=z_mask, sem="relaxed")
    # Publish this block's partial sums before arriving.
    tl.debug_barrier()
    tl.atomic_add(Counters + item, 1, sem="acq_rel")


@triton.jit
def _finish(
    B,
    Y,
    W,
    Plans,
    Sorted,
    Starts,
    Z,
    Counters,
    num_plans,
    item,
    worker,
    offset_start,
    stride_y,
    OUT: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
    WORKERS: tl.constexpr,
    MAX_SPINS: tl.constexpr,
    PLAN_BLOCK: tl.constexpr,
):
    plan, tokens, valid = _tile_tokens(Sorted, Starts, num_plans, item, BM, PLAN_BLOCK)
    counter = Counters + item
    arrived = tl.atomic_add(counter, 0, sem="acquire")
    spins = 0
    while (arrived < WORKERS) & (spins < MAX_SPINS):
        arrived = tl.atomic_add(counter, 0, sem="acquire")
        spins += 1
    if arrived < WORKERS:
        _trap(arrived)
    tl.debug_barrier()

    ranks = tl.arange(0, BR)
    z_ptrs = Z + tokens[:, None] * (K * R) + ranks[None, :]
    z_mask = valid[:, None] & (ranks[None, :] < R)
    shrink_results = ()
    coefficients = ()
    adapters = ()
    for route in tl.static_range(K):
        adapter = tl.load(Plans + plan * K + route).to(tl.int64)
        coeff = tl.load(W + tokens * K + route).to(tl.float32)
        coeff = tl.where((adapter >= 0) & (adapter < E), coeff, 0.0)
        # .cg: sibling blocks wrote these rows through L2 atomics.
        z = tl.load(z_ptrs + route * R, mask=z_mask, other=0.0, cache_modifier=".cg")
        shrink_results += (z.to(B.dtype.element_ty),)  # type: ignore[assignment]
        coefficients += (coeff,)  # type: ignore[assignment]
        adapters += (adapter,)  # type: ignore[assignment]

    # Single-stage loop: SM90 register-A dot workaround (triton-lang/triton#9433).
    for output_tile in tl.range(worker, tl.cdiv(OUT, BN), WORKERS, num_stages=1):
        columns = output_tile * BN + tl.arange(0, BN)
        wrapped = columns % OUT
        if OUT % BN == 0:
            wrapped = tl.max_contiguous(tl.multiple_of(wrapped, BN), BN)
        total = tl.zeros((BM, BN), tl.float32)
        for route in tl.static_range(K):
            adapter = adapters[route]
            if (adapter >= 0) & (adapter < E):
                b = tl.load(
                    B
                    + (adapter * OUT + wrapped[None, :].to(tl.int64)) * R
                    + ranks[:, None],
                    mask=ranks[:, None] < R,
                    other=0.0,
                )
                total += tl.dot(shrink_results[route], b) * coefficients[route][:, None]
        y_ptrs = Y + tokens[:, None] * stride_y + offset_start + columns[None, :]
        mask = valid[:, None] & (columns[None, :] < OUT)
        base = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptrs, base + total, mask=mask)

    # The last block to leave the tile clears its scratch rows and counter.
    tl.debug_barrier()
    if tl.atomic_add(counter, 1, sem="acq_rel") == 2 * WORKERS - 1:
        for route in tl.static_range(K):
            tl.store(z_ptrs + route * R, tl.zeros((BM, BR), tl.float32), mask=z_mask)
        tl.atomic_xchg(counter, 0, sem="relaxed")


@triton.jit(do_not_specialize=["num_plans"])
def _routed_lora_kernel(
    X,
    A,
    B,
    Y,
    W,
    Plans,
    Sorted,
    Starts,
    Z,
    Counters,
    num_plans,
    offset_start,
    stride_x,
    stride_y,
    H: tl.constexpr,
    OUT: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BH: tl.constexpr,
    BR: tl.constexpr,
    WORKERS: tl.constexpr,
    MAX_SPINS: tl.constexpr,
    PLAN_BLOCK: tl.constexpr,
):
    # Siblings share pid // WORKERS and visit identical tiles in order.
    pid = tl.program_id(0)
    worker = pid % WORKERS
    team = pid // WORKERS
    teams = tl.num_programs(0) // WORKERS
    _, plan_tiles = _plan_tiles(Starts, num_plans, BM, PLAN_BLOCK)
    total = tl.sum(plan_tiles, 0)
    # Arrive at the next tile before waiting on this one, so time otherwise
    # spent waiting for slower siblings goes to the next shrink.
    if team < total:
        _shrink(
            X,
            A,
            Plans,
            Sorted,
            Starts,
            Z,
            Counters,
            num_plans,
            team,
            worker,
            stride_x,
            H,
            R,
            K,
            E,
            BM,
            BH,
            BR,
            WORKERS,
            PLAN_BLOCK,
        )
    for item in tl.range(team, total, teams, num_stages=1):
        if item + teams < total:
            _shrink(
                X,
                A,
                Plans,
                Sorted,
                Starts,
                Z,
                Counters,
                num_plans,
                item + teams,
                worker,
                stride_x,
                H,
                R,
                K,
                E,
                BM,
                BH,
                BR,
                WORKERS,
                PLAN_BLOCK,
            )
        _finish(
            B,
            Y,
            W,
            Plans,
            Sorted,
            Starts,
            Z,
            Counters,
            num_plans,
            item,
            worker,
            offset_start,
            stride_y,
            OUT,
            R,
            K,
            E,
            BM,
            BN,
            BR,
            WORKERS,
            MAX_SPINS,
            PLAN_BLOCK,
        )


def _get_config(num_tokens: int, rank: int) -> dict[str, int]:
    """Tuned on H100 BF16 (H=O=4096, ranks 32/128, top-k 1/2/4)."""
    if num_tokens <= 128:
        block_m, block_n, block_h, workers, warps, stages = 16, 128, 128, 32, 4, 1
    elif num_tokens <= 1024:
        block_m, block_n, block_h, workers, warps, stages = 16, 128, 256, 8, 4, 3
    else:
        block_m, block_n, block_h, workers, stages = 32, 256, 256, 4, 3
        warps = 8 if rank > 32 else 4
    return dict(
        block_m=block_m,
        block_n=block_n,
        block_h=block_h,
        workers=workers,
        num_warps=warps,
        num_stages=stages,
    )


@functools.cache
def _libcuda() -> ctypes.CDLL:
    return ctypes.CDLL("libcuda.so.1")


@functools.cache
def is_routed_lora_supported(device: torch.device) -> tuple[bool, str | None]:
    """Check hardware support without compiling or launching a kernel."""
    if device.type != "cuda" or not current_platform.is_cuda() or not HAS_TRITON:
        return False, "requires NVIDIA CUDA and Triton"
    supported = ctypes.c_int()
    device_index = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    result = _libcuda().cuDeviceGetAttribute(
        ctypes.byref(supported),
        ctypes.c_int(_CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH),
        ctypes.c_int(device_index),
    )
    if result != 0:
        raise RuntimeError(f"cuDeviceGetAttribute failed: {result}")
    if not supported.value:
        return False, "device does not support cooperative launches"
    return True, None


def _resident_blocks(compiled, threads: int, device: torch.device) -> int:
    compiled._init_handles()
    blocks = ctypes.c_int()
    result = _libcuda().cuOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(blocks),
        ctypes.c_void_p(compiled.function),
        ctypes.c_int(threads),
        ctypes.c_size_t(compiled.metadata.shared),
    )
    if result != 0:
        raise RuntimeError(
            f"cuOccupancyMaxActiveBlocksPerMultiprocessor failed: {result}"
        )
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return blocks.value * sms


# Per (device, stream) zero-initialized scratch and tile counters. The kernel
# returns both to zero, so they are reused across calls on the same stream.
_workspaces: dict[tuple, dict[str, torch.Tensor]] = {}
_resident_capacity: dict[tuple, tuple[int, str | None]] = {}


class _RoutedLoRALaunch(NamedTuple):
    args: tuple
    options: dict
    grid: int


def _workspace(
    device: torch.device,
    scratch_numel: int,
    counter_numel: int,
    use_static_workspace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_static_workspace:
        from vllm.v1.worker.workspace import (
            current_workspace_manager,
            is_workspace_manager_initialized,
        )

        if is_workspace_manager_initialized():
            manager = current_workspace_manager()
            key = ("routed_lora", current_stream())
            return (
                manager.get_persistent(
                    (*key, "scratch", scratch_numel),
                    (scratch_numel,),
                    torch.float32,
                    zero_init=True,
                ),
                manager.get_persistent(
                    (*key, "counters", counter_numel),
                    (counter_numel,),
                    torch.int32,
                    zero_init=True,
                ),
            )
    key = (device, torch.cuda.current_stream(device).stream_id)
    workspace = _workspaces.setdefault(key, {})
    scratch = workspace.get("scratch")
    if scratch is None or scratch.numel() < scratch_numel:
        workspace["scratch"] = scratch = torch.zeros(
            scratch_numel, dtype=torch.float32, device=device
        )
    counters = workspace.get("counters")
    if counters is None or counters.numel() < counter_numel:
        workspace["counters"] = counters = torch.zeros(
            counter_numel, dtype=torch.int32, device=device
        )
    return scratch, counters


def _prepare_routed_lora(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    route_weights: torch.Tensor,
    plan_lora_indices: torch.Tensor,
    plan_token_indices: torch.Tensor,
    plan_token_start_loc: torch.Tensor,
    offset_start: int = 0,
    use_static_workspace: bool = False,
) -> tuple[_RoutedLoRALaunch | None, str | None]:
    """Prepare a routed-LoRA launch without modifying the output.

    Args:
        inputs: Input activations [T, H], in FP16 or BF16.
        lora_a_weights: Resident adapter weights [E, R, H] or [E, 1, R, H].
        lora_b_weights: Weights [E, O, R] or [E, 1, O, R], with the adapter's
            own alpha/r scaling already applied, as in ordinary vLLM LoRA.
        output_tensor: Base output [T, D], updated in place at the slice
            [offset_start:offset_start + O].
        route_weights: FP32 mixture coefficients [T, K].
        plan_lora_indices: Resident slot per plan and route [P, K]; -1 when
            the route is absent.
        plan_token_indices: Token indices grouped by plan [N].
        plan_token_start_loc: Offsets into plan_token_indices [P + 1].
        offset_start: First output column of this linear slice.
        use_static_workspace: Reserve fixed-capacity scratch for engine graphs.

    Plan metadata comes from vllm.lora.punica_wrapper.utils.make_route_plans.
    Every token appears in at most one plan. Inputs, weights and output share
    a device and dtype; adapter weights, coefficients and plan metadata are
    contiguous. Rank must not exceed MAX_ROUTED_LORA_RANK.

    """
    supported, reason = is_routed_lora_supported(inputs.device)
    if not supported:
        return None, reason

    if lora_a_weights.ndim == 4:
        assert lora_a_weights.size(1) == 1
        lora_a_weights = lora_a_weights.squeeze(1)
    if lora_b_weights.ndim == 4:
        assert lora_b_weights.size(1) == 1
        lora_b_weights = lora_b_weights.squeeze(1)

    num_tokens, hidden_size = inputs.shape
    num_loras, rank, _ = lora_a_weights.shape
    output_size = lora_b_weights.size(1)
    num_plans, num_routes = plan_lora_indices.shape
    if inputs.dtype not in (torch.float16, torch.bfloat16):
        return None, "requires FP16 or BF16 inputs"
    if (
        inputs.dtype == torch.bfloat16
        and torch.cuda.get_device_capability(inputs.device)[0] < 8
    ):
        return None, "BF16 requires compute capability 8.0 or newer"
    if not (
        inputs.dtype
        == lora_a_weights.dtype
        == lora_b_weights.dtype
        == output_tensor.dtype
    ):
        return None, "inputs, output and adapter weights must have matching dtypes"
    if rank > MAX_ROUTED_LORA_RANK:
        return None, f"rank exceeds {MAX_ROUTED_LORA_RANK}"
    if inputs.stride(1) != 1 or output_tensor.stride(1) != 1:
        return None, "requires unit input and output column strides"
    if not lora_a_weights.is_contiguous() or not lora_b_weights.is_contiguous():
        return None, "requires contiguous adapter weights"
    assert lora_a_weights.shape == (num_loras, rank, hidden_size)
    assert lora_b_weights.shape == (num_loras, output_size, rank)
    assert route_weights.size(0) >= num_tokens and route_weights.size(1) == num_routes
    assert route_weights.dtype == torch.float32
    assert plan_token_start_loc.ndim == 1
    assert plan_token_start_loc.numel() >= num_plans + 1
    assert output_tensor.size(0) == num_tokens
    assert 0 <= offset_start <= output_tensor.size(1) - output_size
    assert output_tensor.device == inputs.device
    for tensor in (
        lora_a_weights,
        lora_b_weights,
        route_weights,
        plan_lora_indices,
        plan_token_indices,
        plan_token_start_loc,
    ):
        assert tensor.device == inputs.device and tensor.is_contiguous()
    if num_tokens == 0 or num_plans == 0:
        return _RoutedLoRALaunch((), {}, 0), None

    config = _get_config(num_tokens, rank)
    block_m, workers = config["block_m"], config["workers"]
    # A plan contributes at most one partial tile beyond its full tiles.
    max_tiles = triton.cdiv(num_tokens, block_m) + num_plans
    scratch_tokens = route_weights.size(0) if use_static_workspace else num_tokens
    # Capture sizes trim plans; reserve counters for the full metadata capacity.
    counter_numel = (
        triton.cdiv(scratch_tokens, 16) + plan_token_start_loc.numel() - 1
        if use_static_workspace
        else max_tiles
    )
    scratch, counters = _workspace(
        inputs.device,
        scratch_tokens * num_routes * rank,
        counter_numel,
        use_static_workspace,
    )
    args = (
        inputs,
        lora_a_weights,
        lora_b_weights,
        output_tensor,
        route_weights,
        plan_lora_indices,
        plan_token_indices,
        plan_token_start_loc,
        scratch,
        counters,
        num_plans,
        offset_start,
        inputs.stride(0),
        output_tensor.stride(0),
        hidden_size,
        output_size,
        rank,
        num_routes,
        num_loras,
        block_m,
        config["block_n"],
        config["block_h"],
        max(16, triton.next_power_of_2(rank)),
        workers,
        _MAX_SPINS,
        triton.next_power_of_2(num_plans),
    )
    options = dict(
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        launch_cooperative_grid=True,
    )
    from triton.runtime.errors import OutOfResources

    try:
        compiled = _routed_lora_kernel.warmup(*args, grid=(workers,), **options)
    except OutOfResources as error:
        return None, str(error)
    # Pointer alignment and scalar specialization can change resource usage.
    capacity_key = (inputs.device, compiled.hash)
    cached = _resident_capacity.get(capacity_key)
    if cached is None:
        try:
            capacity = _resident_blocks(
                compiled, 32 * config["num_warps"], inputs.device
            )
        except OutOfResources as error:
            capacity, reason = 0, str(error)
        else:
            reason = (
                f"needs {workers} co-resident blocks; device holds {capacity}"
                if capacity < workers
                else None
            )
        cached = _resident_capacity[capacity_key] = (capacity, reason)
    capacity, reason = cached
    if reason is not None:
        return None, reason
    grid = min(capacity // workers, max_tiles) * workers
    return _RoutedLoRALaunch(args, options, grid), None


def can_implement_routed_lora(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    route_weights: torch.Tensor,
    plan_lora_indices: torch.Tensor,
    plan_token_indices: torch.Tensor,
    plan_token_start_loc: torch.Tensor,
    offset_start: int = 0,
) -> tuple[bool, str | None]:
    """Check and prepare a configuration before any output slice is updated."""
    launch, reason = _prepare_routed_lora(
        inputs,
        lora_a_weights,
        lora_b_weights,
        output_tensor,
        route_weights,
        plan_lora_indices,
        plan_token_indices,
        plan_token_start_loc,
        offset_start,
    )
    return launch is not None, reason


def _routed_lora(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    route_weights: torch.Tensor,
    plan_lora_indices: torch.Tensor,
    plan_token_indices: torch.Tensor,
    plan_token_start_loc: torch.Tensor,
    offset_start: int = 0,
    use_static_workspace: bool = False,
) -> None:
    """Add the routed update; dispatchers must check configuration support first."""
    launch, reason = _prepare_routed_lora(
        inputs,
        lora_a_weights,
        lora_b_weights,
        output_tensor,
        route_weights,
        plan_lora_indices,
        plan_token_indices,
        plan_token_start_loc,
        offset_start,
        use_static_workspace,
    )
    if launch is None:
        raise ValueError(
            f"Unsupported fused routed-LoRA configuration: {reason}. "
            "Use enforce_eager=True for the reference implementation."
        )
    if launch.grid:
        _routed_lora_kernel[(launch.grid,)](*launch.args, **launch.options)


def _routed_lora_fake(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    route_weights: torch.Tensor,
    plan_lora_indices: torch.Tensor,
    plan_token_indices: torch.Tensor,
    plan_token_start_loc: torch.Tensor,
    offset_start: int = 0,
    use_static_workspace: bool = False,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="routed_lora",
        op_func=_routed_lora,
        mutates_args=["output_tensor"],
        fake_impl=_routed_lora_fake,
    )
    routed_lora = torch.ops.vllm.routed_lora
except AttributeError:
    routed_lora = _routed_lora
