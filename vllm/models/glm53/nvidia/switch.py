# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The switch of GLM-5.3's low-latency decode path, read in one place."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Final

import torch

from vllm.logger import init_logger
from vllm.models.glm53.nvidia.ops.cute_dsl.row_table import (
    PACKED_SLOTS,
    segments_per_request,
)
from vllm.models.glm53.nvidia.ops.launch import TIERS, WIDE_SELECTION_TIERS

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

GROUP_ROWS = 4
W8A16_ROWS = (4, 8)
#: Most requests and rows of a step the segment runs: packed tier 16's capacity.
MAX_REQUESTS = max(PACKED_SLOTS.values())
MAX_ROWS = 16
#: Most tokens of one request (MTP K = 7): the 8-row build holds one request.
MAX_PER_REQUEST = 8


@dataclass(frozen=True)
class LowLatencyDecode:
    """The resolved switch of the low-latency decode path, and the deployment its
    kernels are built for.
    """

    DECODE_BACKEND: ClassVar[str] = "latency_specialized"
    WORLD: ClassVar[int] = 8
    KV_CACHE_DTYPES: ClassVar[tuple[str, ...]] = ("fp8", "fp8_e4m3")
    ATTENTION_BACKEND: ClassVar[str] = "FLASHINFER_MLA_SPARSE"

    enabled: bool = False
    row_tiers: tuple[int, ...] = ()
    #: Tokens of one request in a decode step (``1 + K``), which the row tiers follow.
    per_request: int = GROUP_ROWS


def from_env() -> LowLatencyDecode:
    """Return the record with the path off."""
    return LowLatencyDecode()


def requested(vllm_config: "VllmConfig") -> bool:
    """Whether ``decode_backend`` asks for the path; refuses an unknown value."""
    additional = getattr(vllm_config, "additional_config", None)
    backend = (
        additional.get("decode_backend", "auto")
        if isinstance(additional, dict)
        else "auto"
    )
    if backend not in ("auto", LowLatencyDecode.DECODE_BACKEND):
        raise ValueError(
            "GLM-5.3 decode_backend must be 'auto' or "
            f"{LowLatencyDecode.DECODE_BACKEND!r}, "
            f"not {backend!r}."
        )
    return backend == LowLatencyDecode.DECODE_BACKEND


def from_config(vllm_config: "VllmConfig") -> LowLatencyDecode:
    """Read the switch: on for ``decode_backend=latency_specialized``."""
    if not requested(vllm_config):
        return LowLatencyDecode()
    return LowLatencyDecode(
        enabled=True,
        row_tiers=_row_tiers(vllm_config),
        per_request=_per_request(vllm_config),
    )


DRAFT_DECODE_SIZES: tuple[int, ...] = (1, 2, 4, 8)


def draft_capture_sizes(
    seqs: int, mtp: bool, drafts: int = GROUP_ROWS - 1
) -> tuple[int, ...]:
    """Return the capture sizes the drafter's request counts add to the row tiers."""
    per_request = 1 + drafts
    if not mtp or drafts < 2 or (seqs == 1 and per_request <= GROUP_ROWS):
        # One request of at most four tokens keeps vLLM's default sizes (no row tiers).
        return ()
    found = graphs(seqs, per_request)
    claims = {graph.tokens for graph in found}
    sizes: set[int] = set()
    for size in (*DRAFT_DECODE_SIZES, seqs):
        rounded = -(-size // per_request) * per_request
        splits = any(g.requests[0] * per_request <= size < g.tokens for g in found)
        if size <= seqs and not splits and (rounded in claims or rounded > max(claims)):
            sizes.add(size)
    return tuple(sorted(sizes))


def lone_rows(per_request: int) -> int:
    """Return the kernel rows a lone request of ``per_request`` = ``1 + K`` tokens runs
    on an aligned build: the tokens rounded up to whole row groups.
    """
    return -(-per_request // GROUP_ROWS) * GROUP_ROWS


def packs(per_request: int) -> bool:
    """Whether steps of ``per_request`` tokens per request can run on packed builds:
    their rows are not whole row groups, so two requests may share a group.
    """
    return per_request % GROUP_ROWS != 0


class ShapePath(Enum):
    """The one way the segment serves a step of ``B`` requests of ``1 + K`` tokens."""

    #: One request of at most four tokens on the q4 build (W8A16 MoE).
    Q4 = "q4"
    LONE8 = "lone8"
    ALIGNED = "aligned"
    PACK = "pack"
    STOCK = "stock"


def shape_path(requests: int, per_request: int) -> ShapePath:
    """Return the path of a step of ``requests`` requests of ``per_request`` = ``1 + K``
    tokens: the single claim every other predicate of this module derives from
    (:func:`serves_shape`, :func:`shape_tier`, :func:`derived_tiers`, :func:`graphs`,
    :func:`tier_claims`).
    """
    rows = requests * per_request
    too_many = requests > MAX_REQUESTS or rows > MAX_ROWS
    if too_many or per_request > MAX_PER_REQUEST:
        return ShapePath.STOCK
    if requests == 1:
        return ShapePath.Q4 if per_request <= GROUP_ROWS else ShapePath.LONE8
    if not packs(per_request):
        return ShapePath.ALIGNED
    return ShapePath.PACK


def serves_shape(requests: int, per_request: int) -> bool:
    """Whether the segment runs a step of ``requests`` requests of ``per_request``
    tokens: every shape :func:`shape_path` does not leave to vLLM's stock forward.
    """
    return shape_path(requests, per_request) is not ShapePath.STOCK


def served_requests(seqs: int, per_request: int) -> int:
    """Return the most requests of ``per_request`` tokens, up to ``seqs``, whose steps
    the segment runs (0: none); larger steps run vLLM's stock forward.
    """
    counts = range(min(seqs, MAX_REQUESTS), 0, -1)
    return next((b for b in counts if serves_shape(b, per_request)), 0)


def shape_w8a16_rows(requests: int, per_request: int) -> int:
    """Return the row capacity of the W8A16 MoE build (:data:`W8A16_ROWS`) a packed step
    runs on rows 0..capacity-1 of the hidden buffer, the smallest that holds its live
    rows; 0 when the step is not packed or has more rows (the tier's W8A8 MoE).
    """
    if shape_path(requests, per_request) is not ShapePath.PACK:
        return 0
    rows = requests * per_request
    return next((capacity for capacity in W8A16_ROWS if capacity >= rows), 0)


def shape_tier(requests: int, per_request: int) -> int:
    """Return the row tier whose build runs a step of ``requests`` requests."""
    path = shape_path(requests, per_request)
    if path is ShapePath.STOCK:
        raise ValueError(
            f"{requests} requests of {per_request} tokens run vLLM's stock forward"
        )
    rows = lone_rows(per_request) if requests == 1 else requests * per_request
    if path is ShapePath.PACK:
        # The packed tier that holds the rows and a slot per segment.
        slots = requests * segments_per_request(per_request)
        return min(t for t, n in PACKED_SLOTS.items() if t >= rows and n >= slots)
    return min(tier for tier in TIERS if tier >= rows)


def packed_tier(tier: int, per_request: int) -> bool:
    """Whether the build of ``tier`` that runs steps of ``per_request`` tokens is the
    packed one: the tiers with packed builds, for a count that is not whole groups,
    when the tier holds more than a lone request's aligned rows (tier 8 at K = 4-6 is
    the lone request's aligned build, :data:`ShapePath.LONE8`).
    """
    return packs(per_request) and tier in PACKED_SLOTS and tier > lone_rows(per_request)


@dataclass(frozen=True)
class Build:
    """One built segment: the row ``tier`` whose kernels it launches and whether its MoE
    is a W8A16 build's (:attr:`Glm53Geometry.w8a16_rows`; 0: the tier's own).
    """

    tier: int
    w8a16_rows: int = 0


@dataclass(frozen=True)
class Graph:
    """One CUDA graph vLLM captures for the segment: its ``tokens``, the ``build`` whose
    kernels it launches and the request counts that replay it.
    """

    tokens: int
    build: Build
    requests: range

    @property
    def tier(self) -> int:
        return self.build.tier


def shape_build(requests: int, per_request: int) -> Build:
    """Return the build that runs a step of ``requests`` requests of ``per_request``
    tokens.
    """
    return Build(
        shape_tier(requests, per_request), shape_w8a16_rows(requests, per_request)
    )


def graphs(seqs: int, per_request: int) -> list[Graph]:
    """Return the graphs the segment claims in a deployment of up to ``seqs`` requests:
    one per run of request counts that share a build, over the served counts
    (:func:`served_requests`; larger steps are vLLM's own graphs).
    """
    runs: list[Graph] = []
    for requests in range(1, served_requests(seqs, per_request) + 1):
        build = shape_build(requests, per_request)
        if runs and runs[-1].build == build:
            counts = range(runs[-1].requests.start, requests + 1)
            runs[-1] = Graph(requests * per_request, build, counts)
        else:
            runs.append(
                Graph(requests * per_request, build, range(requests, requests + 1))
            )
    return runs


def tier_claims(seqs: int, per_request: int) -> dict[int, Build]:
    """Return the token counts of the graphs the segment claims, each mapped to its
    build.
    """
    return {graph.tokens: graph.build for graph in graphs(seqs, per_request)}


def stock_capture_sizes(
    seqs: int, per_request: int, sizes: Sequence[int]
) -> tuple[int, ...]:
    """Return the capture sizes of the steps vLLM's stock forward runs: those of
    ``sizes`` (vLLM's list, or the one the user passed) above the segment's largest
    graph and up to ``seqs`` requests of ``per_request`` tokens, and that most, so a
    step the segment does not claim replays a stock CUDA graph instead of running
    eagerly.
    """
    top, most = graphs(seqs, per_request)[-1].tokens, seqs * per_request
    stock = {size for size in sizes if top < size <= most}
    return tuple(sorted(stock | ({most} if most > top else set())))


def describe_shapes(seqs: int, per_request: int) -> str:
    """Return the startup line that lists the request counts each build serves."""
    parts = []
    for graph in graphs(seqs, per_request):
        counts = list(graph.requests)
        path = shape_path(counts[0], per_request)
        kind = {
            ShapePath.Q4: "the q4 build",
            ShapePath.LONE8: "the 8-row build",
            ShapePath.ALIGNED: f"whole row groups on tier {graph.tier}",
            ShapePath.PACK: f"packed rows on tier {graph.tier}"
            + (
                f" (q{graph.build.w8a16_rows} MoE, at most "
                f"{graph.build.w8a16_rows} live rows)"
                if graph.build.w8a16_rows
                else ""
            ),
        }[path]
        parts.append(f"B = {counts}: {kind}")
    served = served_requests(seqs, per_request)
    if served < seqs:
        stock = list(range(served + 1, seqs + 1))
        parts.append(
            f"B = {stock}: vLLM's stock forward (R > {MAX_ROWS} or B > {MAX_REQUESTS})"
        )
    return f"MTP K={per_request - 1}: " + "; ".join(parts)


def derived_tiers(seqs: int, per_request: int) -> tuple[int, ...]:
    """Return the row tiers that serve up to ``seqs`` requests of ``per_request``
    tokens: the tier of every served request count (:func:`served_requests`; a batch
    of fewer requests is padded to the next graph up, not to the largest).
    """
    if per_request > MAX_PER_REQUEST:
        raise ValueError(
            f"GLM-5.3 latency-specialized decode serves MTP up to K = "
            f"{MAX_PER_REQUEST - 1}: 1 + K = {per_request} tokens per request"
        )
    return tuple(sorted({g.tier for g in graphs(seqs, per_request)}))


def roles(tiers: tuple[int, ...]) -> tuple[bool, bool]:
    """Return ``(narrow, wide)``: whether the deployment's row tiers (``()``: the one-
    request build) include a build whose index selection runs on ranks 0-2 with push
    roles, and one that selects on every rank.
    """
    if not tiers:
        return True, False
    selects = [tier in WIDE_SELECTION_TIERS for tier in tiers]
    return not all(selects), any(selects)


def _per_request(vllm_config: "VllmConfig") -> int:
    speculative = vllm_config.speculative_config
    return 1 + (0 if speculative is None else speculative.num_speculative_tokens)


def _row_tiers(vllm_config: "VllmConfig") -> tuple[int, ...]:
    seqs = vllm_config.scheduler_config.max_num_seqs
    per_request = _per_request(vllm_config)
    if seqs == 1 and per_request <= GROUP_ROWS:
        return ()
    wanted = derived_tiers(seqs, per_request)
    sizes = tuple(sorted(vllm_config.compilation_config.cudagraph_capture_sizes or ()))
    if not sizes or 1 in sizes:
        return wanted
    found = graphs(seqs, per_request)
    by_size = {graph.tokens: graph.tier for graph in found}
    top = found[-1].tokens
    ours = [size for size in sizes if size <= top]
    if set(ours) <= set(by_size) and top in ours:
        return tuple(sorted({by_size[size] for size in ours}))
    flag = " ".join(str(graph.tokens) for graph in found)
    raise ValueError(
        f"GLM-5.3 latency-specialized decode with --max-num-seqs {seqs} x "
        f"{per_request} tokens needs the CUDA graph capture sizes of its graphs: "
        f"{sorted(by_size)} (a graph holds the tokens of its run of request counts), "
        f"the largest holding {top}, not {list(sizes)}: pass "
        f"--cudagraph-capture-sizes {flag} (or leave the flag out; larger sizes are "
        f"vLLM's stock graphs)"
    )


def _add_capture_size(vllm_config: "VllmConfig", size: int) -> None:
    compilation = vllm_config.compilation_config
    sizes = list(compilation.cudagraph_capture_sizes or ())
    if sizes and size not in sizes:
        compilation.cudagraph_capture_sizes = sorted([*sizes, size])
        compilation.max_cudagraph_capture_size = max(sizes[-1], size)


def _set_capture_sizes(vllm_config: "VllmConfig", tiers: tuple[int, ...]) -> None:
    compilation = vllm_config.compilation_config
    if tiers and tuple(sorted(compilation.cudagraph_capture_sizes or ())) != tiers:
        logger.info_once(
            "GLM-5.3 latency-specialized decode: --cudagraph-capture-sizes %s "
            "(row tiers for --max-num-seqs %d)",
            " ".join(str(tier) for tier in tiers),
            vllm_config.scheduler_config.max_num_seqs,
        )
        compilation.cudagraph_capture_sizes = list(tiers)
        compilation.max_cudagraph_capture_size = tiers[-1]


_resolved: LowLatencyDecode | None = None

DECODE_MLA_AUTOTUNE_OP = "trtllm_batch_decode_mla"


def _skip_decode_mla_autotune() -> None:
    name = "VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS"
    ops = [op.strip() for op in os.environ.get(name, "").split(",") if op.strip()]
    if DECODE_MLA_AUTOTUNE_OP not in ops:
        os.environ[name] = ",".join([*ops, DECODE_MLA_AUTOTUNE_OP])
        logger.info_once(
            "GLM-5.3 latency-specialized decode: %s=%s", name, os.environ[name]
        )


MOE_BACKEND: Final = "flashinfer_trtllm"


def _select_moe_backend(vllm_config: "VllmConfig") -> None:
    kernel = vllm_config.kernel_config
    if kernel.moe_backend == "auto":
        kernel.moe_backend = MOE_BACKEND
        logger.info_once(
            "GLM-5.3 latency-specialized decode: --moe-backend %s", MOE_BACKEND
        )


def current() -> LowLatencyDecode:
    """Return the record ``resolve`` published, or the path off when none was."""
    return _resolved if _resolved is not None else from_env()


def _batch_requirement(
    vllm_config: "VllmConfig", tiers: tuple[int, ...]
) -> dict[str, bool]:
    if tiers:
        return {}
    return {
        "--max-num-seqs 1 (the segment serves one request)": (
            vllm_config.scheduler_config.max_num_seqs == 1
        )
    }


def _unsupported(vllm_config: "VllmConfig", tiers: tuple[int, ...] = ()) -> list[str]:
    import vllm.envs as envs
    from vllm.config import CUDAGraphMode
    from vllm.platforms import current_platform

    parallel = vllm_config.parallel_config
    speculative = vllm_config.speculative_config
    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    cache_dtype = vllm_config.cache_config.cache_dtype
    attention = getattr(vllm_config, "attention_config", None)
    backend = getattr(getattr(attention, "backend", None), "name", None)
    world, attention_backend = (
        LowLatencyDecode.WORLD,
        LowLatencyDecode.ATTENTION_BACKEND,
    )
    moe_backend = vllm_config.kernel_config.moe_backend
    requirements = {
        "an SM100 or SM103 GPU": current_platform.is_cuda()
        and current_platform.is_device_capability_family(100),
        f"tensor parallel size {world} without pipeline parallelism": (
            parallel.tensor_parallel_size == world
            and parallel.pipeline_parallel_size == 1
        ),
        f"--kv-cache-dtype fp8 with --attention-backend {attention_backend} "
        f"(not --kv-cache-dtype {cache_dtype} with {backend})": (
            cache_dtype in LowLatencyDecode.KV_CACHE_DTYPES
            and backend == attention_backend
        ),
        "Model Runner V2 (its model-state hooks complete the index keys)": bool(
            vllm_config.use_v2_model_runner
        ),
        **_batch_requirement(vllm_config, tiers),
        "no expert parallelism": not parallel.enable_expert_parallel,
        "no data, prefill or decode context parallelism, or micro-batching": (
            parallel.data_parallel_size == 1
            and parallel.decode_context_parallel_size == 1
            and parallel.prefill_context_parallel_size == 1
            and not parallel.use_ubatching
        ),
        "FULL CUDA graphs for decode batches only (FULL_AND_PIECEWISE or "
        "FULL_DECODE_ONLY), without --enforce-eager": (
            not vllm_config.model_config.enforce_eager
            and cudagraph_mode is not None
            and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.mixed_mode() != CUDAGraphMode.FULL
        ),
        # The segment binds weight addresses.
        "no sleep mode": not vllm_config.model_config.enable_sleep_mode,
        "MTP speculative decoding, if any, with a static draft count and "
        "without adaptive verification": (
            speculative is None
            or (
                speculative.method == "mtp"
                and not speculative.uses_dynamic_speculative_decoding()
                and not speculative.enable_adaptive_verification
            )
        ),
        "no LoRA": vllm_config.lora_config is None,
        "no KV transfer connector": vllm_config.kv_transfer_config is None,
        "no batch-invariant mode (VLLM_BATCH_INVARIANT)": not envs.VLLM_BATCH_INVARIANT,
        f"--moe-backend {MOE_BACKEND} or auto (not {moe_backend})": (
            moe_backend in ("auto", MOE_BACKEND)
        ),
    }
    return [need for need, ok in requirements.items() if not ok]


def resolve(vllm_config: "VllmConfig") -> LowLatencyDecode:
    """Read the switch once for this process and refuse an unsupported deployment."""
    global _resolved
    record = from_config(vllm_config)
    if record.enabled:
        import torch.distributed as dist

        from vllm.distributed import get_tp_group

        missing = _unsupported(vllm_config, record.row_tiers)
        supported = torch.tensor([0 if missing else 1], dtype=torch.int32, device="cpu")
        dist.all_reduce(supported, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
        if not supported.item():
            reason = (
                "; ".join(missing) or "every other tensor parallel rank to support it"
            )
            raise ValueError(f"GLM-5.3 latency-specialized decode needs {reason}.")
        logger.info("GLM-5.3 latency-specialized decode on")
        _skip_decode_mla_autotune()
        _select_moe_backend(vllm_config)
        speculative = vllm_config.speculative_config
        per_request = _per_request(vllm_config)
        seqs = vllm_config.scheduler_config.max_num_seqs
        extra = draft_capture_sizes(seqs, speculative is not None, per_request - 1)
        if record.row_tiers:
            sizes = {graph.tokens for graph in graphs(seqs, per_request)}
            stock = stock_capture_sizes(
                seqs,
                per_request,
                vllm_config.compilation_config.cudagraph_capture_sizes or (),
            )
            _set_capture_sizes(vllm_config, tuple(sorted({*sizes, *extra, *stock})))
        else:
            _add_capture_size(vllm_config, per_request)
        logger.info_once(
            "GLM-5.3 latency-specialized decode: %s", describe_shapes(seqs, per_request)
        )
    _resolved = record
    return record


__all__ = [
    "LowLatencyDecode",
    "current",
    "from_config",
    "from_env",
    "requested",
    "resolve",
]
