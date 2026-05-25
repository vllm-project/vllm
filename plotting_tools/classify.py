"""Event classification for Chrome-trace JSON exported from Nsight."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

Kind = Literal["compute", "comm", "control"]
Subcategory = Literal[
    "attention_comp",
    "gate_comp",
    "add_norm_comp",
    "matmul_gemm",
    "moe_routing",
    "moe_expert",
    "kv_cache_write",
    "rotary_embedding",
    "sampling_overhead",
    "masking_indexing",
    "other_compute",
    "network_collective",
    "network_p2p",
    "device_copy",
    "host_transfer",
    "control",
]

# Legacy alias used in a few plot paths (maps to network_collective in timelines).
LEGACY_COLLECTIVE_COMM = "network_collective"

COMPUTE_PATTERNS = (
    "vectorized_elementwise_kernel",
    "unrolled_elementwise_kernel",
    "elementwise_kernel",
    "distribution_elementwise_grid_stride_kernel",
    "distributionnormal_cu",
    "normal_and_transform",
    "computeblockwisewithinkcounts",
    "computeblockwisekthcounts",
    "fillfunctor",
    "direct_copy_kernel_cuda",
    "catarraybatchedcopy",
    "gpu_kernel_impl",
    "cunn_softmaxforward",
    "index_elementwise_kernel",
    "index_kernel_impl",
    "index_put_kernel_impl",
    "vllm::modified_torch_code::mbtopk::",
    "flashattnvarlenfunc",
    "_layer_norm_kernel",
    "reduce_fused_impl",
    "reduce_kernel",
    "devicescan",
    "devicescankernel",
    "scatter",
    "gather",
    "per_token_cast_to_fp8",
    "swiglu_forward",
    "compute_attn_ws",
    "apply_penalty_kernel",
    "flash::flash_fwd_splitkv_mla_kernel",
    "flash::flash_fwd_splitkv_mla_combine_kernel",
    "get_mla_metadata_kernel",
    "memcpy32_post kernel",
    "memcpy128 kernel",
    "vllm::mask_top_p_kernel",
    "vllm::revert_output_bin_count_kernel",
    "unbind_impl",
    "cudamemsetasync",
    "memset (device)",
    "gpu_memset",
    "flashinfer",
    "fmha",
    "paged_attention",
)

MATMUL_GEMM_PATTERNS = (
    "nvjet_sm",
    "cublas",
    "cutlass::device_kernel",
    "cutlass::kernel",
    "gemm",
    "wgmma",
    "mma.sync",
)

MOE_ROUTING_PATTERNS = (
    "expandinputrows",
    "finalizemoerouting",
    "computestridiestma",
    "get_dispatch_layout",
    "moe_align",
    "get_fused_mapping",
    "clean_and_count_expert",
)

KV_CACHE_PATTERNS = (
    "reshape_and_cache_flash",
    "reshape_and_cache",
    "cache_flash",
    "slot_mapping",
)

ROTARY_PATTERNS = (
    "rotary_embedding",
    "rotary_emb",
)

SAMPLING_PATTERNS = (
    "compare_scalar",
    "cunn_softmax",
    "softmax",
    "multinomial",
    "distribution_elementwise",
    "distributionnormal",
    "topk",
    "top_k",
    "mbtopk",
    "sort",
    "radix_sort",
    "revert_output_bin_count",
    "mask_top_p",
)

MASKING_INDEX_PATTERNS = (
    "masked_fill",
    "compare_scalar",
    "fillfunctor",
    "index_elementwise",
    "index_kernel_impl",
    "index_put",
    "scatter_gather",
    "_scatter_gather",
    "vectorized_gather",
    "indexselect",
)

MOE_EXPERT_PATTERNS = (
    "moefcgemm",
    "fused_moe",
    "fused_experts",
    "fusedmoe",
    "moe_gemm",
    "fusedmoe",
)

COMM_PATTERNS = (
    "nccl:",
    "nccl::",
    "nccldevkernel",
    "memcpy dtod",
    "gpu_memcpy",
    "dpsk::ep::internode::dispatch_ll",
    "dpsk::ep::internode::combine_ll",
    "notify_dispatch",
    "cached_notify",
    "send_tensor",
    "recv_tensor",
    "isend",
    "irecv",
)

CONTROL_PATTERNS = (
    "c10d::allreduce_",
    "record_param_comms",
    "aten::",
    "detach_",
    "cudaeventquery",
    "cudalaunchkernel",
    "culaunchkernel",
    "cudalaunchkernelexc",
    "cudastreamwaitevent",
    "cudastreamsynchronize",
    "cudastreamiscapturing",
    "cudastreamgetcaptureinfo_v2",
    "cudafuncgetattributes",
    "cudafuncsetattribute",
    "cudadevicegetattribute",
    "cudapointergetattributes",
    "cudaoccupancymaxactiveblockspermultiprocessorwithflags",
    "cudadevicesynchronize",
    "cudapeekatlasterror",
    "cudagetfuncbysymbol",
    "cudagraphlaunch",
    "cudadrivergetversion",
    "allgatherprefix",
    "reducescatterprefix",
    "pytorch profiler",
    "invalid cuda_runtime",
    "cudamemcpyasync",
    "memcpy htod",
    "memcpy dtoh",
)

ATTENTION_PATTERNS = (
    "flashattn",
    "flash_attn",
    "flashinfer",
    "fmha",
    "paged_attention",
    "compute_attn",
    "attention",
    "mla_kernel",
    "get_mla_metadata",
)

GATE_PATTERNS = (
    "topk",
    "fused_topk",
    "moe_align",
    "router",
    "gate",
    "mbtopk",
    "top2_sum_gate",
)

EXPERT_PATTERNS = (
    "fused_moe",
    "fused_experts",
    "fusedmoe",
    "expert",
    "silu_and_mul",
)

NORM_PATTERNS = (
    "layernorm",
    "layer_norm",
    "rms_norm",
    "rmsnorm",
    "_layer_norm",
    "add_norm",
)

_NETWORK_KERNEL_SIGNALS = (
    "nccldevkernel",
    "pnccl",
    "two_shot_all_reduce",
    "one_shot_all_reduce",
    "multimem_all_reduce",
    "allreduce",
    "all_reduce",
    "reducescatter",
    "reduce_scatter",
    "allgather",
    "all_gather",
    "alltoall",
    "all_to_all",
    "sendrecv",
    "broadcast",
)

# NCCL / PyTorch comm op labels for breakdown bar charts (order = first match wins).
_COMM_OP_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("all_to_all", (
        "alltoall",
        "all_to_all",
        "all2all",
        "all-to-all",
        "deepep::all_to_all",
        "deepep_a2a",
    )),
    ("reduce_scatter", (
        "reducescatter",
        "reduce_scatter",
        "reducescatterprefix",
        "nccldevkernel_reducescatter",
    )),
    ("all_gather", (
        "allgather",
        "all_gather",
        "allgatherprefix",
        "allgather_base",
        "nccldevkernel_allgather",
    )),
    ("all_reduce", (
        "allreduce",
        "all_reduce",
        "allreduceprefix",
        "two_shot_all_reduce",
        "one_shot_all_reduce",
        "multimem_all_reduce",
        "nccldevkernel_allreduce",
        "pncclallreduce",
    )),
    ("broadcast", ("broadcast", "nccldevkernel_broadcast")),
    ("point_to_point", (
        "sendrecv",
        "send/recv",
        "isend",
        "irecv",
        "send_tensor",
        "recv_tensor",
        "nccldevkernel_sendrecv",
    )),
    ("host_transfer", (
        "memcpy htod",
        "memcpy dtoh",
        "memcpyhtod",
        "memcpydtoh",
        "cudamemcpyasync",
        "cuda_memcpy_async",
    )),
    ("device_copy", (
        "memcpy dtod",
        "memcpydtod",
        "memcpy device",
        "memcpy default",
        "memcpy unknown",
    )),
    ("nccl_other", ("nccldevkernel", "pnccl", "ncclkernel")),
)

_COLLECTIVE_SIGNALS = (
    "nccl",
    "c10d",
    "collective",
    "allgather",
    "allreduce",
    "all_reduce",
    "reducescatter",
    "reduce_scatter",
    "alltoall",
    "all_to_all",
    "broadcast",
    "sendrecv",
    "two_shot_all_reduce",
    "one_shot_all_reduce",
)

_KERNEL_COLLECTIVE_SIGNALS = _NETWORK_KERNEL_SIGNALS


def _memcpy_is_host_transfer(name: str, *, args: dict[str, Any] | None) -> bool:
    name_l = name.lower().strip()
    if "htod" in name_l or "dtoh" in name_l:
        return True
    kind = int((args or {}).get("copy_kind", 0))
    return kind in (1, 2)


def _is_network_kernel_or_runtime(name: str, cat: str) -> bool:
    s = f"{name} {cat}".lower()
    cat_l = cat.lower()
    if cat_l == "kernel":
        return any(k in s for k in _NETWORK_KERNEL_SIGNALS)
    return any(k in s for k in ("nccldevkernel", "nccl:", "nccl::", "c10d::"))


def classify_comm_operation(
    name: str,
    cat: str = "",
    *,
    args: dict[str, Any] | None = None,
) -> str | None:
    """
    Map a comm-related event name to a collective / transfer bucket.

    Returns None if the event does not look like communication.
    Network collective kernels (custom all-reduce, NCCL) match when cat is kernel.
    """
    name_l = name.lower().strip()
    cat_l = cat.lower().strip()
    s = f"{name_l} {cat_l}"

    if cat_l == "memcpy" or name_l in (
        "memcpy",
        "memcpy unknown",
        "memcpy default",
        "memcpy device",
    ):
        if _memcpy_is_host_transfer(name, args=args):
            return "host_transfer"
        return "device_copy"

    signals = (
        _KERNEL_COLLECTIVE_SIGNALS
        if cat_l == "kernel"
        else _COLLECTIVE_SIGNALS
    )
    if not any(k in s for k in signals):
        return None

    for label, keys in _COMM_OP_RULES:
        if any(k in s for k in keys):
            return label
    return "unclassified_comm"


_COMM_OP_DISPLAY = {
    "all_reduce": "All-Reduce",
    "all_gather": "All-Gather",
    "reduce_scatter": "Reduce-Scatter",
    "all_to_all": "All-to-All",
    "broadcast": "Broadcast",
    "all_scatter": "Scatter",
    "point_to_point": "Point-to-Point",
    "host_transfer": "Host transfer (H↔D)",
    "device_copy": "Device memcpy (CUPTI)",
    "nccl_other": "NCCL (other)",
    "unclassified_comm": "Unclassified comm",
    "other_comm": "Other comm",
}


def comm_operation_label(op: str) -> str:
    return _COMM_OP_DISPLAY.get(op, op.replace("_", " ").title())


# Fabric = network comm events only (not device_copy / host_transfer).
NETWORK_SUBS = frozenset({
    "network_collective",
    "network_p2p",
})

# Op buckets for fabric breakdown bar charts (subset of fabric events).
FABRIC_COMM_OPS = frozenset({
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "broadcast",
    "point_to_point",
})

MOVEMENT_OPS = frozenset({
    "device_copy",
    "host_transfer",
})


def is_fabric_event(e: dict[str, Any]) -> bool:
    """True for network/fabric traffic (rank heatmaps, fabric timing CDFs)."""
    return e.get("kind") == "comm" and e.get("sub") in NETWORK_SUBS


def is_fabric_comm_op(op: str) -> bool:
    return op in FABRIC_COMM_OPS


def is_fabric_op(op: str) -> bool:
    """Alias for is_fabric_comm_op."""
    return is_fabric_comm_op(op)


def is_movement_op(op: str) -> bool:
    return op in MOVEMENT_OPS


# Backward-compatible alias
FABRIC_OPS = FABRIC_COMM_OPS


def classify_kind(
    name: str,
    cat: str,
    *,
    args: dict[str, Any] | None = None,
) -> Kind:
    s = f"{name} {cat}".lower()
    cat_l = cat.lower()

    if cat_l == "memcpy" or (
        name.lower().strip() == "memcpy" and "nccl" not in s
    ):
        if _memcpy_is_host_transfer(name, args=args):
            return "control"
        return "comm"

    if _is_network_kernel_or_runtime(name, cat):
        return "comm"

    if any(k in s for k in COMM_PATTERNS):
        return "comm"

    if "memcpy htod" in s or "memcpy dtoh" in s:
        return "control"

    if cat_l == "kernel":
        return "compute"
    if any(k in s for k in COMPUTE_PATTERNS):
        return "compute"
    if any(k in s for k in CONTROL_PATTERNS):
        return "control"
    return "control"


def classify_subcategory(
    name: str,
    cat: str,
    kind: Kind,
    *,
    args: dict[str, Any] | None = None,
) -> Subcategory:
    s = f"{name} {cat}".lower()

    if kind == "control":
        if "memcpy" in s or _memcpy_is_host_transfer(name, args=args):
            return "host_transfer"
        return "control"

    if kind == "comm":
        op = classify_comm_operation(name, cat, args=args)
        if op == "point_to_point":
            return "network_p2p"
        if op == "device_copy":
            return "device_copy"
        if op == "host_transfer":
            return "host_transfer"
        return "network_collective"

    if any(k in s for k in ATTENTION_PATTERNS):
        return "attention_comp"
    if any(k in s for k in GATE_PATTERNS):
        return "gate_comp"
    if any(k in s for k in MOE_EXPERT_PATTERNS) or any(
        k in s for k in EXPERT_PATTERNS
    ):
        return "moe_expert"
    if any(k in s for k in MOE_ROUTING_PATTERNS):
        return "moe_routing"
    if any(k in s for k in KV_CACHE_PATTERNS):
        return "kv_cache_write"
    if any(k in s for k in ROTARY_PATTERNS):
        return "rotary_embedding"
    if any(k in s for k in MATMUL_GEMM_PATTERNS):
        return "matmul_gemm"
    if any(k in s for k in SAMPLING_PATTERNS):
        return "sampling_overhead"
    if any(k in s for k in MASKING_INDEX_PATTERNS):
        return "masking_indexing"
    if any(k in s for k in NORM_PATTERNS):
        return "add_norm_comp"
    return "other_compute"


def classify_event(
    name: str,
    cat: str,
    unclassified: list[str] | None = None,
    *,
    args: dict[str, Any] | None = None,
) -> tuple[Kind, Subcategory]:
    kind = classify_kind(name, cat, args=args)
    sub = classify_subcategory(name, cat, kind, args=args)
    if unclassified is not None and kind == "control" and sub == "control":
        if not any(k in f"{name} {cat}".lower() for k in CONTROL_PATTERNS):
            if cat.lower() != "kernel":
                if not any(
                    k in f"{name} {cat}".lower()
                    for k in COMM_PATTERNS + COMPUTE_PATTERNS
                ):
                    unclassified.append(f"{name} {cat}".lower())
    return kind, sub


def summarize_unclassified(unclassified: list[str], limit: int = 50) -> None:
    counts = Counter(unclassified)
    print(f"Unclassified strings: {len(counts)}")
    for s, n in counts.most_common(limit):
        print(f"{n:>8}  {s}")
