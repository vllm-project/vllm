"""Event classification for Chrome-trace JSON exported from Nsight."""

from __future__ import annotations

from collections import Counter
from typing import Literal

Kind = Literal["compute", "comm", "control"]
Subcategory = Literal[
    "attention_comp",
    "gate_comp",
    "experts_comp",
    "add_norm_comp",
    "other_compute",
    "collective_comm",
    "control",
]

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
    "gemm",
    "cutlass::device_kernel",
    "cublas",
    "rotary_embedding",
    "top2_sum_gate",
    "reduce_fused_impl",
    "reduce_kernel",
    "devicescan",
    "devicescankernel",
    "scatter",
    "gather",
    "clean_and_count_expert",
    "get_fused_mapping",
    "get_dispatch_layout",
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
    # vLLM / Qwen3 MoE
    "flashinfer",
    "fmha",
    "fused_moe",
    "fused_experts",
    "moe_align",
    "silu_and_mul",
    "paged_attention",
)

COMM_PATTERNS = (
    "nccl:",
    "nccl::",
    "nccldevkernel",
    "memcpy htod",
    "memcpy dtoh",
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
    "clean_and_count_expert",
    "get_fused_mapping",
    "get_dispatch_layout",
    "top2_sum_gate",
)

EXPERT_PATTERNS = (
    "fused_moe",
    "fused_experts",
    "fusedmoe",
    "moe_gemm",
    "expert",
    "swiglu",
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

# NCCL / PyTorch comm op labels for breakdown bar charts (order = first match wins).
_COMM_OP_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("all_to_all", ("alltoall", "all_to_all", "all2all", "all-to-all")),
    ("reduce_scatter", ("reducescatter", "reduce_scatter", "reducescatterprefix")),
    ("all_gather", ("allgather", "all_gather", "allgatherprefix", "allgather_base")),
    ("all_reduce", ("allreduce", "all_reduce", "allreduceprefix")),
    ("broadcast", ("broadcast",)),
    ("all_scatter", ("scatter", "reducescatter")),  # after reduce_scatter
    ("point_to_point", (
        "sendrecv",
        "send/recv",
        "isend",
        "irecv",
        "send_tensor",
        "recv_tensor",
    )),
    ("memcpy_htod", ("memcpy htod", "memcpyhtod", "htod")),
    ("memcpy_dtoh", ("memcpy dtoh", "memcpydtoh", "dtoh")),
    ("memcpy_dtod", ("memcpy dtod", "memcpydtod", "dtod")),
    ("memcpy_async", ("cudamemcpyasync", "cuda_memcpy_async")),
    ("nccl_other", ("nccl",)),
)


def classify_comm_operation(name: str, cat: str = "") -> str | None:
    """
    Map a comm-related event name to a collective / transfer bucket.

    Returns None if the event does not look like communication.
    """
    s = f"{name} {cat}".lower()
    if not any(
        k in s
        for k in (
            "nccl",
            "memcpy",
            "c10d",
            "collective",
            "allgather",
            "allreduce",
            "reducescatter",
            "alltoall",
            "broadcast",
            "sendrecv",
        )
    ):
        return None
    for label, keys in _COMM_OP_RULES:
        if any(k in s for k in keys):
            # all_scatter rule is too broad (matches reducescatter) — skip generic scatter
            if label == "all_scatter" and "reduce" in s:
                continue
            return label
    return "other_comm"


_COMM_OP_DISPLAY = {
    "all_reduce": "All-Reduce",
    "all_gather": "All-Gather",
    "reduce_scatter": "Reduce-Scatter",
    "all_to_all": "All-to-All",
    "broadcast": "Broadcast",
    "all_scatter": "Scatter",
    "point_to_point": "Point-to-Point",
    "memcpy_htod": "Memcpy H→D",
    "memcpy_dtoh": "Memcpy D→H",
    "memcpy_dtod": "Memcpy D→D",
    "memcpy_async": "Memcpy (async)",
    "nccl_other": "NCCL (other)",
    "other_comm": "Other comm",
}


def comm_operation_label(op: str) -> str:
    return _COMM_OP_DISPLAY.get(op, op.replace("_", " ").title())


def classify_kind(name: str, cat: str) -> Kind:
    s = f"{name} {cat}".lower()

    if any(k in s for k in COMM_PATTERNS):
        return "comm"
    if cat.lower() == "kernel":
        return "compute"
    if any(k in s for k in COMPUTE_PATTERNS):
        return "compute"
    if any(k in s for k in CONTROL_PATTERNS):
        return "control"
    return "control"


def classify_subcategory(name: str, cat: str, kind: Kind) -> Subcategory:
    s = f"{name} {cat}".lower()

    if kind == "comm":
        return "collective_comm"
    if kind == "control":
        return "control"

    if any(k in s for k in ATTENTION_PATTERNS):
        return "attention_comp"
    if any(k in s for k in GATE_PATTERNS):
        return "gate_comp"
    if any(k in s for k in EXPERT_PATTERNS):
        return "experts_comp"
    if any(k in s for k in NORM_PATTERNS):
        return "add_norm_comp"
    return "other_compute"


def classify_event(
    name: str,
    cat: str,
    unclassified: list[str] | None = None,
) -> tuple[Kind, Subcategory]:
    kind = classify_kind(name, cat)
    sub = classify_subcategory(name, cat, kind)
    if unclassified is not None and kind == "control" and sub == "control":
        s = f"{name} {cat}".lower()
        if not any(k in s for k in CONTROL_PATTERNS) and cat.lower() != "kernel":
            if not any(k in s for k in COMM_PATTERNS + COMPUTE_PATTERNS):
                unclassified.append(s)
    return kind, sub


def summarize_unclassified(unclassified: list[str], limit: int = 50) -> None:
    counts = Counter(unclassified)
    print(f"Unclassified strings: {len(counts)}")
    for s, n in counts.most_common(limit):
        print(f"{n:>8}  {s}")
