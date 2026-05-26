import torch

import triton
import triton.language as tl

# -----------------------------------------------------------------------------
#                                  Utilities
# -----------------------------------------------------------------------------


@triton.constexpr_function
def get_scaled_dot_format_string(dtype: tl.dtype):
    mapping = {
        tl.float16: "fp16",
        tl.bfloat16: "bf16",
        tl.uint8: "e2m1",
        tl.float8e4nv: "e4m3",
        tl.float8e5: "e5m2",
    }
    return mapping[dtype]


@triton.jit
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: tl.constexpr):
    """
    Swizzle the program id based on integer XCD_SWIZZLE.
    This is useful for reording how blocks are ordered. A scheduler may, for example,
    assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2.
    This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment
    becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to
    the same hardware unit.
    """
    # Number of pids per group in the new arrangement
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE

    # Compute current current and local pid within the group
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE

    # Calculate new pid based on the new grouping
    new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
    return new_pid


@triton.jit
def swizzle2d(pid, grid_m, grid_n, GROUP_M: tl.constexpr):
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    tl.assume(group_size >= 0)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    return pid_m, pid_n


def make_matmul_repr(base_name, order):

    def matmul_repr(specialization):
        signature = specialization.signature
        constants = specialization.constants
        reorder = lambda L: [L[i] for i in order]
        layout = lambda stride: "N" if stride in constants else "T"

        def convert_dtype(dtype):
            if "tensordesc" in dtype:
                ret = convert_dtype(dtype.split("<")[1].split("[")[0])
                return ret
            elif "u8" in dtype:
                return "mxfp4"
            elif dtype[0] == "*":
                return dtype[1:]
            else:
                return dtype

        dtypes = "x".join([convert_dtype(f"{signature[i]}") for i in reorder(["Y", "X", "W"])])
        layouts = "".join([f"{layout(i)}" for i in reorder(["stride_y_n", "stride_x_k", "stride_w_n"])])
        blocks = "x".join([f"{constants[i]}" for i in ["BLOCK_M", "BLOCK_N", "BLOCK_K", "SPLIT_K"]])
        # mode = []
        # if "GatherIndx" not in constants:
        #     mode += ['g']
        # if "ScatterSrcIndx" not in constants:
        #     mode += ['s']
        # suffix = "" if not mode else "_o" + (''.join(mode))
        # if base_name.startswith("_p"):
        #     suffix += "_ptma"
        return f"{base_name}_{layouts}_{dtypes}_{blocks}"

    return matmul_repr


def matmul_launch_metadata(grid, kernel, args):
    from ..proton_opts import launch_metadata_allow_sync

    ret = dict()
    M, N, K = args["M"], args["N"], args["K"]
    Y, X, W = args["YPtr"], args["XPtr"], args["WPtr"]
    tokens_per_expt = args.get("TOKENS_PER_EXPT_FOR_ANNOTATION")
    hist = args["ExptHist"]
    if hist is not None:
        # If annotation is given, use that to generate name for profiling.
        if tokens_per_expt is not None:
            n_rows = f"{tokens_per_expt}*"
        elif launch_metadata_allow_sync():
            n_rows = int(hist.float().mean())
        else:
            n_rows = "unknown"

        if launch_metadata_allow_sync():
            n_tokens = float(hist.sum())
            n_w_bytes = (W.numel() * W.element_size() // hist.numel()) * (hist > 0).sum()
        elif tokens_per_expt is not None:
            n_tokens = tokens_per_expt * args["N_EXPTS_TOT"]
            # This may not be totally correct (e.g., we might not be using all experts)
            # but it's better than nothing.
            n_w_bytes = W.numel() * W.element_size()
        else:
            n_tokens = None
            n_w_bytes = 0

        # If annotation is given, use that to generate name for profiling.
        tokens_per_expt = args.get("TOKENS_PER_EXPT_FOR_ANNOTATION")
        n_rows = f"{tokens_per_expt}*" if tokens_per_expt is not None else n_rows
    else:
        n_tokens = None
        n_w_bytes = W.numel() * W.element_size()
    repr = lambda s, x: f"{s} = {x}" if x is not None else f"E_{len(hist)}({s}) = {n_rows}"
    nbits = X.dtype.itemsize * 8
    batch_repr = ""
    if "batch_size" in args and args["batch_size"] > 1:
        batch_repr = repr("B", args["batch_size"]) + ", "
    ret["name"] = f"{kernel.name} [{batch_repr}{repr('M', M)}, {repr('N', N)}, {repr('K', K)}] stg{kernel.num_stages}"
    ep_subtile = args["EPILOGUE_SUBTILE"]
    if ep_subtile is not None and ep_subtile > 1:
        ret["name"] += f" ep/{ep_subtile}"

    if hist is not None and n_tokens is None:
        return ret  # Don't fill metadata because we can't compute them properly.

    fM = M if M is not None else n_tokens
    fK = K if K is not None else n_tokens
    ret[f"flops{nbits}"] = 2.0 * fM * N * fK

    gindx = args.get("GatherIndx", None)
    # sindx = args.get("WriteBackIndx", None)
    n_x_bytes = X.numel() * X.element_size()
    n_y_bytes = Y.numel() * Y.element_size()
    if hist is not None:
        assert n_tokens is not None
        n_expts_act = args["N_EXPTS_ACT"]

        if (gindx is not None) and launch_metadata_allow_sync():
            # recreate inverse GatherIndx.
            dst = torch.full_like(gindx, -1)
            idx = torch.arange(len(gindx), device=gindx.device, dtype=torch.int32)
            mask = (gindx != -1)
            dst[gindx[mask]] = idx[mask]
            n_read_rows = (dst.view((-1, n_expts_act)) != -1).any(dim=1).sum()
        else:
            n_read_rows = n_tokens
        n_x_bytes = n_read_rows * X.shape[-1] * X.element_size()
        n_y_bytes = n_tokens * Y.shape[-1] * Y.element_size()
    ret["bytes"] = int(n_x_bytes + n_y_bytes + n_w_bytes)

    return ret
