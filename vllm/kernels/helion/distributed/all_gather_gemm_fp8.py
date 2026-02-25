import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
import helion
import helion.language as hl
from vllm.kernels.helion.register import register_kernel
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "helion is required for helion_matmul_w_progress_fp8 kernel. "
        "Install it with: pip install helion"
    )

@register_kernel  # type: ignore[misc]
def helion_matmul_w_progress_fp8(
    a: torch.Tensor,  # [M, K] FP8 (full gathered)
    a_shared: torch.Tensor,  # [M//world_size, K] FP8
    scale_a: torch.Tensor,  # [M//world_size, 1] FP32
    b: torch.Tensor,  # [K, N] FP8 (may be non-contig)
    scale_b: torch.Tensor,  # [1, N] FP32
    progress: torch.Tensor,
    SPLITS_PER_RANK: int,
    RANK: int,
) -> torch.Tensor:
    """
    Performs matrix multiplication with FP8 tensors and tracks progress using Helion.
    """
    M, K = a.size()
    K2, N = b.size()
    assert K2 == K, f"size mismatch {K2} != {K}"
    out = torch.empty(
        [M, N], dtype=torch.bfloat16, device=a.device
    )  # Output buffered as BF16 for performance.
    M_per_rank = a_shared.size(0)

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)  # Initialize accumulator in FP32.
        # Once the progress is filled, we can start doing gemm
        hl.wait(
            progress,
            [tile_m.begin // (M_per_rank // SPLITS_PER_RANK)],  # Wait for certain progress signals.
            signal=1,
        )
        # Load scales once per tile
        sa = scale_a[tile_m, :].to(torch.float32)  # [tile_m, 1]
        sb = scale_b[:, tile_n].to(torch.float32)  # [1, tile_n]

        for tile_k in hl.tile(K):
            # Cast FP8 -> FP32 for accumulation
            acc = torch.addmm(acc, a[tile_m, tile_k].to(torch.float32), 
                                   b[tile_k, tile_n].to(torch.float32))

        # Convert result back to bfloat16
        out[tile_m, tile_n] = (acc * sa * sb).to(torch.bfloat16)

    return out

@helion_matmul_w_progress_fp8.register_config_picker  # type: ignore[misc]
def pick_helion_matmul_w_progress_fp8_config(
    args: tuple, 
    config_keys: list[str],
) -> str | None:
    if not config_keys:
        return None
        """
        Config picker for helion_matmul_w_progress_fp8.

        Args:
            args: tuple containing runtime kernel arguments:
                a_shared: [M_per_rank, K] local input shard
                b: [K, N] weight/projection matrix
                scale_a: [M_per_rank, 1]
                scale_b: [1, N]
                world_size: int
                splits_per_rank: int
            config_keys: list of available pre-autotuned config keys

        Returns:
            str: best matching config key
        """
     # Unpack runtime arguments
    a, a_shared, _ ,b, _ , _, splits_per_rank, rank = args

    # Shapes
    M_per_rank, K = a_shared.shape
    _, N = b.shape

    M, _= a.shape

    # Exact match key
    target_key = (
        f"rank_{rank}_mperrank_{M_per_rank}"
        f"_n_{N}_k_{K}_splits_{splits_per_rank}"
    )
    if target_key in config_keys:
        logger.debug("Found exact config: %s", target_key)
        return target_key

    # Collect candidate distances
    candidates = []
    for key in config_keys:
        try:
            parts = key.split("_")
            candidate_rank = int(parts[1])
            candidate_mpr = int(parts[3])
            candidate_N = int(parts[5])
            candidate_K = int(parts[7])
            candidate_splits = int(parts[9])
        except (ValueError, IndexError):
            continue

        # Only consider same splits_per_rank
        if candidate_splits != splits_per_rank:
            continue

        # Weighted Manhattan distance: M_per_rank dominates
        score = abs(candidate_mpr - M_per_rank) * 1000 \
                + abs(candidate_N - N) * 10 \
                + abs(candidate_K - K)

        candidates.append((score, key))

    if candidates:
        _, best_key = min(candidates)
        logger.debug(
            "No exact config found. Using closest match: %s for rank=%d, M=%d, N=%d, K=%d, splits=%d",
            best_key, rank, M, N, K, splits_per_rank
        )
        return best_key

    # Fallback to default
    if "default" in config_keys:
        logger.debug("Falling back to default config")
        return "default"

    logger.warning("No suitable config found and no default available")
    return None


def copy_engine_all_gather_w_progress(
    output: torch.Tensor,
    inp: torch.Tensor,  # Must be symmetric tensor
    progress: torch.Tensor,
    group_name: ProcessGroup,  
    splits_per_rank: int,
    backend_stream: torch.cuda.Stream | None = None,
) -> torch.cuda.Stream:
    """
    Performs an all-gather operation with progress tracking using symmetric memory.

    - Each rank builds its full output tensor by copying data from all other ranks.
    - Data can be split into smaller chunks (splits_per_rank) for finer-grained progress.
    - The 'progress' 1D tensor signals which splits are ready (1 = ready).
    - GEMM can start operating on a split immediately once its progress flag is set,

    Example (world_size=4, splits_per_rank=2):

        Rank 0: inp0 (8 rows)
        Rank 1: inp1 (8 rows)
        Rank 2: inp2 (8 rows)
        Rank 3: inp3 (8 rows)

        Splits per rank: [A0 A1 | A2 A3], etc.

        Copy order (round-robin) for rank 0:
            Step 0: Copy all splits from rank 1 → output positions [B0 B1 | B2 B3]
            Step 1: Copy all splits from rank 2 → output positions [C0 C1 | C2 C3]
            Step 2: Copy all splits from rank 3 → output positions [D0 D1 | D2 D3]
            Step 3: Copy all splits from rank 0 → output positions [A0 A1 | A2 A3]

        After these steps, rank 0 has the full gathered tensor:
            [A0 A1 | A2 A3 | B0 B1 | B2 B3 | C0 C1 | C2 C3 | D0 D1 | D2 D3]

    Note:
    - This is a partial pipeline: GEMM starts per split as soon as it’s ready.
    - Full pipelined GEMM (all-gather + GEMM fused in Helion kernel) is future work
        (see https://github.com/pytorch/helion/pull/1532), which would eliminate extra
        copies, reduce kernel launch overhead, and maximize overlap of communication
        and computation.
    """
    backend_stream = dist._symmetric_memory._get_backend_stream(priority=-1)
    assert inp.is_contiguous(), "Input tensor 'inp' must be contiguous"
    symm_mem_group = group_name

    if symm_mem_group is None:
        raise RuntimeError("No symmetric memory group available")

    symm_mem_hdl = dist._symmetric_memory.rendezvous(inp, group=symm_mem_group)
    assert symm_mem_hdl is not None, "Failed to obtain symmetric memory handle"

    rank = symm_mem_hdl.rank
    world_size = symm_mem_hdl.world_size
    assert inp.numel() % splits_per_rank == 0, "inp.numel must be divisible by splits_per_rank"
    assert progress.numel() >= world_size * splits_per_rank, "progress size is insufficient"

    output_shape = list(inp.shape)
    output_shape[0] *= world_size
    assert list(output.shape) == output_shape, "Mismatch in output shape"
    chunks = output.chunk(world_size * splits_per_rank)

    symm_mem_hdl.barrier()
    backend_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(backend_stream):
        for step in range(world_size):
            src_rank = (rank + step + 1) % world_size
            for split_id in range(splits_per_rank):
                src_buf = symm_mem_hdl.get_buffer(
                    src_rank, chunks[0].shape, inp.dtype, chunks[0].numel() * split_id
                )
                chunks[src_rank * splits_per_rank + split_id].copy_(src_buf)
                # Write progress signal
                symm_mem_hdl.stream_write_value32(
                    progress,
                    offset=src_rank * splits_per_rank + split_id,
                    val=1,
                )
        symm_mem_hdl.barrier()

    return backend_stream

def _helion_all_gather_fp8_gemm_runtime(
    a_shared: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    world_size: int,
    group_name: ProcessGroup,
    a_out: torch.Tensor | None = None,
    progress: torch.Tensor | None = None,
    SPLITS_PER_RANK: int = 1, 
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs an all-gather on a_shared and matrix multiplication using the Helion library.
    """
    configs = {
        "SPLITS_PER_RANK":SPLITS_PER_RANK,
    }


    symm_mem_group = group_name
    symm_mem_hdl = dist._symmetric_memory.rendezvous(a_shared, group=group_name)

    if symm_mem_hdl is None:
        a_shared_symm = dist._symmetric_memory.empty(
            a_shared.shape,
            dtype=a_shared.dtype,
            device=a_shared.device
        )
        a_shared_symm.copy_(a_shared)
        a_shared_symm._is_symmetric_memory = True

        # Try rendezvous again with the symmetric copy
        symm_mem_hdl = dist._symmetric_memory.rendezvous(a_shared_symm, group=group_name)
        if symm_mem_hdl is None:
            raise RuntimeError("Failed to get symmetric memory handle after copy")
    else:
        a_shared_symm = a_shared  # already usable

    a_shape = list(a_shared_symm.shape)
    a_shape[0] *= symm_mem_hdl.world_size
    configs["RANK"] = symm_mem_hdl.rank
    configs["WORLD_SIZE"] = symm_mem_hdl.world_size

    if a_out is None:
        a_out = torch.empty(a_shape, dtype=a_shared.dtype, device=a_shared.device)
    if progress is None:
        progress = torch.zeros(
            symm_mem_hdl.world_size * configs["SPLITS_PER_RANK"],
            dtype=torch.uint32,
            device=a_shared_symm.device,
        )
    else:
        progress.fill_(0) # Reset progress to 0.
    backend_stream = copy_engine_all_gather_w_progress(
        a_out, a_shared_symm, progress, group_name, configs["SPLITS_PER_RANK"]
    )
    
    c = helion_matmul_w_progress_fp8(
        a_out,
        a_shared_symm,
        scale_a,
        b,
        scale_b,
        progress,
        SPLITS_PER_RANK=configs["SPLITS_PER_RANK"],
        RANK=configs["RANK"],
    )
    assert type(c) is torch.Tensor
    torch.cuda.current_stream().wait_stream(backend_stream)

    return a_out, c

def helion_all_gather_fp8_gemm_fake(
    a_shared: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    world_size: int,
    group_name: ProcessGroup,
    a_out: torch.Tensor | None = None,
    progress: torch.Tensor | None = None,
    SPLITS_PER_RANK: int = 1,      
) -> tuple[torch.Tensor, torch.Tensor]:

    if world_size is None:
        raise RuntimeError("world_size is None")

    M_per_rank, K = a_shared.size()
    K_, N = b.size()
    assert K == K_, f"Shape mismatch: {K} != {K_}"
    a_out_empty_tensor = torch.empty((M_per_rank * world_size, K), dtype=a_shared.dtype, device=a_shared.device)
    c_empty_tensor = torch.empty((M_per_rank * world_size, N), dtype=torch.bfloat16, device=a_shared.device)

    return a_out_empty_tensor, c_empty_tensor

def helion_all_gather_fp8_gemm(
    a_shared: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    world_size: int,
    group_name: str,
    a_out: torch.Tensor | None = None,
    progress: torch.Tensor | None = None,
    SPLITS_PER_RANK: int = 1,      
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.distributed.parallel_state import _groups
    
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")

    # Call the actual runtime with the group object
    return group._helion_all_gather_fp8_gemm(
        a_shared, b, scale_a, scale_b, a_out, progress, SPLITS_PER_RANK
    )

from vllm.utils.torch_utils import (
    direct_register_custom_op,
)
direct_register_custom_op(
    op_name="helion_all_gather_fp8_gemm",
    op_func=helion_all_gather_fp8_gemm,
    fake_impl=helion_all_gather_fp8_gemm_fake,
)
