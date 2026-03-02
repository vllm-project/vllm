from itertools import product
from filelock import FileLock
from pathlib import Path
from vllm.kernels.helion.distributed.all_gather_gemm_fp8 import helion_matmul_w_progress_fp8,copy_engine_all_gather_w_progress
import os
import pytest
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from vllm.kernels.helion.config_manager import ConfigManager
# VLLM_USE_HELION_BACKEND=1  python -m torch.distributed.run --standalone     --nproc-per-node 4     --rdzv-backend c10d --rdzv-endpoint localhost:0     --no_python python3 scripts/autotune_helion_collective.py 
from vllm.kernels.helion.distributed.all_gather_gemm_fp8 import (
    helion_all_gather_fp8_gemm  # This triggers the direct_register_custom_op call
)

from vllm.kernels.helion.utils import get_canonical_gpu_name
platform = get_canonical_gpu_name()

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
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Performs an all-gather on a_shared and matrix multiplication using the Helion library.
    """
    configs = {
        "SPLITS_PER_RANK": SPLITS_PER_RANK,
    }
    M_per_rank, K = a_shared.shape

    # Validate split
    if M_per_rank % SPLITS_PER_RANK != 0:
        raise ValueError(f"SPLITS_PER_RANK={SPLITS_PER_RANK} does not divide M_per_rank={M_per_rank}")


    a_shared_symm = dist._symmetric_memory.empty(
        a_shared.shape,
        dtype=a_shared.dtype,
        device=a_shared.device
    )
    a_shared_symm.copy_(a_shared)

    # Determine group size and rank
    symm_mem_group = group_name
    if symm_mem_group is None:
        raise RuntimeError("No symmetric memory group available")
    symm_mem_hdl = dist._symmetric_memory.rendezvous(a_shared_symm, group=symm_mem_group)
        
    a_shape = list(a_shared.shape)
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
    inputs = (a_out, a_shared_symm, scale_a, b, scale_b, progress,configs["SPLITS_PER_RANK"],configs["RANK"])
    best_config = helion_matmul_w_progress_fp8.run_autotune(inputs)
    best_config.save("best_config.json")
    print("Best config found:", best_config)

    
    torch.cuda.current_stream().wait_stream(backend_stream)

    return a_out, best_config

def autotune(fn=helion_matmul_w_progress_fp8, force=False):
    shapes_to_tune = [
        #(128, 32, 64),
        #(256, 1024, 1024),
        #medium shapes
        (2048, 1024, 2048),
        (2048, 4096, 4096),
        (4096, 2048, 4096),
        #large shapes
        (4096, 5120, 5120),
        #(8192, 8192, 8192), failed probably due to OOM, if needed I need to investigate further
    ]
    #shapes_to_tune[(num_tokens, hidden_size, N)]
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Setup device - : each process gets its own GPU
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Initialize distributed with torchrun's env vars
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    
    # Register dist.group.WORLD in vLLM's _groups registry
    from vllm.distributed.parallel_state import _groups, GroupCoordinator
    import weakref
    
    # Create a minimal GroupCoordinator wrapping WORLD
    world_group = GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=local_rank,
        torch_distributed_backend="nccl",
        use_device_communicator=False,
        group_name="world",
    )
    dist_group = dist.group.WORLD
    assert dist_group is not None
    # Store a weak reference to the GroupCoordinator in _groups so the kernel can access it without preventing garbage collection.
    _groups[dist_group.group_name] = weakref.ref(world_group)
    ConfigManager.reset_instance()
    config_manager = ConfigManager()

    config_path = config_manager.get_config_file_path(
        kernel_name="helion_matmul_w_progress_fp8",
    )

    lock = FileLock(str(config_path) + ".lock")

    for num_tokens, hidden_size, N in (shapes_to_tune):
        try:
            dist.barrier()
            print(f"Start autotuning with num_tokens={num_tokens} and hidden_size={hidden_size}")
            torch.cuda.empty_cache()

            tokens_per_rank = num_tokens // world_size
            # Local shard for this rank
            a_shared = torch.rand(tokens_per_rank, hidden_size, device=device, dtype=torch.float32) * 0.05
            a_shared = a_shared.to(torch.float8_e4m3fn)

            b = (torch.rand(hidden_size, N, device=device, dtype=torch.float32)  * 0.05).T.contiguous().T
            b= b.to(torch.float8_e4m3fn)
            scale_a = torch.rand((tokens_per_rank , 1), device=device, dtype=torch.float32) * 0.05 + 0.01
            scale_b = torch.rand((1, N), device=device, dtype=torch.float32) * 0.05 + 0.01

            #adding clamping to avoid nan, inf (overflow)
            min_val=1e-3 
            max_val = 0.02 * (1024 / max(hidden_size, N))

            scale_a = scale_a.clamp(min=min_val, max=max_val)
            scale_b = scale_b.clamp(min=min_val, max=max_val)

            # Progress tensor
            candidate_splits = [1, 2, 4]
            for sp in candidate_splits:
                if tokens_per_rank % sp != 0:
                    continue

                # Call autotune runtime
                out, best_config = _helion_all_gather_fp8_gemm_runtime(
                    a_shared, b, scale_a, scale_b, world_size, dist_group.group_name, SPLITS_PER_RANK=sp
                )

                config_key = (
                    f"rank_{rank}_mperrank_{tokens_per_rank}"
                    f"_n_{N}_k_{hidden_size}_splits_{sp}"
                )

                torch.cuda.synchronize()

                with lock:
                    config_manager.save_configs(
                        kernel_name="helion_matmul_w_progress_fp8",
                        platform=platform,
                        configs={config_key: best_config},
                    )
                print(f"[Rank {rank}] Autotune done. Best config saved as {config_key}")
            dist.barrier()

        except Exception as e:
            print(f"Autotuning failed for num_tokens={num_tokens}, hidden_size={hidden_size}: {e}")
            continue

if __name__ == "__main__":
    autotune()
