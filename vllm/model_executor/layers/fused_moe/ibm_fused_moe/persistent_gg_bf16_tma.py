import triton
import triton.language as tl
import torch
from .tma_cuda_autotune import CudaUtils, early_config_prune, HOPPER_CONFIGS, STANDARD_CONFIGS

from .tma_autotune import (
    CudaUtils,
    TmaDescriptorHelper,
)

@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, super_group_m):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * super_group_m
    group_size_m = min(num_pid_m - first_pid_m, super_group_m)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["M_TOTAL", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_grouped_gemm_persistent_bf16_tma(
    # Pointers to matrices
    a_desc_ptr,
    b_ptr,
    c_desc_ptr,
    # Pointer to indices array
    indices_ptr,
    workspace,
    # Matrix dimensions
    M_TOTAL: tl.constexpr,  # Total M dimension (sum of all groups)
    N: tl.constexpr,  # N dimension
    K: tl.constexpr,  # K dimension
    NUM_EXPERTS: tl.constexpr, # Number of Experts
    # Tiling parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    TMA_SIZE: tl.constexpr,
    # NUM_CONSUMER_GROUPS: tl.constexpr,
    # Group size (for aligned loads)
    GROUP_SIZE_M: tl.constexpr = 128,
    SUPER_GROUP_M: tl.constexpr = 32, # 32 works best
):
    """
    Contiguous Grouped GEMM kernel forward.
    IMPORTANT: Assumes GROUP_SIZE_M is a multiple of BLOCK_SIZE_M or vice versa,
    and all x are pre-aligned to these block boundaries.
    """
    
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = SUPER_GROUP_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):

            tile_m_idx, tile_n_idx = _compute_pid(tile_id, num_pid_in_group, num_pid_m, SUPER_GROUP_M)
            
            # starting indices for this tile
            m_start = tile_m_idx * BLOCK_SIZE_M
            n_start = tile_n_idx * BLOCK_SIZE_N

            # Only process if in bounds
            if m_start < M_TOTAL:

                # Determine the expert group index and load expert ID
                group_idx = m_start // GROUP_SIZE_M
                expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

                # Device side tensor creation to handle dynamic expert access
                b_desc_ptr_tile = workspace + start_pid * TMA_SIZE 
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=b_desc_ptr_tile,
                    global_address=b_ptr + expert_idx*N*K + n_start*K,
                    load_size=[BLOCK_SIZE_N, BLOCK_SIZE_K],
                    global_size=[NUM_EXPERTS*N, K],
                    element_ty=tl.bfloat16,
                )
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(b_desc_ptr_tile)


                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for ki in range(k_tiles):

                    k_offset = ki * BLOCK_SIZE_K

                    # Load activations (A) with TMA
                    a = tl._experimental_descriptor_load(
                         a_desc_ptr,
                         [m_start, k_offset],
                         [BLOCK_SIZE_M, BLOCK_SIZE_K],
                         tl.bfloat16,
                    )

                    # Load expert weights (B) for the expert assigned to this block
                    b = tl._experimental_descriptor_load(
                         b_desc_ptr_tile,
                         [0, k_offset],
                         [BLOCK_SIZE_N, BLOCK_SIZE_K],
                         tl.bfloat16,
                    )

                    # Accumulate matrix multiplication for this K tile
                    accumulator += tl.dot(a, b.T) 
                
                tile_id_c += NUM_SMS
                tile_m_idx, tile_n_idx = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, SUPER_GROUP_M)

                m_start = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                n_start = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)

                c = accumulator.to(tl.float32)
                
                # Store output (C) with TMA
                tl._experimental_descriptor_store(
                        c_desc_ptr,
                        c.to(tl.bfloat16),
                        [m_start, n_start],
                    )

# =============== Wrapper for FP8 GGEMM =================
def _grouped_gemm_persistent(
    x: torch.Tensor,  # [M_total, K]
    w: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Contiguous grouped GEMM forward pass for MoE.
    All tokens mapped to the same expert must be in contiguous blocks of size group_size_m.

    Args:
        x: Input tensor of shape [M_total, K]
        w: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        group_size_m: Size of contiguous token blocks for each expert (default: 128)
    
    Returns:
        Output tensor of shape [M_total, N]
    """
    # Validate x
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert w.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"


    # Check if x are properly aligned
    M_total, K = x.shape
    assert (
        M_total % group_size_m == 0
    ), f"M_total ({M_total}) must be a multiple of group_size_m ({group_size_m})"

    # Convert expert_indices to int32 if needed
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    # Get dimensions
    num_experts, N, K_weights = w.shape

    # Validate dimensions
    assert K == K_weights, f"Input K ({K}) must match weight K ({K_weights})"
    assert (
        expert_indices.shape[0] == M_total
    ), "Expert indices length must match M_total"

    # Get NUM_SMs
    NUM_SMS = CudaUtils.get_num_sms()

    # Create output tensor
    c = torch.empty((M_total, N), device=x.device, dtype=torch.bfloat16)

    # TMA Setup
    desc_helper = None
    desc_x = x
    desc_w = w
    desc_c = c
    workspace = None

    tma_size = 128
    desc_helper = TmaDescriptorHelper(tma_size=tma_size)

    desc_helper.init_tma_descriptor("x")
    desc_helper.init_tma_descriptor("w")
    desc_helper.init_tma_descriptor("c")

    desc_x = desc_helper.get_tma_descriptor_kernel_param("x")
    desc_w = desc_helper.get_tma_descriptor_kernel_param("w")
    desc_c = desc_helper.get_tma_descriptor_kernel_param("c")

    workspace = torch.empty(
        NUM_SMS * desc_helper.tma_size,
        device=x.device,
        dtype=torch.uint8,
    )


    def grid(META):
        nonlocal desc_helper
        desc_helper.fill_2d_tma_descriptor(
            "x",
            x.data_ptr(),
            M_total,
            K,
            META["BLOCK_SIZE_M"],
            META["BLOCK_SIZE_K"],
            x.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "w",
            w.data_ptr(),
            num_experts*N,
            K,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            w.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "w",
            w.data_ptr(),
            N,
            K,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            w.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "c",
            c.data_ptr(),
            M_total,
            N,
            META["BLOCK_SIZE_M"],
            META["BLOCK_SIZE_N"],
            c.element_size(),
        )

        return (NUM_SMS,)
    
    # Launch kernel
    _kernel_grouped_gemm_persistent_bf16_tma[grid](
        desc_x,
        w,
        desc_c,
        expert_indices,
        workspace,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=group_size_m,
        TMA_SIZE=tma_size,
        NUM_SMS=NUM_SMS,
    )
    return c


def is_kernel_supported(x: torch.Tensor,
                        w: torch.Tensor):
    """
    Supports cuda architectures >= 9.0, bf16 tensors and multiple of 128 hidden size.
    """
    return (torch.cuda.get_device_capability()[0] >= 9 and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and x.size(1) % 128 == 0)

def grouped_gemm_persistent(
    x: torch.Tensor,  # [M_total, K]
    w: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
) -> torch.Tensor:
    return _grouped_gemm_persistent(x, w, expert_indices)
