"""HIP paged attention kernel for TurboQuant k8v4 decode on ROCm.

Provides an optimized HIP decode path for TQ k8v4 (FP8 key + 4-bit value)
on AMD MI300X (gfx942) and MI355X (gfx950) GPUs. Uses a hybrid dispatch:
  - Per-Q-head kernel with grid split-K for low batch / short context
  - MFMA GQA kernel for high batch + long context

The kernel is compiled at first use via torch.utils.cpp_extension and
cached for subsequent calls.
"""

import ctypes
import glob
import logging
import os
import shutil
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_tq_k8v4_loaded = False


def _load_tq_k8v4_kernel() -> bool:
    """Build and load the TQ k8v4 HIP paged attention kernel.

    Returns True if the kernel is loaded and ready.
    """
    global _tq_k8v4_loaded
    if _tq_k8v4_loaded:
        return True

    if not hasattr(torch.ops, "tq_k8v4") or not hasattr(
        torch.ops.tq_k8v4, "paged_attention"
    ):
        hip_src = Path(__file__).parent / "tq_k8v4_rocm_decode.hip"
        if not hip_src.exists():
            logger.warning("TQ k8v4 HIP kernel source not found: %s", hip_src)
            return False

        # Detect GPU architecture
        arch = "gfx942"  # default
        try:
            props = torch.cuda.get_device_properties(0)
            if hasattr(props, "gcnArchName"):
                arch = props.gcnArchName.split(":")[0]
        except Exception:
            pass

        # gfx942 needs -O2 (hipcc crashes with -O3 on large kernels)
        opt_level = "-O2" if arch == "gfx942" else "-O3"

        try:
            import torch.utils.cpp_extension as ext

            os.environ["PYTORCH_ROCM_ARCH"] = arch
            ext.load(
                name=f"tq_k8v4_pa_{arch}",
                sources=[str(hip_src)],
                extra_cuda_cflags=[f"--offload-arch={arch}", opt_level],
                verbose=False,
                is_python_module=False,
            )

            # Find and cache the built .so
            cache_dir = Path.home() / ".cache" / "vllm" / "tq_k8v4"
            cache_dir.mkdir(parents=True, exist_ok=True)
            so_path = cache_dir / f"tq_k8v4_pa_{arch}.so"

            sos = glob.glob(
                str(
                    Path.home()
                    / ".cache"
                    / "torch_extensions"
                    / f"**/tq_k8v4_pa_{arch}*.so"
                ),
                recursive=True,
            )
            if sos:
                shutil.copy2(sos[0], so_path)

            logger.info("TQ k8v4 HIP kernel built for %s: %s", arch, so_path)
        except Exception as e:
            logger.warning("Failed to build TQ k8v4 HIP kernel: %s", e)
            return False

    _tq_k8v4_loaded = True
    logger.info("TQ k8v4 HIP kernel loaded")
    return True


def tq_k8v4_rocm_decode_attention(
    query: torch.Tensor,  # [B, Hq, D]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, slot_size]
    block_table: torch.Tensor,  # [B, max_num_blocks]
    seq_lens: torch.Tensor,  # [B]
    scale: float,
    key_packed_size: int,  # 128 for FP8
    max_num_kv_splits: int = 32,
    output_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run TQ k8v4 decode attention using the HIP kernel.

    Args:
        query: Query tensor [B, Hq, D] in bf16 or fp16.
        kv_cache: TQ k8v4 packed KV cache [num_blocks, bs, Hk, slot_size].
        block_table: Block table [B, max_num_blocks] int32.
        seq_lens: Context lengths [B] int32.
        scale: Attention scale (1/sqrt(head_dim)).
        key_packed_size: Bytes per key (128 for FP8 D=128).
        max_num_kv_splits: Fixed split count for CUDA graph compatibility.
        output_buf: Pre-allocated output buffer (optional).

    Returns:
        Output tensor [B, Hq, D] in same dtype as query.
    """
    if not _load_tq_k8v4_kernel():
        raise RuntimeError("TQ k8v4 HIP kernel not available")

    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    gqa_ratio = Hq // Hk

    # Validate constraints of the HIP kernel.
    if D != 128:
        raise ValueError(
            f"TQ k8v4 HIP kernel requires head_dim=128, got {D}. "
            "Falling back to Triton is recommended for other head dims."
        )
    if gqa_ratio > 16:
        raise ValueError(
            f"TQ k8v4 HIP MFMA kernel supports GQA ratio ≤ 16, "
            f"got {gqa_ratio} (Hq={Hq}, Hk={Hk}). "
            "Falling back to Triton is recommended."
        )

    if output_buf is not None and output_buf.shape[0] >= B:
        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=query.dtype, device=query.device)

    # val_data_offset = key_packed_size (value data starts after key)
    # val_scale_offset = key_packed_size + val_data_bytes
    #   = 128 + 64 = 192 for D=128, 4-bit values
    val_data_offset = key_packed_size
    val_scale_offset = key_packed_size + (D // 2)  # 4-bit: D/2 bytes

    torch.ops.tq_k8v4.paged_attention(
        query.contiguous(),
        kv_cache,
        block_table,
        seq_lens,
        output,
        scale,
        max_num_kv_splits,
        key_packed_size,
        val_data_offset,
        val_scale_offset,
    )

    return output
