# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from SGLang (Apache-2.0):
# https://github.com/sgl-project/sglang/blob/656331295619a4d0267b40b7da0b5cd5b549fcbc/python/sglang/kernels/ops/attention/deep_select.py  # noqa: E501
"""The dense indexers' token top-k on DeepSelect's bf16 cluster kernel.

``cluster_topk`` selects, like ``decode_topk``, the indices of each row's ``k``
largest BF16 logits in any order (exact: every index above the k-th value,
then enough equal to it, chosen by atomic order), or ``0 .. n-1`` and -1 for a
row of at most ``k`` (a row holding a NaN gets index 0 in every slot). The
kernel is DeepSelect's bf16 cluster kernel as SGLang rewrote it
(``vllm/third_party/deep_select``, see its README): a cluster of 16 CTAs a row
while every row's cluster is resident at once, else 8; each CTA takes a
contiguous slice of the row's valid length, every CTA pushes its 256-bin
radix histograms to every other with ``st.async`` so each finds the k-th
value itself, and each warp stores its picks at slots claimed on CTA 0's
cursor. It waits on its predecessor grid before its first global access and
lets its dependents launch once its first stages have landed.

``precompile`` builds the op with nvcc through torch's extension loader in the
warm-up: the build is cached on disk, and the TP ranks share it.
"""

from collections.abc import Callable
from functools import cache, partial
from pathlib import Path

import torch

import vllm.third_party

from .decode_topk_cute import _resident_clusters

_VENDOR = Path(vllm.third_party.__file__).parent / "deep_select"
_SOURCE = Path(__file__).with_name("cluster_topk.cu")
CLUSTERS = (16, 8)
MAX_K = 1024  # the kernel's top-k bucket
MAX_WIDTH = 1 << 23  # DeepSelect's MAX_VOCAB_SIZE (exclusive)
# deep_select.py's flags for this kernel (--ftz=false must follow
# --use_fast_math), and the BF16 conversions it uses, which torch's extension
# flags hide.
_CUDA_FLAGS = (
    "-O3",
    "-std=c++20",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--ftz=false",
    "-Xptxas=--register-usage-level=10",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
)


@cache
def _launcher(device_index: int) -> Callable[..., None]:
    """Build (or load the cached build of) the op for the device's
    architecture, bound to its shared memory per SM."""
    from flashinfer.jit.env import CUTLASS_INCLUDE_DIRS
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability(device_index)
    arch = f"{major}{minor}a"
    load(
        name=f"sd_cluster_topk_sm{arch}",
        sources=[str(_SOURCE)],
        extra_include_paths=[
            str(_VENDOR),
            str(_VENDOR / "3rdparty" / "kerutils" / "include"),
            *(str(path) for path in CUTLASS_INCLUDE_DIRS),
        ],
        extra_cuda_cflags=[
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            *_CUDA_FLAGS,
        ],
        is_python_module=False,
    )
    properties = torch.cuda.get_device_properties(device_index)
    return partial(
        torch.ops.sd_cluster_topk.topk,
        smem_per_sm=properties.shared_memory_per_block_optin,
    )


def cluster_for(rows: int, device_index: int) -> int | None:
    """CTAs a row: the most of ``CLUSTERS`` whose clusters of every row are
    resident at once (16 up to 7 rows and 8 up to 15 on a GB300), else
    None."""
    for cluster in CLUSTERS:
        if rows <= _resident_clusters(device_index, cluster):
            return cluster
    return None


def precompile(device_index: int) -> None:
    """Build the op and read the cluster occupancy it plans with, before
    graph capture."""
    _launcher(device_index)
    for cluster in CLUSTERS:
        _resident_clusters(device_index, cluster)


def supports_cluster_topk(logits: torch.Tensor, out: torch.Tensor) -> bool:
    """Whether `cluster_topk` writes the top-k ``out`` of these logits: BF16 rows
    of 128-byte aligned starts and widths, at most ``MAX_K`` (a multiple of 8)
    indices into 32-byte aligned rows, and few enough rows for one wave."""
    rows, width = logits.shape
    k = out.shape[1]
    return (
        logits.dtype == torch.bfloat16
        and logits.stride(1) == 1
        and logits.stride(0) % 64 == 0
        and logits.data_ptr() % 128 == 0
        and width % 64 == 0
        and width < MAX_WIDTH
        and 0 < k <= MAX_K
        and k % 8 == 0
        and out.stride(1) == 1
        and out.stride(0) % 8 == 0
        and out.data_ptr() % 32 == 0
        and cluster_for(rows, logits.device.index) is not None
    )


def cluster_topk(
    logits: torch.Tensor, lengths: torch.Tensor, out: torch.Tensor
) -> None:
    """Launch the cluster top-k on the current stream (see
    `supports_cluster_topk`).

    Args:
        logits: [rows, width] BF16, row ``r`` valid below ``lengths[r]``.
        lengths: Contiguous [rows] int32 valid lengths.
        out: [rows, k] int32 for the indices.

    """
    rows = logits.shape[0]
    assert supports_cluster_topk(logits, out)
    assert out.dtype == torch.int32 and out.shape[0] == rows
    assert (
        lengths.dtype == torch.int32
        and lengths.is_contiguous()
        and lengths.numel() == rows
    )
    device_index = logits.device.index
    _launcher(device_index)(logits, lengths, out, cluster_for(rows, device_index))
