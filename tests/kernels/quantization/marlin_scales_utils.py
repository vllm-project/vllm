# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch references for Marlin scale correctness tests and benchmarks."""

import torch


def get_scale_perms():
    grouped = [i + 8 * j for i in range(8) for j in range(8)]
    single = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
    return grouped, single


def reference_permute_scales(s, size_k, size_n, group_size, is_a_8bit=False):
    grouped, single = get_scale_perms()
    perm = (
        grouped
        if group_size < size_k and group_size != -1 and not is_a_8bit
        else single
    )
    return s.reshape(-1, len(perm))[:, perm].reshape(-1, size_n).contiguous()


def reference_process_scales(s, input_dtype=None):
    if input_dtype is None or input_dtype.itemsize == 2:
        s = s.view(-1, 4)[:, [0, 2, 1, 3]].view(s.size(0), -1)
    s = s.to(torch.float8_e8m0fnu)
    if input_dtype == torch.float8_e4m3fn:
        s = s.view(torch.uint8)
        assert s.max() <= 249
        s = (s + 6).view(torch.float8_e8m0fnu)
    return s


def prepare_scales(
    raw,
    group_size,
    param_dtype,
    input_dtype=None,
    *,
    reference=False,
    padded_k=None,
    padded_n=None,
):
    """Prepare raw [N, G] or [E, N, G] MXFP4 scales at the production boundary."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_pad_scales,
    )

    permute = reference_permute_scales if reference else ops.marlin_permute_scales
    process = reference_process_scales if reference else ops.mxfp4_marlin_process_scales
    n, g = raw.shape[-2:]
    k = g * group_size
    pk = k if padded_k is None else padded_k
    pn = n if padded_n is None else padded_n
    decoded = raw.view(torch.float8_e8m0fnu).to(param_dtype)
    experts = decoded.unsqueeze(0) if raw.ndim == 2 else decoded
    outputs = []
    for expert in experts:
        s = marlin_pad_scales(expert.T, n, k, pn, pk, group_size)
        s = permute(s, pk, pn, group_size, input_dtype == torch.float8_e4m3fn)
        outputs.append(process(s, input_dtype))
    return (
        outputs[0] if raw.ndim == 2 else torch.cat([s.unsqueeze(0) for s in outputs], 0)
    )
