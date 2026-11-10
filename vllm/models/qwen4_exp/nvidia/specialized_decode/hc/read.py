# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's hyper-connection read: what
`GatedResidual.mix` and `GatedResidual.combine_and_mix` compute, with the
kernels of this package for steps of 1-64 rows, and at TP1 for steps of up to
128 rows.

A read normalizes the four streams with the stock kernels (the grouped Gemma
RMSNorm, or the combine-norm when a block output is pending), then runs the
two projections:

* `hc_down` and `hc_up_mix` (warp MMAs, 1-64 rows), or
* with `tensor_cores`, from `TC_MIN_ROWS` rows: `hc_down_tc`
  (tcgen05 UMMAs of one 128-row tile into tensor memory) and the fused up
  projection and gate mix of `..ops.hc` (`hc_up_gate_mix`, the CUDA
  kernel the stock read takes from 512 rows; FP32 gate, tcgen05).
  `TC_MIN_ROWS` is where this pair overtakes the warp-MMA kernels. The
  executor sets `tensor_cores` at TP1 only (`use_tensor_core_reads`): at
  TP4 the row split serves large steps, whose ranks read at most 64 rows
  each (`..runtime.split`). The down
  kernel runs 16-CTA clusters of about 180 KiB of shared memory a CTA; a
  device or build that cannot run the pair (`tensor_core_reads_unsupported`)
  keeps the stock reads above 64 rows, logged once.

The combine-norm runs with its option to launch dependents right after its
own grid-dependency wait, so the down kernel streams its weights during the
whole norm; the down kernel in turn launches the up projection early, which
waits for it (programmatic dependent launch).

The combined streams are the stock ones bit for bit; the low-rank activation
and the injection logits round as `hc_down_silu` rounds them; the gate stays
in FP32 where the stock read rounds it to BF16.
"""

import cutlass
import torch

from vllm.logger import init_logger

from ...hyperconnection import GatedResidual
from ...ops.hc import grouped_gemma_rmsnorm, hc_combine_norm, hc_up_gate_mix
from .. import HC_LOW_RANK, HC_STREAMS
from .down import GROUP_ROWS, MAX_ROWS, hc_down
from .down_tc import MAX_ROWS as TC_MAX_ROWS
from .down_tc import SPLITS, HcDownTc, hc_down_tc
from .up_mix import HcUpMixGather, hc_up_mix

__all__ = [
    "MAX_ROWS",
    "TC_MAX_ROWS",
    "TC_MIN_ROWS",
    "hc_project",
    "hc_read",
    "tensor_core_reads_unsupported",
    "use_tensor_core_reads",
    "warm_up_hc_read",
]

logger = init_logger(__name__)

# The smallest read (token rows) that takes the tensor-core kernels when the
# executor enables them: from 52 rows they beat the warp-MMA kernels by 1.2
# us or more a read, at 48 and below they lose (TP1 microbenchmark, GB300).
TC_MIN_ROWS = 52


def hc_read(
    hc: GatedResidual,
    hidden_states: torch.Tensor,
    block_output: torch.Tensor | None,
    injection: torch.Tensor | None,
    tensor_cores: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """`hc.combine_and_mix(hidden_states, block_output, injection)`, or
    `hc.mix(hidden_states)` when no `block_output` is pending, for 1-64
    rows (`tensor_cores`: 1-128 rows).

    Args:
        hc: A decoder layer's hyper-connection or the final mixer.
        hidden_states: The streams, BF16 `[rows, 10240]`.
        block_output: The previous block's pending output, BF16
            `[rows, 2560]`, or None.
        injection: Its injection logits, BF16 `[rows, 4]`.
        tensor_cores: Run the projections of reads of `TC_MIN_ROWS` rows
            and more with the tensor-core kernels (`hc_project`).

    Returns:
        The streams (combined with the pending block output), the block input
        and the injection logits of this read (None for the final mixer).

    """
    norm_weight = hc.hc_norm.weight
    eps = hc.config.rms_norm_eps
    if block_output is None:
        xn = grouped_gemma_rmsnorm(hidden_states, norm_weight, eps, hc.hc_count)
    else:
        hidden_states, xn = hc_combine_norm(
            hidden_states,
            block_output,
            injection,
            norm_weight,
            eps,
            hc.hc_count,
            launch_dependents_early=True,
        )
    block_input, injection = hc_project(hc, xn, tensor_cores=tensor_cores)
    return hidden_states, block_input, injection


def hc_project(
    hc: GatedResidual,
    xn: torch.Tensor,
    gather: HcUpMixGather | None = None,
    tensor_cores: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The projections of a read from its normalized streams: `hc_down`,
    then `hc_up_mix`, for 1-64 rows; with `tensor_cores`,
    `hc_down_tc` and `hc_up_gate_mix` for `TC_MIN_ROWS` to 128 rows.

    Args:
        hc: A decoder layer's hyper-connection or the final mixer.
        xn: The normalized streams, BF16 `[rows, 10240]`.
        gather: For a row-split read (`..runtime.split`), the up/mix that
            all-gathers the block input over the TP group: `xn` then holds
            this rank's rows only.
        tensor_cores: Take the tensor-core kernels from `TC_MIN_ROWS` rows
            (TP1 reads only; not with `gather`).

    Returns:
        The block input (`gather`: of every rank's rows, in rank order) and
        the injection logits of the read (None for the final mixer; `gather`:
        of this rank's rows).

    """
    tc = tensor_cores and xn.shape[0] >= TC_MIN_ROWS
    assert not (tc and gather is not None), "a row-split read takes hc_up_mix"
    down = hc_down_tc if tc else hc_down
    if hc.use_combine:
        low, injection = down(
            xn, hc.input_mix_weight_down_block_inject.weight, hc.hc_lowrank, hc.hc_count
        )
    else:
        # The final mixer's down projection has no injection columns.
        low, _ = down(xn, hc.input_mix_weight_down.weight, hc.hc_lowrank, 0)
        injection = None
    up = hc.input_mix_weight_up.weight
    if gather is not None:
        return gather(low, xn, up), injection
    if tc:
        return hc_up_gate_mix(low, xn, up), injection
    return hc_up_mix(low, xn, up), injection


def tensor_core_reads_unsupported(device_index: int) -> str | None:
    """What the tensor-core projections need that the device or the build
    lacks, or None: tcgen05 (an SM100-family GPU), the fused up projection's
    CUDA kernel, the down kernel's shared memory a CTA, and clusters of
    `SPLITS` CTAs that each take a whole SM's shared memory.

    Args:
        device_index: The CUDA device.

    """
    if torch.cuda.get_device_capability(device_index)[0] != 10:
        return "tcgen05 (an SM100-family GPU)"
    if not hasattr(torch.ops._C, "qwen4_exp_hc_up_gate_mix"):
        return "the fused up projection kernel (_C.qwen4_exp_hc_up_gate_mix)"
    need = HcDownTc(HC_LOW_RANK, HC_STREAMS).smem_bytes
    have = torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin
    if have < need:
        return f"{need} bytes of shared memory a CTA ({have} available)"
    if cutlass.utils.HardwareInfo(device_index).get_max_active_clusters(SPLITS) < 1:
        return f"clusters of {SPLITS} CTAs"
    return None


def use_tensor_core_reads(max_rows: int, tp_size: int, device_index: int) -> bool:
    """Whether a decode of steps of up to `max_rows` rows at TP `tp_size`
    runs the tensor-core projections (`hc_project`'s `tensor_cores`): at
    TP1, when its steps reach `TC_MIN_ROWS` rows and the device and build
    can run them; one that cannot is logged once.

    Args:
        max_rows: Rows of the decode's largest step.
        tp_size: The tensor-parallel size.
        device_index: The CUDA device.

    """
    if tp_size != 1 or max_rows < TC_MIN_ROWS:
        return False
    missing = tensor_core_reads_unsupported(device_index)
    if missing is not None:
        logger.warning_once(
            "Qwen4Exp specialized decode: the tensor-core hyper-connection read "
            "kernels need %s; steps of more than %d rows run the stock reads.",
            missing,
            MAX_ROWS,
        )
        return False
    return True


def warm_up_hc_read(
    layer_hc: GatedResidual,
    final_hc: GatedResidual,
    max_rows: int,
    tensor_cores: bool = False,
) -> None:
    """Run every variant of the read once on zero inputs, so that no CUDA graph
    capture compiles: the CuTe DSL kernels of each row group count up to
    `max_rows` rows (with `tensor_cores`, the tensor-core down kernel from
    `TC_MIN_ROWS` rows), with and without injection columns, and the
    combine-norm with its early launch of dependents.

    Args:
        layer_hc: A decoder layer's hyper-connection.
        final_hc: The final mixer.
        max_rows: The rows of the largest step the reads serve (at most 64;
            with `tensor_cores` at most 128).
        tensor_cores: The reads take the tensor-core kernels
            (`hc_project`).

    """
    weight = layer_hc.hc_norm.weight
    # The warp-MMA kernels compile per row group, the tensor-core one once.
    last = min(max_rows, TC_MIN_ROWS - 1) if tensor_cores else max_rows
    counts = list(range(1, last + 1, GROUP_ROWS))
    if tensor_cores and max_rows >= TC_MIN_ROWS:
        counts.append(max_rows)
    for rows in counts:
        streams = weight.new_zeros(rows, layer_hc.hyper_hidden_size)
        block_output = weight.new_zeros(rows, layer_hc.hidden_size)
        # A mix, then the combining reads with that mix's injection logits:
        # the strides the decode steps run.
        streams, _, injection = hc_read(layer_hc, streams, None, None, tensor_cores)
        streams, _, injection = hc_read(
            layer_hc, streams, block_output, injection, tensor_cores
        )
        hc_read(final_hc, streams, block_output, injection, tensor_cores)
