# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the DFlash/DSpark input-prep Triton kernel.

``_prepare_dflash_inputs_kernel`` specializes on ``BLOCK_SIZE =
min(256, next_power_of_2(max_tokens_per_req))``, which follows the largest
per-request span in the batch. The worker warmup's tiny synthetic steps only
cover the smallest buckets, so the first large prefill chunk JIT-compiles a
new variant mid-inference — on PP ranks hosting the Mooncake bootstrap server
that compile stalls the in-process HTTP server past the 5s bootstrap query
timeout (vllm-project/vllm#58543).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


def dflash_triton_warmup(worker: "Worker") -> None:
    speculator = getattr(worker.model_runner, "speculator", None)
    if speculator is None:
        return
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    if isinstance(speculator, DFlashSpeculator):
        speculator.warmup_prepare_inputs()
