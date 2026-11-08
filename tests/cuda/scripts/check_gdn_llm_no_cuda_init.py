#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that LLM() for a GDN model does not initialize CUDA in the caller.

There is deliberately no __main__ guard, as in many user scripts: if CUDA were
initialized here, vLLM would have to spawn the engine and this script would
fail to bootstrap.
"""

import torch

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3.5-0.8B",
    load_format="dummy",
    language_model_only=True,
    max_model_len=256,
    max_num_seqs=1,
    enforce_eager=True,
    gpu_memory_utilization=0.5,
)
assert not torch.cuda.is_initialized(), "LLM() initialized CUDA in the caller"
output = llm.generate("Hello", SamplingParams(max_tokens=1))[0]
assert len(output.outputs[0].token_ids) == 1

# The forked engine inherited the FLA autotune configs derived from this
# CUDA-free probe, so it must still see the device's shared memory limit.
from vllm.third_party.flash_linear_attention.ops.utils import (  # noqa: E402
    Backend,
    check_shared_mem,
)

props = torch.cuda.get_device_properties(0)
assert check_shared_mem() == (
    props.shared_memory_per_block_optin >= Backend.DEFAULT.value
)
print("OK")
