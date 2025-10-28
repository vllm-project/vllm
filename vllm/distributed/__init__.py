# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .communication_op import *
from .parallel_state import *
from .utils import *

from vllm.platforms.tpu import USE_TPU_INFERENCE
if USE_TPU_INFERENCE:
    from tpu_inference.distributed import jax_parallel_state
    get_pp_group = jax_parallel_state.get_pp_group