# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# set some common config/environment variables that should be set
# for all processes created by vllm and all processes
# that interact with vllm workers.
# they are executed whenever `import vllm` is called.

# see https://github.com/vllm-project/vllm/pull/15951
# it avoids unintentional cuda initialization from torch.cuda.is_available()
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

# see https://github.com/vllm-project/vllm/issues/10480
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
# see https://github.com/vllm-project/vllm/issues/10619
torch._inductor.config.compile_threads = 1

# Set FlashInfer workspace directory to be inside vLLM's cache directory
# This ensures FlashInfer kernels are stored alongside other vLLM cache files
if 'FLASHINFER_WORKSPACE_BASE' not in os.environ:
    import vllm.envs as envs

    os.environ['FLASHINFER_WORKSPACE_BASE'] = envs.VLLM_CACHE_ROOT
