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

if os.environ.get('NCCL_CUMEM_ENABLE', '0') != '0':
    logger.warning(
        "NCCL_CUMEM_ENABLE is set to %s, skipping override. "
        "This may increase memory overhead with cudagraph+allreduce: "
        "https://github.com/NVIDIA/nccl/issues/1234",
        os.environ['NCCL_CUMEM_ENABLE'])
elif not os.path.exists('/dev/nvidia-caps-imex-channels'):
    # NCCL requires NCCL_CUMEM_ENABLE to work with
    # multi-node NVLink, typically on GB200-NVL72 systems.
    # The ultimate way to detect multi-node NVLink is to use
    # NVML APIs, which are too expensive to call here.
    # As an approximation, we check the existence of
    # /dev/nvidia-caps-imex-channels, used by
    # multi-node NVLink to communicate across nodes.
    # This will still cost some GPU memory, but it is worthwhile
    # because we can get very fast cross-node bandwidth with NVLink.
    os.environ['NCCL_CUMEM_ENABLE'] = '0'

# see https://github.com/vllm-project/vllm/pull/15951
# it avoids unintentional cuda initialization from torch.cuda.is_available()
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'

# see https://github.com/vllm-project/vllm/issues/10480
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
# see https://github.com/vllm-project/vllm/issues/10619
torch._inductor.config.compile_threads = 1
