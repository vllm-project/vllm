# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

from tests.utils import init_test_distributed_environment
from tests.v1.spec_decode.test_ngram import test_ngram_proposer
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

if __name__ == "__main__":
    pp_size = 1
    local_rank = int(os.environ["LOCAL_RANK"])
    tp_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(
        tp_size=tp_size,
        pp_size=pp_size,
        rank=local_rank,
        distributed_init_port=None,
        local_rank=local_rank,
    )

    test_ngram_proposer()
    cleanup_dist_env_and_memory()
    print("test_ngram_distributed() successfully passed!")
