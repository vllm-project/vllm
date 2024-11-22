import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import msgspec
import torch

from vllm.distributed.device_communicators.shm_broadcast import Handle


@dataclass
class SamplerOutput:

    # [num_reqs]
    sampled_token_ids: List[int]

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: Optional[torch.Tensor]
    # [num_reqs, max_num_logprobs + 1]
    logprobs: Optional[torch.Tensor]

    # TODO: Support prompt logprobs.
    prompt_logprob_token_ids: Optional[torch.Tensor]
    prompt_logprobs: Optional[torch.Tensor]


# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use List instead.
class ModelRunnerOutput(msgspec.Struct,
                        array_like=True,
                        omit_defaults=True,
                        gc=False):

    # [num_reqs]
    req_ids: List[str]
    # req_id -> index
    req_id_to_index: Dict[str, int]

    # [num_reqs]
    sampled_token_ids_cpu: List[int]

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids_cpu: Optional[torch.Tensor]
    # [num_reqs, max_num_logprobs + 1]
    logprobs_cpu: Optional[torch.Tensor]


# Below are data structures used for serializing initiailization-related
# data structures to send between workers and the core engine process
class NumBlocksMsg(msgspec.Struct):
    num_blocks: Tuple[int, int]


class NumGPUBlocks(msgspec.Struct):
    num_gpu_blocks: int


class ShmHandleMsg(msgspec.Struct):
    handle: Handle


class WorkerInitRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """
    DETERMINE_NUM_BLOCKS = b'\x00'
    INIT_CACHE = b'\x01'
    BEGIN_MODEL_EXECUTION = b'\x02'


class WorkerInitOutputType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """
    NUM_BLOCKS = b'\x00'
    MODEL_OUTPUT_MSG_QUEUE = b'\x01'
