# SPDX-License-Identifier: Apache-2.0
"""Adapt day0-kit 152c2c0's sender API to the current vLLM NCCL primitives."""

from dataclasses import dataclass

import torch

from vllm.distributed.weight_transfer.nccl_common import trainer_init
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferUpdateInfo as CurrentUpdateInfo,
)
from vllm.distributed.weight_transfer.packed_tensor import packed_nccl_broadcast_producer


def NCCLWeightTransferUpdateInfo(*, packed, **kwargs):
    # Packing is negotiated at communicator initialization in current vLLM.
    return CurrentUpdateInfo(**kwargs)


@dataclass
class NCCLTrainerSendWeightsArgs:
    group: object
    packed: bool
    packed_buffer_size_bytes: int
    packed_num_buffers: int


class NCCLWeightTransferEngine:
    trainer_init = staticmethod(trainer_init)

    @staticmethod
    def trainer_send_weights(iterator, args):
        if args.packed:
            packed_nccl_broadcast_producer(
                iterator=iterator, group=args.group, src=0,
                post_iter_func=lambda item: item[1],
                buffer_size_bytes=args.packed_buffer_size_bytes,
                num_buffers=args.packed_num_buffers,
            )
        else:
            for _, tensor in iterator:
                args.group.broadcast(tensor.contiguous(), src=0,
                                     stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
