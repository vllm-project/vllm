from collections import deque
from typing import List, Optional

import torch
import hashlib

from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.logger import init_logger
from vllm import _custom_ops as ops
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase

logger = init_logger(__name__)

class KVDatabaseTransfer(KVLookupBufferBase):
    def __init__(self, ip: str, port:int, local_rank: int,
                 data_pipe: KVPipeBase) -> None:
        self.ip = ip
        self.port = port

        self.init_valkey = False
        self.local_rank = local_rank
        self.recv_device = torch.device(f"cuda:{self.local_rank}")
        self.data_pipe = data_pipe

    def _encode_tensors(self, tensor1, tensor2):
        tensor1_bytes = tensor1.cpu().numpy().tobytes()
        tensor2_bytes = tensor2.cpu().numpy().tobytes()

        combined_bytes = tensor1_bytes + tensor2_bytes
        hash_object = hashlib.sha256(combined_bytes)
        hash_value = hash_object.hexdigest()

        return hash_value

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:
        if not self.init_valkey:
            ops.valkey_init(self.ip, self.port, True)
            self.init_valkey = True

        tensor_key = self._encode_tensors(input_tokens, roi) + "/" + str(self.local_rank)
        key_key = tensor_key + "/key"
        val_key = tensor_key + "/value"
        hid_key = tensor_key + "/hidden"

        self.data_pipe.send_tensor(key, key_key)
        self.data_pipe.send_tensor(value, val_key)
        self.data_pipe.send_tensor(hidden, hid_key)

    def drop_select(self, input_tokens: torch.Tensor,
                    roi: torch.Tensor) -> List[Optional[torch.Tensor]]:

        if not self.init_valkey:
            ops.valkey_init(self.ip, self.port, True)
            self.init_valkey = True

        tensor_key = self._encode_tensors(input_tokens, roi) + "/" + str(self.local_rank)
        key_key = tensor_key + "/key"
        val_key = tensor_key + "/value"
        hid_key = tensor_key + "/hidden"

        key = self.data_pipe.recv_tensor(key_key)
        val = self.data_pipe.recv_tensor(val_key)
        hid = self.data_pipe.recv_tensor(hid_key)
        res = [input_tokens, roi, key, val, hid]

        return [tensor.to(self.recv_device) for tensor in res]

    def close(self):
        pass
