# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Literal, Optional

import msgpack_numpy
import redis

from vllm.config import VllmConfig
from vllm.separated_encode.ec_transfer.connector.template import (
    ECConnectorTemplate)
from vllm.logger import init_logger
import torch

logger = init_logger(__name__)

class RedisECConnector(ECConnectorTemplate):

    def __init__(self,
                 vllm_config: "VllmConfig",
                 device: Optional[torch.device],
                 intra_instance_type: Literal["scheduler", "model-runner"],
                 preallocate_callback: Optional[Callable[[str, int, int, str],
                                                         None]],
                 injection_callback: Optional[Callable[
                     [str, int, torch.Tensor, str], None]],
                 redis_host: str = "localhost",
                 redis_port: int = 6379):
        
        if redis_host is None or redis_port is None:
            raise RuntimeError("Redis Encoder Cache Connector is used, "
                            "but redis_host or redis_port is not specified")

        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port)
        self.rank = vllm_config.epd_disagg_config.epd_rank
        super().__init__(
            vllm_config,
            device,
            intra_instance_type,
            preallocate_callback,
            injection_callback,
        )
    
    def _get_request_ranks(self, request_id: str):
        """Extract E_RANK and PD_RANK from a proxy-formatted request ID.
    
        Extracts the request_id with format $ACTUAL_REQUEST_ID|$E_RANK|$PD_RANK
        
        Args:
            request_id: The formatted request ID string from the proxy.
            
        Returns:
            Tuple containing (E_RANK, PD_RANK).
        """
        result = request_id.split("|")
        return int(result[-2]), int(result[-1])

    def _send_prealloc_notification(self, request_id: str, input_id: int, 
                                    successful: bool, mm_hash: str) -> None:
        """
        Send pre-allocation notification from PD to E instance via Redis.

        Notifies the encoder instance whether pre-allocation was successful
        and whether the encoder cache should be sent.

        Args:
            request_id: The formatted request ID containing rank information.
            input_id: Index of the multimodal input within the request.
            successful: Whether pre-allocation succeeded and cache should be sent.
            mm_hash: Hash of the multimodal input.
        """
        transfer_data = {
            "request_id": request_id, 
            "input_id": input_id, 
            "successful": successful,
            "mm_hash": mm_hash
        }
        rank = self._get_request_ranks(request_id)[0]
        logger.debug(f"Sent prealloc notification -> {rank}, {request_id}, {successful}")        
        self.redis_client.lpush(f"prealloc{rank}", 
                                msgpack_numpy.packb(transfer_data))

    def _send_encoder_cache_metas(
        self, request_id: str, input_id: int,
        num_encoder_tokens: int, mm_hash: str
    ) -> None:
        """
        Send encoder cache metadata from E to PD instance via Redis.
        
        Transfers metadata needed for pre-allocating space for the encoder cache
        on the prefill/decode instance.
        
        Args:
            request_id: The formatted request ID containing rank information.
            input_id: Index of the multimodal input within the request.
            num_encoder_tokens: Number of tokens in the encoder cache.
            mm_hash: Hash of the multimodal input.
        """
        transfer_data = {
            "request_id": request_id,
            "input_id": input_id,
            "num_encoder_tokens": num_encoder_tokens,
            "mm_hash": mm_hash
        }
        rank = self._get_request_ranks(request_id)[1]
        logger.debug(f"Sent encode cache metadata -> {rank}, {request_id}")
        self.redis_client.lpush(f"cache_metas{rank}",
                                msgpack_numpy.packb(transfer_data))

    def _send_encoder_cache(
        self, request_id: str, input_id: int,
        encoder_cache: torch.Tensor, mm_hash: str) -> None:
        """
        Send encoder cache tensor from E to PD instance via Redis.
        
        Converts the encoder cache to CPU float16 numpy array before sending
        to optimize transfer size.
        
        Args:
            request_id: The formatted request ID containing rank information.
            input_id: Index of the multimodal input within the request.
            encoder_cache: The encoder output tensor to transfer.
            mm_hash: Hash of the multimodal input.
        """
        encoder_cache_numpy = encoder_cache.to("cpu", dtype=torch.float16).numpy()
        transfer_data = msgpack_numpy.packb({
            "request_id": request_id,
            "input_id": input_id,
            "encoder_cache": encoder_cache_numpy,
            "mm_hash": mm_hash
        })
        rank = self._get_request_ranks(request_id)[1]
        logger.debug(f"Sent encode cache -> {rank}, {request_id}")
        self.redis_client.lpush(f"cache{rank}", transfer_data)

    def _recv_prealloc_notification(
            self, maybe_send_cache_callback: Callable[[str, int, bool, str],
                                                      None]) -> None:
        """
        Receive pre-allocation notification on E instance from Redis.
        
        Blocks until a notification is received, then unpacks the data and
        invokes the callback to handle cache sending logic.
        
        Args:
            maybe_send_cache_callback: Callback to determine whether to send
                the encoder cache based on the pre-allocation result.
        """
        transfered_data = self.redis_client.blpop(f"prealloc{self.rank}")[1]
        transfered_data = msgpack_numpy.unpackb(transfered_data, raw=False)
        request_id, input_id, successful, mm_hash = (
            transfered_data["request_id"],
            transfered_data["input_id"],
            transfered_data["successful"],
            transfered_data["mm_hash"]
        )
        logger.debug(f"Received prealloc notif -> {self.rank}, {request_id}")
        maybe_send_cache_callback(request_id, input_id, successful, mm_hash)

    def _recv_encoder_cache_metas(
            self, preallocate_callback: Callable[[str, int, int, str],
                                                 None]) -> None:
        """
        Receive encoder cache metadata on PD instance from Redis.
        
        Blocks until metadata is received, then unpacks the data and invokes
        the callback to pre-allocate space in the scheduler.
        
        Args:
            preallocate_callback: Scheduler callback to pre-allocate space
                for the incoming encoder cache.
        """
        transfered_data = self.redis_client.blpop(f"cache_metas{self.rank}")[1]
        transfered_data = msgpack_numpy.unpackb(transfered_data, raw=False)
        request_id, input_id, num_encoder_tokens, mm_hash = (
            transfered_data["request_id"], 
            transfered_data["input_id"],
            transfered_data["num_encoder_tokens"],
            transfered_data["mm_hash"]
        )
        logger.debug(f"Received encoder metadata -> {self.rank}, {request_id}")
        preallocate_callback(request_id, input_id, num_encoder_tokens, mm_hash)

    def _recv_encoder_cache(
        self, 
        injection_callback: Callable[[str, int, torch.Tensor, str],None]
    ) -> None:
        """
        Receive encoder cache tensor on PD instance from Redis.
        
        Blocks until cache data is received, converts it from numpy back to
        the appropriate torch tensor format, then invokes the injection callback.
        
        Args:
            injection_callback: Model runner callback to inject the encoder
                cache into the cache dictionary.
        """
        transfered_data = self.redis_client.blpop(f"cache{self.rank}")[1]
        transfered_data = msgpack_numpy.unpackb(transfered_data, raw=False)
        request_id, input_id, encoder_cache, mm_hash = (
            transfered_data["request_id"], 
            transfered_data["input_id"],
            transfered_data["encoder_cache"],
            transfered_data["mm_hash"]
        )
        encoder_cache = torch.from_numpy(encoder_cache).to(
                device=self.device, dtype=self.dtype)   
        logger.debug(f"Received encoder cache -> {self.rank}, {request_id}")
        injection_callback(request_id, input_id, encoder_cache, mm_hash)
