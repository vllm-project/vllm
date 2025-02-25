# SPDX-License-Identifier: Apache-2.0
"""
LMCache KV Cache Connector for Distributed Machine Learning Inference

The LMCacheConnector can (1) transfer KV caches between prefill vLLM worker
(KV cache producer) and decode vLLM worker (KV cache consumer) using LMCache;
(2) offload and share KV caches.
"""

from typing import TYPE_CHECKING, List, Tuple, Union

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class LMCacheConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):

        self.transfer_config = config.kv_transfer_config
        self.vllm_config = config

        from lmcache.experimental.cache_engine import LMCacheEngineBuilder
        from lmcache.integration.vllm.utils import ENGINE_NAME
        from lmcache.integration.vllm.vllm_adapter import (
            RetrieveStatus, StoreStatus, init_lmcache_engine,
            lmcache_retrieve_kv, lmcache_should_store, lmcache_store_kv)
        logger.info("Initializing LMCacheConfig under kv_transfer_config %s",
                    self.transfer_config)

        # TODO (Jiayi): Find model_config, parallel_config, and cache_config
        self.engine = init_lmcache_engine(config.model_config,
                                          config.parallel_config,
                                          config.cache_config)
        self.lmcache_engine_name = ENGINE_NAME
        self.lmcache_engine_builder = LMCacheEngineBuilder

        self.model_config = config.model_config
        self.parallel_config = config.parallel_config
        self.cache_config = config.cache_config
        self.lmcache_retrieve_kv = lmcache_retrieve_kv
        self.lmcache_store_kv = lmcache_store_kv
        self.lmcache_should_store = lmcache_should_store
        self.store_status = StoreStatus
        self.retrieve_status = RetrieveStatus

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        hidden_or_intermediate_states = None

        # TODO (Jiayi): Need to support chunked prefill
        retrieve_status = self.retrieve_status.PREFILL

        model_input, bypass_model_exec = self.lmcache_retrieve_kv(
            model_executable, model_input, self.cache_config, kv_caches,
            retrieve_status)

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        num_reqs = 0
        seq_group_list = model_input.sampling_metadata.seq_groups
        assert seq_group_list is not None
        for seq_group in seq_group_list:
            seq_ids = seq_group.seq_ids
            for seq_id in seq_ids:
                num_reqs += 1

        # TODO (Jiayi): Only normal prefill is supported for now
        store_status = self.lmcache_should_store(model_input)
        self.lmcache_store_kv(
            self.model_config,
            self.parallel_config,
            self.cache_config,
            model_executable,
            model_input,
            kv_caches,
            store_status,
        )

    def close(self):
        self.lmcache_engine_builder.destroy(self.lmcache_engine_name)
