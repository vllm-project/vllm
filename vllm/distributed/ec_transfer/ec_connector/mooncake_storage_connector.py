# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase, ECConnectorMetadata, ECConnectorRole)
from vllm.distributed.ec_transfer.ec_lookup_buffer.mooncake_store import (
    ECMooncakeStore)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class MMMeta:
    mm_hash: str
    num_token: int

    @staticmethod
    def make_meta(mm_hash, num_token) -> "MMMeta":
        return MMMeta(
            mm_hash=mm_hash,
            num_token=num_token,
        )


@dataclass
class ECMooncakeStorageConnectorMetadata(ECConnectorMetadata):
    mm_datas: list[MMMeta]

    def __init__(self):
        self.mm_datas = []

    def add_mm_data(self, mm_data:MMMeta):
        self.mm_datas.append(mm_data)


class ECMooncakeStorageConnector(ECConnectorBase):
    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        # mm_hash -> num_tokens
        self._mm_datas_need_loads: dict[str, int] = {}
        self.store = ECMooncakeStore(vllm_config)

    def start_load_caches(self, **kwargs) -> None:
        """Start loading the EC cache from the connector buffer to
        worker encoder_cache

        Args:
            **kwargs: additional arguments for the load operation
        """

        # Get the metadata
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMooncakeStorageConnectorMetadata)
        if not metadata:
            return

        encoder_cache = kwargs.get("encoder_cache")   # returns None if missing
        assert encoder_cache is not None

        mm_hashes = [mm_data.mm_hash for mm_data in metadata.mm_datas
                    if mm_data.mm_hash not in encoder_cache]
        tensors = self.store.batch_get(mm_hashes)

        for mm_hash, ec_cache in zip(mm_hashes, tensors):
            encoder_cache[mm_hash] = ec_cache
            if ec_cache is None:
                logger.error(f"Load failed for {mm_hash}")
            logger.debug(f"Load tensor for {mm_hash} successfully")

    def save_caches(self, **kwargs) -> None:
        """Start saving the KV cache of the layer from encoder cache
        
        NOTE: this is for saving (mm_hash, torch.Tensor) into cache,
        not (mm_hash, [torch.Tensor])

        Args:
            **kwargs: additional arguments for the save operation.
        """
        if not self.is_producer:
            return
        encoder_cache = kwargs.get("encoder_cache")   # returns None if missing
        mm_hash = kwargs.get("mm_hash")
        assert encoder_cache is not None
        assert mm_hash is not None
        self.store.batch_put([mm_hash], [encoder_cache[mm_hash]])
    
    def wait_for_save(self):
        self.store.wait_for_put()

    def check_caches_exist(
        self,
        request: "Request",
    ) -> list[bool]:
        """
        Check if cache exist externally for each mm_data of request
        
        Args:
            request (Request): the request object.

        Returns:
            List of bool indicate that ith mm_data exist in cache or not
        """
        return self.store.batch_exists(request.mm_hashes)

    def update_state_after_alloc(self, 
                                 request: "Request",
                                 index: int,
                                ) -> None:
        """
        Update ECConnector state after encoder cache allocation.
        """
        mm_hash = request.mm_hashes[index]
        num_encoder_token = request.get_num_encoder_tokens(index)
        # Insert mm_hash only if this block has not been recorded yet.
        self._mm_datas_need_loads[mm_hash] = num_encoder_token

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.
        This only build for load mm_data only
        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = ECMooncakeStorageConnectorMetadata()
        for mm_hash, num_encoder_token in self._mm_datas_need_loads.items():
            meta.add_mm_data(MMMeta.make_meta(mm_hash, num_encoder_token))
        self._mm_datas_need_loads.clear()
        return meta

