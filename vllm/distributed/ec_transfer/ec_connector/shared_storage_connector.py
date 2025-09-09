# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import safetensors
import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase, ECConnectorMetadata, ECConnectorRole)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
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
            num_token=num_token
        )

    

@dataclass
class ECSharedStorageConnectorMetadata(ECConnectorMetadata):
    mm_datas: list[MMMeta]

    def __init__(self):
        self.mm_datas = []

    def add_mm_data(self, mm_data:MMMeta):
        self.mm_datas.append(mm_data)


class ECSharedStorageConnector(ECConnectorBase):
    # NOTE: This is Simple debug implementation of the EC connector.
    # It save / load the EC cache to / from the disk.

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorBase):
        super().__init__(vllm_config=vllm_config, role=role)
        # req_id -> index -> MMMeta
        self._mm_datas_need_loads: dict[str, dict[int, MMMeta]] = {}
        transfer_config = vllm_config.ec_transfer_config
        self.is_producer = (transfer_config.ec_role == 'ec_producer')
        self._storage_path = transfer_config.get_from_extra_config("shared_storage_path", "/tmp")
        logger.debug(transfer_config)
        logger.debug("Shared storage path is %s", self._storage_path)

    def start_load_caches(self, **kwargs) -> None:
        """Start loading the EC cache from the connector buffer to worker 
        encoder_cache

        Args:
            **kwargs: additional arguments for the load operation
        """

        # Get the metadata 
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ECSharedStorageConnectorMetadata)
        encoder_cache = kwargs.get("encoder_cache")   # returns None if missing
        assert encoder_cache is not None
        if metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the connector metadata is None"
            )
            return
        # Load the KV for each request each layer
        for mm_data in metadata.mm_datas:
            # TODO: Only load it once
            # if mm_data in self._mm_datas_need_loads:
            #     continue
            filename = self._generate_filename_debug(mm_data.mm_hash)
            ec_cache = safetensors.torch.load_file(filename)["ec_cache"].cuda()
            encoder_cache[mm_data.mm_hash] = ec_cache
            logger.debug(f"Success load encoder cache for hash {mm_data.mm_hash}")

    def save_caches(self, **kwargs) -> None:
        """Start saving the KV cache of the layer from encoder cache

        Args:
            **kwargs: additional arguments for the save operation.
        """
        # Return if it is PD Instance
        if not self.is_producer:
            return
        encoder_cache = kwargs.get("encoder_cache") 
        mm_hash = kwargs.get("mm_hash")
        assert encoder_cache is not None
        assert mm_hash is not None
        filename = self._generate_filename_debug(mm_hash)
        ec_cache = encoder_cache[mm_hash]
        tensors = {"ec_cache": ec_cache.detach().cpu()}
        safetensors.torch.save_file(tensors, filename)
        logger.debug(f"Save cache successful for mm_hash {mm_hash}")
    
    def wait_for_save(self):
        return

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
        result = []
        for mm_hash in request.mm_hashes:
            result.append(self._found_match_for_mm_data(mm_hash))
        return result

    def update_state_after_alloc(self, 
                                 request: "Request",
                                 cache_exists: list[bool],
                                ) -> None:
        """
        Update ECConnector state after block allocation.

        If cache exist for mm_data, need to load it
        """
        # Create the per-request mapping only if it does not exist.
        loads_for_request = self._mm_datas_need_loads.setdefault(
            request.request_id, {}
        )

        for index, has_cache in enumerate(cache_exists):
            if has_cache:
                mm_hash = request.mm_hashes[index]
                num_encoder_token = request.get_num_encoder_tokens(index)
                # Insert mm_hash only if this block has not been recorded yet.
                loads_for_request.setdefault(index, MMMeta.make_meta(mm_hash,num_encoder_token))
        logger.info(f"After update the _mm_datas_need_loads is {self._mm_datas_need_loads}")

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
        meta = ECSharedStorageConnectorMetadata()
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            for mm_input_id in encoder_input_ids:
                if req_id in self._mm_datas_need_loads and mm_input_id in self._mm_datas_need_loads[req_id]:
                    mm_data = self._mm_datas_need_loads[req_id][mm_input_id]
                    meta.add_mm_data(mm_data)
        self._mm_datas_need_loads.clear()
        return meta

    # ==============================
    # Helper functions
    # ==============================

    def _found_match_for_mm_data(
        self,
        mm_hash
    ) -> bool:
        """Check if the cache is hit for the request.
        """
        foldername = self._generate_foldername_debug(mm_hash,create_folder=False)
        return os.path.exists(foldername)

    def _generate_foldername_debug(
        self,
        mm_hash: str,
        create_folder: bool = True,   # <- now defaults to True
    ) -> str:
        """
        Return the folder in which the cache for this mm_hash lives.
        If `create_folder` is True (default) the directory is created
        recursively the first time it is needed.
        """
        foldername = os.path.join(self._storage_path, mm_hash)
        if create_folder:
            os.makedirs(foldername, exist_ok=True)
        return foldername

    def _generate_filename_debug(self, mm_hash: str) -> str:
        """
        Return the full path of the safetensors file for this mm_hash.
        Ensures the parent directory exists because
        `_generate_foldername_debug` is called with its default
        (`create_folder=True`).
        """
        foldername = self._generate_foldername_debug(mm_hash)   # <- folder auto-created
        return os.path.join(foldername, "encoder_cache.safetensors")

