# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

import safetensors

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase, ECConnectorMetadata, ECConnectorRole)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class MMMeta:
    request_id: str = ""
    input_ids: list[int] = field(default_factory=list)

    @staticmethod
    def make_mm_meta(request_id: str, input_ids: list[int]) -> "MMMeta":
        return MMMeta(request_id=request_id, input_ids=input_ids)


@dataclass
class ECSharedStorageConnectorMetadata(ECConnectorMetadata):
    mm_datas: list[MMMeta]

    def __init__(self):
        self.mm_datas = []

    def add_mm_data(self, mm_data: MMMeta):
        self.mm_datas.append(mm_data)

    def add_mm_metadata(self, request_id: str, input_ids: list[int]):
        self.mm_datas.append(MMMeta.make_mm_meta(request_id, input_ids))


class ECSharedStorageConnector(ECConnectorBase):
    # NOTE: This is Simple debug implementation of the EC connector.
    # It save / load the EC cache to / from the disk.

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        # req_id -> index -> MMMeta
        self._mm_datas_need_loads: dict[str, int] = {}
        self._mm_datas: dict[str, list[int]] = {}
        transfer_config = vllm_config.ec_transfer_config
        self._storage_path = transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp")
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
        encoder_cache = kwargs.get("encoder_cache")  # returns None if missing
        assert encoder_cache is not None
        if metadata is None:
            logger.warning(
                "In connector.start_load_caches, but the connector metadata "
                "is None")
            return

        for mm_data in metadata.mm_datas:
            for input_id in mm_data.input_ids:
                if input_id in encoder_cache.get(mm_data.request_id, {}):
                    continue
                filename = self._generate_filename_debug(
                    f"{mm_data.request_id}_{input_id}")
                if not os.path.exists(filename):
                    logger.warning("Encoder cache file %s does not exist",
                                   filename)
                    continue
                ec_cache = safetensors.torch.load_file(
                    filename)["ec_cache"].npu()
                if mm_data.request_id not in encoder_cache:
                    encoder_cache[mm_data.request_id] = {}
                encoder_cache[mm_data.request_id][input_id] = ec_cache
                logger.debug(
                    "Success load encoder cache for request_id %s, input_id %d",
                    mm_data.request_id, input_id)

    def save_caches(self, **kwargs) -> None:
        """Start saving the EC cache for each mm_datas from encoder cache

        Args:
            **kwargs: additional arguments for the save operation.
        """
        # Return if it is PD Instance
        if not self.is_producer:
            return
        encoder_cache = kwargs.get("encoder_cache")
        mm_hash = kwargs.get("mm_hash")
        assert encoder_cache is not None
        if mm_hash:
            filename = self._generate_filename_debug(mm_hash)
            ec_cache = encoder_cache[mm_hash]
        else:
            request_id = kwargs.get("request_id")
            input_id = kwargs.get("input_id")
            filename = self._generate_filename_debug(
                f"{request_id}_{input_id}")
            ec_cache = encoder_cache[request_id][input_id]
        tensors = {"ec_cache": ec_cache.detach().cpu()}
        safetensors.torch.save_file(tensors, filename)
        logger.debug(
            "Save cache successful for mm_hash %s, request_id %s, input_id %s",
            mm_hash, request_id, input_id)

    def wait_for_save(self):
        return

    def check_caches_exist(
        self,
        request: "Request",
        index: Optional[int] = None,
    ) -> Union[bool, list[bool]]:
        """
        Check if cache exist externally for each mm_data of request
        
        Args:
            request (Request): the request object.
            index (Optional[int]): the index of the request in the batch.

        Returns:
            List of bool indicate that ith mm_data exist in cache or not
        """
        result = []
        request_id = request.request_id
        if index is not None:
            return self._found_match_for_mm_data(f"{request_id}_{index}")

        for input_id in range(len(request.mm_positions)):
            if self._found_match_for_mm_data(f"{request_id}_{input_id}"):
                result.append(True)
            else:
                result.append(False)

        # for mm_hash in request.mm_hashes:
        #     result.append(self._found_match_for_mm_data(mm_hash))
        return result

    def update_state_after_alloc(
        self,
        request: "Request",
        index: int,
    ) -> None:
        """
        Update ECConnector state after encoder cache allocation.
        """
        # mm_hash = request.mm_hashes[index]
        # num_encoder_token = request.get_num_encoder_tokens(index)
        # # Insert mm_hash only if this block has not been recorded yet.
        # self._mm_datas_need_loads[mm_hash] = num_encoder_token
        self._mm_datas.setdefault(request.request_id, []).append(index)

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
        for mm_data in self._mm_datas:
            meta.add_mm_metadata(mm_data, self._mm_datas[mm_data])
        self._mm_datas.clear()
        # for mm_hash, num_encoder_token in self._mm_datas_need_loads.items():
        #     meta.add_mm_data(MMMeta.make_meta(mm_hash, num_encoder_token))
        # self._mm_datas_need_loads.clear()
        return meta

    # ==============================
    # Helper functions
    # ==============================

    def _found_match_for_mm_data(self, mm_hash) -> bool:
        """Check if the cache is hit for the request.
        """
        filename = self._generate_filename_debug(mm_hash)
        return os.path.exists(filename)

    def _generate_foldername_debug(
            self,
            mm_hash: str,
            create_folder: bool = True,  # <- now defaults to True
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
        foldername = self._generate_foldername_debug(
            mm_hash)  # <- folder auto-created
        return os.path.join(foldername, "encoder_cache.safetensors")
