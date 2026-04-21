# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import safetensors
import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
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
        return MMMeta(mm_hash=mm_hash, num_token=num_token)


@dataclass
class ECExampleConnectorMetadata(ECConnectorMetadata):
    mm_datas: list[MMMeta]

    def __init__(self):
        self.mm_datas = []

    def add_mm_data(self, mm_data: MMMeta):
        self.mm_datas.append(mm_data)


class ECExampleConnector(ECConnectorBase):
    # NOTE: This is Simple debug implementation of the EC connector.
    # It save / load the EC cache to / from the disk.

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        # req_id -> index
        self._mm_datas_need_loads: dict[str, int] = {}
        self._model_config = vllm_config.model_config
        self._metadata_fields_cache: dict[str, set[str]] = {}
        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is not None:
            self._storage_path = transfer_config.get_from_extra_config(
                "shared_storage_path", "/tmp"
            )
            logger.debug(transfer_config)
            logger.debug("Shared storage path is %s", self._storage_path)
        else:
            raise ValueError("ec_transfer_config must be set for ECConnectorBase")

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        """
        Start loading the cache from the connector into vLLM's encoder cache.

        This method loads the encoder cache based on metadata provided by the scheduler.
        It is called before `_gather_mm_embeddings` for the EC Connector. For EC,
        the `encoder_cache` and `mm_hash` are stored in `kwargs`.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            kwargs (dict): Additional keyword arguments for the connector.
        """
        from vllm.platforms import current_platform

        # Get the metadata
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ECExampleConnectorMetadata)
        assert encoder_cache is not None
        if metadata is None:
            logger.warning(
                "In connector.start_load_caches, but the connector metadata is None"
            )
            return
        # A bare "cuda" makes safetensors load onto cuda:0. Pin the load to the
        # selected device so tensor-parallel workers load the cache locally.
        device = current_platform.device_type
        if current_platform.is_cuda_alike():
            device = f"{device}:{current_platform.current_device()}"
        # Load the EC for each mm data
        for mm_data in metadata.mm_datas:
            if mm_data.mm_hash in encoder_cache:
                continue
            filename = self._generate_filename_debug(mm_data.mm_hash)
            ec_cache = safetensors.torch.load_file(filename, device=device)["ec_cache"]
            encoder_cache[mm_data.mm_hash] = ec_cache
            logger.debug("Success load encoder cache for hash %s", mm_data.mm_hash)

    def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
        """
        Save the encoder cache to the connector.

        This method saves the encoder cache from the worker's local storage
        to shared storage or another external connector.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            mm_hash (str): The hash of the multimodal data whose cache is being saved.
            kwargs (dict): Additional keyword arguments for the connector.
        """
        # Return if it is PD Instance
        if not self.is_producer:
            return
        filename = self._generate_filename_debug(mm_hash)
        ec_cache = encoder_cache[mm_hash]
        tensors = {"ec_cache": ec_cache.detach().cpu()}
        safetensors.torch.save_file(tensors, filename)
        logger.debug("Save cache successful for mm_hash %s", mm_hash)

    def has_cache_item(
        self,
        identifier: str,
    ) -> bool:
        """
        Check if cache exist externally for the media

        Args:
            identifier (str): the identifier of the media.

        Returns:
            Bool indicate that media exists in cache or not
        """
        return self._found_match_for_mm_data(identifier)

    def update_state_after_alloc(
        self,
        request: "Request",
        index: int,
    ) -> None:
        """
        Update ECConnector state after encoder cache allocation.
        """
        mm_hash = request.mm_features[index].identifier
        # Only load cache if it is consumer and cache exists
        if not self.is_consumer or not self.has_cache_item(mm_hash):
            return
        num_encoder_token = request.get_num_encoder_embeds(index)
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
        meta = ECExampleConnectorMetadata()
        for mm_hash, num_encoder_token in self._mm_datas_need_loads.items():
            meta.add_mm_data(MMMeta.make_meta(mm_hash, num_encoder_token))
        self._mm_datas_need_loads.clear()
        return meta

    def _placeholder_metadata_fields(self, modality: str) -> set[str]:
        """Which processed keys this model needs published for `modality`.

        Read from `MultiModalDataParser.embedding_fields`, the same declaration
        the consumer's parser requires, so the two cannot drift. An empty set
        means the modality cannot be delivered out of band, and the consumer
        will process the media itself.
        """
        if modality in self._metadata_fields_cache:
            return self._metadata_fields_cache[modality]

        fields: set[str] = set()
        try:
            from vllm.multimodal import MULTIMODAL_REGISTRY

            info = MULTIMODAL_REGISTRY.create_processor(self._model_config).info
            fields = info.data_parser.placeholder_metadata_fields(modality)
        except Exception:
            # Reporting nothing is a safe degradation: the consumer falls back to
            # processing the media itself.
            logger.warning(
                "Could not determine the placeholder metadata fields for "
                "modality %s; the consumer will preprocess the media itself.",
                modality,
                exc_info=True,
            )

        self._metadata_fields_cache[modality] = fields
        return fields

    def request_finished(
        self,
        request: "Request",
    ) -> tuple[bool, dict[str, Any] | None]:
        """Report each item's cache key and grid so a consumer can skip the
        image transform.

        A consumer only needs the grid to size the prompt's placeholder range;
        the embedding itself arrives through this connector. Reporting the grid
        the producer actually computed keeps the two sides in agreement without
        the caller re-deriving it from the raw media.
        """
        if not self.is_producer:
            return False, None

        items = []
        for feature in request.mm_features:
            metadata = {}
            # `data` is None for items served from the processor cache, in which
            # case the metadata is unavailable here and the consumer has to fall
            # back to processing the media itself.
            if feature.data is not None:
                wanted = self._placeholder_metadata_fields(feature.modality)
                metadata = {
                    key: value.tolist()
                    for key, value in feature.data.get_data().items()
                    if key in wanted and isinstance(value, torch.Tensor)
                }
            items.append({"mm_hash": feature.identifier, **metadata})

        if not items:
            return False, None
        return False, {"ec_items": items}

    # ==============================
    # Helper functions
    # ==============================

    def _found_match_for_mm_data(self, mm_hash) -> bool:
        """Check if the cache is hit for the request."""
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
        foldername = self._generate_foldername_debug(mm_hash)  # <- folder auto-created
        return os.path.join(foldername, "encoder_cache.safetensors")
