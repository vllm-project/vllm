# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import threading
import time

import torch
from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.distributed.kv_transfer.kv_connector.v1.weight_transfer_connector import (
    WeightTransferConnector,
)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group, get_world_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading_mla,
    process_weights_after_loading_quant,
)
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class RemoteInstanceModelLoader(BaseModelLoader):
    """
    Get model weights from GPUs of other vLLM instances
    Only support loading weights from instance with same parallelism strategy
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        extra_config = (
            {}
            if load_config.model_loader_extra_config is None
            else load_config.model_loader_extra_config.copy()
        )

        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{load_config.model_loader_extra_config.keys()}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        raise NotImplementedError

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        """Load a model with the given configurations."""
        self.client_id = vllm_config.instance_id
        self.trigger(model_config)
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )

        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )
            process_weights_after_loading_quant(model, model_config, target_device)
            begin = time.perf_counter()
            self.load_weights(model, model_config)
            end = time.perf_counter()

            logger.info("Loading weights on %s using %s s", load_device, end - begin)
            # process_weights_after_loading(model, model_config, target_device)
            process_weights_after_loading_mla(model, model_config)
            # model.load_state_dict(model.state_dict(), strict=True)
        return model.eval()

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load a model with the given configurations."""
        global_rank = _get_rank()
        url = f"{_get_seed_instance_ip()}:{_get_instance_ports()[global_rank]}"
        logger.info("url: %s", url)

        with WeightTransferConnector(url) as client:
            self.load_model_from_remote_instance(model, client, model_config)

    def load_model_from_remote_instance(
        self,
        model: nn.Module,
        client: WeightTransferConnector,
        model_config: ModelConfig,
    ) -> None:
        start_build_group_tic = time.time()
        # To support tp, pp
        global_rank = _get_rank()
        success, message = client.build_group(
            gpu_id=torch.cuda.current_device(),
            client_rank=global_rank,
            client_id=self.client_id,
        )
        if not success:
            raise RuntimeError(f"Failed to build group for remote instance: {message}")
        # Wait for rank0 to complete trigger()
        get_world_group().barrier()

        end_build_group_tic = time.time()
        logger.info(
            "finish building group for remote instance, time used: %.4fs",
            end_build_group_tic - start_build_group_tic,
        )
        import threading

        from vllm.model_executor.model_loader.remote_instance_loader_utils import (
            trigger_transferring_weights_request,
        )

        if global_rank == 0:
            t = threading.Thread(
                target=trigger_transferring_weights_request,
                args=(
                    _get_seed_instance_ip(),
                    _get_instance_service_port(),
                    _get_instance_ports(),
                    self.client_id,
                    sum(1 for v in model.state_dict().values() if v.numel() > 0),
                ),
            )
            t.start()

        try:
            logger.info("Recv weight in %s", client._model_update_group)
            start_get_weights_tic = time.time()
            with set_default_torch_dtype(model_config.dtype):
                state_dict = model.state_dict()
                for key, tensor in state_dict.items():
                    if tensor.numel():
                        torch.distributed.broadcast(
                            tensor,
                            src=0,
                            group=client._model_update_group,
                        )

            end_get_weights_tic = time.time()
            logger.info(
                "finish getting all weights from remote instance, time used: %.4fs",
                end_get_weights_tic - start_get_weights_tic,
            )
            # torch.cuda.empty_cache()
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)

    def trigger(self, model_config: ModelConfig):
        global_rank = _get_rank()
        if global_rank != 0:
            return

        from vllm.model_executor.model_loader.remote_instance_loader_utils import (
            get_remote_instance_model,
            trigger_init_weights_send_group_for_remote_instance_request,
        )

        try:
            remote_model_id = get_remote_instance_model(
                _get_seed_instance_ip(), _get_instance_service_port()
            )
        except Exception as e:
            raise ValueError(f"Failed to get remote model info: {e}") from e

        if _normalize_model_id(remote_model_id) != _normalize_model_id(
            model_config.model
        ):
            raise ValueError(
                f"Model mismatch: remote model '{remote_model_id}' "
                f"does not match local model '{model_config.model}'"
            )

        t = threading.Thread(
            target=trigger_init_weights_send_group_for_remote_instance_request,
            args=(
                _get_seed_instance_ip(),
                _get_instance_service_port(),
                _get_instance_ports(),
                self.client_id,
            ),
        )
        t.start()


def _get_seed_instance_ip() -> str:
    ip = os.environ.get("REMOTE_INSTANCE_IP")
    if ip is None:
        raise ValueError(
            "REMOTE_INSTANCE_IP environment variable is not set. "
            "Please set REMOTE_INSTANCE_IP to the IP address of the remote instance."
        )
    return ip


def _get_instance_ports() -> list[int]:
    import json

    ports_str = os.environ.get(
        "REMOTE_INSTANCE_PORTS", "[50000,50001,50002,50003,50004,50005,50006,50007]"
    )
    return json.loads(ports_str)


def _get_instance_service_port() -> int:
    return int(os.environ.get("REMOTE_INSTANCE_SERVER_PORT", "30000"))


def _get_rank() -> int:
    tp_rank = get_tp_group().rank_in_group
    tp_size = get_tp_group().world_size

    pp_rank = get_pp_group().rank_in_group
    global_rank = pp_rank * tp_size + tp_rank

    return global_rank


def _normalize_model_id(model_id: str) -> str:
    """Normalize model ID, remove path prefix, keep only model name"""
    # If it's a path, extract the last directory name
    if "/" in model_id:
        return model_id.rstrip("/").split("/")[-1]
    return model_id
