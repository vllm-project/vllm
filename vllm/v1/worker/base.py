# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from abc import abstractmethod
from collections.abc import Iterable
from typing import cast

import torch

from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.online_quantization import (
    restore_weights_for_reloading,
)
from vllm.model_executor.model_loader.utils import (
    process_weights_after_loading as _process_after_loading,
)

logger = init_logger(__name__)


class ModelRunnerBase:
    vllm_config: VllmConfig
    load_config: LoadConfig
    model_config: ModelConfig

    @abstractmethod
    def get_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    def reload_weights(
        self,
        weights_iterator: Iterable[tuple[str, torch.Tensor]] | None = None,
        process_weights_after_loading: bool = True,
    ) -> None:
        """ """
        # argument validation
        weights_from_disk = weights_iterator is None
        if weights_from_disk and not process_weights_after_loading:
            logger.warning(
                "Loading from disk means that weights will be in checkpoint format"
            )

        if getattr(self, "model", None) is not None:
            raise ValueError("Cannot reload weights before model is loaded.")

        model = self.get_model()
        logger.info("Reloading weights inplace...")
        counter_before_loading_weights = time.perf_counter()

        # load weights from disk if none are provided
        if weights_iterator is None:
            model_loader = get_model_loader(self.load_config)
            weights_iterator = model_loader.get_all_weights(self.model_config, model)
            weights_iterator = cast(
                Iterable[tuple[str, torch.Tensor]], weights_iterator
            )

        if process_weights_after_loading:
            # restore model to checkpoint format
            if hasattr(model, "weight_loading_metadata"):
                restore_weights_for_reloading(model)
            else:
                logger.warning("Quant config is not supported")

            # load weights from checkpoint format
            loaded_weights = model.load_weights(weights_iterator)

            # process weights into kernel format
            device_config = self.vllm_config.device_config
            load_config = self.vllm_config.load_config
            load_device = (
                device_config.device
                if load_config.device is None
                else load_config.device
            )
            _process_after_loading(model, self.model_config, load_device)

        else:
            # load weights from kernel format
            loaded_weights = set()
            for name, loaded_weight in weights_iterator:
                param = model.get_parameter(name)
                param.weight_loader(param, loaded_weight)
                loaded_weights.add(loaded_weight)

        # logging
        counter_after_loading_weights = time.perf_counter()
        diff_seconds = counter_after_loading_weights - counter_before_loading_weights
        logger.info_once(
            f"Loading {len(loaded_weights)} weights took %.2f seconds",
            diff_seconds,
            scope="local",
        )
