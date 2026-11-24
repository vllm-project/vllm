# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import time
from abc import ABC, abstractmethod
from collections.abc import Generator
from weakref import WeakKeyDictionary

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload import finalize_layerwise_processing
from vllm.model_executor.model_loader.reload.direct import (
    direct_reload_enabled,
    track_loaded_params,
)
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    log_layer_backends,
    process_weights_after_loading,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    def get_external_weight_memory(self, vllm_config: VllmConfig) -> int:
        """Get weights memory from external process;
        0 when the weights are not external."""
        return 0

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    def on_sleep(self, level: int) -> None:  # noqa: B027
        """Called before `Worker.sleep(level)` frees the GPU memory holding
        the weights of the model this loader loaded.

        No-op by default. Override to release or pause any external state
        that depends on that memory remaining valid before it becomes
        invalid.
        """

    def on_wake_up(self, tags: list[str] | None) -> None:  # noqa: B027
        """Called after `Worker.wake_up(tags)` returns.

        No-op by default. Override to resume whatever `on_sleep` paused.
        `tags` is the same argument `wake_up` was called with; the weights
        are woken only when it is `None` or includes `"weights"`. Once woken
        after a level-1 sleep, the weights are valid here. After a level-2
        sleep they are not: they only become valid once the `reload_model`
        call that follows has completed.
        """

    def reload_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        model: nn.Module,
        *,
        is_checkpoint_format: bool = True,
    ) -> None:
        """Reload `model`'s weights in place from this loader's source.

        Used by `reload_weights()` when no weights are provided. After a
        level-2 sleep this is the point where weights become valid again.
        The default streams `get_all_weights()` through the generic in-place
        reload. Loaders that write final weights into the model themselves
        override it; the instance that loaded `model` is then reused, so
        state it keeps across `on_sleep`/`on_wake_up` sees the reload.
        """
        from vllm.model_executor.model_loader.reload.inplace import (
            reload_weights_from_iterator,
        )

        reload_weights_from_iterator(
            vllm_config,
            model,
            self.get_all_weights(model_config, model),
            is_checkpoint_format,
        )

    def create_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Create a model with the given configurations."""
        model = initialize_model(
            vllm_config=vllm_config,
            model_config=model_config,
            prefix=prefix,
        )
        if vllm_config.parallel_config.fine_grained_tp_config.enabled:
            from vllm.model_executor.layers.vocab_parallel_plan import (
                validate_vocab_parallel_model,
            )

            validate_vocab_parallel_model(model, vllm_config)
        log_online_quantization(vllm_config)
        log_model_inspection(model)
        return model

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Iterate over all weights by checkpoint name, for filtered
        partial reloads (e.g. FT expert reload after scale_down)."""
        raise NotImplementedError(
            f"load_format={self.load_config.load_format!r} does not support "
            "iterating weights by checkpoint name"
        )

    @instrument(span_name="Load model")
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                time_before_load = time.perf_counter()
                model = self.create_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
                time_after_load = time.perf_counter()
                logger.info_once(
                    "Initializing model took %.6f seconds",
                    time_after_load - time_before_load,
                )

            try:
                logger.debug("Loading weights on %s ...", load_device)
                if direct_reload_enabled():
                    track_loaded_params(model)  # what every update must load again
                self.load_weights(model, model_config)

                # Release transient tensors held by PyTorch's caching allocator
                # before weight post-processing and raw CUDA driver allocations.
                gc.collect()
                torch.accelerator.empty_cache()

                # Log peak GPU memory after loading weights. This is needed
                # to have test coverage on peak memory for online quantization.
                if current_platform.is_cuda_alike() or current_platform.is_xpu():
                    peak_memory = torch.accelerator.max_memory_allocated()
                    logger.debug_once(
                        "Peak GPU memory after loading weights: %s GiB",
                        format_gib(peak_memory),
                    )

                # Process weights into kernel format. Note that when using online
                # quantization, weights are (typically) quantized as they are loaded.
                if _has_online_quant(model):
                    finalize_layerwise_processing(model, model_config)

                process_weights_after_loading(model, model_config, target_device)
                log_online_quantization_time(vllm_config)
            except Exception:
                # The model runner logs the backends of a loaded model. A
                # failed load never reaches it, but its selections were made.
                log_layer_backends(model, model_config, load_failed=True)
                raise

            gc.collect()
            torch.accelerator.empty_cache()

        model = model.eval()
        bind_model_loader(model, self)
        return model


_MODEL_LOADERS: WeakKeyDictionary[nn.Module, BaseModelLoader] = WeakKeyDictionary()


def bind_model_loader(model: nn.Module, loader: BaseModelLoader) -> None:
    """Record `loader` as the loader that loaded `model`."""
    _MODEL_LOADERS[model] = loader


def get_bound_model_loader(model: nn.Module | None) -> BaseModelLoader | None:
    """Return the loader that loaded `model`, if it was recorded."""
    return None if model is None else _MODEL_LOADERS.get(model)


def unbind_model_loader(model: nn.Module | None) -> None:
    """Forget the loader recorded for `model`. A loader that references its
    model keeps the entry, and so the model, alive until this is called."""
    if model is not None:
        _MODEL_LOADERS.pop(model, None)


def log_model_inspection(model: nn.Module) -> None:
    """Log model structure if VLLM_LOG_MODEL_INSPECTION=1."""
    if not envs.VLLM_LOG_MODEL_INSPECTION:
        return

    from vllm.model_inspection import format_model_inspection

    logger.info("vLLM model structure:\n%s", format_model_inspection(model))


def _get_online_quantization_config(vllm_config: VllmConfig):
    """Return the online quantization config when one is configured."""
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )

    quant_config = vllm_config.quant_config
    online_quantization_config = getattr(
        quant_config, "online_quantization_config", None
    )
    if isinstance(online_quantization_config, OnlineQuantizationConfig):
        quant_config = online_quantization_config
    if not isinstance(quant_config, OnlineQuantizationConfig):
        return None
    return quant_config


def log_online_quantization(vllm_config: VllmConfig) -> None:
    """Log the online-quantized layer count and types, when applicable."""
    quant_config = _get_online_quantization_config(vllm_config)
    if quant_config is None:
        return

    logger.info(
        "Quantizing %d layers of types: %s",
        len(quant_config.quantized_layers),
        "; ".join(quant_config.quantized_layer_summaries),
    )


def log_online_quantization_time(vllm_config: VllmConfig) -> None:
    """Log online quantization processing time, when applicable."""
    quant_config = _get_online_quantization_config(vllm_config)
    if quant_config is None:
        return

    logger.info(
        "Online quantization as part of model loading took %.2f seconds",
        quant_config.online_quantization_time,
    )


def _has_online_quant(model: nn.Module):
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if getattr(quant_method, "uses_meta_device", False):
            return True

    return False
