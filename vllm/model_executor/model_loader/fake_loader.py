# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FakeTensorMode wrapper for model loaders.

Wraps any BaseModelLoader so that its full pipeline (initialize_model →
load_weights → process_weights_after_loading) runs under FakeTensorMode.
All tensors appear as CUDA tensors with the correct post-processing
dtypes/shapes but consume no GPU memory and perform no file I/O.

This is used for compile-only mode where we only need the model graph
structure to drive torch.compile / Inductor compilation and populate
the cache.
"""

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

logger = init_logger(__name__)


def wrap_loader_with_fake(real_loader: BaseModelLoader) -> BaseModelLoader:
    """Wrap a model loader so it runs entirely under FakeTensorMode.

    The returned loader delegates to *real_loader* for model
    initialization and weight post-processing, but replaces
    ``load_weights`` with fake materialization (no file I/O) and
    runs everything under ``FakeTensorMode`` (no GPU memory).
    """

    def _fake_load_weights(model: nn.Module, model_config: ModelConfig) -> None:
        """Materialize meta-device parameters as fake tensors.

        Online quantization methods (e.g. Fp8OnlineLinearMethod) create
        weights on the meta device, expecting load_weights to
        materialize them.  We replace them with fake tensors on the
        target device so that process_weights_after_loading can run
        the full quantization/post-processing pipeline.
        """
        target_device = torch.get_default_device()
        for name, param in list(model.named_parameters()):
            if param.device == torch.device("meta"):
                parts = name.split(".")
                mod = model
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                new_data = torch.empty(
                    param.shape,
                    dtype=param.dtype,
                    device=target_device,
                )
                mod.register_parameter(
                    parts[-1],
                    nn.Parameter(new_data, requires_grad=param.requires_grad),
                )

    original_load_model = real_loader.load_model

    def _fake_load_model(
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        from torch._subclasses.fake_tensor import FakeTensorMode

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        with fake_mode:
            return original_load_model(vllm_config, model_config, prefix)

    real_loader.load_weights = _fake_load_weights  # type: ignore[assignment]
    real_loader.load_model = _fake_load_model  # type: ignore[assignment]
    real_loader.download_model = lambda *a, **kw: None  # type: ignore[assignment]

    return real_loader
