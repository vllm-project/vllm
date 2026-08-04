# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from lmcache.v1.gpu_connector.utils import LayoutHints, need_gpu_interm_buffer
from lmcache.v1.metadata import LMCacheMetadata

# Boolean config flags whose underlying implementations exist only on a
# subset of accelerators. Each entry is ``(attr_name, human_label,
# supported_devices)``; the human label is what appears in the error message.
_DEVICE_SCOPED_VLLM_BOOL_FEATURES: tuple[tuple[str, str, frozenset[str]], ...] = (
    ("enable_blending", "config.enable_blending", frozenset({"cuda", "xpu"})),
    ("use_gpu_connector_v3", "config.use_gpu_connector_v3", frozenset({"cuda", "xpu"})),
)


def _validate_vllm_device_features(config: LMCacheEngineConfig) -> None:
    """Reject vLLM configurations that can't run on the active device.

    Three classes of misconfiguration are rejected here, all *before* any
    device-specific construction so the error surfaces as a plain
    ``ValueError`` instead of a deep ``torch.cuda.Stream()`` crash or a
    ``RuntimeError`` from ``torch.device('musa:0')`` on a non-MUSA build:

    1. Any flag in :data:`_DEVICE_SCOPED_VLLM_BOOL_FEATURES` requested on
       a device that has no implementation for that path.
    2. ``config.use_layerwise=True`` on HPU — HPU only ships
       :class:`VLLMPagedMemHPUConnectorV2`; previously, the dispatch silently
       fell through to the CUDA layerwise connector.

    Args:
        config: The LMCache engine configuration.

    Raises:
        ValueError: For any of the unsupported combinations above.
    """
    for attr, label, supported_devices in _DEVICE_SCOPED_VLLM_BOOL_FEATURES:
        if torch_device_type not in supported_devices and getattr(config, attr):
            supported = "/".join(sorted(supported_devices))
            raise ValueError(
                f"{label}=True is only supported on {supported}; the active "
                f"device is '{torch_device_type}'. Set {attr}=False or run "
                "on a supported accelerator build."
            )

    if torch_device_type == "hpu" and config.use_layerwise:
        raise ValueError(
            "config.use_layerwise=True is not supported on HPU; LMCache "
            "ships no layerwise HPU connector. Set use_layerwise=False or "
            "run on a CUDA-capable build."
        )


def CreateGPUConnector(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    engine: EngineType,
    layout_hints: LayoutHints | None = None,
) -> GPUConnectorInterface:
    """
    Create a GPU Connector based on the configuration and metadata.

    Args:
        config: The LMCache engine configuration.
        metadata: The LMCache metadata.
        engine: The serving engine type (EngineType.VLLM, EngineType.SGLANG,
                EngineType.TRTLLM, or EngineType.MOCK).
        layout_hints: Optional hints from the serving engine about KV cache
            layout (e.g. ``{"kv_layout": "HND"}``).
    """
    use_gpu = need_gpu_interm_buffer(config)

    if engine == EngineType.SGLANG:
        if torch_device_type == "musa":
            raise ValueError(
                "SGLang on MUSA is not supported; only the vLLM MUSA "
                "connector is available."
            )

        num_layer, _, chunk_size, num_kv_head, head_dim = metadata.kv_shape
        hidden_dim_size = num_kv_head * head_dim
        local_worker_id = metadata.local_worker_id
        torch_dev.set_device(local_worker_id)
        device = torch.device(f"{torch_device_type}:{local_worker_id}")
        kv_dtype = metadata.kv_dtype

        if torch_device_type == "xpu":
            # First Party
            from lmcache.v1.gpu_connector.xpu_connectors import (
                SGLangLayerwiseXPUConnector,
                SGLangXPUConnector,
            )

            if config.use_layerwise:
                return SGLangLayerwiseXPUConnector(
                    hidden_dim_size,
                    num_layer,
                    use_gpu=use_gpu,
                    chunk_size=chunk_size,
                    dtype=kv_dtype,
                    device=device,
                )
            else:
                return SGLangXPUConnector(
                    hidden_dim_size,
                    num_layer,
                    use_gpu=use_gpu,
                    chunk_size=chunk_size,
                    dtype=kv_dtype,
                    device=device,
                )
        else:  # GPU for SGLang
            # First Party
            from lmcache.v1.gpu_connector.gpu_connectors import (
                SGLangGPUConnector,
                SGLangLayerwiseGPUConnector,
            )

            if config.use_layerwise:
                return SGLangLayerwiseGPUConnector(
                    hidden_dim_size,
                    num_layer,
                    use_gpu=use_gpu,
                    chunk_size=chunk_size,
                    dtype=kv_dtype,
                    device=device,
                )
            else:
                return SGLangGPUConnector(
                    hidden_dim_size,
                    num_layer,
                    use_gpu=use_gpu,
                    chunk_size=chunk_size,
                    dtype=kv_dtype,
                    device=device,
                )
    elif engine == EngineType.VLLM:
        _validate_vllm_device_features(config)

        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import (
            VLLMBufferLayerwiseGPUConnector,
            VLLMPagedMemGPUConnectorV2,
            VLLMPagedMemGPUConnectorV3,
            VLLMPagedMemLayerwiseGPUConnector,
        )

        local_worker_id = metadata.local_worker_id
        torch_dev.set_device(local_worker_id)
        device = torch.device(f"{torch_device_type}:{local_worker_id}")

        if torch_device_type == "cuda":
            # First Party
            from lmcache.v1.gpu_connector.gpu_connectors import (
                VLLMBufferLayerwiseGPUConnector,
                VLLMPagedMemGPUConnectorV2,
                VLLMPagedMemGPUConnectorV3,
                VLLMPagedMemLayerwiseGPUConnector,
            )

            if config.use_layerwise:
                if config.enable_blending:
                    return VLLMBufferLayerwiseGPUConnector.from_metadata(
                        metadata, use_gpu, device, layout_hints=layout_hints
                    )
                else:
                    return VLLMPagedMemLayerwiseGPUConnector.from_metadata(
                        metadata, use_gpu, device, layout_hints=layout_hints
                    )

            if config.use_gpu_connector_v3:
                return VLLMPagedMemGPUConnectorV3.from_metadata(
                    metadata, use_gpu, device, layout_hints=layout_hints
                )
            else:
                return VLLMPagedMemGPUConnectorV2.from_metadata(
                    metadata, use_gpu, device, layout_hints=layout_hints
                )
        elif torch_device_type == "xpu":
            # First Party
            from lmcache.v1.gpu_connector.xpu_connectors import (
                VLLMBufferLayerwiseXPUConnector,
                VLLMPagedMemLayerwiseXPUConnector,
                VLLMPagedMemXPUConnectorV2,
                VLLMPagedMemXPUConnectorV3,
            )

            if config.use_layerwise:
                if config.enable_blending:
                    return VLLMBufferLayerwiseXPUConnector.from_metadata(
                        metadata, use_gpu, device
                    )
                else:
                    return VLLMPagedMemLayerwiseXPUConnector.from_metadata(
                        metadata, use_gpu, device
                    )

            if config.use_gpu_connector_v3:
                return VLLMPagedMemXPUConnectorV3.from_metadata(
                    metadata, use_gpu, device
                )
            else:
                return VLLMPagedMemXPUConnectorV2.from_metadata(
                    metadata, use_gpu, device
                )
        elif torch_device_type == "musa":
            # First Party
            from lmcache.v1.gpu_connector.musa_connectors import (
                VLLMPagedMemLayerwiseMUSAConnector,
                VLLMPagedMemMUSAConnectorV2,
            )

            if config.use_layerwise:
                return VLLMPagedMemLayerwiseMUSAConnector.from_metadata(
                    metadata, use_musa=use_gpu, device=device
                )
            return VLLMPagedMemMUSAConnectorV2.from_metadata(metadata, use_gpu, device)
        elif torch_device_type == "hpu":
            # First Party
            from lmcache.v1.gpu_connector.hpu_connector import (
                VLLMPagedMemHPUConnectorV2,
            )

            return VLLMPagedMemHPUConnectorV2.from_metadata(metadata, use_gpu, device)
        else:
            raise RuntimeError(f"No supported {torch_device_type} connector found.")

    elif engine == EngineType.TRTLLM:
        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector

        local_worker_id = metadata.local_worker_id
        torch_dev.set_device(local_worker_id)
        device = torch.device(f"{torch_device_type}:{local_worker_id}")
        return TRTLLMGPUConnector.from_metadata(metadata, device=device)

    elif engine == EngineType.MOCK:
        kv_shape = metadata.kv_shape
        return MockGPUConnector(kv_shape=kv_shape)
    else:
        raise RuntimeError(f"Unsupported engine: {engine}")
