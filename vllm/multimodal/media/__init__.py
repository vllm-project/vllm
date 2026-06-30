# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from .audio import AudioEmbeddingMediaIO, AudioMediaIO
from .base import MediaIO, MediaWithBytes
from .connector import MEDIA_CONNECTOR_REGISTRY, MediaConnector
from .image import ImageEmbeddingMediaIO, ImageMediaIO
from .video import VIDEO_LOADER_REGISTRY, VideoMediaIO

try:
    from .tq_connector import TQMediaConnector
except ImportError:
    # transfer_queue is an optional dependency; TQ connector will not
    # be available if the package is not installed.
    TQMediaConnector = None  # type: ignore[assignment,misc]

__all__ = [
    "MediaIO",
    "MediaWithBytes",
    "AudioEmbeddingMediaIO",
    "AudioMediaIO",
    "ImageEmbeddingMediaIO",
    "ImageMediaIO",
    "VIDEO_LOADER_REGISTRY",
    "VideoMediaIO",
    "MEDIA_CONNECTOR_REGISTRY",
    "MediaConnector",
]

# Only export TQMediaConnector when the optional dependency is available,
# so that ``from vllm.multimodal.media import *`` never exposes ``None``.
if TQMediaConnector is not None:
    __all__.append("TQMediaConnector")
