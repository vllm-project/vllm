from dataclasses import dataclass
from typing import Literal
from urllib.parse import urljoin

import numpy.typing as npt

from vllm.utils import PlaceholderModule

from .base import VLLM_S3_BUCKET_URL, get_vllm_public_assets

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

ASSET_DIR = "multimodal_asset"


@dataclass(frozen=True)
class AudioAsset:
    name: Literal["winning_call", "mary_had_lamb"]

    @property
    def audio_and_sample_rate(self) -> tuple[npt.NDArray, int]:
        audio_path = get_vllm_public_assets(filename=f"{self.name}.ogg",
                                            s3_prefix=ASSET_DIR)
        y, sr = librosa.load(audio_path, sr=None)
        assert isinstance(sr, int)
        return y, sr

    @property
    def url(self) -> str:
        return urljoin(VLLM_S3_BUCKET_URL, f"{ASSET_DIR}/{self.name}.ogg")
