from dataclasses import dataclass
from typing import Literal, Tuple
from urllib.parse import urljoin

import librosa
import numpy as np

from vllm.assets.base import get_vllm_public_assets, vLLM_S3_BUCKET_URL

ASSET_DIR = "multimodal_asset"


@dataclass(frozen=True)
class AudioAsset:
    name: Literal["winning_call", "mary_had_lamb"]

    @property
    def audio_and_sample_rate(self) -> Tuple[np.ndarray, int]:

        audio_path = get_vllm_public_assets(filename=f"{self.name}.ogg",
                                            s3_prefix=ASSET_DIR)
        y, sr = librosa.load(audio_path, sr=None)
        assert isinstance(sr, int)
        return y, sr

    @property
    def url(self) -> str:
        return urljoin(vLLM_S3_BUCKET_URL, f"{ASSET_DIR}/{self.name}.ogg")
