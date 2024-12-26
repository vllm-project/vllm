from typing import Any

import numpy as np
import numpy.typing as npt

from vllm.inputs.registry import InputContext

from .base import MultiModalPlugin
from .inputs import AudioItem, MultiModalData, MultiModalKwargs


class AudioPlugin(MultiModalPlugin):
    """Plugin for audio data."""

    def get_data_key(self) -> str:
        return "audio"

    def _default_input_mapper(
        self,
        ctx: InputContext,
        data: MultiModalData[AudioItem],
        **mm_processor_kwargs,
    ) -> MultiModalKwargs:
        raise NotImplementedError("There is no default audio input mapper")

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        raise NotImplementedError(
            "There is no default maximum multimodal tokens")


def try_import_audio_packages() -> tuple[Any, Any]:
    try:
        import librosa
        import soundfile
    except ImportError as exc:
        raise ImportError(
            "Please install vllm[audio] for audio support.") from exc
    return librosa, soundfile


def resample_audio(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
) -> npt.NDArray[np.floating]:
    try:
        import librosa
    except ImportError as exc:
        msg = "Please install vllm[audio] for audio support."
        raise ImportError(msg) from exc

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
