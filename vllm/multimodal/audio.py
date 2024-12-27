import numpy as np
import numpy.typing as npt

from vllm.inputs.registry import InputContext
from vllm.utils import PlaceholderModule

from .base import MultiModalPlugin
from .inputs import AudioItem, MultiModalData, MultiModalKwargs

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]


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


def resample_audio(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
) -> npt.NDArray[np.floating]:
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
