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
