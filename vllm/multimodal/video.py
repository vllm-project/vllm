from vllm.inputs.registry import InputContext
from vllm.multimodal.base import MultiModalInputs, MultiModalPlugin


class VideoPlugin(MultiModalPlugin):
    """Plugin for video data."""

    def get_data_key(self) -> str:
        return "video"

    def _default_input_mapper(self, ctx: InputContext,
                              data: object) -> MultiModalInputs:
        raise NotImplementedError("There is no default video input mapper")

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        raise NotImplementedError(
            "There is no default maximum multimodal tokens")
