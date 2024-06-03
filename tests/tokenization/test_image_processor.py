import pytest
from transformers.image_processing_utils import BaseImageProcessor

from vllm.transformers_utils.image_processor import get_image_processor

IMAGE_PROCESSOR_NAMES = [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-v1.6-34b-hf",
]


@pytest.mark.parametrize("processor_name", IMAGE_PROCESSOR_NAMES)
def test_image_processor_revision(processor_name: str):
    # Assume that "main" branch always exists
    image_processor = get_image_processor(processor_name, revision="main")
    assert isinstance(image_processor, BaseImageProcessor)

    # Assume that "never" branch always does not exist
    with pytest.raises(OSError, match='not a valid git identifier'):
        get_image_processor(processor_name, revision="never")
