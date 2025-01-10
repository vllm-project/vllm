from functools import lru_cache
from typing import Any, cast

from transformers.processing_utils import ProcessorMixin


def get_processor(
    processor_name: str,
    *args: Any,
    trust_remote_code: bool = False,
    processor_cls: type[ProcessorMixin] = ProcessorMixin,
    **kwargs: Any,
):
    """Load a processor for the given model name via HuggingFace."""
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoProcessor

    processor_factory = (AutoProcessor
                         if processor_cls == ProcessorMixin else processor_cls)

    try:
        processor = processor_factory.from_pretrained(
            processor_name,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the processor. If the processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e

    return cast(ProcessorMixin, processor)


cached_get_processor = lru_cache(get_processor)


def get_image_processor(
    processor_name: str,
    *args: Any,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    """Load an image processor for the given model name via HuggingFace."""
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoImageProcessor
    from transformers.image_processing_utils import BaseImageProcessor

    try:
        processor = AutoImageProcessor.from_pretrained(
            processor_name,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs)
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoImageProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the image processor. If the image processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e

    return cast(BaseImageProcessor, processor)


def get_video_processor(
    processor_name: str,
    *args: Any,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    """Load a video processor for the given model name via HuggingFace."""
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers.image_processing_utils import BaseImageProcessor

    processor = get_processor(
        processor_name,
        *args,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )

    return cast(BaseImageProcessor, processor.video_processor)
