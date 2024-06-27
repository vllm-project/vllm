from functools import lru_cache
from typing import Optional

from transformers import WhisperProcessor

from vllm.logger import init_logger

logger = init_logger(__name__)


def get_whisper_processor(
    processor_name: str,
    *args,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    **kwargs,
) -> BaseImageProcessor:
    """Gets an image processor for the given model name via HuggingFace."""
    try:
        processor: WhisperProcessor = WhisperProcessor.from_pretrained(
            processor_name,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs)
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoImageProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the whisper processor. If the whisper processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e

    return processor


get_whisper_processor = lru_cache(get_whisper_processor)
