# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import Counter

from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

logger = init_logger(__name__)


def _get_interleaved_text_prompt(
    placeholder_storage: dict[str, list], texts: list[str]
) -> str:
    for idx, elem in enumerate(texts):
        if elem in placeholder_storage:
            slots = placeholder_storage[elem]
            if not slots:
                raise VLLMValidationError(
                    f"Found more '{elem}' placeholders in input prompt than "
                    "actual multimodal data items."
                )
            texts[idx] = slots.pop(0)

    return "\n".join(texts)


# TODO: Let user specify how to insert multimodal tokens into prompt
# (similar to chat template)
def _get_full_multimodal_text_prompt(
    placeholder_storage: dict[str, list],
    texts: list[str],
    interleave_strings: bool,
    multimodal_content_part_separator: str = "\n",
) -> str:
    """Combine multimodal prompts for a multimodal language model."""
    # flatten storage to make it looks like
    # {
    #   "<|image|>": 2,
    #   "<|audio|>": 1
    # }
    placeholder_counts = Counter(
        [v for elem in placeholder_storage.values() for v in elem]
    )

    if interleave_strings:
        text_prompt = _get_interleaved_text_prompt(placeholder_storage, texts)
    else:
        text_prompt = "\n".join(texts)

    # Pass interleaved text further in case the user used image placeholders
    # himself, but forgot to disable the 'interleave_strings' flag

    # Look through the text prompt to check for missing placeholders
    missing_placeholders: list[str] = []
    for placeholder in placeholder_counts:
        # For any existing placeholder in the text prompt, we leave it as is
        placeholder_counts[placeholder] -= text_prompt.count(placeholder)

        if placeholder_counts[placeholder] < 0:
            logger.error(
                "Placeholder count is negative! "
                "Ensure that the 'interleave_strings' flag is disabled "
                "(current value: %s) "
                "when manually placing image placeholders.",
                interleave_strings,
            )
            logger.debug("Input prompt: %s", text_prompt)
            raise VLLMValidationError(
                f"Found more '{placeholder}' placeholders in input prompt than "
                "actual multimodal data items.",
                parameter="messages",
            )

        missing_placeholders.extend([placeholder] * placeholder_counts[placeholder])

    # NOTE: Default behaviour: we always add missing placeholders
    # at the front of the prompt, if interleave_strings=False
    if text_prompt:
        return multimodal_content_part_separator.join(
            missing_placeholders + [text_prompt]
        )
    else:
        return multimodal_content_part_separator.join(missing_placeholders)
