from collections.abc import Mapping
from typing import Generic, TypeVar

from vllm import envs
from vllm.inputs import DummyData
from vllm.logger import init_logger

from .inputs import MultiModalInputsV2
from .processing import BaseProcessingInfo
from .processor import BaseMultiModalProcessor
from .profiling import BaseDummyInputsBuilder

logger = init_logger(__name__)

_I = TypeVar("_I", bound=BaseProcessingInfo)


class MultiModalProfiler(Generic[_I]):

    def __init__(
        self,
        processor: BaseMultiModalProcessor[_I],
    ) -> None:
        super().__init__()

        self.processor = processor

    @property
    def processing(self) -> BaseProcessingInfo:
        return self.processor.info

    @property
    def dummy_inputs(self) -> BaseDummyInputsBuilder[_I]:
        return self.processor.dummy_inputs

    def _get_mm_limits(self) -> Mapping[str, int]:
        mm_config = self.processing.ctx.get_mm_config()
        mm_limit_per_prompt = mm_config.limit_per_prompt

        supported_mm_limits = self.processing.get_supported_mm_limits()

        mm_limits = {
            modality: mm_limit_per_prompt.get(modality, 1)
            for modality in supported_mm_limits
        }

        for modality, supported_limit in supported_mm_limits.items():
            limit = mm_limits[modality]
            if supported_limit is not None and supported_limit < limit:
                raise ValueError(
                    f"You set {modality}={limit} (or defaulted to 1) in "
                    f"`--limit-mm-per-prompt`, but this model only supports "
                    f"at most {supported_limit} {modality} items.")

        return mm_limits

    def _get_dummy_mm_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalInputsV2:
        factory = self.dummy_inputs
        processor_inputs = factory.get_dummy_processor_inputs(
            seq_len, mm_counts)

        return self.processor.apply(
            prompt_text=processor_inputs.prompt_text,
            mm_data=processor_inputs.mm_data,
            hf_processor_mm_kwargs=processor_inputs.hf_processor_mm_kwargs,
        )

    def get_dummy_data(self, seq_len: int) -> DummyData:
        # Avoid circular import
        from vllm.sequence import SequenceData

        mm_counts = self._get_mm_limits()

        processing = self.processing
        mm_max_tokens_per_item = processing.get_mm_max_tokens_per_item(seq_len)

        if mm_counts.keys() != mm_max_tokens_per_item.keys():
            raise AssertionError(
                "The keys returned by `get_supported_mm_limits`"
                f"({set(mm_counts.keys())}) should be the same as those "
                "returned by `get_mm_max_tokens_per_item` "
                f"({set(mm_max_tokens_per_item.keys())})")

        mm_inputs = self._get_dummy_mm_inputs(seq_len, mm_counts)
        prompt_token_ids = mm_inputs["prompt_token_ids"]
        placeholders_by_modality = mm_inputs["mm_placeholders"]

        total_placeholders_by_modality = {
            modality: sum(item["length"] for item in placeholders)
            for modality, placeholders in placeholders_by_modality.items()
        }
        expected_placeholders_by_modality = {
            modality: mm_max_tokens_per_item[modality] * mm_counts[modality]
            for modality in placeholders_by_modality
        }
        if total_placeholders_by_modality != expected_placeholders_by_modality:
            raise AssertionError(
                f"The processed dummy data has a total of "
                f"{total_placeholders_by_modality} placeholder tokens, which "
                f"is not the expected {expected_placeholders_by_modality} "
                "tokens.")

        total_len = len(prompt_token_ids)

        # V0 does not support chunked prefill.
        if total_len > seq_len and not envs.VLLM_USE_V1:
            logger.warning(
                "The context length (%d) of the model is too short "
                "to hold the multi-modal embeddings in the worst case "
                "(%d tokens in total, out of which %s are reserved for "
                "multi-modal embeddings). This may cause certain multi-modal "
                "inputs to fail during inference, even when the input text is "
                "short. To avoid this, you should increase `max_model_len`, "
                "reduce `max_num_seqs`, and/or reduce `mm_counts`.", seq_len,
                total_len, total_placeholders_by_modality)

            return DummyData(
                seq_data=SequenceData.from_prompt_token_counts((0, seq_len)),
                multi_modal_data=None,
                multi_modal_placeholders=None,
            )

        prompt_token_ids.extend([0] * (seq_len - len(prompt_token_ids)))

        return DummyData(
            seq_data=SequenceData.from_seqs(prompt_token_ids),
            multi_modal_data=mm_inputs["mm_kwargs"],
            multi_modal_placeholders=placeholders_by_modality,
        )
