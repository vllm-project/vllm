# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, deferred word alignment for synchronous offline generation."""

from _thread import LockType
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakKeyDictionary

from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionWord
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.multimodal.audio_alignment import AUDIO_ALIGNMENT_HANDLE_KEY
from vllm.multimodal.parse import AudioProcessorItems
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils import random_uuid

if TYPE_CHECKING:
    from vllm import LLM

logger = init_logger(__name__)
_alignment_locks: WeakKeyDictionary["LLM", LockType] = WeakKeyDictionary()
_alignment_locks_guard = Lock()


@contextmanager
def _alignment_session(llm: "LLM") -> Iterator[None]:
    with _alignment_locks_guard:
        lock = _alignment_locks.setdefault(llm, Lock())
    if not lock.acquire(blocking=False):
        raise RuntimeError("Another word-alignment call is using this LLM")
    try:
        if llm.llm_engine.has_unfinished_requests():
            raise ValueError(
                "Complete queued requests before requesting word timestamps"
            )
        yield
    finally:
        lock.release()


@dataclass
class WordTimestampResult:
    """A generated transcript and its optional word alignment.

    ``output`` is the original generation result. An empty ``words`` list with
    ``error=None`` is a successful empty transcript; a nonempty error explains
    why timestamps could not be produced. Chinese and Japanese units follow
    the model's character segmentation rather than semantic word segmentation.
    """

    output: RequestOutput
    words: list[TranscriptionWord]
    error: str | None = None


def _alignment_sampling_params(params: SamplingParams, handle: str) -> SamplingParams:
    if params.n != 1:
        raise ValueError("Word timestamps require SamplingParams(n=1)")
    if params.extra_args and AUDIO_ALIGNMENT_HANDLE_KEY in params.extra_args:
        raise ValueError(
            "The audio alignment handle is reserved for the offline helper"
        )
    copied = params.clone()
    copied.extra_args = dict(params.extra_args or {})
    copied.extra_args[AUDIO_ALIGNMENT_HANDLE_KEY] = handle
    copied.output_kind = RequestOutputKind.FINAL_ONLY
    return copied


def transcribe_with_word_timestamps(
    llm: "LLM",
    prompts: PromptType | Sequence[PromptType],
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
) -> list[WordTimestampResult]:
    """Generate transcripts and align them after the generation batch finishes.

    Initialize the model with ``hf_overrides={"enable_ctc_timestamps": True}``,
    ``enforce_eager=True`` and ``enable_prefix_caching=False``. The initial
    implementation supports one CUDA worker, one audio item and one completion
    per prompt. MTP is supported because only final accepted token IDs enter
    alignment. The helper is synchronous and must not share its engine with
    another concurrent generation or alignment call.

    Args:
        llm: An initialized native vLLM model supporting audio alignment.
        prompts: The same text/audio prompts accepted by ``LLM.generate``.
        sampling_params: Generation parameters, optionally one per prompt.

    Returns:
        Generated outputs and alignment results in input order. Alignment
        failures preserve the generated output and set the corresponding error.

    """
    with _alignment_session(llm):
        if llm.get_world_size(include_dp=True) != 1:
            raise ValueError("Offline word timestamps currently require one worker")
        config = llm.llm_engine.vllm_config
        if not config.use_v2_model_runner:
            raise ValueError("Offline word timestamps require Model Runner V2")
        if (
            config.speculative_config is not None
            and config.speculative_config.method != "mtp"
        ):
            raise ValueError(
                "Offline word timestamps currently support MTP speculation only"
            )
        tokenizer = llm.get_tokenizer()
        if config.cache_config.enable_prefix_caching:
            raise ValueError(
                "Offline word timestamps require enable_prefix_caching=False"
            )
        if not llm.model_config.enforce_eager:
            raise ValueError(
                "Offline word timestamps currently require enforce_eager=True"
            )
        stats = llm.collective_rpc("get_audio_alignment_stats")
        if len(stats) != 1 or not stats[0].get("enabled", False):
            raise ValueError(
                "Initialize an alignment-capable model with "
                "hf_overrides={'enable_ctc_timestamps': True}"
            )

        source_batch = (
            [prompts] if isinstance(prompts, (str, Mapping)) else list(prompts)
        )
        if not source_batch:
            return []
        batch: list[PromptType] = []
        parser = llm.renderer.get_mm_processor().info.get_data_parser()
        for source_prompt in source_batch:
            if not isinstance(source_prompt, Mapping):
                raise ValueError("Word timestamps require a text/audio prompt")
            # Rendering merges processor kwargs into its prompt containers. Copy
            # those containers without copying hour-long waveform storage.
            prompt: dict[str, Any] = dict(source_prompt)
            for field in (
                "mm_processor_kwargs",
                "multi_modal_data",
                "multi_modal_uuids",
            ):
                value = prompt.get(field)
                if isinstance(value, Mapping):
                    prompt[field] = {
                        key: dict(item)
                        if isinstance(item, Mapping)
                        else list(item)
                        if isinstance(item, list)
                        else item
                        for key, item in value.items()
                    }
            mm_data = prompt.get("multi_modal_data")
            if not isinstance(mm_data, Mapping):
                raise ValueError("Each timestamp prompt must contain one audio item")
            items = parser.parse_mm_data(mm_data)
            if set(items) != {"audio"} or items.get_count("audio") != 1:
                raise ValueError(
                    "Each timestamp prompt must contain exactly one audio item"
                )
            audio_items = items.get_items("audio", AudioProcessorItems)
            if audio_items.get(0) is None:
                raise ValueError("Word timestamps require the original audio waveform")
            scoped = prompt.get("mm_processor_kwargs", {})
            if scoped and (
                "capture_ctc_timestamps" in scoped
                or any(
                    isinstance(value, Mapping) and "capture_ctc_timestamps" in value
                    for value in scoped.values()
                )
            ):
                raise ValueError("The capture option is managed by the offline helper")
            batch.append(cast(PromptType, prompt))

        if sampling_params is None:
            base_params = [llm.get_default_sampling_params()] * len(batch)
        elif isinstance(sampling_params, SamplingParams):
            base_params = [sampling_params] * len(batch)
        else:
            base_params = list(sampling_params)
            if len(base_params) != len(batch):
                raise ValueError("sampling_params must have one entry per prompt")
        handles = [random_uuid() for _ in batch]
        params = [
            _alignment_sampling_params(value, handle)
            for value, handle in zip(base_params, handles, strict=True)
        ]
        succeeded = False
        try:
            outputs = llm.generate(
                batch,
                params,
                use_tqdm=False,
                mm_processor_kwargs={"capture_ctc_timestamps": True},
            )
            if len(outputs) != len(handles):
                raise RuntimeError("Generation returned an unexpected output count")
            texts: list[str] = []
            for output in outputs:
                if not output.finished or len(output.outputs) != 1:
                    raise RuntimeError("Alignment requires one finished completion")
                # Keep RPC handle order, but do not score incomplete transcripts.
                texts.append(
                    ""
                    if output.outputs[0].finish_reason == "length"
                    else tokenizer.decode(
                        output.outputs[0].token_ids, skip_special_tokens=False
                    )
                )
            replies = llm.collective_rpc(
                "align_audio", args=(list(zip(handles, texts, strict=True)),)
            )
            if len(replies) != 1 or len(replies[0]) != len(outputs):
                raise RuntimeError("Alignment returned an unexpected result count")
            results: list[WordTimestampResult] = []
            for output, reply in zip(outputs, replies[0], strict=True):
                error = reply["error"]
                if output.outputs[0].finish_reason == "length":
                    error = "truncated_transcript"
                results.append(
                    WordTimestampResult(
                        output=output,
                        words=[]
                        if output.outputs[0].finish_reason == "length"
                        else [
                            TranscriptionWord.model_validate(w) for w in reply["words"]
                        ],
                        error=error,
                    )
                )
            succeeded = True
            return results
        finally:
            try:
                llm.collective_rpc("release_audio_alignment", args=(handles,))
            except Exception:
                if succeeded:
                    raise
                logger.exception(
                    "Failed to release captures after offline alignment failed"
                )
