# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from collections.abc import Callable

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from vllm.benchmarks.datasets.abstractions import BenchmarkDataset, SampleRequest
from vllm.benchmarks.datasets.utils import (
    apply_multimodal_chat_transformation,
    is_valid_sequence,
    process_image,
    process_video,
)
from vllm.utils.import_utils import PlaceholderModule

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")

logger = logging.getLogger(__name__)


class HuggingFaceDataset(BenchmarkDataset):
    """Base class for datasets hosted on HuggingFace."""

    SUPPORTED_DATASET_PATHS: set[str] | dict[str, Callable] = set()

    def __init__(
        self,
        dataset_path: str,
        dataset_split: str,
        no_stream: bool = False,
        dataset_subset: str | None = None,
        hf_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(dataset_path=dataset_path, **kwargs)

        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset
        self.load_stream = not no_stream
        self.hf_name = hf_name or dataset_path
        self.data: Dataset = self.load_data()

    def load_data(self) -> Dataset:
        """Load data from HuggingFace datasets."""
        data = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=self.load_stream,
        )
        if not getattr(self, "disable_shuffle", False):
            data = data.shuffle(seed=self.random_seed)
        return data


# -----------------------------------------------------------------------------
# Conversation Dataset Implementation
# -----------------------------------------------------------------------------


class ConversationDataset(HuggingFaceDataset):
    """Dataset for text-only conversation data."""

    SUPPORTED_DATASET_PATHS = {
        "Aeala/ShareGPT_Vicuna_unfiltered",
    }
    IS_MULTIMODAL = False

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list[SampleRequest]:
        # Filter examples with at least 2 conversations
        filtered_data = self.data.filter(lambda x: len(x["conversations"]) >= 2)
        sampled_requests: list[SampleRequest] = []
        ind = 0
        dynamic_output = output_len is None

        for item in filtered_data:
            if len(sampled_requests) >= num_requests:
                break
            conv = item["conversations"]
            prompt, completion = conv[0]["value"], conv[1]["value"]

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)
            output_len = completion_len if dynamic_output else output_len
            assert isinstance(output_len, int) and output_len > 0
            if dynamic_output and not is_valid_sequence(prompt_len, completion_len):
                continue
            mm_content = process_image(item["image"]) if "image" in item else None
            if enable_multimodal_chat:
                # Note: when chat is enabled the request prompt_len is no longer
                # accurate and we will be using request output to count the
                # actual prompt len and output len
                prompt = apply_multimodal_chat_transformation(prompt, mm_content)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                    request_id=request_id_prefix + str(ind),
                )
            )
            ind += 1
        self.maybe_oversample_requests(
            sampled_requests, num_requests, request_id_prefix, no_oversample
        )
        return sampled_requests


class MultiModalConversationDataset(HuggingFaceDataset):
    """Dataset for multimodal conversation data."""

    SUPPORTED_DATASET_PATHS = {
        "lmms-lab/LLaVA-OneVision-Data",
    }
    IS_MULTIMODAL = True

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list[SampleRequest]:
        # Filter examples with at least 2 conversations
        filtered_data = self.data.filter(lambda x: len(x["conversations"]) >= 2)
        sampled_requests: list[SampleRequest] = []
        ind = 0
        dynamic_output = output_len is None

        for item in filtered_data:
            if len(sampled_requests) >= num_requests:
                break
            conv = item["conversations"]
            prompt, completion = conv[0]["value"], conv[1]["value"]

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)
            output_len = completion_len if dynamic_output else output_len
            assert isinstance(output_len, int) and output_len > 0
            if dynamic_output and not is_valid_sequence(prompt_len, completion_len):
                continue
            mm_content = process_image(item["image"]) if "image" in item else None
            if enable_multimodal_chat:
                # Note: when chat is enabled the request prompt_len is no longer
                # accurate and we will be using request output to count the
                # actual prompt len and output len
                prompt = apply_multimodal_chat_transformation(prompt, mm_content)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                    request_id=request_id_prefix + str(ind),
                )
            )
            ind += 1
        self.maybe_oversample_requests(
            sampled_requests, num_requests, request_id_prefix, no_oversample
        )
        return sampled_requests


# -----------------------------------------------------------------------------
# Vision Arena Dataset Implementation
# -----------------------------------------------------------------------------


class VisionArenaDataset(HuggingFaceDataset):
    """
    Vision Arena Dataset.
    """

    DEFAULT_OUTPUT_LEN = 128
    SUPPORTED_DATASET_PATHS = {
        "lmarena-ai/VisionArena-Chat": lambda x: x["conversation"][0][0]["content"],
        "lmarena-ai/vision-arena-bench-v0.1": lambda x: x["turns"][0][0]["content"],
    }
    IS_MULTIMODAL = True

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list[SampleRequest]:
        output_len = output_len if output_len is not None else self.DEFAULT_OUTPUT_LEN
        sampled_requests: list[SampleRequest] = []
        for i, item in enumerate(self.data):
            if len(sampled_requests) >= num_requests:
                break
            parser_fn = self.SUPPORTED_DATASET_PATHS.get(self.hf_name)
            if parser_fn is None:
                raise ValueError(f"Unsupported dataset path: {self.hf_name}")
            prompt = parser_fn(item)
            mm_content = process_image(item["images"][0])
            prompt_len = len(tokenizer(prompt).input_ids)
            if enable_multimodal_chat:
                # Note: when chat is enabled the request prompt_len is no longer
                # accurate and we will be using request output to count the
                # actual prompt len
                prompt = apply_multimodal_chat_transformation(prompt, mm_content)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                    request_id=request_id_prefix + str(i),
                )
            )
        self.maybe_oversample_requests(
            sampled_requests, num_requests, request_id_prefix, no_oversample
        )
        return sampled_requests


class MMVUDataset(HuggingFaceDataset):
    """
    MMVU Dataset.
    https://huggingface.co/datasets/yale-nlp/MMVU
    """

    DEFAULT_OUTPUT_LEN = 128
    SUPPORTED_DATASET_PATHS = {
        "yale-nlp/MMVU": lambda x: x["question"]
        + " "
        + (" ".join(f"{k}.{v}" for k, v in x["choices"].items())),
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list[SampleRequest]:
        output_len = output_len if output_len is not None else self.DEFAULT_OUTPUT_LEN
        sampled_requests: list[SampleRequest] = []
        for i, item in enumerate(self.data):
            if len(sampled_requests) >= num_requests:
                break
            parser_fn = self.SUPPORTED_DATASET_PATHS.get(self.hf_name)
            if parser_fn is None:
                raise ValueError(f"Unsupported dataset path: {self.hf_name}")
            prompt = parser_fn(item)
            mm_content = process_video(item["video"])
            prompt_len = len(tokenizer(prompt).input_ids)
            if enable_multimodal_chat:
                # Note: when chat is enabled the request prompt_len is no longer
                # accurate and we will be using request output to count the
                # actual prompt len
                prompt = apply_multimodal_chat_transformation(prompt, mm_content)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                    request_id=request_id_prefix + str(i),
                )
            )
        self.maybe_oversample_requests(
            sampled_requests, num_requests, request_id_prefix, no_oversample
        )
        return sampled_requests


# -----------------------------------------------------------------------------
# Instruct Coder Dataset Implementation
# -----------------------------------------------------------------------------


class InstructCoderDataset(HuggingFaceDataset):
    """
    InstructCoder Dataset.
    https://huggingface.co/datasets/likaixin/InstructCoder

    InstructCoder is the dataset designed for general code editing.  It consists
    of 114,239 instruction-input-output triplets, and covers multiple distinct
    code editing scenario.
    """

    DEFAULT_OUTPUT_LEN = 200  # this is the average default output length
    SUPPORTED_DATASET_PATHS = {
        "likaixin/InstructCoder",
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        skip_chat_template: bool = False,
        **kwargs,
    ) -> list[SampleRequest]:
        output_len = output_len if output_len is not None else self.DEFAULT_OUTPUT_LEN
        sampled_requests: list[SampleRequest] = []
        for i, item in enumerate(self.data):
            if len(sampled_requests) >= num_requests:
                break
            prompt = (
                f"{item['input']}\n\n{item['instruction']} Just output "
                "the code, do not include any explanation."
            )

            # apply template
            if not skip_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )

            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    request_id=request_id_prefix + str(i),
                )
            )
        self.maybe_oversample_requests(
            sampled_requests, num_requests, request_id_prefix, no_oversample
        )
        return sampled_requests


# -----------------------------------------------------------------------------
# MT-Bench Dataset Implementation
# -----------------------------------------------------------------------------


class MTBenchDataset(HuggingFaceDataset):
    """
    MT-Bench Dataset.
    https://huggingface.co/datasets/philschmid/mt-bench

    We create a single turn dataset for MT-Bench.
    This is similar to Spec decoding benchmark setup in vLLM
    https://github.com/vllm-project/vllm/blob/9d98ab5ec/examples/offline_inference/eagle.py#L14-L18
    """  # noqa: E501

    DEFAULT_OUTPUT_LEN = 256  # avg len used in SD bench in vLLM
    SUPPORTED_DATASET_PATHS = {
        "philschmid/mt-bench",
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        skip_chat_template: bool = False,
        **kwargs,
    ) -> list[SampleRequest]:
        output_len = output_len if output_len is not None else self.DEFAULT_OUTPUT_LEN
        sampled_requests: list[SampleRequest] = []

        for i, item in enumerate(self.data):
            if len(sampled_requests) >= num_requests:
                break
            prompt = item["turns"][0]

            # apply template
            if not skip_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )

            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    request_id=request_id_prefix + str(i),
                )
            )
        self.maybe_oversample_requests(
            sampled_requests, num_requests, request_id_prefix, no_oversample
        )
        return sampled_requests


# -----------------------------------------------------------------------------
# Blazedit Dataset Implementation
# -----------------------------------------------------------------------------


class BlazeditDataset(HuggingFaceDataset):
    """
    Blazedit Dataset.
    https://github.com/ise-uiuc/blazedit

    5k char version: vdaita/edit_5k_char
    10k char version: vdaita/edit_10k_char
    """  # noqa: E501

    # 5k char version will have output as ~5k chars
    # 10k char version will have output as ~10k chars
    # Assuming 3 char per token, 10k chars will be 3333 tokens
    # We set default to 4000 to be safe
    DEFAULT_OUTPUT_LEN = 4000
    SUPPORTED_DATASET_PATHS = {
        "vdaita/edit_5k_char",
        "vdaita/edit_10k_char",
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        output_len: int | None = None,
        skip_chat_template: bool = False,
        min_distance: float = 0.0,
        max_distance: float = 1.0,
        **kwargs,
    ) -> list[SampleRequest]:
        output_len = output_len if output_len is not None else self.DEFAULT_OUTPUT_LEN
        sampled_requests: list[SampleRequest] = []

        for i, item in enumerate(self.data):
            if len(sampled_requests) >= num_requests:
                break
            code = item["code"]
            change_request = item["change_request"]
            norm_distance = item["norm_distance"]

            # compare the levenshtein distance normalized by code length
            if norm_distance < min_distance or norm_distance > max_distance:
                continue

            # template copied from
            # https://github.com/ise-uiuc/blazedit/blob/7765137e656fd62de877422d2e4cf8de51228054/dataset/create_refined_dataset.py#L94-L105 # noqa: E501
            prompt = f"""Given a code file, please apply the change requests and generate the new file.

Original file:
```python
{code}
```

Change request:
{change_request}

Please generate the new code file in the "New file" section below."""  # noqa: E501

            # apply template
            if not skip_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )

            prompt_len = len(tokenizer(prompt).input_ids)

            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    request_id=request_id_prefix + str(i),
                )
            )
        self.maybe_oversample_requests(
            sampled_requests, num_requests, request_id_prefix, no_oversample
        )

        return sampled_requests


# -----------------------------------------------------------------------------
# AIMO Dataset Implementation
# -----------------------------------------------------------------------------


class AIMODataset(HuggingFaceDataset):
    """
    Dataset class for processing a AIMO dataset with reasoning questions.
    """

    SUPPORTED_DATASET_PATHS = {
        "AI-MO/aimo-validation-aime",
        "AI-MO/NuminaMath-1.5",
        "AI-MO/NuminaMath-CoT",
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        output_len: int | None = None,
        **kwargs,
    ) -> list[SampleRequest]:
        sampled_requests: list[SampleRequest] = []
        ind = 0
        dynamic_output = output_len is None

        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            prompt, completion = item["problem"], item["solution"]

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)
            output_len = completion_len if dynamic_output else output_len
            assert isinstance(output_len, int) and output_len > 0
            if dynamic_output and not is_valid_sequence(
                prompt_len, completion_len, max_prompt_len=2048, max_total_len=32000
            ):
                continue
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=None,
                    request_id=request_id_prefix + str(ind),
                )
            )
            ind += 1
        self.maybe_oversample_requests(
            sampled_requests, num_requests, request_id_prefix, no_oversample
        )
        return sampled_requests


# -----------------------------------------------------------------------------
# Next Edit Prediction Dataset Implementation
# -----------------------------------------------------------------------------


zeta_prompt = """### Instruction:
You are a code completion assistant and your task is to analyze user edits and then rewrite an excerpt that the user provides, suggesting the appropriate edits within the excerpt, taking into account the cursor location.

### User Edits:

{}

### User Excerpt:

{}

### Response:

"""  # noqa: E501


def _format_zeta_prompt(
    sample: dict, original_start_marker: str = "<|editable_region_start|>"
) -> dict:
    """Format the zeta prompt for the Next Edit Prediction (NEP) dataset.

    This function formats examples from the NEP dataset
    into prompts and expected outputs. It could be
    further extended to support more NEP datasets.

    Args:
        sample: The dataset sample containing events,
            inputs, and outputs.
        original_start_marker: The marker indicating the
            start of the editable region. Defaults to
            "<|editable_region_start|>".

    Returns:
        A dictionary with the formatted prompts and expected outputs.
    """
    events = sample["events"]
    input = sample["input"]
    output = sample["output"]
    prompt = zeta_prompt.format(events, input)

    # following the original implementation, extract the focused region
    # from the raw output
    output_start_index = output.find(original_start_marker)
    output_focused_region = output[output_start_index:]
    expected_output = output_focused_region

    return {"prompt": prompt, "expected_output": expected_output}


class NextEditPredictionDataset(HuggingFaceDataset):
    """
    Dataset class for processing a Next Edit Prediction dataset.
    """

    SUPPORTED_DATASET_PATHS = {
        "zed-industries/zeta",
    }
    MAPPING_PROMPT_FUNCS = {
        "zed-industries/zeta": _format_zeta_prompt,
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ):
        formatting_prompt_func = self.MAPPING_PROMPT_FUNCS.get(self.hf_name)
        if formatting_prompt_func is None:
            raise ValueError(f"Unsupported dataset path: {self.hf_name}")
        samples = []
        for i, sample in enumerate(self.data):
            sample = formatting_prompt_func(sample)
            samples.append(
                SampleRequest(
                    prompt=sample["prompt"],
                    prompt_len=len(tokenizer(sample["prompt"]).input_ids),
                    expected_output_len=len(
                        tokenizer(sample["expected_output"]).input_ids
                    ),
                    request_id=request_id_prefix + str(i),
                )
            )
            if len(samples) >= num_requests:
                break
        self.maybe_oversample_requests(
            samples, num_requests, request_id_prefix, no_oversample
        )
        return samples


# -----------------------------------------------------------------------------
# ASR Dataset Implementation
# -----------------------------------------------------------------------------


class ASRDataset(HuggingFaceDataset):
    """
    Dataset class for processing a ASR dataset for transcription.
    Tested on the following set:

    +----------------+----------------------------------------+--------------------------+-----------------------------+
    | Dataset        | Domain                                 | Speaking Style           | hf-subset                   |
    +----------------+----------------------------------------+--------------------------+-----------------------------+
    | TED-LIUM       | TED talks                              | Oratory                  | release1, release2, release3|
    |                |                                        |                          | release3-speaker-adaptation |
    | VoxPopuli      | European Parliament                    | Oratory                  | en, de, it, fr,  ...        |
    | LibriSpeech    | Audiobook                              | Narrated                 | "LIUM/tedlium"              |
    | GigaSpeech     | Audiobook, podcast, YouTube            | Narrated, spontaneous    | xs, s, m, l, xl, dev, test  |
    | SPGISpeech     | Financial meetings                     | Oratory, spontaneous     | S, M, L, dev, test          |
    | AMI            | Meetings                               | Spontaneous              | ihm, sdm                    |
    +----------------+----------------------------------------+--------------------------+-----------------------------+

    """  # noqa: E501

    SUPPORTED_DATASET_PATHS = {
        "openslr/librispeech_asr",
        "facebook/voxpopuli",
        "LIUM/tedlium",
        "edinburghcstr/ami",
        "speechcolab/gigaspeech",
        "kensho/spgispeech",
    }

    DEFAULT_OUTPUT_LEN = 128
    IS_MULTIMODAL = True

    # TODO Whisper-specific. Abstract interface when more models are supported.
    TRANSCRIPTION_PREAMBLE = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    skip_long_audios: bool = True

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        output_len: int | None = None,
        **kwargs,
    ) -> list[SampleRequest]:
        output_len = output_len if output_len is not None else self.DEFAULT_OUTPUT_LEN
        prompt = ASRDataset.TRANSCRIPTION_PREAMBLE
        prompt_len = len(tokenizer(prompt).input_ids)
        sampled_requests: list[SampleRequest] = []
        ind = 0
        skipped = 0
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            audio = item["audio"]
            y, sr = audio["array"], audio["sampling_rate"]
            duration_s = librosa.get_duration(y=y, sr=sr)
            # Whisper max supported duration
            if self.skip_long_audios and duration_s > 30:
                skipped += 1
                continue

            mm_content = {"audio": (y, sr)}
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                    request_id=request_id_prefix + str(ind),
                )
            )
            ind += 1
        if skipped:
            logger.warning(
                "%d samples discarded from dataset due to"
                " their length being greater than"
                " what Whisper supports.",
                skipped,
            )
        self.maybe_oversample_requests(
            sampled_requests, num_requests, request_id_prefix, no_oversample
        )
        return sampled_requests


# -----------------------------------------------------------------------------
# MLPerf Dataset Implementation
# -----------------------------------------------------------------------------


class MLPerfDataset(HuggingFaceDataset):
    """
    MLPerf Inference Dataset.

    Dataset on HF:
    https://huggingface.co/datasets/mgoin/mlperf-inference-llama2-data
    https://huggingface.co/datasets/mgoin/mlperf-inference-llama3.1-data

    Each record contains:
      - "system_prompt": system role instruction.
      - "question": user question.
      - "output": reference answer.

    We combine the system prompt and question into a chat-formatted prompt
    (using the tokenizer's chat template) and set the expected output length to
    the tokenized length of the provided reference answer.
    """

    SUPPORTED_DATASET_PATHS = {
        "mgoin/mlperf-inference-llama2-data",
        "mgoin/mlperf-inference-llama3.1-data",
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        output_len: int | None = None,
        **kwargs,
    ) -> list[SampleRequest]:
        # Force dynamic output length based on reference completion.
        dynamic_output = output_len is None
        sampled_requests: list[SampleRequest] = []
        ind = 0

        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break

            system_prompt = item["system_prompt"]
            question = item["question"]
            reference_answer = item["output"]

            # Build chat-style prompt using tokenizer template, if available.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            prompt_formatted = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompt_len = len(tokenizer(prompt_formatted).input_ids)

            # Determine output length from reference answer tokens.
            ref_out_len = len(
                tokenizer(reference_answer, add_special_tokens=False).input_ids
            )
            expected_output_len = ref_out_len if dynamic_output else output_len

            # Validate sequence lengths.
            if not is_valid_sequence(prompt_len, expected_output_len or 0):
                continue

            sampled_requests.append(
                SampleRequest(
                    prompt=prompt_formatted,
                    prompt_len=prompt_len,
                    expected_output_len=expected_output_len,
                    request_id=request_id_prefix + str(ind),
                )
            )
            ind += 1

        self.maybe_oversample_requests(
            sampled_requests, num_requests, request_id_prefix, no_oversample
        )
        return sampled_requests


# -----------------------------------------------------------------------------
# MMStar Dataset Implementation
# -----------------------------------------------------------------------------


class MMStarDataset(HuggingFaceDataset):
    """
    Lin-Chen/MMStar: https://huggingface.co/datasets/Lin-Chen/MMStar
    refer to: https://github.com/sgl-project/SpecForge/pull/106
    """

    DEFAULT_OUTPUT_LEN = 128
    SUPPORTED_DATASET_PATHS = {"Lin-Chen/MMStar"}
    IS_MULTIMODAL = True

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list[SampleRequest]:
        # If --hf-output-len is not set, use the default output length.
        output_len = output_len if output_len is not None else self.DEFAULT_OUTPUT_LEN
        sampled_requests: list[SampleRequest] = []

        for ind, item in enumerate(self.data):
            if len(sampled_requests) >= num_requests:
                break
            # Split the question text from options
            # (keep only the part before "Options:").
            full_q: str = item.get("question", "")
            question_text = full_q.split("Options:", 1)[0].strip()

            # Multimodal image content.
            mm_content = process_image(item["image"])

            # Compute prompt token length (note: this is plain text length
            # if enable_multimodal_chat is False).
            prompt_len = len(tokenizer(question_text).input_ids)

            prompt: str | list[dict]
            if enable_multimodal_chat:
                # If multimodal content should be embedded in the chat message,
                # convert to [{"role":"user","content":[...]}]
                prompt = apply_multimodal_chat_transformation(question_text, mm_content)
                mm_for_request = None  # Already embedded in chat content.
            else:
                # Default: prompt is plain text,
                # image is in mm_content for the bench to assemble.
                prompt = question_text
                mm_for_request = mm_content

            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_for_request,
                    request_id=request_id_prefix + str(ind),
                )
            )

        self.maybe_oversample_requests(
            sampled_requests, num_requests, request_id_prefix, no_oversample
        )
        return sampled_requests
