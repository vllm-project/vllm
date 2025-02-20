# SPDX-License-Identifier: Apache-2.0
import base64
import io
import json
import random
from abc import ABC, abstractmethod
from typing import Any, Collection, Iterable, Optional

import numpy as np
import pandas as pd
from datasets import IterableDataset, load_dataset, load_dataset_builder
from PIL import Image
from transformers import PreTrainedTokenizerBase


def pil_image_to_mm_content(image: Image.Image):
    image = image.convert("RGB")
    image_data = io.BytesIO()
    image.save(image_data, format='JPEG')
    image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
    mm_content = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image_base64}"
        },
    }
    return mm_content


def image_url_to_mm_content(image_url: str):
    if (image_url.startswith("http://") or \
        image_url.startswith("file://")):
        image_url = image_url
    else:
        image_url = f"file://{image_url}"

    mm_content = {
        "type": "image_url",
        "image_url": {
            "url": image_url
        },
    }
    return mm_content


class DatasetSampler(ABC):

    dataset: Optional[Iterable[Any]]
    tokenizer: PreTrainedTokenizerBase

    def filter_func(self, data: dict) -> bool:
        """Filter function to filter out unsatisfied rows from dataset."""
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        num_samples: int,
        fixed_output_len: Optional[int] = None
    ) -> list[tuple[str, int, int, dict[str, Collection[str]]]]:
        """Function to sample online requests from the dataset."""
        raise NotImplementedError

    # TODO(Isotr0py): Add sample function to sample offline request.
    def sample_offline(self,
                       num_samples: int,
                       fixed_output_len: Optional[int] = None) -> list[Any]:
        """Function to sample offline requests from the dataset."""
        raise NotImplementedError


class BurstGPTSampler(DatasetSampler):
    """Dataset sampler for creating burst request."""

    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizerBase,
        seed: Optional[int] = None,
    ):
        df = pd.read_csv(dataset_path)
        self.dataset = df[self.filter_func(df)]
        self.tokenizer = tokenizer
        self.seed = seed

    def filter_func(self, data: pd.DataFrame) -> pd.DataFrame:
        return (data["Model"] == "GPT-4") & (data["Response tokens"] > 0)

    def sample(
        self,
        num_samples: int,
        fixed_output_len: Optional[int] = None,
    ) -> list[tuple[str, int, int, None]]:
        self.dataset: pd.DataFrame
        # Randomly sample num_requests from the dataset
        if num_samples <= len(self.dataset):
            gpt4_df = self.dataset.sample(n=num_samples,
                                          random_state=self.seed)
        else:
            gpt4_df = self.dataset.sample(n=num_samples,
                                          random_state=self.seed,
                                          replace=True)
        # Convert the dataframe to a list of tuples
        dataset = gpt4_df.values.tolist()
        input_requests = []
        for i in range(num_samples):
            input_len = int(dataset[i][2])
            output_len = int(dataset[i][3])
            prompt = self.tokenizer.decode([(i + j) % self.tokenizer.vocab_size
                                            for j in range(input_len)])
            input_requests.append((prompt, input_len, output_len, None))
        return input_requests


class RandomSampler(DatasetSampler):
    """Dataset sampler for creating random request."""

    def __init__(
        self,
        prefix_len: int,
        input_len: int,
        range_ratio: float,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.prefix_len = prefix_len
        self.input_len = input_len
        self.range_ratio = range_ratio
        self.tokenizer = tokenizer

    def sample(
        self,
        num_samples: int,
        fixed_output_len: Optional[int] = 128,
    ) -> list[tuple[str, int, int, dict[str, Collection[str]]]]:
        prefix_token_ids = np.random.randint(0,
                                             self.tokenizer.vocab_size,
                                             size=self.prefix_len).tolist()

        input_lens = np.random.randint(
            int(self.input_len * self.range_ratio),
            self.input_len + 1,
            size=num_samples,
        )
        output_lens = np.random.randint(
            int(fixed_output_len * self.range_ratio),
            fixed_output_len + 1,
            size=num_samples,
        )
        offsets = np.random.randint(0,
                                    self.tokenizer.vocab_size,
                                    size=num_samples)
        input_requests = []
        for i in range(num_samples):
            prompt = self.tokenizer.decode(prefix_token_ids +
                                           [(offsets[i] + i + j) %
                                            self.tokenizer.vocab_size
                                            for j in range(input_lens[i])])

            input_requests.append(
                (prompt, int(self.prefix_len + input_lens[i]),
                 int(output_lens[i]), None))
        return input_requests


class ShareGPTSampler(DatasetSampler):
    """Dataset sampler for ShareGPT local datasets."""

    def __init__(self,
                 dataset_path: str,
                 tokenizer: PreTrainedTokenizerBase,
                 seed: Optional[int] = None):
        with open(dataset_path, encoding='utf-8') as f:
            dataset = json.load(f)
        dataset = [data for data in dataset if self.filter_func(data)]
        # seed was set in main function
        random.shuffle(dataset)
        self.dataset = dataset
        self.tokenizer = tokenizer

    def filter_func(self, data: dict) -> bool:
        return len(data["conversations"]) >= 2

    def _get_mm_content(self,
                        data: dict) -> Optional[dict[str, Collection[str]]]:
        if "image" in data and isinstance(data["image"], Image.Image):
            return pil_image_to_mm_content(data["image"])
        elif "image" in data and isinstance(data["image"], str):
            return image_url_to_mm_content(data["image"])
        return None

    def sample(
        self,
        num_samples: int,
        fixed_output_len: Optional[int] = 128,
    ) -> list[tuple[str, int, int, dict[str, Collection[str]]]]:
        sampled_requests: list[tuple[str, int, int,
                                     dict[str, Collection[str]]]] = []
        for data in self.dataset:
            if len(sampled_requests) == num_samples:
                break

            # Tokenize the prompts and completions.
            prompt = data["conversations"][0]["value"]
            prompt_token_ids = self.tokenizer(prompt).input_ids
            completion = data["conversations"][1]["value"]
            completion_token_ids = self.tokenizer(completion).input_ids
            prompt_len = len(prompt_token_ids)
            output_len = len(
                completion_token_ids
            ) if fixed_output_len is None else fixed_output_len
            if fixed_output_len is None and (prompt_len < 4 or output_len < 4):
                # Prune too short sequences.
                continue
            if fixed_output_len is None and \
                (prompt_len > 1024 or prompt_len + output_len > 2048):
                # Prune too long sequences.
                continue

            mm_content = self._get_mm_content(data)

            sampled_requests.append(
                (prompt, prompt_len, output_len, mm_content))
        return sampled_requests


class SonnetSampler(DatasetSampler):
    """Dataset sampler for Sonnet local datasets."""

    def __init__(
        self,
        dataset_path: str,
        input_len: int,
        prefix_len: int,
        tokenizer: PreTrainedTokenizerBase,
    ):
        assert input_len > prefix_len, (
            "'args.sonnet-input-len' must be "
            "greater than 'args.prefix-input-len'.")

        self.tokenizer = tokenizer

        # Load the dataset.
        with open(dataset_path, encoding='utf-8') as f:
            self.dataset = f.readlines()

        # Tokenize the poem lines.
        poem_token_ids = tokenizer(self.dataset).input_ids
        average_poem_len = sum(
            len(token_ids)
            for token_ids in poem_token_ids) / len(poem_token_ids)

        # Base prefix for all requests.
        base_prompt = "Pick as many lines as you can from these poem lines:\n"
        base_message = [{
            "role": "user",
            "content": base_prompt,
        }]
        base_prompt_formatted = tokenizer.apply_chat_template(
            base_message, add_generation_prompt=True, tokenize=False)
        base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

        assert input_len > base_prompt_offset, (
            "Please set 'args.sonnet-input-len' "
            f"higher than {base_prompt_offset}.")
        num_input_lines = round(
            (input_len - base_prompt_offset) / average_poem_len)

        # First approximately `prefix_len` number of tokens in the
        # prompt are fixed poem lines.
        assert prefix_len > base_prompt_offset, (
            f"Please set 'args.sonnet-prefix-len' "
            f"higher than {base_prompt_offset}.")

        num_prefix_lines = round(
            (prefix_len - base_prompt_offset) / average_poem_len)
        num_input_lines = round(
            (input_len - base_prompt_offset) / average_poem_len)

        self.base_prompt = base_prompt
        self.num_lines_needed = num_input_lines - num_prefix_lines
        self.prefix_lines = self.dataset[:num_prefix_lines]

    def sample(
        self,
        num_samples: int,
        output_len: int = 128,
    ) -> list[tuple[str, int, int, dict[str, Collection[str]]]]:
        sampled_requests: list[tuple[str, int, int]] = []
        for _ in range(num_samples):
            sampled_lines = "".join(
                self.prefix_lines +
                random.choices(self.dataset, k=self.num_lines_needed))

            prompt = f"{self.base_prompt}{sampled_lines}"
            message = [
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            prompt_formatted = self.tokenizer.apply_chat_template(
                message, add_generation_prompt=True, tokenize=False)
            prompt_len = len(self.tokenizer(prompt_formatted).input_ids)
            sampled_requests.append(
                (prompt, prompt_formatted, prompt_len, output_len, None))

        return sampled_requests


class HFDatasetSampler(DatasetSampler):
    """Base class for sampling Hugging Face Datasets."""

    def __init__(self,
                 hf_dataset: IterableDataset,
                 tokenizer: PreTrainedTokenizerBase,
                 seed: Optional[int] = None):
        self.tokenizer = tokenizer
        self.dataset = hf_dataset.shuffle(seed=seed).filter(self.filter_func)


class ShareGPTHFSampler(ShareGPTSampler, HFDatasetSampler):
    """
    Dataset sampler for ShareGPT-style remote datasets on hf hub.
    - Text-only dataset like: 'RyokoAI/ShareGPT52K' etc.
    - Vision dataset like: 'lmms-lab/LLaVA-OneVision-Data' etc.
    """

    def __init__(self,
                 dataset: IterableDataset,
                 tokenizer: PreTrainedTokenizerBase,
                 seed: Optional[int] = None):
        assert "conversations" in dataset.features, (
            "Sonnet-style Dataset must have 'conversations' column.")
        HFDatasetSampler.__init__(self,
                                  dataset,
                                  tokenizer=tokenizer,
                                  seed=seed)


class VisionArenaBenchSampler(HFDatasetSampler):
    """Dataset sampler for 'lmarena-ai/vision-arena-bench-v0.1' dataset."""

    def filter_func(self, data: dict) -> bool:
        # vision-arena-bench always has an image and one turn conversation.
        return True

    def sample(
        self,
        num_samples: int,
        fixed_output_len: Optional[int] = 128,
    ):
        sampled_requests: list[tuple[str, int, int,
                                     dict[str, Collection[str]]]] = []
        for data in self.dataset:
            if len(sampled_requests) == num_samples:
                break

            prompt = data["turns"][0][0]['content']
            prompt_token_ids = self.tokenizer(prompt).input_ids

            prompt_len = len(prompt_token_ids)
            output_len = fixed_output_len

            # lmarena-ai/vision-arena-bench-v0.1 always has an image.
            mm_content = pil_image_to_mm_content(data["images"][0])

            sampled_requests.append(
                (prompt, prompt_len, output_len, mm_content))

        return sampled_requests


HF_DATASET_SAMPLE_FUNC: dict[str, HFDatasetSampler] = {
    "lmarena-ai/vision-arena-bench-v0.1": VisionArenaBenchSampler,
}


def get_hf_dataset_sampler(
    dataset_path: str,
    dataset_subset: Optional[str],
    dataset_split: str,
    tokenizer: PreTrainedTokenizerBase,
    seed: Optional[int] = None,
) -> HFDatasetSampler:
    ds_builder = load_dataset_builder(dataset_path, name=dataset_subset)
    ds_info = ds_builder.info
    assert dataset_split in ds_info.splits, (
        f"Split '{dataset_split}' not found in dataset '{dataset_path}'")
    dataset = load_dataset(dataset_path,
                           name=dataset_subset,
                           split=dataset_split,
                           streaming=True)

    if dataset_path in HF_DATASET_SAMPLE_FUNC:
        return HF_DATASET_SAMPLE_FUNC[dataset_path](dataset,
                                                    tokenizer=tokenizer,
                                                    seed=seed)
    else:
        return ShareGPTHFSampler(dataset, tokenizer=tokenizer, seed=seed)
