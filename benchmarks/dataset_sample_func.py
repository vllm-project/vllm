import base64
import io
from abc import ABC, abstractmethod
from typing import Collection, Optional

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


class HFDatasetSampler(ABC):

    def __init__(self, dataset: IterableDataset, seed: Optional[int] = None):
        self.dataset = dataset.shuffle(seed=seed).filter(self.filter_func)

    @abstractmethod
    def filter_func(self, data: dict) -> bool:
        return True

    @abstractmethod
    def sample(
        self,
        num_samples: int,
        tokenizer: PreTrainedTokenizerBase,
        fixed_output_len: Optional[int] = None
    ) -> list[tuple[str, int, int, dict[str, Collection[str]]]]:
        raise NotImplementedError


class SonnetSampler(HFDatasetSampler):

    def __init__(self, dataset: IterableDataset, seed: Optional[int] = None):
        assert "conversations" in dataset.features, (
            "Sonnet-style Dataset must have 'conversations' column.")
        super().__init__(dataset, seed=seed)

    def filter_func(self, data: dict) -> bool:
        return len(data["conversations"]) >= 2

    def _get_mm_content(self,
                        data: dict) -> Optional[dict[str, Collection[str]]]:
        if "image" in data and isinstance(data["image"], Image):
            return pil_image_to_mm_content(data["image"])
        elif "image" in data and isinstance(data["image"], str):
            return image_url_to_mm_content(data["image"])
        return None

    def sample(
        self,
        num_samples: int,
        tokenizer: PreTrainedTokenizerBase,
        fixed_output_len: Optional[int] = 128,
    ) -> list[tuple[str, int, int, dict[str, Collection[str]]]]:
        sampled_requests: list[tuple[str, int, int,
                                     dict[str, Collection[str]]]] = []
        for data in self.dataset:
            if len(sampled_requests) == num_samples:
                break

            # Tokenize the prompts and completions.
            prompt = data["conversations"][0]["value"]
            prompt_token_ids = tokenizer(prompt).input_ids
            completion = data["conversations"][1]["value"]
            completion_token_ids = tokenizer(completion).input_ids
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


class VisionArenaBenchSampler(HFDatasetSampler):

    def sample(
        self,
        num_samples: int,
        tokenizer: PreTrainedTokenizerBase,
        fixed_output_len: Optional[int] = 128,
    ):
        sampled_requests: list[tuple[str, int, int,
                                     dict[str, Collection[str]]]] = []
        for data in self.dataset:
            if len(sampled_requests) == num_samples:
                break

            prompt = data["turns"][0][0]['content']
            prompt_token_ids = tokenizer(prompt).input_ids

            prompt_len = len(prompt_token_ids)
            output_len = fixed_output_len

            # lmarena-ai/vision-arena-bench-v0.1 always has an image.
            mm_content = pil_image_to_mm_content(data["images"][0])

            sampled_requests.append(
                (prompt, prompt_len, output_len, mm_content))

        return sampled_requests


DATASET_SAMPLE_FUNC: dict[str, HFDatasetSampler] = {
    "lmarena-ai/vision-arena-bench-v0.1": VisionArenaBenchSampler,
}


def get_hf_dataset_sampler(
    dataset_path: str,
    dataset_subset: Optional[str],
    dataset_split: str,
    seed: Optional[int] = None,
) -> HFDatasetSampler:
    ds_builder = load_dataset_builder(dataset_path, name=dataset_subset)
    ds_info = ds_builder.info
    assert dataset_split in ds_info.splits, (
        f"Split '{dataset_split}' not found in dataset '{dataset_path}'")
    dataset = load_dataset(dataset_path,
                           name=dataset_subset,
                           split=dataset_split)

    if dataset_path in DATASET_SAMPLE_FUNC:
        return DATASET_SAMPLE_FUNC[dataset_path](dataset, seed=seed)
    else:
        return SonnetSampler(dataset, seed=seed)
