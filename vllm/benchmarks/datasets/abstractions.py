# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any

from transformers import PreTrainedTokenizerBase

from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalDataDict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    prompt: str | list[str] | list[dict]
    prompt_len: int
    expected_output_len: int | None
    multi_modal_data: MultiModalDataDict | dict | list[dict] | None = None
    lora_request: LoRARequest | None = None
    request_id: str | None = None


class BenchmarkDataset(ABC):
    DEFAULT_SEED = 0
    IS_MULTIMODAL = False

    def __init__(
        self,
        dataset_path: str | None = None,
        random_seed: int = DEFAULT_SEED,
        disable_shuffle: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the BenchmarkDataset with an optional dataset path and random
        seed.

        Args:
            dataset_path (Optional[str]): Path to the dataset. If None, it
                indicates that a default or random dataset might be used.
            random_seed (int): Seed value for reproducible shuffling or
                sampling. Defaults to DEFAULT_SEED.
        """
        self.dataset_path = dataset_path
        # Set the random seed, ensuring that a None value is replaced with the
        # default seed.
        self.random_seed = random_seed if random_seed is not None else self.DEFAULT_SEED
        self.disable_shuffle = disable_shuffle
        self.data: Any

    def load_data(self) -> None:
        """
        Load data from the dataset path into self.data.

        This method must be overridden by subclasses since the method to load
        data will vary depending on the dataset format and source.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        # TODO (jenniferzhao): add support for downloading data
        raise NotImplementedError("load_data must be implemented in subclasses.")

    @abstractmethod
    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list[SampleRequest]:
        """
        Abstract method to generate sample requests from the dataset.

        Subclasses must override this method to implement dataset-specific logic
        for generating a list of SampleRequest objects.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used
                for processing the dataset's text.
            num_requests (int): The number of sample requests to generate.
            request_id_prefix (str): The prefix of request_id.

        Returns:
            list[SampleRequest]: A list of sample requests generated from the
            dataset.
        """
        raise NotImplementedError("sample must be implemented in subclasses.")

    def maybe_oversample_requests(
        self,
        requests: list[SampleRequest],
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
    ) -> None:
        """
        Oversamples the list of requests if its size is less than the desired
        number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
                requests.
            num_requests (int): The target number of requests.
            request_id_prefix (str): The prefix applied to generated request
                identifiers.

        """
        if no_oversample:
            logger.info("Skipping oversampling. Total samples: %d.", len(requests))
            return

        if len(requests) < num_requests:
            random.seed(self.random_seed)
            needed = num_requests - len(requests)
            additional = []
            for i in range(needed):
                req = replace(
                    random.choice(requests),
                    request_id=request_id_prefix + str(len(requests) + i),
                )
                additional.append(req)
            requests.extend(additional)
            logger.info("Oversampled requests to reach %d total samples.", num_requests)

        ids = [req.request_id for req in requests]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate request_id found in the sampled "
                "requests. Please ensure that each request_id "
                "is unique."
            )
