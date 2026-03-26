# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Mapping
from itertools import groupby
from typing import Protocol

from .inputs import MultiModalFieldConfig, MultiModalKwargsItems, NestedTensors
from .utils import group_and_batch_mm_items


class MultiModalBatcher(Protocol):
    """
    Interface defining how to batch and unbatch model inputs.

    The following invariants must hold:
    ```
    for modality, num_items, batch in group_batches(x):
        assert len(split_batch(batch)) == num_items

        subbatches = list(group_batches(batch))
        assert len(subbatches) == 1

        m, n, b = subbatches[0]
        assert (m, n) == (modality, num_items)
        assert nested_tensors_equal(b, batch)
    ```
    """

    def group_batches(
        self,
        items: Iterable[dict[str, NestedTensors]],
    ) -> Iterable[tuple[str, int, dict[str, NestedTensors]]]:
        """
        Group an iterator of model inputs into batches.

        Args:
            items: An iterator of model inputs. Each element may contain the inputs
              for one or more multi-modal items.

        Yields:
            A tuple `(modality, num_items, batch)`, where:
            - `modality` is the modality of the batch;
            - `num_items` is the corresponding number of multi-modal items;
            - `batch` is a dictionary containing model inputs.
        """
        raise NotImplementedError

    def split_batch(
        self,
        batch: dict[str, NestedTensors],
    ) -> list[dict[str, NestedTensors]]:
        """
        Split model inputs for a single batch into
        one set of model inputs for each multi-modal item in the batch.

        Args:
            batch: A dictionary containing model inputs
              for one or more multi-modal items.

        Returns:
            A list of model inputs. Each element contains the inputs
            for just one multi-modal item.
        """
        raise NotImplementedError


class MultiModalFieldBatcher(MultiModalBatcher):
    def __init__(self, config_by_key: Mapping[str, MultiModalFieldConfig]) -> None:
        super().__init__()

        self.config_by_key = config_by_key

    def group_batches(
        self,
        items: Iterable[dict[str, NestedTensors]],
    ) -> Iterable[tuple[str, int, dict[str, NestedTensors]]]:
        config_by_key = self.config_by_key

        it = (
            (modality, parsed_item)
            for item in items
            for modality, parsed_items in MultiModalKwargsItems.from_hf_inputs(
                item, config_by_key
            ).items()
            for parsed_item in parsed_items
        )

        for modality, group in groupby(it, key=lambda x: x[0]):
            parsed_items = [item for _, item in group]
            for num_items, batch in group_and_batch_mm_items(parsed_items):
                yield modality, num_items, batch

    def split_batch(
        self,
        batch: dict[str, NestedTensors],
    ) -> list[dict[str, NestedTensors]]:
        config_by_key = self.config_by_key
        mm_parsed_items = MultiModalKwargsItems.from_hf_inputs(batch, config_by_key)

        return [
            parsed_item.get_data()
            for parsed_items in mm_parsed_items.values()
            for parsed_item in parsed_items
        ]
