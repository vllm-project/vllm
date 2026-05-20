# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import chain

from vllm.inputs import MultiModalHashes

from ..hasher import MultiModalHasher
from ..parse import MultiModalDataItems, MultiModalUUIDItems


@dataclass
class ProcessorInputs:
    """
    Represents the keyword arguments to
    [`vllm.multimodal.processing.BaseMultiModalProcessor.apply`][].
    """

    prompt: str | list[int]
    mm_data_items: MultiModalDataItems
    mm_uuid_items: MultiModalUUIDItems | None = None
    hf_processor_mm_kwargs: Mapping[str, object] = field(default_factory=dict)
    tokenization_kwargs: Mapping[str, object] = field(default_factory=dict)

    def get_mm_hashes(self, model_id: str) -> MultiModalHashes:
        mm_data_items = self.mm_data_items
        mm_uuid_items = self.mm_uuid_items or {}
        hf_processor_mm_kwargs = self.hf_processor_mm_kwargs

        mm_hashes = dict[str, list[str]]()
        hasher = MultiModalHasher
        shared_items = [("model_id", model_id), *hf_processor_mm_kwargs.items()]
        shared_items.sort(key=lambda kv: kv[0])
        shared_keys = [key for key, _ in shared_items]
        shared_key_set = set(shared_keys)

        for modality, data_items in mm_data_items.items():
            if modality in shared_key_set:

                def hash_item(item_to_hash: object, modality: str = modality) -> str:
                    # Preserve dict unpack overwrite semantics in the unlikely
                    # case of a key collision with processor kwargs.
                    return hasher.hash_kwargs(
                        model_id=model_id,
                        **{modality: item_to_hash},
                        **hf_processor_mm_kwargs,
                    )

            else:
                insert_idx = bisect_left(shared_keys, modality)
                prefix = shared_items[:insert_idx]
                suffix = shared_items[insert_idx:]

                def hash_item(
                    item_to_hash: object,
                    prefix: list[tuple[str, object]] = prefix,
                    modality: str = modality,
                    suffix: list[tuple[str, object]] = suffix,
                ) -> str:
                    return hasher.hash_ordered_items(
                        chain(prefix, ((modality, item_to_hash),), suffix)
                    )

            if modality in mm_uuid_items:
                uuid_items = mm_uuid_items[modality]

                # For None entries, compute a hash; otherwise, use provided ID.
                hashes: list[str] = []
                for i, item in enumerate(data_items.get_all_items_for_hash()):
                    uuid_item = uuid_items[i]

                    # NOTE: Even if a uuid_item is provided, we still compute a hash
                    # if `hf_processor_mm_kwargs` is provided.
                    # This is because the processed multimodal inputs can be different
                    # depending on the processor kwargs.
                    if uuid_item is None or hf_processor_mm_kwargs:
                        # NOTE: use provided hash string to hash with kwargs
                        # if available for better performance.
                        item = uuid_item if uuid_item is not None else item
                        hashes.append(hash_item(item))
                    else:
                        hashes.append(uuid_item)

                mm_hashes[modality] = hashes
            else:
                mm_hashes[modality] = [hash_item(item) for item in data_items]

        return mm_hashes
