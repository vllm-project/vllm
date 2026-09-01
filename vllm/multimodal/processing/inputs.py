# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from dataclasses import dataclass, field

from vllm.config.multimodal import MMHasherAlgorithm
from vllm.inputs import MultiModalHashes

from ..hasher import MultiModalHasher
from ..parse import MultiModalDataItems, MultiModalUUIDItems


@dataclass
class ProcessorInputs:
    """
    Represents the keyword arguments to
    [`vllm.multimodal.processing.BaseMultiModalProcessor.apply`][].
    """

    prompt: list[int]
    mm_data_items: MultiModalDataItems
    mm_uuid_items: MultiModalUUIDItems | None = None
    hf_processor_mm_kwargs: Mapping[str, object] = field(default_factory=dict)
    media_io_kwargs: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def get_mm_hashes(
        self,
        model_id: str,
        hash_algorithm: MMHasherAlgorithm,
    ) -> MultiModalHashes:
        mm_data_items = self.mm_data_items
        mm_uuid_items = self.mm_uuid_items or {}

        hash_factors = {
            "media_io_kwargs": dict(self.media_io_kwargs),
            "mm_processor_kwargs": dict(self.hf_processor_mm_kwargs),
        }
        has_hash_factors = any(hash_factors.values())
        hash_factor_kwargs = hash_factors if has_hash_factors else {}

        mm_hashes = dict[str, list[str]]()
        hasher = MultiModalHasher

        for modality, data_items in mm_data_items.items():
            uuid_items = (
                mm_uuid_items[modality]
                if modality in mm_uuid_items
                else ([None] * len(data_items))
            )

            # For None entries, compute a hash; otherwise, use provided ID.
            hashes: list[str] = []
            for i, item in enumerate(data_items.get_all_items_for_hash()):
                uuid_item = uuid_items[i]

                # NOTE: Even if a uuid_item is provided, model output
                # depends on hash_factor_kwargs, so they are taken into account.
                if uuid_item is None or has_hash_factors:
                    # NOTE: use provided hash string to hash with kwargs
                    # if available for better performance.
                    item = uuid_item if uuid_item is not None else item
                    hashes.append(
                        hasher.hash_kwargs(
                            hash_algorithm,
                            model_id=model_id,
                            **{modality: item},
                            **hash_factor_kwargs,
                        )
                    )
                else:
                    # If there are no extra kwargs, use the client-provided UUID.
                    hashes.append(uuid_item)

            mm_hashes[modality] = hashes

        return mm_hashes
