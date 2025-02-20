# SPDX-License-Identifier: Apache-2.0
import json
from typing import Any, List, Optional, Union

import msgspec


# Helper function to check if all elements in a list are integers or None
def valid_prefix_prompt_ids(lst: List[Any]) -> bool:
    return (all(isinstance(x, int) for x in lst) or all(x is None for x in lst)
            or all(
                isinstance(x, list) and all(isinstance(i, int) for i in x)
                for x in lst if x is not None)
            or all(x is None or
                   (isinstance(x, list) and all(isinstance(i, int) for i in x))
                   for x in lst))


# Helper function to check if all elements in a list are strings or None
def valid_kvcache_keys(lst: List[Any]) -> bool:
    return (all(isinstance(x, str) for x in lst) or all(x is None for x in lst)
            or all(
                isinstance(x, list) and all(isinstance(i, str) for i in x)
                for x in lst if x is not None)
            or all(x is None or
                   (isinstance(x, list) and all(isinstance(i, str) for i in x))
                   for x in lst))


def verify_input_from_optional(input_dict: dict) -> None:
    prefix_prompt_ids = input_dict.get("prefix_prompt_ids")
    if prefix_prompt_ids is not None and not all(
            isinstance(x, int) for x in prefix_prompt_ids):
        raise ValueError(
            f"prefix_prompt_ids: {prefix_prompt_ids} should be a list of "
            "integers or None.")

    kvcache_load_keys = input_dict.get("kvcache_load_keys")
    if kvcache_load_keys is not None and not all(
            isinstance(x, str) for x in kvcache_load_keys):
        raise ValueError(
            f"kvcache_load_keys: {kvcache_load_keys} should be a list of "
            "strings or None.")

    kvcache_store_keys = input_dict.get("kvcache_store_keys")
    if kvcache_store_keys is not None and not all(
            isinstance(x, str) for x in kvcache_store_keys):
        raise ValueError(
            f"kvcache_store_keys: {kvcache_store_keys} should be a list of "
            "strings or None.")


class KVTransferParams(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        dict=True):  # type: ignore[call-arg]
    """API parameters for KV transfer scenarios such as disaggregated prefill.

    Args:
        prefix_prompt_ids: If provided, the engine will send/recv the KVCache
            for the given prefix token ids. For disaggregated prefilling,
            prefix_prompt_ids equals to prompt_ids.
        kvcache_load_keys: If provided, then it contains the keys of the
            KVCache that need to be read.
        kvcache_store_keys: If provided, then it contains the keys of the
            KVCache that need to be send.
    """
    prefix_prompt_ids: Optional[Any] = None
    kvcache_load_keys: Optional[Any] = None
    kvcache_store_keys: Optional[Any] = None

    def __post_init__(self):
        self._verify_args()

    def _verify_args(self) -> None:
        # Check prefix_prompt_ids
        if self.prefix_prompt_ids is not None and not (
                isinstance(self.prefix_prompt_ids, list)
                and valid_prefix_prompt_ids(self.prefix_prompt_ids)):
            raise ValueError(
                f"prefix_prompt_ids: {self.prefix_prompt_ids} is not valid.")

        # Check kvcache_load_keys
        if self.kvcache_load_keys is not None and not (
                isinstance(self.kvcache_load_keys, list)
                and valid_kvcache_keys(self.kvcache_load_keys)):
            raise ValueError(
                f"kvcache_load_keys: {self.kvcache_load_keys} is not valid.")

        # Check kvcache_store_keys
        if self.kvcache_store_keys is not None and not (
                isinstance(self.kvcache_store_keys, list)
                and valid_kvcache_keys(self.kvcache_store_keys)):
            raise ValueError(
                f"kvcache_store_keys: {self.kvcache_store_keys} is not valid.")

    @classmethod
    def from_optional(cls, input_json: Optional[Union[str, dict]]):

        if input_json is None:
            return None

        if isinstance(input_json, str):
            kv_transfer_params_dict = json.loads(input_json)
        elif isinstance(input_json, dict):
            kv_transfer_params_dict = input_json
        else:
            raise ValueError(
                "The provided kv_transfer_params is not a valid JSON "
                "string or dictionary")

        # Items of kv_transfer_params_dict should not be nested list
        verify_input_from_optional(kv_transfer_params_dict)

        return cls(**kv_transfer_params_dict)
