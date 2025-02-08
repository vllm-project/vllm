# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Union

import msgspec


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
    prefix_prompt_ids: Optional[Union[List[int], List[List[int]]]] = None
    kvcache_load_keys: Optional[Union[List[str], List[List[str]]]] = None
    kvcache_store_keys: Optional[Union[List[str], List[List[str]]]] = None
