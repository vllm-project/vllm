# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer KV dequant tile helpers.

This PR only creates the scaffold. Wiring stays in a follow-up PR.
"""

from vllm.v1.kv_cache_interface import KVQuantMode

SUPPORTED_MODES: frozenset[KVQuantMode] = frozenset({KVQuantMode.NONE})
