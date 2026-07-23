# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for ``--enable-return-indexer-topk``.

Verifies that the OpenAI-compatible endpoints (``/v1/completions`` and
``/v1/chat/completions``) return per-token sparse-attention indexer topk
indices when the server is launched with ``--enable-return-indexer-topk``.

The indices are base64-encoded ``.npy`` bytes; after decoding the ndarray
has shape ``(num_tokens, num_indexer_layers, index_topk)`` and dtype
``int32`` (see :class:`IndexerTopkManager`).

Only models with :class:`SparseAttnIndexer` layers (DeepSeek-V32 / V4)
emit indexer topk. To keep the test CI-friendly we launch
``deepseek-ai/DeepSeek-V3.2`` with ``--load-format dummy`` (random weights)
and shrink it via ``--hf-overrides`` to a tiny 2-layer / 128-hidden config.
The GPU memory fraction can be tuned via ``VLLM_TEST_GPU_MEMORY_UTILIZATION``
(defaults to 0.5).
"""

import io
import os

import numpy as np
import pybase64 as base64
import pytest

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm

from ...utils import RemoteOpenAIServer

# DeepSeek-V3.2 carries a SparseAttnIndexer in every backbone layer when
# ``index_topk`` is set. Shrink the model so it fits in CI:
#   - 2 hidden layers  -> 2 indexer layers (default freq=1, offset=2
#     still produces an indexer in every layer because
#     max(layer_id - offset + 1, 0) % freq == 0 for all layer ids).
#   - index_topk=128 keeps the per-token payload small (smallest value
#     that still exercises multi-slot selection).
#   - hidden_size=128 / intermediate=256 / 2 experts keeps memory tiny.
#   - num_experts_per_tok=2 must be <= n_routed_experts.
SMALL_V32_OVERRIDES = {
    "num_hidden_layers": 2,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "n_routed_experts": 2,
    "num_experts_per_tok": 2,
    "index_topk": 128,
}

MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
NUM_INDEXER_LAYERS = 2
INDEX_TOPK = 128

# Allow CI / developers to override the GPU memory fraction used by the
# server. Default 0.5 is conservative enough for H100 80GB but small enough
# to coexist with other GPU processes on smaller cards.
DEFAULT_GPU_MEM_UTIL = float(os.environ.get("VLLM_TEST_GPU_MEMORY_UTILIZATION", "0.5"))

# SparseAttnIndexer currently requires CUDA + DeepGEMM (Hopper+).
pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not has_deep_gemm(),
    reason="indexer_topk e2e test requires CUDA with DeepGEMM (Hopper+).",
)


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "4",
        "--enforce-eager",
        "--load-format",
        "dummy",
        "--trust-remote-code",
        "--enable-return-indexer-topk",
        "--gpu-memory-utilization",
        str(DEFAULT_GPU_MEM_UTIL),
    ]
    with RemoteOpenAIServer(
        MODEL_NAME,
        args,
        override_hf_configs=SMALL_V32_OVERRIDES,
    ) as remote_server:
        yield remote_server


def _decode_indexer_topk(b64: str) -> np.ndarray:
    """Decode the base64 ``.npy`` payload returned by the server."""
    return np.load(io.BytesIO(base64.b64decode(b64)))


@pytest.mark.asyncio
async def test_indexer_topk_completions(server):
    """``/v1/completions`` returns base64 indexer_topk with valid shape."""
    async with server.get_async_client() as client:
        result = await client.completions.create(
            model=MODEL_NAME,
            prompt="Hello, world",
            max_tokens=5,
            temperature=0,
            extra_body={
                "return_token_ids": True,
                # Request indexer topk for the full prompt + generated tokens.
                "indexer_topk_prompt_start": 0,
            },
        )

        choice = result.model_dump()["choices"][0]

        # Field must exist and be populated when the flag is enabled.
        assert choice["indexer_topk"] is not None, (
            "indexer_topk is None; ensure --enable-return-indexer-topk is set "
            "and the model has SparseAttnIndexer layers."
        )
        assert choice["token_ids"] is not None

        indexer_topk = _decode_indexer_topk(choice["indexer_topk"])

        # Shape contract: (num_tokens, num_indexer_layers, index_topk).
        assert indexer_topk.ndim == 3
        num_tokens, num_indexer_layers, index_topk = indexer_topk.shape
        # prompt (>=1) + generated tokens (5)
        assert num_tokens > 0
        assert num_indexer_layers == NUM_INDEXER_LAYERS
        assert index_topk == INDEX_TOPK

        # dtype is int32 (KV slot indices can exceed 65535).
        assert indexer_topk.dtype == np.int32

        # KV-slot indices are non-negative; -1 is the documented "no valid
        # slot" sentinel used by the top-k kernels (see
        # ``sparse_attn_indexer.py`` and ``dcp_indexer_cutedsl.py``). When the
        # sequence is shorter than ``index_topk``, unfilled positions remain
        # -1, so we require at least one valid index rather than all.
        assert (indexer_topk >= 0).any(), (
            "Expected at least one valid (non-negative) indexer_topk entry, "
            "but all values are -1 sentinels."
        )


@pytest.mark.asyncio
async def test_indexer_topk_chat_completions(server):
    """``/v1/chat/completions`` also surfaces indexer_topk."""
    async with server.get_async_client() as client:
        result = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=5,
            temperature=0,
            extra_body={
                "return_token_ids": True,
                "indexer_topk_prompt_start": 0,
            },
        )

        choice = result.model_dump()["choices"][0]
        assert choice["indexer_topk"] is not None

        indexer_topk = _decode_indexer_topk(choice["indexer_topk"])
        assert indexer_topk.ndim == 3
        assert indexer_topk.shape[0] > 0
        assert indexer_topk.shape[1] == NUM_INDEXER_LAYERS
        assert indexer_topk.shape[2] == INDEX_TOPK
        assert indexer_topk.dtype == np.int32
        # -1 is the expected sentinel for unfilled top-k positions.
        assert (indexer_topk >= 0).any(), (
            "Expected at least one valid (non-negative) indexer_topk entry."
        )


@pytest.mark.asyncio
async def test_indexer_topk_prompt_start_offset(server):
    """``indexer_topk_prompt_start`` trims the leading prompt tokens."""
    prompt = "The quick brown fox jumps over the lazy dog."
    prompt_start = 4

    async with server.get_async_client() as client:
        full = await client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=3,
            temperature=0,
            extra_body={"indexer_topk_prompt_start": 0},
        )
        trimmed = await client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=3,
            temperature=0,
            extra_body={"indexer_topk_prompt_start": prompt_start},
        )

        full_topk = _decode_indexer_topk(
            full.model_dump()["choices"][0]["indexer_topk"]
        )
        trimmed_topk = _decode_indexer_topk(
            trimmed.model_dump()["choices"][0]["indexer_topk"]
        )

        # The trimmed output should drop exactly `prompt_start` tokens
        # from the leading prompt (same number of generated tokens).
        assert full_topk.shape[0] - trimmed_topk.shape[0] == prompt_start
        # Layer / index_topk dimensions are unaffected.
        assert full_topk.shape[1:] == trimmed_topk.shape[1:]
