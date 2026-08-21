# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ExampleConnector must not claim more externally-matched tokens than the
producer actually persisted.

A store request is only recorded on its first scheduling step, so a chunked
prefill saves just the blocks allocated by then while the folder for the
full prompt already exists. Before the cap, get_num_new_matched_tokens
returned the full aligned prefix for any existing folder, the scheduler
marked unsaved tokens as computed, and the worker-side injection failed on
the shape mismatch between the saved file and the slot mapping.
"""

import safetensors.torch
import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
    align_to_block_size,
)

from .utils import create_request, create_vllm_config

BLOCK_SIZE = 16
NUM_PROMPT_TOKENS = 100


def _make_connector(tmp_path) -> ExampleConnector:
    vllm_config = create_vllm_config(
        block_size=BLOCK_SIZE,
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
    )
    return ExampleConnector(
        vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config=None
    )


def _save_layer_file(connector, request, num_saved_tokens: int) -> None:
    """Write one layer file the way the producer side would, holding
    num_saved_tokens tokens."""
    num_tokens_to_check = align_to_block_size(
        len(request.prompt_token_ids) - 1, BLOCK_SIZE
    )
    foldername = connector._generate_foldername_debug(
        torch.tensor(request.prompt_token_ids)[:num_tokens_to_check],
        [],
        create_folder=True,
    )
    kv = torch.zeros(num_saved_tokens, 2, 1, 8, dtype=torch.float16)
    safetensors.torch.save_file(
        {"kv_cache": kv}, f"{foldername}/model.layers.0.self_attn.safetensors"
    )


def test_no_match_without_saved_files(tmp_path):
    connector = _make_connector(tmp_path)
    request = create_request(num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE)
    matched, load_async = connector.get_num_new_matched_tokens(request, 0)
    assert matched == 0
    assert load_async is False


def test_full_save_claims_aligned_prefix(tmp_path):
    connector = _make_connector(tmp_path)
    request = create_request(num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE)
    full = align_to_block_size(NUM_PROMPT_TOKENS - 1, BLOCK_SIZE)
    _save_layer_file(connector, request, full)
    matched, _ = connector.get_num_new_matched_tokens(request, 0)
    assert matched == full


def test_partial_save_caps_claim(tmp_path):
    """The chunked-prefill case: files hold fewer tokens than the aligned
    prompt prefix, and the claim must shrink to what exists."""
    connector = _make_connector(tmp_path)
    request = create_request(num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE)
    num_saved = 2 * BLOCK_SIZE  # first chunk only
    _save_layer_file(connector, request, num_saved)

    matched, load_async = connector.get_num_new_matched_tokens(request, 0)
    assert matched == num_saved
    assert load_async is False

    # And never negative once some tokens are locally computed past the cap.
    matched, _ = connector.get_num_new_matched_tokens(request, 3 * BLOCK_SIZE)
    assert matched == 0


def test_unaligned_partial_save_rounds_down(tmp_path):
    connector = _make_connector(tmp_path)
    request = create_request(num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE)
    _save_layer_file(connector, request, 2 * BLOCK_SIZE + 5)
    matched, _ = connector.get_num_new_matched_tokens(request, 0)
    assert matched == 2 * BLOCK_SIZE


def test_folder_without_layer_files_claims_nothing(tmp_path):
    connector = _make_connector(tmp_path)
    request = create_request(num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE)
    num_tokens_to_check = align_to_block_size(
        len(request.prompt_token_ids) - 1, BLOCK_SIZE
    )
    connector._generate_foldername_debug(
        torch.tensor(request.prompt_token_ids)[:num_tokens_to_check],
        [],
        create_folder=True,
    )
    matched, _ = connector.get_num_new_matched_tokens(request, 0)
    assert matched == 0
