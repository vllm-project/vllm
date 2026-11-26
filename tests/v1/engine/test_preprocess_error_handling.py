# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch
import torch.cuda

from vllm import LLM, SamplingParams
from vllm.config import (
    KVTransferConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.utils import (
    MROPE_POSITION_DELTA_KEY,
    get_mrope_position_limit,
)
from vllm.exceptions import VLLMValidationError
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.platforms import current_platform
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore

MODEL_NAME = "vllm-project/tiny-random-LlamaForCausalLM"


@pytest.mark.parametrize(
    ("delta", "position_limit", "accepted"),
    [
        (-5, 12, False),
        (-4, 12, True),
        (0, 12, True),
        (6.0, 12, True),
        (7, 12, False),
        (torch.iinfo(torch.int32).max - 5, None, True),
        (torch.iinfo(torch.int32).max - 4, None, False),
    ],
)
def test_transferred_mrope_delta_fits_decode_context(delta, position_limit, accepted):
    core = object.__new__(EngineCore)
    core.vllm_config = VllmConfig(
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_consumer",
        ),
    )
    core.vllm_config.model_config = MagicMock(
        spec=ModelConfig, uses_mrope=True, max_model_len=8
    )
    core.mm_receiver_cache = None
    core.request_block_hasher = None
    core.prefix_hash_config = None
    core._mrope_position_limit = position_limit
    request = EngineCoreRequest(
        request_id="test-request",
        prompt_token_ids=[1, 2, 3, 4],
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=2,
            extra_args={"kv_transfer_params": {MROPE_POSITION_DELTA_KEY: delta}},
        ),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )

    if not accepted:
        with pytest.raises(VLLMValidationError, match=MROPE_POSITION_DELTA_KEY):
            core.preprocess_add_request(request)
        return

    processed, _ = core.preprocess_add_request(request)
    assert processed.kv_transfer_params == {MROPE_POSITION_DELTA_KEY: delta}


def test_mrope_position_limit_uses_loaded_target_and_draft_tables():
    def rope(positions_factor: int):
        return MRotaryEmbedding(
            head_size=6,
            rotary_dim=6,
            max_position_embeddings=8,
            base=10_000,
            is_neox_style=True,
            dtype=torch.float32,
            mrope_section=[1, 1, 1],
            positions_factor=positions_factor,
        )

    worker = MagicMock()
    with set_current_vllm_config(VllmConfig()):
        worker.get_model.return_value = rope(4)
        worker.model_runner.get_draft_model.return_value = rope(2)
    assert get_mrope_position_limit(worker) == 16


def test_preprocess_error_handling(monkeypatch: pytest.MonkeyPatch):
    """Test that preprocessing errors are handled gracefully."""
    if current_platform.is_xpu():
        pytest.skip(
            "Skipped on XPU: this test only works with 'fork', but XPU uses 'spawn'."
        )

    assert not torch.cuda.is_initialized(), (
        "fork needs to be used for the engine "
        "core process and this isn't possible if cuda is already initialized"
    )

    # Store original method to call for non-failing requests
    original_preprocess = EngineCore.preprocess_add_request

    # Monkeypatch to make preprocess_add_request raise an exception
    # only for requests with "FAIL" in the first token
    def conditional_failing_preprocess(self, request: EngineCoreRequest):
        # Fail if the first token id is 333
        if request.prompt_token_ids is not None and request.prompt_token_ids[0] == 333:
            raise ValueError("Simulated preprocessing error!")
        return original_preprocess(self, request)

    monkeypatch.setattr(
        EngineCore, "preprocess_add_request", conditional_failing_preprocess
    )
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")

    llm = LLM(model=MODEL_NAME)

    # Create a failing request by crafting a request with an invalid token
    # We need to use a direct approach since LLM.generate tokenizes for us
    from vllm.inputs import TokensPrompt

    # This should raise an exception due to the preprocessing failure
    # Special token id to trigger the failure
    failing_prompt = TokensPrompt(prompt_token_ids=[333])
    outputs = llm.generate(failing_prompt, SamplingParams(max_tokens=10))
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) == 0
    assert outputs[0].finished
    assert outputs[0].outputs[0].finish_reason == "error"

    # Verify the engine is still functional with a normal request
    outputs = llm.generate("Hello, my name is", SamplingParams(max_tokens=10))
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0
    assert outputs[0].outputs[0].finish_reason in ("stop", "length")
