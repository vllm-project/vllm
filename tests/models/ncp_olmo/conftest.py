# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.models.ncp_olmo.model import NCPOlmo3ForCausalLM
from vllm.utils.torch_utils import set_default_torch_dtype


@pytest.fixture
def ncp_config(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    config = EngineArgs(
        model=os.environ.get(
            "NCP_OLMO_TEST_MODEL",
            "ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage2_v1",
        ),
        trust_remote_code=True,
        skip_tokenizer_init=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        dtype="bfloat16",
        hf_overrides={
            "hidden_size": 128,
            "ffn_hidden_size": 256,
            "intermediate_size": 256,
            "num_attention_heads": 4,
            "num_query_groups": 4,
            "num_key_value_heads": 4,
            "num_layers": 4,
            "num_hidden_layers": 4,
            "conceptlm_encoder_layers": 2,
            "conceptlm_decoder_layers": 2,
            "conceptlm_special_layers": 2,
            "conceptlm_v22_vq_num_codebooks": 4,
            "conceptlm_v22_vq_codebook_size": 4,
            "vocab_size": 128,
        },
    ).create_engine_config()
    with set_current_vllm_config(config):
        yield config


@pytest.fixture
def ncp_model(ncp_config, dist_init):
    """Use real layers and loaders with small, deterministic dummy weights."""
    with (
        set_current_vllm_config(ncp_config),
        set_default_torch_dtype(ncp_config.model_config.dtype),
        torch.device("cuda"),
    ):
        model = NCPOlmo3ForCausalLM(vllm_config=ncp_config)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                initialize_single_dummy_weight(parameter, name=name)
        yield model
