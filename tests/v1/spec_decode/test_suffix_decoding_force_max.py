# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import torch

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.config.compilation import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
)
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer

PAD_TOKEN_ID = 151645
SUFFIX_DECODING_CACHE = "arctic_inference.suffix_decoding.SuffixDecodingCache"
CACHED_TOKENIZER_FROM_CONFIG = (
    "vllm.v1.spec_decode.suffix_decoding.cached_tokenizer_from_config"
)


def _make_vllm_config(
    force_max_spec_tokens: bool = True,
    num_speculative_tokens: int = 3,
    max_model_len: int = 4096,
) -> VllmConfig:
    """
    Build a VllmConfig with minimal fields for testing.
    Uses VllmConfig.__new__() to bypass Pydantic validation,
    since we supply MagicMock objects instead of real ModelConfig
    / SchedulerConfig instances to avoid HF model downloads in CI.
    """
    cfg = VllmConfig.__new__(VllmConfig)
    cfg.compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        custom_ops=["none"],
        splitting_ops=[],
        compile_sizes=[],
        cudagraph_mode=CUDAGraphMode.NONE,
    )
    cfg.speculative_config = SpeculativeConfig(
        method="suffix",
        num_speculative_tokens=num_speculative_tokens,
        force_max_spec_tokens=force_max_spec_tokens,
    )
    mock_model_config = MagicMock()
    mock_model_config.max_model_len = max_model_len
    cfg.model_config = mock_model_config
    mock_scheduler_config = MagicMock()
    mock_scheduler_config.max_num_seqs = 256
    cfg.scheduler_config = mock_scheduler_config
    return cfg


def _make_mock_tokenizer(eos_token_id: int = PAD_TOKEN_ID):
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = eos_token_id
    return mock_tokenizer


class TestSuffixDecodingForceMaxSpecTokens:
    def test_init_reads_force_max_spec_tokens(self):
        vllm_config = _make_vllm_config(force_max_spec_tokens=True)
        mock_tokenizer = _make_mock_tokenizer(eos_token_id=PAD_TOKEN_ID)
        with (
            patch(SUFFIX_DECODING_CACHE, autospec=True),
            patch(
                CACHED_TOKENIZER_FROM_CONFIG,
                return_value=mock_tokenizer,
            ),
        ):
            proposer = SuffixDecodingProposer(vllm_config)
            assert proposer.force_max_spec_tokens is True
            assert proposer._pad_token_id == PAD_TOKEN_ID
            assert proposer._pad_template == [PAD_TOKEN_ID] * 3

    def test_init_tokenizer_none_disables_force(self):
        vllm_config = _make_vllm_config(force_max_spec_tokens=True)
        with (
            patch(SUFFIX_DECODING_CACHE, autospec=True),
            patch(
                CACHED_TOKENIZER_FROM_CONFIG,
                return_value=None,
            ),
        ):
            proposer = SuffixDecodingProposer(vllm_config)
            assert proposer.force_max_spec_tokens is False
            assert proposer._pad_token_id == -1
            assert proposer._pad_template == []

    def test_init_default_disabled(self):
        vllm_config = _make_vllm_config(force_max_spec_tokens=False)
        with (
            patch(SUFFIX_DECODING_CACHE, autospec=True),
            patch(
                CACHED_TOKENIZER_FROM_CONFIG,
                return_value=MagicMock(),
            ),
        ):
            proposer = SuffixDecodingProposer(vllm_config)
            assert proposer.force_max_spec_tokens is False
            assert proposer._pad_token_id == -1
            assert proposer._pad_template == []

    def test_propose_pads_short_lists(self):
        """Verify short draft lists are padded to num_speculative_tokens."""
        vllm_config = _make_vllm_config(force_max_spec_tokens=True)
        mock_tokenizer = _make_mock_tokenizer()
        with (
            patch(SUFFIX_DECODING_CACHE, autospec=True) as mock_cache_cls,
            patch(
                CACHED_TOKENIZER_FROM_CONFIG,
                return_value=mock_tokenizer,
            ),
        ):
            mock_cache = mock_cache_cls.return_value
            mock_cache.active_requests = set()
            mock_cache.cached_requests = set()
            mock_cache.speculate.side_effect = [
                MagicMock(token_ids=[1]),
                MagicMock(token_ids=[4, 5]),
            ]

            proposer = SuffixDecodingProposer(vllm_config)

            mock_batch = MagicMock()
            mock_batch.req_ids = {0: "req_0", 1: "req_1", 2: "req_2"}
            mock_batch.num_tokens_no_spec = [5, 5, 5]
            mock_batch.req_id_to_index = {"req_0": 0, "req_1": 1, "req_2": 2}
            mock_batch.num_prompt_tokens = [100, 100, 100]
            mock_batch.token_ids_cpu = torch.zeros((3, 4096), dtype=torch.long)
            sampled_token_ids = [[100], [200, 300], []]

            result = proposer.propose(mock_batch, sampled_token_ids, None)

            assert result == [
                [1, PAD_TOKEN_ID, PAD_TOKEN_ID],
                [4, 5, PAD_TOKEN_ID],
                [],
            ]

    def test_propose_disabled_does_nothing(self):
        """Verify no-op when force_max_spec_tokens is False."""
        vllm_config = _make_vllm_config(force_max_spec_tokens=False)
        with (
            patch(SUFFIX_DECODING_CACHE, autospec=True) as mock_cache_cls,
            patch(
                CACHED_TOKENIZER_FROM_CONFIG,
                return_value=MagicMock(),
            ),
        ):
            mock_cache = mock_cache_cls.return_value
            mock_cache.active_requests = set()
            mock_cache.cached_requests = set()
            mock_cache.speculate.return_value = MagicMock(
                token_ids=[1, 2, 3, 4, 5, 6],
            )

            proposer = SuffixDecodingProposer(vllm_config)

            mock_batch = MagicMock()
            mock_batch.req_ids = {0: "req_0"}
            mock_batch.num_tokens_no_spec = [5]
            mock_batch.req_id_to_index = {"req_0": 0}
            mock_batch.num_prompt_tokens = [100]
            mock_batch.token_ids_cpu = torch.zeros((1, 4096), dtype=torch.long)
            sampled_token_ids = [[100, 200]]

            result = proposer.propose(mock_batch, sampled_token_ids, None)

            assert result == [[1, 2, 3, 4, 5, 6]]
