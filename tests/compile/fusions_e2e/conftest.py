# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging

import pytest

from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

from ..fusion_test_utils import run_model
from .common import FUSION_LOG_PATTERNS, AttentionBackendCase, Matches


@pytest.fixture
def run_e2e_fusion_test(monkeypatch, caplog_mp_spawn):
    def run(
        model_name: str,
        matches: Matches,
        model_kwargs: dict,
        attn_backend: AttentionBackendCase,
        compilation_config: dict,
        matches_check: list[str],
        tp_size: int = 1,
    ):
        # Disable, compile cache to make sure custom passes run.
        # Otherwise, we can't verify fusion happened through the logs.
        monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

        # To capture subprocess logs, we need to know whether spawn or fork is used.
        # Force spawn as it is more general.
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        model_kwargs = {**attn_backend.model_kwargs, **model_kwargs}
        model_kwargs["attention_config"] = {"backend": attn_backend.backend.name}
        model_kwargs["tensor_parallel_size"] = tp_size

        full_compilation_config = CompilationConfig(
            cudagraph_mode=CUDAGraphMode.NONE,
            mode=CompilationMode.VLLM_COMPILE,
            inductor_compile_config={"force_disable_caches": True},
            **compilation_config,
        )

        with caplog_mp_spawn(logging.DEBUG) as log_holder:
            run_model(full_compilation_config, model_name, **model_kwargs)

        for match_name in matches_check:
            pattern = FUSION_LOG_PATTERNS[match_name]
            log_matches = pattern.findall(log_holder.text)

            assert len(log_matches) == tp_size, (
                f"Could not find {match_name} in \n: {log_holder.text}"
            )

            for i, m in enumerate(log_matches):
                expected_matches = getattr(matches, match_name)
                assert int(m) == expected_matches, (
                    f"{match_name}[{i}] expected: {expected_matches}, found: {int(m)}"
                )

    return run
