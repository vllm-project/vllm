# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging

import pytest
import regex as re

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

        # Always compile the full graph instead of piecewise
        if not compilation_config["use_inductor_graph_partition"]:
            compilation_config["splitting_ops"] = []

        full_compilation_config = CompilationConfig(
            cudagraph_mode=CUDAGraphMode.NONE,
            mode=CompilationMode.VLLM_COMPILE,
            inductor_compile_config={"force_disable_caches": True},
            **compilation_config,
        )

        with caplog_mp_spawn(logging.DEBUG) as log_holder:
            run_model(full_compilation_config, model_name, **model_kwargs)

        num_compile_ranges = len(full_compilation_config.get_compile_ranges())
        assert num_compile_ranges in [1, 2]

        print(f"Compile ranges: {full_compilation_config.get_compile_ranges()}")
        print("Fusion results:")

        # Iterate through all so printing happens before asserting
        log_matches_dict = {}
        for match_name, pattern in FUSION_LOG_PATTERNS.items():
            log_matches_dict[match_name] = list(pattern.findall(log_holder.text))
            print(f"- {match_name}={','.join(log_matches_dict[match_name])}")

        # Now check the matches
        for match_name in matches_check:
            # ar-rms-norm only activates in one range
            num_ranges_activated = (
                1 if match_name == "ar_rms_fusion" else num_compile_ranges
            )
            n_expected = tp_size * num_ranges_activated

            log_matches = log_matches_dict[match_name]
            assert len(log_matches) == n_expected, (
                f"Could not find {n_expected} {match_name} "
                f"(found {len(log_matches)}) in:\n {log_holder.text}"
            )

            expected_matches = getattr(matches, match_name)
            for i, m in enumerate(log_matches):
                assert int(m) == expected_matches, (
                    f"{match_name}[{i}] expected: {expected_matches}, found: {int(m)}"
                )

            if match_name == "ar_rms_fusion":
                log_matches = re.findall(
                    r"pass_manager.py:\d+] Skipping "
                    r".*AllReduceFusionPass.* with compile range",
                    log_holder.text,
                )

                n_expected = 2 * (num_compile_ranges - num_ranges_activated)
                assert len(log_matches) == n_expected, (
                    f'Could not find {n_expected} "Skipping AllReduceFusionPass" '
                    f"(found {len(log_matches)}) in:\n {log_holder.text}"
                )

    return run
