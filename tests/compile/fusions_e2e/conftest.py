# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from collections import defaultdict

import pytest
import regex as re

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

from .common import FUSION_LOG_PATTERNS, AttentionBackendCase, Matches


def run_model(compile_config: int | CompilationConfig, model: str, **model_kwargs):
    """Run a model with the given compilation config for E2E fusion tests."""
    compilation_config = (
        compile_config
        if isinstance(compile_config, CompilationConfig)
        else CompilationConfig(mode=compile_config)
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0)
    # Allow override from model_kwargs
    model_kwargs = {"tensor_parallel_size": 1, **model_kwargs}
    model_kwargs = {"disable_custom_all_reduce": True, **model_kwargs}

    # No cudagraphs by default
    if compilation_config.cudagraph_mode is None:
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    llm = LLM(
        model=model,
        compilation_config=compilation_config,
        **model_kwargs,
    )
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Get the compile ranges endpoints after vllm config post init
    # in order to compute compile ranges correctly
    compilation_config.compile_ranges_endpoints = (
        llm.llm_engine.vllm_config.compilation_config.compile_ranges_endpoints
    )

    # Fetch match table from each worker via RPC and sum across workers.
    worker_tables = llm.llm_engine.engine_core.collective_rpc(
        "get_compilation_match_table"
    )
    combined: defaultdict[str, int] = defaultdict(int)
    for table in worker_tables:
        for k, v in table.items():
            combined[k] += v
    return dict(combined)


@pytest.fixture
def run_e2e_fusion_test(monkeypatch, caplog_mp_spawn):
    def run(
        model_name: str,
        matches: Matches,
        model_kwargs: dict,
        attn_backend: AttentionBackendCase,
        compilation_config: dict,
        matches_check: list[str],
        use_deepgemm: bool = False,
        use_aiter: bool = False,
        tp_size: int = 1,
    ):
        monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1" if use_deepgemm else "0")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1" if use_aiter else "0")
        from vllm._aiter_ops import rocm_aiter_ops

        rocm_aiter_ops.refresh_env_variables()

        # Filter here to reduce code duplication
        backend_name = attn_backend.backend.name.lower()
        requires_mla = "deepseek" in model_name.lower()
        is_mla = "mla" in backend_name
        # DeepSeek V3.2 uses sparse MLA
        requires_sparse = "v3.2" in model_name.lower()
        is_sparse = "sparse" in backend_name

        if requires_mla != is_mla or requires_sparse != is_sparse:
            pytest.skip(
                f"Incompatible model '{model_name}' and "
                f"attention backend '{attn_backend.backend.name}'"
            )

        if attn_backend.backend.name == "FLASHINFER":
            from vllm.utils.flashinfer import supports_trtllm_attention

            if not supports_trtllm_attention():
                matches = matches._replace(attn_quant_fusion=0)

        # TODO: remove this after finishing migration from envs to model kwargs
        if model_name == "openai/gpt-oss-20b":
            from .common import is_blackwell

            if is_blackwell():
                monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8", "1")

        # Disable, compile cache to make sure custom passes run.
        # Otherwise, we can't verify fusion happened through the logs.
        monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

        # To capture subprocess logs, we need to know whether spawn or fork is used.
        # Force spawn as it is more general.
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        model_kwargs = {**attn_backend.model_kwargs, **model_kwargs}
        model_kwargs["attention_config"] = {"backend": attn_backend.backend.name}
        model_kwargs["tensor_parallel_size"] = tp_size

        # Cap warmup memory: tests use small max_model_len (1024) but the
        # engine default max_num_batched_tokens is 16384. Warming up large
        # models (e.g. Llama-4-Scout-FP8) at 16384 tokens may trigger OOM.
        model_kwargs.setdefault("max_num_batched_tokens", 8192)

        # Sparse MLA models (DSv3.2) hit an over-strict inductor assertion in
        # decompose_auto_functionalized when +rotary_embedding is forced into
        # the compile graph. Disable qk_norm+rope fusion (which auto-enables
        # +rotary_embedding) for this combo to avoid the known torch bug.
        # TODO: remove once upstream torch fix lands.
        if requires_sparse:
            if "pass_config" in compilation_config:
                compilation_config["pass_config"].enable_qk_norm_rope_fusion = False
                matches_check = [m for m in matches_check if m != "norm_rope_fusion"]
            # DSv3.2 sparse indexer uses persistent_topk with k=config.index_topk
            # (2048 for the default config). max_model_len must be >= index_topk
            # or the topk kernel raises "k out of range" at runtime.
            model_kwargs["max_model_len"] = max(
                model_kwargs.get("max_model_len", 0), 2048
            )

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
            match_table = run_model(full_compilation_config, model_name, **model_kwargs)

        num_compile_ranges = len(full_compilation_config.get_compile_ranges())
        assert num_compile_ranges in [1, 2, 3]

        print(f"Compile ranges: {full_compilation_config.get_compile_ranges()}")
        print("Fusion results:")

        # Iterate through all so printing happens before asserting
        log_matches_dict = {}
        for match_name, pattern in FUSION_LOG_PATTERNS.items():
            log_matches_dict[match_name] = list(pattern.findall(log_holder.text))
            print(f"- {match_name}={','.join(log_matches_dict[match_name])}")

        # Now check the matches
        for match_name in matches_check:
            log_matches = list(int(ms) for ms in log_matches_dict[match_name])

            # AR+RMS skips the largest range; SP skips the smallest.
            # When both are enabled, AR+RMS activation count is
            # model-dependent (hidden_size affects threshold), so derive
            # from log data.
            if (
                match_name == "ar_rms_fusion"
                and "sequence_parallel" in matches_check
                and num_compile_ranges >= 2
            ):
                assert (
                    len(log_matches) >= tp_size and len(log_matches) % tp_size == 0
                ), (
                    f"Expected multiple of {tp_size} ar_rms log entries, "
                    f"found {len(log_matches)}"
                )
                num_ranges_activated = len(log_matches) // tp_size
            elif (
                match_name in ("ar_rms_fusion", "sequence_parallel")
                and num_compile_ranges >= 2
            ):
                num_ranges_activated = num_compile_ranges - 1
            else:
                num_ranges_activated = num_compile_ranges

            # TODO: Remove log counting in unit tests
            # once all matchers implement VllmFusionPatternMatcherPass
            n_expected = tp_size * num_ranges_activated
            if match_name not in ("attn_quant_fusion", "act_quant_fusion"):
                assert len(log_matches) == n_expected, (
                    f"Could not find {n_expected} {match_name} "
                    f"(found {len(log_matches)}) in:\n {log_holder.text}"
                )

            expected_matches = getattr(matches, match_name)

            if match_name == "rms_quant_fusion" and "ar_rms_fusion" in matches_check:
                # AR+rms+quant takes precedence over rms+quant if activated.
                # That means we get full matching where ar+rms+quant was not
                # activated, and less where it was (only the smallest range).
                assert sum(m == expected_matches for m in log_matches) == tp_size * (
                    num_ranges_activated - 1
                ), "Expecting full rms+quant fusion where ar+rms+quant not activated"

                assert all(
                    expected_matches - matches.ar_rms_fusion <= m <= expected_matches
                    for m in log_matches
                ), (
                    f"Expecting at least {expected_matches - matches.ar_rms_fusion} "
                    f"where ar+rms+quant was activated"
                )
            elif (
                match_name == "async_tp"
                and "sequence_parallel" in matches_check
                and num_compile_ranges >= 2
            ):
                # AsyncTP only finds patterns on ranges where SP ran.
                n_sp_ranges = num_compile_ranges - 1
                assert (
                    sum(m == expected_matches for m in log_matches)
                    == tp_size * n_sp_ranges
                ), (
                    f"Expecting {expected_matches} async_tp on "
                    f"{tp_size * n_sp_ranges} SP-range entries, "
                    f"found: {log_matches}"
                )
                assert sum(m == 0 for m in log_matches) == tp_size, (
                    f"Expecting 0 async_tp on {tp_size} small-range entries "
                    f"(no SP), found: {log_matches}"
                )
            elif (
                match_name == "ar_rms_fusion"
                and "sequence_parallel" in matches_check
                and num_compile_ranges >= 2
            ):
                # SP consumes allreduce patterns first, so AR+RMS finds
                # full matches only on the smallest range (no SP).
                assert sum(m == expected_matches for m in log_matches) == tp_size, (
                    f"Expecting {expected_matches} ar_rms on "
                    f"{tp_size} small-range entries, found: {log_matches}"
                )
                assert sum(m == 0 for m in log_matches) == tp_size * (
                    num_ranges_activated - 1
                ), (
                    f"Expecting 0 ar_rms on "
                    f"{tp_size * (num_ranges_activated - 1)} large-range "
                    f"entries (SP took precedence), found: {log_matches}"
                )

            elif match_name == "act_quant_fusion":
                actual_match = match_table.get("activation_quant_fusion_pass", 0)
                assert actual_match == expected_matches * n_expected, (
                    f"Could not find {expected_matches * n_expected} "
                    f"{match_name} (found {actual_match})."
                )
            elif match_name == "attn_quant_fusion":
                actual_match = match_table.get(
                    "attn_quant_fusion", 0
                ) + match_table.get("mla_attn_quant_fusion", 0)
                assert actual_match == expected_matches * n_expected, (
                    f"Could not find {expected_matches * n_expected} "
                    f"{match_name} (found {actual_match})."
                )
            else:
                expected_matches_list = [expected_matches] * n_expected
                assert sorted(log_matches) == expected_matches_list, (
                    f"{match_name} expected: {expected_matches_list}, "
                    f"found: {sorted(log_matches)}"
                )

            if match_name == "ar_rms_fusion" and num_compile_ranges >= 2:
                log_matches = re.findall(
                    r"pass_manager.py:\d+] Skipping "
                    r".*AllReduceFusionPass.* with compile range",
                    log_holder.text,
                )

                n_expected = tp_size * (num_compile_ranges - num_ranges_activated)
                assert len(log_matches) == n_expected, (
                    f'Could not find {n_expected} "Skipping AllReduceFusionPass" '
                    f"(found {len(log_matches)}) in:\n {log_holder.text}"
                )

            if match_name == "sequence_parallel" and num_compile_ranges >= 2:
                log_matches = re.findall(
                    r"pass_manager.py:\d+] Skipping "
                    r".*SequenceParallelismPass.* with compile range",
                    log_holder.text,
                )

                n_expected = tp_size * (num_compile_ranges - num_ranges_activated)
                assert len(log_matches) == n_expected, (
                    f'Could not find {n_expected} "Skipping SequenceParallelismPass" '
                    f"(found {len(log_matches)}) in:\n {log_holder.text}"
                )

    return run
