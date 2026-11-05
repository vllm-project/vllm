# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron 3.5 Embed vision kernel autotuning regression tests."""

import json

import pytest

from tests.models.utils import check_embeddings_close


@pytest.mark.usefixtures("enable_pickle")
@pytest.mark.parametrize(
    "tensor_parallel_size,mm_encoder_tp_mode",
    [(1, "weights"), (2, "weights"), (2, "data")],
    ids=["single-rank", "tensor-parallel", "unsupported-data-fallback"],
)
def test_flashinfer_autotune_vision_gemms(
    vllm_runner, monkeypatch, tmp_path, tensor_parallel_size, mm_encoder_tp_mode
):
    """Opt-in tuning must cover vision projections and preserve embeddings."""
    from PIL import Image

    def vision_tuning_profiles(model):
        import ast
        import json
        import os
        import tempfile
        from pathlib import Path

        from flashinfer.autotuner import AutoTuner

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tuning.json"
            AutoTuner.get().save_configs(str(path))
            config_tactics = {
                key: value[1]
                for key, value in json.loads(path.read_text()).items()
                if key.startswith("('fp4_gemm',")
            }

        # Managed-cache winners are persisted separately from legacy configs.
        cache_root = Path(os.environ["VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"])
        for path in cache_root.glob("v2/*/entries/*.json"):
            entry = json.loads(path.read_text())
            config_tactics[entry["key"]] = entry["tactic"]

        qkv = model.vision_tower._first_attention.qkv_proj
        packed_k = qkv.input_size_per_partition // 2
        output_size = qkv.output_size_per_partition
        profiles = {}
        for key, tactic in config_tactics.items():
            if key.startswith("('fp4_gemm',"):
                fields = ast.parse(key, mode="eval").body.elts
                shapes = ast.literal_eval(fields[2])
                # QKV projects from the vision hidden size. The language
                # model has a different input width, so it cannot satisfy this.
                if shapes[0][1] == packed_k and shapes[1] == (packed_k, output_size):
                    profiles[key] = tactic
        return profiles

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))
    embeddings = []
    previous_snapshot = None
    # Changing processor cache capacity must not retune the last start.
    for startup, skip_mm_profiling in enumerate((True, False, False)):
        with vllm_runner(
            "nvidia/Nemotron-3.5-Embed-1B-NVFP4",
            runner="pooling",
            dtype="bfloat16",
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=1,
            tensor_parallel_size=tensor_parallel_size,
            mm_encoder_tp_mode=mm_encoder_tp_mode,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
            compilation_config={"mode": 0},
            kernel_config={
                "enable_flashinfer_autotune": True,
                "enable_flashinfer_mm_encoder_autotune": True,
                "linear_backend": "flashinfer_cutlass",
            },
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={"size": {"longest_edge": 128}},
            mm_processor_cache_gb=0 if startup == 2 else 4,
            skip_mm_profiling=skip_mm_profiling,
        ) as model:
            profiles = model.collective_rpc(
                lambda worker: vision_tuning_profiles(worker.get_model())
            )
            assert all(
                bool(rank_profiles) != skip_mm_profiling for rank_profiles in profiles
            )
            assert all(rank_profiles == profiles[0] for rank_profiles in profiles)
            embeddings.append(
                model.embed(
                    ["<s>passage: [IMG]"],
                    images=[Image.new("RGB", (128, 128), color="red")],
                )
            )

        if tensor_parallel_size > 1:
            snapshots = {
                path: (path.stat().st_mtime_ns, path.read_bytes())
                for path in tmp_path.glob("*/autotune_configs_dp*_rank*.json")
            }
            # Skipped and enabled profiling have separate config identities,
            # with one complete cache file per rank in each directory.
            assert len(snapshots) == tensor_parallel_size * (
                1 if skip_mm_profiling else 2
            )
            if not skip_mm_profiling:
                assert (
                    sum(
                        bool(set(json.loads(contents)).intersection(profiles[0]))
                        for _, contents in snapshots.values()
                    )
                    == tensor_parallel_size
                )
            if startup == 2:
                assert snapshots == previous_snapshot
            previous_snapshot = snapshots

    check_embeddings_close(
        embeddings_0_lst=embeddings[0],
        embeddings_1_lst=embeddings[1],
        name_0="encoder-profiling-skipped",
        name_1="encoder-autotuned",
    )
    check_embeddings_close(
        embeddings_0_lst=embeddings[1],
        embeddings_1_lst=embeddings[2],
        name_0="fresh-tuning",
        name_1="cached-tuning",
    )
