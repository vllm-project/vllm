# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer vision kernel autotuning regression tests."""

import ast
import json

import pytest

from tests.models.utils import check_embeddings_close, check_logprobs_close
from tests.utils import multi_gpu_marks, multi_gpu_test


@pytest.mark.usefixtures("enable_pickle")
@pytest.mark.parametrize(
    "tensor_parallel_size,mm_encoder_tp_mode",
    [
        (1, "weights"),
        pytest.param(2, "weights", marks=multi_gpu_marks(num_gpus=2)),
        pytest.param(2, "data", marks=multi_gpu_marks(num_gpus=2)),
    ],
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


@multi_gpu_test(num_gpus=2)
@pytest.mark.usefixtures("enable_pickle")
@pytest.mark.parametrize("num_items", [1, 3])
def test_flashinfer_autotune_data_parallel_encoder_cache(
    vllm_runner, monkeypatch, tmp_path, num_items
):
    """Uneven data-mode batches must tune all ranks and reuse the saved cache."""
    from PIL import Image

    def encoder_info(worker):
        from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
            FlashInferCutlassMxfp8LinearKernel,
        )
        from vllm.multimodal.encoder_budget import MultiModalBudget

        model = worker.get_model()
        budget = MultiModalBudget(worker.vllm_config, worker.model_runner.mm_registry)
        qkv = model.visual.blocks[0].attn.qkv
        assert isinstance(qkv.quant_method.kernel, FlashInferCutlassMxfp8LinearKernel)
        return (
            model.use_data_parallel,
            model.visual.tp_size,
            budget.mm_max_items_per_batch["image"],
            qkv.input_size_per_partition,
            qkv.output_size_per_partition,
        )

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))
    previous_snapshot = None
    outputs = []
    for _ in range(2):
        with vllm_runner(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            dtype="bfloat16",
            quantization="online",
            quantization_config={
                "targets": {"visual.blocks.*.attn.qkv": "mxfp8"},
            },
            tensor_parallel_size=2,
            mm_encoder_tp_mode="data",
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=num_items,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
            compilation_config={"mode": 0},
            kernel_config={
                "enable_flashinfer_autotune": True,
                "enable_flashinfer_mm_encoder_autotune": True,
                "linear_backend": "flashinfer_cutlass",
            },
            limit_mm_per_prompt={"image": 1, "video": 0},
            mm_processor_kwargs={"min_pixels": 3136, "max_pixels": 3136},
        ) as model:
            info = model.collective_rpc(encoder_info)
            assert len(info) == 2
            assert all(rank_info[:3] == (True, 1, num_items) for rank_info in info)
            outputs.append(
                model.generate_greedy_logprobs(
                    ["<|vision_start|><|image_pad|><|vision_end|>Describe the image."],
                    max_tokens=8,
                    num_logprobs=5,
                    images=[Image.new("RGB", (56, 56), color="red")],
                )
            )

        snapshot = {
            path: (path.stat().st_mtime_ns, path.read_bytes())
            for path in tmp_path.glob("*/autotune_configs_dp*_rank*.json")
        }
        assert len(snapshot) == 2
        profiles = []
        for _, contents in snapshot.values():
            vision_profiles = set()
            for key in json.loads(contents):
                if key.startswith("('mxfp8_gemm',"):
                    shapes = ast.literal_eval(ast.parse(key, mode="eval").body.elts[2])
                    # Vision QKV dimensions differ from the language model's.
                    if shapes[0][-1] == info[0][3] and shapes[1][-2:] == info[0][3:]:
                        vision_profiles.add(key)
            assert vision_profiles
            profiles.append(vision_profiles)
        assert profiles[0] == profiles[1]
        if previous_snapshot is not None:
            assert snapshot == previous_snapshot
        previous_snapshot = snapshot

    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="fresh-tuning",
        name_1="cached-tuning",
    )
