# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fnmatch
import multiprocessing as mp
import os
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.model_executor.model_loader import ShardedStateLoader
from vllm.transformers_utils.repo_utils import hf_api

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
    ignore_eos=True,
)


def test_filter_subtensors():
    state_dict = {
        "a": torch.empty(2),
        "b": torch.empty((2, 4)),
        "c": torch.empty((2, 4, 8)),
    }
    state_dict.update(
        {
            "x": state_dict["b"],
            "y": state_dict["c"][1, 2, :],
            "z": state_dict["c"][1, :, 4],
        }
    )
    filtered_state_dict = ShardedStateLoader._filter_subtensors(state_dict)
    assert tuple(filtered_state_dict.keys()) == ("a", "b", "c")
    for key, tensor in filtered_state_dict.items():
        # NOTE: don't use `equal` here, as the tensor might contain NaNs
        assert tensor is state_dict[key]


def _mock_pp_group(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    *,
    pp_rank: int = 0,
    tp_rank: int = 0,
) -> None:
    """Stub the parallel-state accessors used by the sharded state loader."""
    monkeypatch.setattr(
        "vllm.distributed.get_pp_group",
        lambda: SimpleNamespace(world_size=world_size, rank_in_group=pp_rank),
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_rank",
        lambda: tp_rank,
    )
    monkeypatch.setattr(
        "vllm.distributed.get_pipeline_model_parallel_rank",
        lambda: pp_rank,
    )


@pytest.mark.parametrize(
    "world_size,expected",
    [
        (1, ShardedStateLoader.TP_ONLY_PATTERN),
        (2, ShardedStateLoader.PP_AND_TP_PATTERN),
    ],
)
def test_get_default_pattern(monkeypatch, world_size, expected):
    """The default pattern is chosen from the pipeline parallel group size."""
    _mock_pp_group(monkeypatch, world_size)
    assert ShardedStateLoader._get_default_pattern() == expected


def test_render_default_pattern_default_tp(monkeypatch):
    """Without a custom pattern and PP=1, legacy TP-only names are rendered."""
    _mock_pp_group(monkeypatch, 1)
    assert (
        ShardedStateLoader._render_default_pattern(part=0)
        == "model-rank-0-part-0.safetensors"
    )


def test_render_default_pattern_default_pp(monkeypatch):
    """Without a custom pattern and PP>1, pp/tp-aware names are rendered."""
    _mock_pp_group(monkeypatch, 2)
    assert (
        ShardedStateLoader._render_default_pattern(part=0)
        == "model-pp-0-tp-0-part-0.safetensors"
    )


def test_render_default_pattern_custom_tp_only(monkeypatch):
    """A custom TP-only pattern is honored when pipeline parallelism is off."""
    _mock_pp_group(monkeypatch, 1)
    assert (
        ShardedStateLoader._render_default_pattern(
            ShardedStateLoader.TP_ONLY_PATTERN, part=0
        )
        == "model-rank-0-part-0.safetensors"
    )


def test_render_default_pattern_tp_only_with_pp_fails(monkeypatch):
    """TP-only filenames would collide across pipeline stages; fail fast."""
    _mock_pp_group(monkeypatch, 2)
    with pytest.raises(ValueError, match="does not embed the pipeline parallel"):
        ShardedStateLoader._render_default_pattern(
            ShardedStateLoader.TP_ONLY_PATTERN, part=0
        )


def test_render_default_pattern_pp_pattern_without_pp_fails(monkeypatch):
    """A pp/tp-aware pattern cannot be rendered when pipeline parallelism is off."""
    _mock_pp_group(monkeypatch, 1)
    with pytest.raises(ValueError, match="requires pipeline parallelism"):
        ShardedStateLoader._render_default_pattern(
            ShardedStateLoader.PP_AND_TP_PATTERN, part=0
        )


def test_render_default_pattern_custom_pp(monkeypatch):
    """A custom pp/tp-aware pattern is honored when pipeline parallelism is on."""
    _mock_pp_group(monkeypatch, 2)
    assert (
        ShardedStateLoader._render_default_pattern(
            ShardedStateLoader.PP_AND_TP_PATTERN, part=0
        )
        == "model-pp-0-tp-0-part-0.safetensors"
    )


@pytest.mark.parametrize(
    "pp_rank,tp_rank,expected",
    [
        (0, 0, "model-pp-0-tp-0-part-0.safetensors"),
        (1, 2, "model-pp-1-tp-2-part-0.safetensors"),
        (3, 5, "model-pp-3-tp-5-part-0.safetensors"),
    ],
)
def test_render_default_pattern_non_zero_ranks(monkeypatch, pp_rank, tp_rank, expected):
    """The rendered filename must embed the actual pp/tp ranks, not zeros."""
    _mock_pp_group(monkeypatch, 2, pp_rank=pp_rank, tp_rank=tp_rank)
    assert (
        ShardedStateLoader._render_default_pattern(
            ShardedStateLoader.PP_AND_TP_PATTERN, part=0
        )
        == expected
    )


def test_render_default_pattern_part_wildcard(monkeypatch):
    """The glob wildcard '*' passes through as the part token for file lookup."""
    _mock_pp_group(monkeypatch, 1, tp_rank=4)
    assert (
        ShardedStateLoader._render_default_pattern(part="*")
        == "model-rank-4-part-*.safetensors"
    )


def test_save_model_pp_aware_filenames(monkeypatch, tmp_path):
    """save_model writes pp/tp-aware filenames for pipeline-parallel models."""
    _mock_pp_group(monkeypatch, 2, pp_rank=1, tp_rank=2)

    saved: list[str] = []

    def _capture(tensors, path, **kwargs):
        saved.append(os.path.basename(path))

    monkeypatch.setattr("safetensors.torch.save_file", _capture)

    model = SimpleNamespace(
        state_dict=lambda: {"w": torch.zeros(4, dtype=torch.float32)}
    )
    ShardedStateLoader.save_model(model, str(tmp_path))
    assert saved == ["model-pp-1-tp-2-part-0.safetensors"]


def test_save_model_part_split(monkeypatch, tmp_path):
    """save_model rolls over to a new part when max_size is exceeded."""
    _mock_pp_group(monkeypatch, 1, tp_rank=0)

    saved: list[str] = []

    def _capture(tensors, path, **kwargs):
        saved.append(os.path.basename(path))

    monkeypatch.setattr("safetensors.torch.save_file", _capture)

    # Two float32[4] tensors (16 B each); max_size=20 fits one but not two.
    model = SimpleNamespace(
        state_dict=lambda: {
            "a": torch.zeros(4, dtype=torch.float32),
            "b": torch.zeros(4, dtype=torch.float32),
        }
    )
    ShardedStateLoader.save_model(model, str(tmp_path), max_size=20)
    assert saved == [
        "model-rank-0-part-0.safetensors",
        "model-rank-0-part-1.safetensors",
    ]


@pytest.fixture(scope="module")
def llama_3p2_1b_files():
    input_dir = hf_api().snapshot_download(
        "meta-llama/Llama-3.2-1B-Instruct", ignore_patterns=["*.bin*", "original/*"]
    )

    yield input_dir


def _run_writer(input_dir, output_dir, weights_patterns, **kwargs):
    llm_sharded_writer = LLM(model=input_dir, **kwargs)

    # Dump worker states to output directory
    llm_sharded_writer.llm_engine.engine_core.save_sharded_state(path=output_dir)

    # Copy metadata files to output directory
    for file in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, file)):
            shutil.copytree(
                os.path.join(input_dir, file), os.path.join(output_dir, file)
            )
        elif not any(fnmatch.fnmatch(file, ext) for ext in weights_patterns):
            shutil.copy(os.path.join(input_dir, file), output_dir)


def _run_generate(input_dir, queue: mp.Queue, **kwargs):
    llm = LLM(model=input_dir, **kwargs)
    gen = llm.generate(prompts, sampling_params)
    queue.put([g.outputs[0].__dict__ for g in gen])
    queue.close()
    queue.join_thread()


@pytest.mark.parametrize("enable_lora", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_sharded_state_loader(
    enable_lora, tp_size, num_gpus_available, llama_3p2_1b_files
):
    if num_gpus_available < tp_size:
        pytest.skip(f"Not enough GPUs for tensor parallelism {tp_size}")

    weights_patterns = ("*.safetensors",)
    gpu_memory_utilization = 0.8
    input_dir = llama_3p2_1b_files
    ctx = mp.get_context("spawn")

    # Keep batching deterministic: this test compares exact greedy outputs
    # across separate engine processes, whose schedulers may otherwise form
    # different batches depending on prompt-processing timing.
    platform_args = {"max_num_seqs": 1}

    # Run in separate processes for memory & CUDA isolation
    with TemporaryDirectory() as output_dir:
        p = ctx.Process(
            target=_run_writer,
            args=(input_dir, output_dir, weights_patterns),
            kwargs=dict(
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=True,
                **platform_args,
            ),
        )
        p.start()
        p.join()

        queue = ctx.Queue()

        p = ctx.Process(
            target=_run_generate,
            args=(input_dir, queue),
            kwargs=dict(
                enable_lora=enable_lora,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tp_size,
                **platform_args,
            ),
        )
        p.start()
        # Call queue.get() before p.join() to prevent deadlock:
        # If p.join() is called before queue.get() and the queue is full,
        # the child process may block while writing to the queue and never
        # terminate, causing the parent to wait indefinitely on p.join().
        # See: https://github.com/vllm-project/vllm/pull/22371#discussion_r2257773814
        out_before = queue.get()
        p.join()
        queue.close()
        queue.join_thread()

        queue = ctx.Queue()

        p = ctx.Process(
            target=_run_generate,
            args=(output_dir, queue),
            kwargs=dict(
                enable_lora=enable_lora,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tp_size,
                load_format="sharded_state",
                **platform_args,
            ),
        )
        p.start()
        # Call queue.get() before p.join() to prevent deadlock:
        # If p.join() is called before queue.get() and the queue is full,
        # the child process may block while writing to the queue and never
        # terminate, causing the parent to wait indefinitely on p.join().
        # See: https://github.com/vllm-project/vllm/pull/22371#discussion_r2257773814
        out_after = queue.get()
        p.join()
        queue.close()
        queue.join_thread()

        assert out_before == out_after


def _run_sharded_state_round_trip(
    input_dir: str,
    *,
    tp_size: int,
    pp_size: int,
    num_gpus_available: int,
) -> None:
    """Save a TP/PP sharded checkpoint, reload it, and compare outputs.

    Args:
        input_dir: directory containing the original (un-sharded) model.
        tp_size: tensor parallel size used for both save and load.
        pp_size: pipeline parallel size used for both save and load.
        num_gpus_available: number of GPUs on the current machine.
    """
    if num_gpus_available < tp_size * pp_size:
        pytest.skip(f"Not enough GPUs for TP={tp_size}, PP={pp_size}")

    weights_patterns = ("*.safetensors",)
    gpu_memory_utilization = 0.8
    ctx = mp.get_context("spawn")

    # Keep batching deterministic: this test compares exact greedy outputs
    # across separate engine processes, whose schedulers may otherwise form
    # different batches depending on prompt-processing timing.
    platform_args = {"max_num_seqs": 1}
    parallel_args = dict(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
    )

    with TemporaryDirectory() as output_dir:
        p = ctx.Process(
            target=_run_writer,
            args=(input_dir, output_dir, weights_patterns),
            kwargs=dict(
                **parallel_args,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=True,
                **platform_args,
            ),
        )
        p.start()
        p.join()
        assert p.exitcode == 0, f"writer process failed with exit code {p.exitcode}"

        if pp_size > 1:
            # PP checkpoints must use the pp/tp-aware naming scheme so that
            # different pipeline stages write distinct files.
            sharded_files = [
                f for f in os.listdir(output_dir) if f.endswith(".safetensors")
            ]
            assert sharded_files, "no sharded checkpoint files were written"
            assert any(
                fnmatch.fnmatch(f, "model-pp-*-tp-*-part-*.safetensors")
                for f in sharded_files
            ), f"expected PP-aware filenames, got {sharded_files}"

        queue = ctx.Queue()
        p = ctx.Process(
            target=_run_generate,
            args=(input_dir, queue),
            kwargs=dict(
                enable_lora=False,
                gpu_memory_utilization=gpu_memory_utilization,
                **parallel_args,
                **platform_args,
            ),
        )
        p.start()
        # Call queue.get() before p.join() to prevent deadlock:
        # If p.join() is called before queue.get() and the queue is full,
        # the child process may block while writing to the queue and never
        # terminate, causing the parent to wait indefinitely on p.join().
        # See: https://github.com/vllm-project/vllm/pull/22371#discussion_r2257773814
        out_before = queue.get()
        p.join()
        assert p.exitcode == 0, f"generate process failed with exit code {p.exitcode}"
        queue.close()
        queue.join_thread()

        queue = ctx.Queue()
        p = ctx.Process(
            target=_run_generate,
            args=(output_dir, queue),
            kwargs=dict(
                enable_lora=False,
                gpu_memory_utilization=gpu_memory_utilization,
                load_format="sharded_state",
                **parallel_args,
                **platform_args,
            ),
        )
        p.start()
        out_after = queue.get()
        p.join()
        assert p.exitcode == 0, f"generate process failed with exit code {p.exitcode}"
        queue.close()
        queue.join_thread()

        assert out_before == out_after


def test_sharded_state_loader_pp(num_gpus_available, llama_3p2_1b_files):
    """Round-trip sharded state save/load with pipeline parallelism (PP=2)."""
    _run_sharded_state_round_trip(
        llama_3p2_1b_files,
        tp_size=1,
        pp_size=2,
        num_gpus_available=num_gpus_available,
    )


def test_sharded_state_loader_pp_tp(num_gpus_available, llama_3p2_1b_files):
    """Round-trip sharded state save/load with pipeline + tensor parallelism
    (PP=2, TP=2)."""
    _run_sharded_state_round_trip(
        llama_3p2_1b_files,
        tp_size=2,
        pp_size=2,
        num_gpus_available=num_gpus_available,
    )
