# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from tests.utils import multi_gpu_test
from tests.v1.e2e.spec_decode.utils import compute_acceptance_len
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DRAFT = "nm-testing/Llama3_2_1B_speculator.eagle3"
PROMPTS = [
    "The capital of France is",
    "2 + 2 equals",
    "In one word, the color of the sky is",
    "Q: If a train travels 60 miles in 1.5 hours, what is its average speed?\nA:",
]

# Acceptance held within 2% across PP=1..4; a dropped state cost ~11%.
ACCEPTANCE_TOLERANCE = 0.95


def _run(pp_size: int) -> float:
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        pipeline_parallel_size=pp_size,
        max_model_len=512,
        gpu_memory_utilization=0.45,
        disable_log_stats=False,
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
        speculative_config={
            "method": "eagle3",
            "model": DRAFT,
            "num_speculative_tokens": 3,
        },
    )
    try:
        llm.generate(
            PROMPTS,
            SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True),
        )
        acceptance = compute_acceptance_len(llm.get_metrics())
        assert acceptance > 1
        return acceptance
    finally:
        del llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def draft_without_embed_tokens(tmp_path_factory) -> str:
    """``DRAFT`` with its input embedding stripped out.

    Most EAGLE3 checkpoints ship no ``embed_tokens`` and alias the target's --
    yuhuili/EAGLE3-LLaMA3.1-Instruct-8B, which test_eagle_correctness.py runs, is
    one. ``DRAFT`` carries its own, so on its own it never exercises that path.
    Removing the key reproduces the shared-embedding class at 1B.
    """
    from safetensors.torch import load_file, save_file

    from vllm.transformers_utils.repo_utils import hf_api

    src = Path(
        hf_api().snapshot_download(
            repo_id=DRAFT,
            repo_type="model",
            allow_patterns=["*.json", "*.py", "*.safetensors"],
        )
    )
    dst = tmp_path_factory.mktemp("eagle3_shared_embed")
    for path in src.iterdir():
        if path.is_file() and path.suffix in (".json", ".py"):
            shutil.copy(path, dst / path.name)

    tensors = load_file(src / "model.safetensors")
    stripped = [name for name in tensors if "embed_tokens" in name]
    assert stripped, f"{DRAFT} has no embed_tokens to strip"
    for name in stripped:
        del tensors[name]
    save_file(tensors, str(dst / "model.safetensors"), metadata={"format": "pt"})
    return str(dst)


@multi_gpu_test(num_gpus=2)
def test_eagle3_pipeline_parallel_shared_embedding(draft_without_embed_tokens: str):
    """A drafter with no embedding of its own must still get the target's.

    Sharing is what supplies it, and under PP the target's embedding lives on the
    first stage while the drafter runs on the last. Get this wrong and the
    embedding is never written at all: the load is skipped for a key the
    checkpoint lacks, and the drafter proposes from uninitialized memory, which
    costs acceptance without failing.
    """
    baseline = _run(1, MODEL, draft_without_embed_tokens, "FULL_AND_PIECEWISE")
    parallel = _run(2, MODEL, draft_without_embed_tokens, "FULL_AND_PIECEWISE")

    assert parallel >= baseline * ACCEPTANCE_TOLERANCE, (
        f"acceptance length regressed under PP=2 for a drafter sharing the "
        f"target's embedding: {parallel:.3f} < {baseline:.3f} * "
        f"{ACCEPTANCE_TOLERANCE}"
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model,draft", [(MODEL, DRAFT)])
@pytest.mark.parametrize("cudagraph_mode", [None, "FULL_AND_PIECEWISE"])
def test_eagle3_pipeline_parallel_acceptance(
    model: str,
    draft: str,
    cudagraph_mode: str | None,
):
    """Aux hidden states must survive the pipeline handoff.

    Compares acceptance length at PP=2 against PP=1 on the same model. This
    feature fails quietly: stale, out-of-order, or missing states still yield
    well-formed proposals but lower acceptance.

    Greedy text parity deliberately is not asserted. bf16 argmax ties break
    differently once the batch shape changes, so even two runs with spec decode
    off diverge from each other, which would make such an assertion flaky for
    reasons unrelated to this feature.
    """
    baseline = _run(1, model, draft, cudagraph_mode)
    parallel = _run(2, model, draft, cudagraph_mode)

    assert parallel >= baseline * ACCEPTANCE_TOLERANCE, (
        f"acceptance length regressed under PP=2 "
        f"(cudagraph_mode={cudagraph_mode}): "
        f"{parallel:.3f} < {baseline:.3f} * {ACCEPTANCE_TOLERANCE}"
    )


@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize("model,draft", [(MODEL, DRAFT)])
def test_eagle3_pipeline_parallel_far_stage_acceptance(model: str, draft: str):
    """Cover auxiliary states produced more than one stage before the drafter.

    PP=2 exercises none of this: its only producer is the stage right before the
    last one. PP=4 is the smallest size where an auxiliary state must be relayed
    through a middle stage.

    Full cudagraph is the mode to run this in. The layout is resolved at setup
    precisely so the forward stays capturable, and 16 layers over 4 stages splits
    evenly, so an uneven-split regression would not show up here.
    """
    baseline = _run(1, model, draft, "FULL_AND_PIECEWISE")
    parallel = _run(4, model, draft, "FULL_AND_PIECEWISE")

    assert parallel >= baseline * ACCEPTANCE_TOLERANCE, (
        f"acceptance length regressed under PP=4: "
        f"{parallel:.3f} < {baseline:.3f} * {ACCEPTANCE_TOLERANCE}"
    )
