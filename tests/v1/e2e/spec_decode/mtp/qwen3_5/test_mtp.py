# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf
from vllm.platforms import current_platform
from vllm.v1.metrics.reader import Vector

from ...utils import (
    assert_request_outputs_match,
    compute_acceptance_len,
    compute_acceptance_rate,
    get_spec_decode_metric_value,
    get_test_prompts,
)
from .._correctness import check_mtp_correctness


@single_gpu_only
@pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(90)
    ),
    reason="Qwen GDN batch invariance currently requires CUDA SM90.",
)
@pytest.mark.parametrize("num_spec_tokens", [0, 1, 2, 3])
def test_mtp_batch_invariance(
    monkeypatch: pytest.MonkeyPatch,
    num_spec_tokens: int,
    vllm_runner,
):
    """MTP must preserve seeded token/logprob outputs when requests are regrouped."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    monkeypatch.setenv("VLLM_GPU_SYNC_CHECK", "error")
    manual = (
        "Routine facility inspections are recorded in the audit ledger every day. "
        * 100
    )
    prompts = [
        manual + "\nWhat is the capital of France?",
        "Explain the water cycle.",
        manual + "\nCompute 17 + 29.",
        "Write a short story about a lighthouse.",
    ]
    params = [
        SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=32,
            ignore_eos=True,
            logprobs=5,
            seed=1000 + i,
        )
        for i in range(len(prompts))
    ]
    cold_params = [param.clone() for param in params]
    for param in cold_params:
        param.skip_reading_prefix_cache = True
    with vllm_runner(
        "Qwen/Qwen3.5-0.8B-Base",
        block_size=None,
        enable_chunked_prefill=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=2048,
        max_num_seqs=2,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        attention_config={"backend": "FLASH_ATTN"},
        additional_config={"gdn_prefill_backend": "triton"},
        **(
            {
                "speculative_config": {
                    "method": "mtp",
                    "num_speculative_tokens": num_spec_tokens,
                }
            }
            if num_spec_tokens
            else {}
        ),
        disable_log_stats=False,
    ) as runner:
        llm = runner.llm
        assert len(llm.get_tokenizer().encode(manual)) > 1024

        def selected(output):
            completion = output.outputs[0]
            tokens = list(completion.token_ids)
            assert completion.logprobs is not None
            logprobs = [
                step[token].logprob
                for token, step in zip(tokens, completion.logprobs, strict=True)
            ]
            return tokens, logprobs

        serial = [
            selected(
                llm.generate(
                    [prompt],
                    [param],
                    use_tqdm=False,
                )[0]
            )
            for prompt, param in zip(prompts, cold_params, strict=True)
        ]
        warm_outputs = llm.generate(prompts, params, use_tqdm=False)
        cached_tokens = [warm_outputs[i].num_cached_tokens for i in (0, 2)]
        assert all(count is not None and count > 0 for count in cached_tokens)
        batched = [selected(output) for output in warm_outputs]
        reversed_batch = [
            selected(output)
            for output in llm.generate(prompts[::-1], params[::-1], use_tqdm=False)
        ][::-1]
        cold_batch = [
            selected(output)
            for output in llm.generate(
                prompts,
                cold_params,
                use_tqdm=False,
            )
        ]
        if num_spec_tokens:
            assert compute_acceptance_rate(llm.get_metrics()) > 0
        assert serial == batched == reversed_batch == cold_batch


def _add_random_mtp_module(model_name: str, path: Path) -> Path:
    """Copy of `model_name` with a second MTP module of random weights."""
    src = Path(download_weights_from_hf(model_name, None, allow_patterns=["*"]))
    for file in src.iterdir():
        if file.name not in ("config.json", "model.safetensors.index.json"):
            (path / file.name).symlink_to(file)
    config = json.loads((src / "config.json").read_text())
    config["text_config"]["mtp_num_hidden_layers"] = 2
    (path / "config.json").write_text(json.dumps(config))

    index = json.loads((src / "model.safetensors.index.json").read_text())
    generator = torch.Generator().manual_seed(0)
    module_1 = {}
    for name, shard in index["weight_map"].items():
        if name.startswith("mtp.layers.0."):
            with safe_open(src / shard, framework="pt") as f:
                weight = f.get_tensor(name)
            module_1[name.replace("layers.0.", "layers.1.")] = torch.randn(
                weight.shape, generator=generator
            ).to(weight.dtype)
    save_file(module_1, path / "mtp_module_1.safetensors")
    index["weight_map"].update(dict.fromkeys(module_1, "mtp_module_1.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    return path


@single_gpu_only
def test_mtp_draft_steps_use_their_own_module(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    tmp_path: Path,
    vllm_runner,
):
    """With two MTP modules, draft step 2 must run module 1, also with the
    default drafter CUDA graphs. Module 1 has random weights, so its drafts are
    rejected while module 0's are still accepted."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    model = _add_random_mtp_module("Qwen/Qwen3.5-0.8B-Base", tmp_path)
    with vllm_runner(
        str(model),
        block_size=None,
        enable_chunked_prefill=None,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 2,
            # Also covers get_top_tokens(..., spec_step_idx) on this path.
            "use_local_argmax_reduction": True,
        },
        disable_log_stats=False,
    ) as runner:
        runner.llm.chat(get_test_prompts(mm_enabled=False), sampling_config)
        metrics = runner.llm.get_metrics()

    num_drafts = get_spec_decode_metric_value(metrics, "vllm:spec_decode_num_drafts")
    accepted_per_pos = next(
        m.values
        for m in metrics
        if isinstance(m, Vector)
        and m.name == "vllm:spec_decode_num_accepted_tokens_per_pos"
    )
    step_1, step_2 = (accepted / num_drafts for accepted in accepted_per_pos)
    assert step_1 > 0.3, f"module 0 acceptance {step_1:.3f}"
    assert step_2 < 0.05, f"module 1 acceptance {step_2:.3f}: module 0 was reused"


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        (
            ("mtp", "Qwen/Qwen3.5-0.8B-Base", 1, None),
            False,
            0.20,
        ),  # hybrid + MTP, ref: ~34%-35%
    ],
    ids=["qwen3_5-hybrid"],
)
@single_gpu_only
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int, str | None],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    vllm_runner,
):
    check_mtp_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        vllm_runner,
    )


@single_gpu_only
def test_mtp_draft_lm_head_quantization(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    vllm_runner,
):
    """A quantized draft copy of the shared lm_head only changes drafting:
    greedy outputs match stock MTP and acceptance stays within 10%."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    prompts = get_test_prompts(mm_enabled=False)
    results = {}
    # NVFP4 needs a CUDA W4A16 kernel.
    quants = (None, "fp8", "nvfp4") if current_platform.is_cuda() else (None, "fp8")
    for quant in quants:
        with vllm_runner(
            "Qwen/Qwen3.5-0.8B-Base",
            block_size=None,
            enable_chunked_prefill=None,
            max_model_len=2048,
            limit_mm_per_prompt={"image": 0, "video": 0},
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": 3,
                "draft_lm_head_quantization": quant,
            },
            disable_log_stats=False,
        ) as runner:
            outputs = runner.llm.chat(prompts, sampling_config)
            results[quant] = (outputs, compute_acceptance_len(runner.llm.get_metrics()))

    ref_outputs, ref_al = results.pop(None)
    for quant, (outputs, al) in results.items():
        assert_request_outputs_match(
            ref_outputs,
            outputs,
            # Not bit-exact: a different acceptance length changes the target's
            # verify batch shapes, which can flip a near-tied greedy token.
            required_matches=int(0.8 * len(ref_outputs)) + 1,
            context=f"draft_lm_head_quantization={quant}",
        )
        assert al >= 0.9 * ref_al, (
            f"{quant}: acceptance length {al:.3f} vs {ref_al:.3f}"
        )


@single_gpu_only
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="nvfp4 needs a CUDA W4A16 kernel"
)
def test_mtp_draft_lm_head_quantization_survives_sleep_and_reload(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    vllm_runner,
):
    """The quantized draft head is derived from the target head: it must come
    back intact after a level 1 sleep and be re-derived after a target weight
    reload, so outputs and acceptance match the run before."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    prompts = get_test_prompts(mm_enabled=False)
    with vllm_runner(
        "Qwen/Qwen3.5-0.8B-Base",
        block_size=None,
        enable_chunked_prefill=None,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
            "draft_lm_head_quantization": "nvfp4",
        },
        enable_sleep_mode=True,
        disable_log_stats=False,
    ) as runner:
        llm = runner.llm

        def run():
            before = llm.get_metrics()
            outputs = llm.chat(prompts, sampling_config)
            return outputs, compute_acceptance_len(llm.get_metrics(), before)

        ref_outputs, ref_al = run()

        llm.sleep(level=1)
        llm.wake_up()
        outputs, al = run()
        assert_request_outputs_match(
            ref_outputs,
            outputs,
            required_matches=int(0.8 * len(ref_outputs)) + 1,
            context="after level 1 sleep",
        )
        assert al == pytest.approx(ref_al, rel=0.03)

        llm.collective_rpc("reload_weights")
        outputs, al = run()
        assert_request_outputs_match(
            ref_outputs,
            outputs,
            required_matches=int(0.8 * len(ref_outputs)) + 1,
            context="after weight reload",
        )
        assert al == pytest.approx(ref_al, rel=0.03)
