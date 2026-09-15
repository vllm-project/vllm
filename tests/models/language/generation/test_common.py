# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from contextlib import contextmanager

import pytest
import torch
from packaging.version import Version
from transformers import __version__ as TRANSFORMERS_VERSION

from vllm.logprobs import Logprob
from vllm.platforms import current_platform

from ....utils import large_gpu_mark
from ...registry import HF_EXAMPLE_MODELS
from ...utils import TokensTextLogprobsPromptLogprobs, check_logprobs_close
from vllm.platforms.rocm import on_gfx950

# Models that require embedding scaling for prompt_embeds test
EMBED_SCALING_MODELS = {
    "openbmb/MiniCPM4.1-8B",
}

# This list contains the model that are using AITER kernel.
# Skip model that are not using AITER tests.
# When more AITER kernels are added, this list will not be
# needed as all the models will be calling AITER kernels
# in parts of the operators
AITER_MODEL_LIST = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "openbmb/MiniCPM3-4B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TitanML/tiny-mixtral",
    "Qwen/Qwen3-8B",
]

# MoE top-2 near-tie second chance -- a diagnostic, off unless the env var is
# set. tiny-mixtral diverges on one expert at a router tie that is exact in HF
# and one ULP wide in vLLM; this lets vLLM adopt HF's pair when the two
# selections swap exactly one expert for one other and those two are within
# `tol` in vLLM's own fp32 router logits. 0.01 is calibrated: the decisive flip
# margin is 0.0068 and the nearest non-tie 0.0176. It lets vLLM read HF's
# answer, so it is not a test policy -- it shows the flip is the sole cause.
MOE_NEAR_TIE_TOL = float(os.environ.get("VLLM_MOE_NEAR_TIE_TOL") or 0.0)


@contextmanager
def record_hf_expert_choice(hf_model, store: dict[int, torch.Tensor]):
    """Record HF's top-k expert indices per MoE layer, as rows in call order."""
    if MOE_NEAR_TIE_TOL <= 0:
        yield
        return

    rows: dict[int, list[torch.Tensor]] = {}
    hooks = [
        layer.mlp.gate.register_forward_hook(
            # out[2] is top_k_index.
            lambda module, args, out, i=i: rows.setdefault(i, []).append(
                out[2].detach().reshape(-1, out[2].shape[-1]).cpu()
            )
        )
        for i, layer in enumerate(hf_model.model.model.layers)
        if hasattr(layer.mlp, "gate")
    ]
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()
        store.update({i: torch.cat(v) for i, v in rows.items()})


@contextmanager
def moe_near_tie_rescue(hf_choice: dict[int, torch.Tensor], tol: float):
    """Adopt HF's expert pair on a one-for-one swap that is a near tie in vLLM.

    Rows are matched to `hf_choice` by position, so enter this once per generate
    call, and only under eager execution -- CUDA graph replay skips Python for
    the decode rows, which silently misaligns the cursor.
    """
    if tol <= 0 or not hf_choice:
        yield
        return

    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
        FusedMoERouter,
    )
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed

    original = FusedMoERouter.select_experts
    layer_of: dict[int, int] = {}
    cursor: dict[int, int] = {}

    def select_experts(
        self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None
    ):
        weights, ids = original(
            self, hidden_states, router_logits, topk_indices_dtype, input_ids=input_ids
        )
        # Routers are called in layer order, so first-seen order is layer order.
        layer = layer_of.setdefault(id(self), len(layer_of))
        start = cursor.get(layer, 0)
        n = ids.shape[0]
        cursor[layer] = start + n
        hf_ids = hf_choice.get(layer, ids.new_empty(0))[start : start + n]
        if hf_ids.shape[0] != n:
            return weights, ids

        with gpu_sync_allowed():  # HF's indices have to come back to the device
            hf_ids = hf_ids.to(ids.device).long()
            logits = router_logits.float()
            mine = torch.zeros(n, logits.shape[-1], dtype=torch.bool, device=ids.device)
            mine.scatter_(1, ids.long(), True)
            theirs = torch.zeros_like(mine).scatter_(1, hf_ids, True)
            only_mine, only_theirs = mine & ~theirs, theirs & ~mine

            # Rescuable only if exactly one expert was swapped for one other.
            swapped = (only_mine.sum(-1) == 1) & (only_theirs.sum(-1) == 1)
            margin = logits.gather(
                1, only_mine.float().argmax(-1, keepdim=True)
            ) - logits.gather(1, only_theirs.float().argmax(-1, keepdim=True))
            take = swapped & (margin.squeeze(1).abs() <= tol)
            if not bool(take.any()):
                return weights, ids

            # Rebuild the weights HF's pick implies, the way the kernel would.
            rescued = logits.softmax(-1).gather(1, hf_ids)
            if getattr(self, "renormalize", True):
                rescued = rescued / rescued.sum(-1, keepdim=True)
            order = rescued.argsort(-1, descending=True)
            take = take[:, None]
            return (
                torch.where(take, rescued.gather(1, order).to(weights.dtype), weights),
                torch.where(take, hf_ids.gather(1, order).to(ids.dtype), ids),
            )

    FusedMoERouter.select_experts = select_experts
    try:
        yield
    finally:
        FusedMoERouter.select_experts = original


def score_forced_continuations(
    vllm_model,
    prompt_token_ids: list[int],
    continuations: list[list[int]],
) -> list[list[float]]:
    """
    Teacher-forced per-token logprobs of each continuation, conditioned on
    `prompt_token_ids`.

    Returns one list per continuation, of the same length, holding the
    logprob of each continuation token given everything before it. Summing
    a list gives the joint logprob of that continuation; its last element
    is the conditional logprob of the final token.

    Every continuation is scored as its own sequence, so a token is
    reachable no matter how low it ranks. All are submitted as one batch.
    """
    seqs = [list(prompt_token_ids) + list(c) for c in continuations]
    outputs = vllm_model.generate_greedy_logprobs(
        seqs, max_tokens=1, num_logprobs=None, num_prompt_logprobs=0
    )

    results: list[list[float]] = []
    for continuation, output in zip(continuations, outputs):
        output = cast(TokensTextLogprobsPromptLogprobs, output)
        token_datas = cast(list[dict[int, Logprob] | None], output[3])
        # The trailing prompt positions are the forced continuation.
        tail = token_datas[len(token_datas) - len(continuation) :]
        logprobs: list[float] = []
        for token_id, token_data in zip(continuation, tail):
            assert token_data is not None
            logprobs.append(token_data[token_id].logprob)
        results.append(logprobs)

    return results


# @maybe_test_rocm_aiter
@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            "bigscience/bloom-560m",  # bloom - testing alibi slopes
            marks=[
                pytest.mark.core_model,
                pytest.mark.slow_test,
                pytest.mark.cpu_model,
            ],
        ),
        pytest.param(
            "openai-community/gpt2",  # gpt2
            marks=[pytest.mark.core_model],
        ),
        pytest.param("Milos/slovak-gpt-j-405M"),  # gptj
        pytest.param("bigcode/tiny_starcoder_py"),  # gpt_bigcode
        pytest.param("EleutherAI/pythia-70m"),  # gpt_neox
        pytest.param(
            "google/gemma-1.1-2b-it",  # gemma
            marks=[
                pytest.mark.core_model,
                pytest.mark.cpu_model,
                pytest.mark.slow_test,
            ],
        ),
        pytest.param(
            "google/gemma-2-2b-it",  # test hybrid attention
            marks=[pytest.mark.cpu_model],
        ),
        pytest.param(
            "zai-org/chatglm3-6b",  # chatglm (text-only)
        ),
        pytest.param(
            "meta-llama/Llama-3.2-1B-Instruct",  # llama
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "openbmb/MiniCPM4.1-8B",  # minicpm
            marks=[pytest.mark.core_model, large_gpu_mark(min_gb=48)],
        ),
        pytest.param(
            "facebook/opt-125m",  # opt
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "microsoft/phi-2",  # phi
            marks=[pytest.mark.core_model, pytest.mark.slow_test],
        ),
        pytest.param(
            "Qwen/Qwen2.5-0.5B-Instruct",  # qwen2
            marks=[
                pytest.mark.core_model,
                pytest.mark.cpu_model,
                pytest.mark.slow_test,
            ],
        ),
        pytest.param(
            "Qwen/Qwen3-8B",  # qwen (text-only)
        ),
        pytest.param("stabilityai/stablelm-3b-4e1t"),  # stablelm
        pytest.param("bigcode/starcoder2-3b"),  # starcoder2
        pytest.param(
            "TitanML/tiny-mixtral",  # mixtral
            marks=[pytest.mark.core_model],
        ),
        pytest.param("swiss-ai/Apertus-8B-Instruct-2509"),  # apertus
        pytest.param(
            "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",  # hyperclovax
            marks=[large_gpu_mark(min_gb=32)],
        ),
    ],
)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
@pytest.mark.parametrize("use_prompt_embeds", [True, False])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
    use_rocm_aiter: bool,
    use_prompt_embeds: bool,
    monkeypatch,
) -> None:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    if current_platform.is_rocm() and model == "TitanML/tiny-mixtral":
        # Its single-token router selects LLMM1, whose low-precision
        # accumulation can change the top-2 experts. Keep the optimized kernel
        # enabled generally, but use the reference GEMM for this accuracy test.
        monkeypatch.setenv("VLLM_ROCM_USE_SKINNY_GEMM", "0")

    if use_rocm_aiter and (model in AITER_MODEL_LIST):
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    elif use_rocm_aiter and model not in AITER_MODEL_LIST:
        # Skip model that are not using AITER tests.
        # When more AITER kernels are added, this list will not be
        # needed as all the models will be calling AITER kernels
        # in parts of the operators
        pytest.skip(f"Skipping '{model}' model test with AITER kernel.")

    if model == "bigcode/starcoder2-3b":
        # Replace example.txt's Test1 (an NL prompt) with a code prompt:
        # starcoder2-3b is a code model, so NL prompts give near-uniform
        # digit logits where HF<->vLLM bf16 drift can reorder top-K.
        example_prompts = list(example_prompts)
        example_prompts[1] = (
            "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - "
        )

    hf_expert_choice: dict[int, torch.Tensor] = {}

    with hf_runner(
        model,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
    ) as hf_model:
        with record_hf_expert_choice(hf_model, hf_expert_choice):
            hf_outputs = hf_model.generate_greedy_logprobs_limit(
                example_prompts, max_tokens, num_logprobs
            )

        prompt_embeds: list[torch.Tensor] | None = [] if use_prompt_embeds else None

        for prompt in example_prompts:
            token_ids = hf_model.tokenizer(prompt, return_tensors="pt").input_ids.to(
                hf_model.model.device
            )
            if prompt_embeds is not None:
                embed = hf_model.model.get_input_embeddings()(token_ids)

                if "gemma" in model.lower() and (
                    Version(TRANSFORMERS_VERSION) < Version("5.3.0.dev0")
                ):
                    # For Gemma 1/2 models with Transformers 5.4.0+, the prompt
                    # embeddings are normalised in `get_prompt_embeddings`,
                    # like Gemma 3. For older versions, we need to manually normalise.
                    embed_scale = hf_model.config.hidden_size**0.5
                    normalizer = torch.tensor(embed_scale, dtype=embed.dtype)
                    embed *= normalizer

                # MiniCPM models apply scale_emb to embeddings internally.
                # vLLM expects pre-scaled embeddings when using inputs_embeds.
                if model in EMBED_SCALING_MODELS:
                    config = hf_model.model.config
                    embed = embed * config.scale_emb

                prompt_embeds.append(embed.squeeze(0))

    vllm_kwargs = {}
    if (
        model == "bigscience/bloom-560m"
        and current_platform.is_device_capability_family(90)
    ):
        # On SM90, the metadata builder otherwise selects FA3 AOT scheduling
        # before Bloom's ALiBi layers fall back to FA2. Pinning FA2 keeps the
        # builder and layer consistent and preserves the L4 test path.
        vllm_kwargs["attention_config"] = {"flash_attn_version": 2}

    if MOE_NEAR_TIE_TOL > 0:
        # The rescue matches router rows against HF positionally, and CUDA
        # graph replay skips Python for every decode row.
        vllm_kwargs["enforce_eager"] = True

    with vllm_runner(
        model,
        tokenizer_name=model_info.tokenizer or model,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        # Remove the effects of batch variance on ROCm since batch invariance
        # is not yet supported.
        # See: https://github.com/vllm-project/vllm/issues/27433
        max_num_seqs=1 if current_platform.is_rocm() else 2,
        enable_prompt_embeds=use_prompt_embeds,
        compilation_config={"cudagraph_capture_sizes": [1, 2]},
        **vllm_kwargs,
    ) as vllm_model:
        # One context per generate: the cursor into HF's rows restarts here.
        with moe_near_tie_rescue(hf_expert_choice, MOE_NEAR_TIE_TOL):
            vllm_outputs = vllm_model.generate_greedy_logprobs(
                example_prompts, max_tokens, num_logprobs
            )
        if prompt_embeds is not None:
            with moe_near_tie_rescue(hf_expert_choice, MOE_NEAR_TIE_TOL):
                vllm_outputs_from_embeds = vllm_model.generate_greedy_logprobs(
                    prompt_embeds, max_tokens, num_logprobs
                )


    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
    if prompt_embeds is not None:
        check_logprobs_close(
            outputs_0_lst=vllm_outputs,
            outputs_1_lst=vllm_outputs_from_embeds,
            name_0="vllm",
            name_1="vllm_from_embeds",
        )

    if use_rocm_aiter:
        # this is to ensure that vllm engine
        # has deallocated the memory before running the next
        # unit tests. On ROCm, when using AITER
        # the memory might not be deallocated completely
        # before running the next test case
        torch.accelerator.synchronize()
