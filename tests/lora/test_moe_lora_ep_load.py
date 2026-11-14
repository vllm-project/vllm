# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end correctness test for 2D MoE LoRA expert-parallel
load-time slicing
"""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm.lora.lora_model import LoRAModel, MoEEPLoadSpec
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager

NUM_LAYERS = 48
GLOBAL_NUM_EXPERTS = 128
LOCAL_NUM_EXPERTS = 64  # ep_size = 2
EXPERT_PROJECTIONS = ("down_proj", "gate_proj", "up_proj")
NON_EXPERT_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate")


@pytest.mark.parametrize(
    "use_ep,requested_placement,num_expert_group,resolved_placement",
    [
        (False, "linear", 2, "linear"),
        (False, "round_robin", 2, "linear"),
        (True, "linear", 2, "linear"),
        (True, "round_robin", 1, "linear"),
        (True, "round_robin", 2, "round_robin"),
    ],
)
def test_moe_lora_ep_requires_contiguous_expert_placement(
    use_ep, requested_placement, num_expert_group, resolved_placement
):
    """Reject non-contiguous ownership, but allow requests resolved to linear."""
    parallel_config = replace(
        FusedMoEParallelConfig.make_no_parallel(),
        use_ep=use_ep,
        ep_size=2 if use_ep else 1,
    )
    expert_map = ExpertMapManager(
        max_num_batched_tokens=16,
        top_k=2,
        global_num_experts=4,
        num_redundant_experts=0,
        num_expert_group=num_expert_group,
        moe_parallel_config=parallel_config,
        placement_strategy=requested_placement,
        enable_eplb=False,
    )
    assert expert_map.placement_strategy == resolved_placement

    layer = FusedMoEWithLoRA.__new__(FusedMoEWithLoRA)
    torch.nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(moe_parallel_config=parallel_config)
    layer.base_layer = SimpleNamespace(
        expert_placement_strategy=expert_map.placement_strategy
    )
    error = (
        pytest.raises(
            ValueError,
            match="Set expert_placement_strategy='linear' to use MoE LoRA "
            "with expert parallelism",
        )
        if use_ep and resolved_placement != "linear"
        else nullcontext()
    )
    with error:
        layer._ep_check()


def _expected_lora_modules() -> set[str]:
    """Replicate the set `WorkerLoRAManager._load_adapter` would build
    from this model's `packed_modules_mapping`."""
    expected: set[str] = set(NON_EXPERT_MODULES)
    for expert in range(GLOBAL_NUM_EXPERTS):
        for proj in EXPERT_PROJECTIONS:
            expected.add(f"experts.{expert}.{proj}")
    return expected


def _load(lora_dir, peft_helper, *, moe_ep_spec, lora_id):
    return LoRAModel.from_local_checkpoint(
        lora_dir,
        _expected_lora_modules(),
        peft_helper=peft_helper,
        lora_model_id=lora_id,
        device="cpu",
        moe_ep_spec=moe_ep_spec,
    )


@pytest.mark.parametrize("ep_rank", [0, 1])
def test_moe_lora_ep2_real_qwen3moe(qwen3moe_lora_files, ep_rank):
    """ep_size=2 against the real Qwen3-MoE adapter: each rank's loaded
    LoRA has the right size, the right expert membership, and the
    right tensor values."""
    peft_helper = PEFTHelper.from_local_dir(
        qwen3moe_lora_files, max_position_embeddings=4096
    )

    # Baseline: no spec → loads every expert × projection × layer plus
    # all non-expert LoRA modules.
    ground_truth = _load(qwen3moe_lora_files, peft_helper, moe_ep_spec=None, lora_id=1)
    expected_baseline = (
        GLOBAL_NUM_EXPERTS * len(EXPERT_PROJECTIONS) * NUM_LAYERS
        + len(NON_EXPERT_MODULES) * NUM_LAYERS
    )
    assert len(ground_truth.loras) == expected_baseline

    # Sliced load: only this rank's experts; non-expert LoRA is untouched.
    spec = MoEEPLoadSpec(
        ep_rank=ep_rank,
        local_num_experts=LOCAL_NUM_EXPERTS,
        global_num_experts=GLOBAL_NUM_EXPERTS,
    )
    sliced = _load(
        qwen3moe_lora_files,
        peft_helper,
        moe_ep_spec=spec,
        lora_id=100 + ep_rank,
    )

    expected_sliced = (
        LOCAL_NUM_EXPERTS * len(EXPERT_PROJECTIONS) * NUM_LAYERS
        + len(NON_EXPERT_MODULES) * NUM_LAYERS
    )
    assert len(sliced.loras) == expected_sliced

    expert_start = ep_rank * LOCAL_NUM_EXPERTS
    expert_end = expert_start + LOCAL_NUM_EXPERTS

    for name, lora in sliced.loras.items():
        gt = ground_truth.loras[name]
        torch.testing.assert_close(lora.lora_a, gt.lora_a)
        torch.testing.assert_close(lora.lora_b, gt.lora_b)
        if ".experts." in name:
            expert_idx = int(name.split(".experts.")[-1].split(".")[0])
            assert expert_start <= expert_idx < expert_end, (
                f"non-local expert {expert_idx} leaked: {name}"
            )


def _make_moe_lora_checkpoint(root):
    import json

    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(123)
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=4,
        num_experts_per_tok=4,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    model = Qwen3MoeForCausalLM(config)
    model_path = root / "model"
    model.save_pretrained(model_path)
    paths = []
    for adapter_id in (1, 2):
        path = root / f"adapter_{adapter_id}"
        path.mkdir()
        (path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "r": 8,
                    "lora_alpha": 8,
                    "target_modules": ["gate_proj", "up_proj", "down_proj"],
                    "bias": "none",
                }
            )
        )
        weights = {}
        generator = torch.Generator().manual_seed(100 + adapter_id)
        for layer in range(2):
            for expert in range(4):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    prefix = (
                        f"base_model.model.model.layers.{layer}.mlp."
                        f"experts.{expert}.{proj}"
                    )
                    weights[f"{prefix}.lora_A.weight"] = (
                        torch.randn(8, 256, generator=generator) * 0.15
                    )
                    weights[f"{prefix}.lora_B.weight"] = (
                        torch.randn(256, 8, generator=generator) * 0.15
                    )
        save_file(weights, str(path / "adapter_model.safetensors"))
        paths.append(str(path))
    return str(model_path), paths


async def _check_lora_slots_dp_ep(model_path, paths, *, preload=True):
    import asyncio

    import torch

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.lora.request import LoRARequest
    from vllm.v1.engine.async_llm import AsyncLLM

    prompts = [[1, 7, 13, 29, 31], [1, 5, 17, 23, 43, 61]]
    requests = {
        0: None,
        1: LoRARequest("a", 1, paths[0]),
        2: LoRARequest("b", 2, paths[1]),
    }
    reference = {}

    async def generate(engine, tag, rank, adapter, prompt):
        final = None
        async for output in engine.generate(
            {"prompt_token_ids": prompts[prompt]},
            SamplingParams(
                temperature=0,
                max_tokens=3,
                ignore_eos=True,
                logprobs=-1,
                detokenize=False,
            ),
            request_id=tag,
            lora_request=requests[adapter],
            data_parallel_rank=rank,
        ):
            final = output.outputs[0]
        assert final is not None
        assert final.logprobs is not None
        logits = torch.tensor(
            [[step[token].logprob for token in range(128)] for step in final.logprobs]
        )
        return final.token_ids, logits

    def engine_args(dp):
        return AsyncEngineArgs(
            model=model_path,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=64,
            max_logprobs=128,
            max_num_seqs=4,
            max_num_batched_tokens=64,
            gpu_memory_utilization=0.1,
            enforce_eager=True,
            enable_prefix_caching=False,
            enable_lora=True,
            max_loras=2,
            max_lora_rank=8,
            enable_mixed_moe_lora_format=True,
            data_parallel_size=dp,
            enable_expert_parallel=dp > 1,
            all2all_backend="allgather_reducescatter",
            kernel_config={
                "moe_backend": "triton",
                "enable_flashinfer_autotune": False,
            },
        )

    engine = AsyncLLM.from_engine_args(engine_args(1))
    try:
        for adapter in (0, 1, 2):
            for prompt in range(2):
                reference[adapter, prompt] = await generate(
                    engine, f"ref-{adapter}-{prompt}", 0, adapter, prompt
                )
    finally:
        engine.shutdown()

    # A dropped or swapped adapter must fail the numerical comparison.
    for left, right in ((0, 1), (0, 2), (1, 2)):
        separation = (reference[left, 0][1][0] - reference[right, 0][1][0]).abs().max()
        assert separation > 0.1, f"Adapters are insufficiently distinct: {separation}"

    engine = AsyncLLM.from_engine_args(engine_args(2))
    try:
        if preload:
            for adapter in (1, 2):
                assert await engine.add_lora(requests[adapter])
        scenarios = [
            [(0, 1, 0), (1, 2, 1)],
            [(0, 2, 0), (1, 1, 1)],
            [(0, 0, 0), (1, 1, 1)],
            [(1, 2, 0)],
            [(0, 1, 1)],
        ]
        for step, scenario in enumerate(scenarios):
            outputs = await asyncio.gather(
                *(
                    generate(engine, f"dp-{step}-{rank}", rank, adapter, prompt)
                    for rank, adapter, prompt in scenario
                )
            )
            for (_, adapter, prompt), (tokens, logits) in zip(scenario, outputs):
                expected_tokens, expected_logits = reference[adapter, prompt]
                first_error = (logits[0] - expected_logits[0]).abs().max().item()
                assert tokens == expected_tokens, (
                    f"scenario={step}, adapter={adapter}: tokens={tokens}, "
                    f"expected={expected_tokens}, first_logprob_error={first_error}"
                )
                torch.testing.assert_close(logits, expected_logits, atol=0.02, rtol=0)
                max_error = (logits - expected_logits).abs().max().item()
                print(
                    f"DP/EP scenario={step} adapter={adapter} prompt={prompt} "
                    f"max_logprob_error={max_error:.6f}"
                )
        from vllm.outputs import RequestOutput
        from vllm.sampling_params import RequestOutputKind

        async def layout():
            layouts = await engine.engine_core.call_utility_all_async(
                "collective_rpc", "get_lora_slot_ids"
            )
            assert layouts[0] == layouts[1], layouts
            return layouts[0][0]

        await engine.pin_lora(1)
        await engine.add_lora(LoRARequest("c", 3, paths[0]))
        assert set(await layout()) == {1, 3}
        await engine.add_lora(requests[2])
        before = await layout()
        with pytest.raises(Exception, match="DP LoRA preparation failed"):
            await engine.add_lora(LoRARequest("missing", 4, "/missing/adapter"))
        assert await layout() == before

        # Only the remote rank has a queued request: its weights are protected.
        await engine.pause_generation(mode="keep", clear_cache=False)
        collector = await engine.add_request(
            request_id="protected-remote",
            prompt={"prompt_token_ids": prompts[0]},
            params=SamplingParams(
                temperature=0,
                max_tokens=3,
                ignore_eos=True,
                detokenize=False,
                output_kind=RequestOutputKind.FINAL_ONLY,
            ),
            lora_request=requests[1],
            data_parallel_rank=1,
        )
        assert await engine.is_paused()
        with pytest.raises(Exception, match="unfinished requests"):
            await engine.remove_lora(1)
        with pytest.raises(Exception, match="unfinished requests"):
            await engine.add_lora(
                LoRARequest("replace", 1, paths[1], load_inplace=True)
            )
        await engine.resume_generation()
        output = await asyncio.wait_for(collector.get(), timeout=60)
        assert isinstance(output, RequestOutput)
        assert output.outputs[0].token_ids == reference[1, 0][0]

        assert await engine.add_lora(
            LoRARequest("replace", 1, paths[1], load_inplace=True)
        )
        requests[1] = LoRARequest("replace", 1, paths[1])
        tokens, logits = await generate(engine, "replaced", 0, 1, 0)
        assert tokens == reference[2, 0][0]
        torch.testing.assert_close(logits, reference[2, 0][1], atol=0.02, rtol=0)
        assert await engine.remove_lora(1)
        assert 1 not in await layout()
        requests[1] = LoRARequest("restored", 1, paths[0])
        assert await engine.add_lora(requests[1])
        requests[3] = LoRARequest("c", 3, paths[0])
        # Three global IDs must queue through two slots while base remains valid.
        mixed = [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 3, 1), (1, 0, 1)]
        outputs = await asyncio.wait_for(
            asyncio.gather(
                *(
                    generate(engine, f"mixed-{i}", rank, adapter, prompt)
                    for i, (rank, adapter, prompt) in enumerate(mixed)
                )
            ),
            timeout=60,
        )
        for (_, adapter, prompt), (tokens, logits) in zip(mixed, outputs):
            expected = reference[1 if adapter == 3 else adapter, prompt]
            assert tokens == expected[0]
            torch.testing.assert_close(logits, expected[1], atol=0.02, rtol=0)
        await layout()
    finally:
        engine.shutdown()


@pytest.mark.parametrize("preload", [True, False], ids=["preloaded", "on-demand"])
def test_lora_slots_dp2_ep(tmp_path, preload):
    """Different DP request histories must preserve remote expert LoRA identity."""
    import asyncio

    model_path, paths = _make_moe_lora_checkpoint(tmp_path)
    asyncio.run(_check_lora_slots_dp_ep(model_path, paths, preload=preload))
