# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json
import os
import shutil
from collections.abc import Sequence

import pytest
import torch
from safetensors.torch import load_file, save_file

import vllm
from vllm.lora.request import LoRARequest, LoRARoutingRequest
from vllm.platforms import current_platform
from vllm.v1.metrics.loggers import AggregateStatLoggerBase

from ..models.utils import check_logprobs_close
from ..utils import multi_gpu_test


class GraphModeRecorder(AggregateStatLoggerBase):
    def __init__(self, vllm_config, engine_indexes):
        self.modes: set[str] = set()

    def record(
        self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0
    ):
        if scheduler_stats is not None and scheduler_stats.cudagraph_stats is not None:
            self.modes.add(scheduler_stats.cudagraph_stats.runtime_mode)

    def log_engine_initialized(self):
        pass


@pytest.fixture
def tiny_distributed_moe(tmp_path):
    from transformers import OlmoeConfig, OlmoeForCausalLM

    torch.manual_seed(42)
    model = OlmoeForCausalLM(
        OlmoeConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=128,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    base = tmp_path / "base"
    model.save_pretrained(base)
    for adapter in (1, 2):
        path = tmp_path / str(adapter)
        path.mkdir()
        (path / "adapter_config.json").write_text(
            json.dumps(
                dict(
                    peft_type="LORA",
                    task_type="CAUSAL_LM",
                    r=8,
                    lora_alpha=8,
                    target_modules=["gate_proj", "up_proj", "down_proj"],
                    bias="none",
                    lora_dropout=0.0,
                )
            )
        )
        weights = {}
        for expert in range(4):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = (
                    f"base_model.model.model.layers.0.mlp.experts.{expert}.{projection}"
                )
                weights[f"{prefix}.lora_A.weight"] = torch.randn(8, 64) * 0.1
                weights[f"{prefix}.lora_B.weight"] = torch.randn(64, 8) * 0.1
        save_file(weights, str(path / "adapter_model.safetensors"))
    return str(tmp_path)


def _distributed_generation_rank(
    rank, root, port, enable_ep, eager, barrier, tp=1, runner="0"
):
    os.environ.update(
        VLLM_DP_SIZE="2",
        VLLM_DP_RANK=str(rank),
        VLLM_DP_RANK_LOCAL=str(rank),
        VLLM_DP_MASTER_IP="127.0.0.1",
        VLLM_DP_MASTER_PORT=str(port),
        VLLM_USE_V2_MODEL_RUNNER=runner,
        VLLM_GPU_SYNC_CHECK="error",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    llm = vllm.LLM(
        model=f"{root}/base",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=64,
        max_logprobs=64,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        enable_lora=True,
        max_loras=2,
        max_loras_per_request=2,
        max_lora_rank=8,
        lora_target_modules=["experts", "gate_proj", "up_proj", "down_proj"],
        enable_expert_parallel=enable_ep,
        tensor_parallel_size=tp,
        all2all_backend="allgather_reducescatter",
        enforce_eager=eager,
        gpu_memory_utilization=0.25,
        kv_cache_memory_bytes=16 * 1024 * 1024,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8]},
        disable_log_stats=False,
        cudagraph_metrics=True,
    )
    try:
        assert llm.llm_engine.vllm_config.use_v2_model_runner == (runner == "1")
        _check_distributed_generation(llm, rank, root, barrier, eager)
    finally:
        llm.llm_engine.engine_core.shutdown()


def _check_distributed_generation(llm, rank, root, barrier, eager):
    from transformers import OlmoeForCausalLM

    recorder = GraphModeRecorder(llm.llm_engine.vllm_config, [0])
    manager = llm.llm_engine.logger_manager
    assert manager is not None
    manager.stat_loggers.append(recorder)
    adapters = tuple(LoRARequest(str(i), i, f"{root}/{i}") for i in (1, 2))
    # Each rank starts with a different resident adapter.
    llm.llm_engine.add_lora(adapters[rank])
    barrier.wait(timeout=180)
    prompt = [3, 4, 5]
    for step in range(3):
        recorder.modes.clear()
        coefficients = ((0.7, 0.3), (0.3, 0.7), (0.0, 0.0))[step]
        request = (
            LoRARoutingRequest("mixture", 100, adapters, coefficients)
            if step != 2
            else None
        )
        # Unequal completion lengths force the finished rank through idle steps.
        params = vllm.SamplingParams(
            temperature=0,
            max_tokens=1 if rank == 0 else 3,
            logprobs=64,
            ignore_eos=True,
        )
        outputs = llm.generate(
            [{"prompt_token_ids": prompt}, {"prompt_token_ids": prompt}],
            params,
            lora_request=[request, None],
            use_tqdm=False,
        )
        output, base_output = (result.outputs[0] for result in outputs)
        if not eager and rank == 1:
            assert "FULL" in recorder.modes, recorder.modes
        reference = OlmoeForCausalLM.from_pretrained(f"{root}/base").float().eval()
        with torch.no_grad():
            # Mixed routes must follow the same TP token partition as activations.
            base_prefix = prompt + list(base_output.token_ids[:-1])
            base_expected = (
                reference(torch.tensor([base_prefix]))
                .logits[0, len(prompt) - 1 :]
                .log_softmax(-1)
            )
            assert base_output.logprobs is not None
            base_actual = torch.tensor(
                [
                    [scores[i].logprob for i in range(64)]
                    for scores in base_output.logprobs
                ]
            )
            torch.testing.assert_close(base_actual, base_expected, atol=2e-2, rtol=2e-3)
            experts = reference.model.layers[0].mlp.experts
            for adapter, coefficient in zip(adapters, coefficients):
                weights = load_file(f"{adapter.lora_path}/adapter_model.safetensors")
                for expert in range(4):
                    for name in ("gate_proj", "up_proj", "down_proj"):
                        prefix = "base_model.model.model.layers.0.mlp.experts."
                        prefix += f"{expert}.{name}"
                        delta = (
                            weights[f"{prefix}.lora_B.weight"]
                            @ weights[f"{prefix}.lora_A.weight"]
                        )
                        if name == "down_proj":
                            target = experts.down_proj[expert]
                        elif name == "up_proj":
                            target = experts.gate_up_proj[expert, 64:]
                        else:
                            target = experts.gate_up_proj[expert, :64]
                        target.add_(delta, alpha=coefficient)
            prefix = prompt + list(output.token_ids[:-1])
            expected = (
                reference(torch.tensor([prefix]))
                .logits[0, len(prompt) - 1 :]
                .log_softmax(-1)
            )
        assert output.logprobs is not None
        actual = torch.tensor(
            [[scores[i].logprob for i in range(64)] for scores in output.logprobs]
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-3)
        barrier.wait(timeout=180)


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Distributed composition requires CUDA"
)
@pytest.mark.parametrize("enable_ep", [False, True])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("runner", ["0", "1"], ids=["v1", "v2"])
def test_distributed_moe_composition_generation(
    tiny_distributed_moe, enable_ep, eager, runner
):
    """Real distributed generation, including idle ranks, matches merged weights."""
    from vllm.utils.network_utils import get_open_port

    context = torch.multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    torch.multiprocessing.spawn(
        _distributed_generation_rank,
        args=(
            tiny_distributed_moe,
            get_open_port(),
            enable_ep,
            eager,
            barrier,
            1,
            runner,
        ),
        nprocs=2,
    )


@multi_gpu_test(num_gpus=4)
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Distributed composition requires CUDA"
)
@pytest.mark.parametrize("runner", ["0", "1"], ids=["v1", "v2"])
def test_distributed_moe_composition_generation_tp2_dp2(tiny_distributed_moe, runner):
    """TP peers share adapter identities with both DP ranks during graph replay."""
    from vllm.utils.network_utils import get_open_port

    context = torch.multiprocessing.get_context("spawn")
    torch.multiprocessing.spawn(
        _distributed_generation_rank,
        args=(
            tiny_distributed_moe,
            get_open_port(),
            True,
            False,
            context.Barrier(2),
            2,
            runner,
        ),
        nprocs=2,
    )


MODEL_PATH = "allenai/OLMoE-1B-7B-0125-Instruct"

PROMPT_TEMPLATE = """I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me. Do not return any additional explanation. Below is an instruction that describes a task, Write a response that appropriately completes the request.
"
##Instruction:
candidate_poll contains tables such as candidate, people. Table candidate has columns such as Candidate_ID, People_ID, Poll_Source, Date, Support_rate, Consider_rate, Oppose_rate, Unsure_rate. Candidate_ID is the primary key.
Table people has columns such as People_ID, Sex, Name, Date_of_Birth, Height, Weight. People_ID is the primary key.
The People_ID of candidate is the foreign key of People_ID of people.


###Input:
{context}

###Response:"""  # noqa: E501

EXPECTED_LORA_OUTPUT = [
    "SELECT count(*) FROM candidate",
    "SELECT count(*) FROM candidate",
    "SELECT poll_source FROM candidate GROUP BY poll_source ORDER BY count(*) DESC LIMIT 1",  # noqa: E501
    "SELECT poll_source FROM candidate GROUP BY poll_source ORDER BY count(*) DESC LIMIT 1",  # noqa: E501
]

EXPECTED_BASE_MODEL_OUTPUT = [
    "SELECT COUNT(Candidate_ID) FROM candidate",
    "SELECT COUNT(Candidate_ID) FROM candidate",
    "SELECT Candidate_ID, COUNT(*) as Total_Candidates\nFROM candidate\nINNER JOIN people ON candidate.People_ID = people.People_ID",  # noqa: E501
    # There are multiple acceptable responses
    (
        "SELECT Candidate_ID, Poll_Source FROM candidate WHERE People_ID IN (SELECT People_ID FROM people) ORDER BY COUNT(*) DESC LIMIT 1",  # noqa: E501
        "SELECT Candidate_ID, Poll_Source FROM candidate WHERE COUNT(People_ID) = (SELECT COUNT(People_ID) FROM people) ORDER BY Candidate_ID DESC LIMIT 1",  # noqa: E501
    ),
]


def _output_matches(generated: str, accepted: str | Sequence[str]) -> bool:
    if isinstance(accepted, str):
        accepted = (accepted,)
    return any(generated.startswith(s) for s in accepted)


def generate_and_test(
    llm: vllm.LLM,
    lora_path: str,
    lora_id: list[int | None] | int | None,
) -> None:
    prompts = [
        PROMPT_TEMPLATE.format(context="How many candidates are there?"),
        PROMPT_TEMPLATE.format(context="Count the number of candidates."),
        PROMPT_TEMPLATE.format(
            context="Which poll resource provided the most number of candidate information?"  # noqa: E501
        ),
        PROMPT_TEMPLATE.format(
            context="Return the poll resource associated with the most candidates."
        ),
    ]

    lora_request = None
    if isinstance(lora_id, int):
        lora_request = LoRARequest(str(lora_id), lora_id, lora_path)
    elif isinstance(lora_id, list):
        lora_request = [
            LoRARequest(str(i), i, lora_path) if i is not None else None
            for i in lora_id
        ]

    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=64)
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    # Print the outputs.
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    for i in range(len(EXPECTED_LORA_OUTPUT)):
        req_lora_id = lora_id[i] if isinstance(lora_id, list) else lora_id
        generated_text = generated_texts[i]
        expected_output = (
            EXPECTED_LORA_OUTPUT[i]
            if req_lora_id is not None
            else EXPECTED_BASE_MODEL_OUTPUT[i]
        )
        assert _output_matches(generated_text, expected_output), (
            f"Output {i}: {generated_text!r} does not match any of {expected_output!r}"
        )


def test_olmoe_lora_mixed(olmoe_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        hf_overrides={"head_dtype": "float32"},
        enable_chunked_prefill=True,
    )

    generate_and_test(llm, olmoe_lora_files, lora_id=[1, None, 3, None])


def test_olmoe_lora_mixed_random(
    olmoe_lora_files, tmp_path, maybe_enable_lora_dual_stream
):
    # Create a dummy LoRA with random weights based on the real one
    random_lora_path = tmp_path / "random_lora"
    shutil.copytree(olmoe_lora_files, random_lora_path)

    weights_path = random_lora_path / "adapter_model.safetensors"
    weights = load_file(str(weights_path))
    random_weights = {k: torch.randn_like(v) for k, v in weights.items()}
    save_file(random_weights, str(weights_path))

    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
    )

    prompts = [
        PROMPT_TEMPLATE.format(context="How many candidates are there?"),
        PROMPT_TEMPLATE.format(context="Count the number of candidates."),
    ]

    lora_requests = [
        LoRARequest("real", 1, olmoe_lora_files),
        LoRARequest("random", 2, str(random_lora_path)),
    ]

    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=64)
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_requests)
    assert outputs[0].outputs[0].text.strip().startswith(EXPECTED_LORA_OUTPUT[0])


@pytest.mark.parametrize("fully_sharded_loras", [False, True])
@multi_gpu_test(num_gpus=2)
def test_olmoe_lora_tp2(olmoe_lora_files, fully_sharded_loras):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        tensor_parallel_size=2,
        fully_sharded_loras=fully_sharded_loras,
    )

    generate_and_test(llm, olmoe_lora_files, lora_id=1)
    generate_and_test(llm, olmoe_lora_files, lora_id=2)


COMPOSITION_WEIGHTS = (0.7, 0.3)


def write_composition_adapters(source: str, root) -> tuple[str, str, str]:
    """Writes adapter B, a perturbed copy of the source adapter A, and adapter C,
    the rank-concatenated merge of COMPOSITION_WEIGHTS * (A, B). Composing A and
    B with those weights must therefore match C served as one adapter."""
    generator = torch.Generator().manual_seed(0)
    adapter_a = load_file(f"{source}/adapter_model.safetensors")
    adapter_b, adapter_c = {}, {}
    for name, a_weight in adapter_a.items():
        if ".lora_B." in name:
            continue
        b_name = name.replace(".lora_A.", ".lora_B.")
        b_weight = adapter_a[b_name]
        noise = torch.randn(a_weight.shape, generator=generator).to(a_weight.dtype)
        adapter_b[name] = a_weight.roll(1, dims=0) + noise * a_weight.std()
        adapter_b[b_name] = -b_weight
        adapter_c[name] = torch.cat((a_weight, adapter_b[name]))
        adapter_c[b_name] = torch.cat(
            (
                COMPOSITION_WEIGHTS[0] * b_weight,
                COMPOSITION_WEIGHTS[1] * adapter_b[b_name],
            ),
            dim=1,
        )
    paths = []
    for name, weights, rank_factor in (("b", adapter_b, 1), ("c", adapter_c, 2)):
        path = root / name
        shutil.copytree(source, path)
        save_file(weights, str(path / "adapter_model.safetensors"))
        config = json.loads((path / "adapter_config.json").read_text())
        # Doubling r and alpha keeps the LoRA scaling alpha / r unchanged.
        config["r"] *= rank_factor
        config["lora_alpha"] *= rank_factor
        (path / "adapter_config.json").write_text(json.dumps(config))
        paths.append(str(path))
    return source, paths[0], paths[1]


def generate_composition_and_merged(
    llm: vllm.LLM, prompts: list[str], adapters: tuple[str, str, str]
):
    path_a, path_b, path_c = adapters
    composition = LoRARoutingRequest(
        routing_name="composition",
        routing_int_id=100,
        lora_requests=(
            LoRARequest("a", 1, path_a),
            LoRARequest("b", 2, path_b),
        ),
        lora_weights=COMPOSITION_WEIGHTS,
    )
    merged = LoRARequest("merged", 3, path_c)
    params = vllm.SamplingParams(temperature=0, max_tokens=32, logprobs=5)
    results = []
    for lora_request in (composition, merged):
        outputs = llm.generate(prompts, params, lora_request=lora_request)
        results.append(
            [
                (
                    list(output.outputs[0].token_ids),
                    output.outputs[0].text,
                    output.outputs[0].logprobs,
                )
                for output in outputs
            ]
        )
    return results


COMPOSITION_ENGINE_ARGS = dict(
    enforce_eager=current_platform.is_xpu(),
    max_model_len=1024,
    enable_lora=True,
    max_loras=4,
    max_loras_per_request=2,
    max_lora_rank=16,
    lora_target_modules=[
        "experts",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    trust_remote_code=True,
)


def test_olmoe_lora_composition_matches_merged_adapter(olmoe_lora_files, tmp_path):
    """Composed expert and attention adapters match the merged adapter."""
    adapters = write_composition_adapters(olmoe_lora_files, tmp_path)
    llm = vllm.LLM(MODEL_PATH, **COMPOSITION_ENGINE_ARGS)
    prompts = [
        PROMPT_TEMPLATE.format(context="How many candidates are there?"),
        PROMPT_TEMPLATE.format(context="Count the number of candidates."),
    ]
    composition, merged = generate_composition_and_merged(llm, prompts, adapters)
    check_logprobs_close(
        outputs_0_lst=merged,
        outputs_1_lst=composition,
        name_0="merged",
        name_1="composition",
    )


def test_olmoe_lora_composition_rejects_unsupported_dispatch():
    """Reject unsupported composition before launching distributed workers."""
    with pytest.raises(ValueError, match="Distributed routed LoRA requires"):
        vllm.LLM(
            MODEL_PATH,
            enable_expert_parallel=True,
            all2all_backend="deepep_low_latency",
            **COMPOSITION_ENGINE_ARGS,
        )
