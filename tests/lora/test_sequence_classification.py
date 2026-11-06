# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from safetensors.torch import save_file

from tests.utils import multi_gpu_test
from vllm import LLM, PoolingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.lora.request import LoRARequest

MODEL_NAME = "Qwen/Qwen3-0.6B"
NATIVE_MODEL_NAME = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
BERT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
PROMPTS = {
    "star_trek": "Does warp drive appear in Star Trek?",
    "new_zealand": "Wellington is the capital of New Zealand.",
}
EXPECTED_OUT = {
    "star_trek": [13.6640625, -13.6640625],
    "new_zealand": [-4.80078125, 8.6171875],
    "native": [-1.52734375],
}


def _classify_logits(
    llm: LLM,
    prompts: list[str],
    requests: LoRARequest | list[LoRARequest],
) -> torch.Tensor:
    params = [PoolingParams(task="classify", use_activation=False) for _ in prompts]
    outputs = llm.classify(
        prompts,
        pooling_params=params,
        lora_request=requests,
        use_tqdm=False,
    )
    return torch.tensor([output.outputs.probs for output in outputs])


def test_converted_model_with_modules_to_save(
    qwen3_guard_star_trek_lora_files: str,
) -> None:
    prompt = PROMPTS["star_trek"]
    llm = LLM(
        model=MODEL_NAME,
        runner="pooling",
        convert="classify",
        hf_overrides={"num_labels": 2},
        dtype="float16",
        enable_lora=True,
        max_lora_rank=16,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.5,
    )
    actual = _classify_logits(
        llm,
        [prompt],
        LoRARequest("star-trek", 1, qwen3_guard_star_trek_lora_files),
    )[0]

    torch.testing.assert_close(
        actual,
        torch.tensor(EXPECTED_OUT["star_trek"]),
        atol=2e-2,
        rtol=2e-2,
    )

    del llm
    cleanup_dist_env_and_memory()


def test_native_classification_model_with_modules_to_save(
    skywork_qwen3_reward_lora_files: str,
) -> None:
    prompt = "Which response is more helpful and correct?"
    llm = LLM(
        model=NATIVE_MODEL_NAME,
        runner="pooling",
        dtype="float16",
        enable_lora=True,
        max_lora_rank=8,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.5,
    )
    actual = _classify_logits(
        llm,
        [prompt],
        LoRARequest("native", 1, skywork_qwen3_reward_lora_files),
    )[0]

    torch.testing.assert_close(
        actual,
        torch.tensor(EXPECTED_OUT["native"]),
        atol=2e-2,
        rtol=2e-2,
    )

    del llm
    cleanup_dist_env_and_memory()


def _save_dense_adapter(path, hidden_size: int, modules: list[str]) -> str:
    path.mkdir()
    config = {"r": 8, "lora_alpha": 16, "target_modules": ["dense"]}
    (path / "adapter_config.json").write_text(json.dumps(config))
    tensors = {}
    for module in modules:
        tensors[f"base_model.model.{module}.lora_A.weight"] = torch.zeros(
            8, hidden_size
        )
        tensors[f"base_model.model.{module}.lora_B.weight"] = torch.zeros(
            hidden_size, 8
        )
    save_file(tensors, str(path / "adapter_model.safetensors"))
    return str(path)


def test_lora_on_unsupported_module_is_rejected(tmp_path) -> None:
    """BERT's pooler.dense is an nn.Linear that LoRA does not wrap. Its LoRA
    weights must fail the load instead of being ignored, while the encoder's
    dense layers with the same suffix still load."""
    llm = LLM(
        model=BERT_MODEL_NAME,
        runner="pooling",
        dtype="float16",
        enable_lora=True,
        max_lora_rank=8,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.3,
    )
    hidden_size = llm.llm_engine.model_config.get_hidden_size()
    encoder_dense = "bert.encoder.layer.0.attention.output.dense"
    pooler_adapter = _save_dense_adapter(
        tmp_path / "pooler", hidden_size, [encoder_dense, "bert.pooler.dense"]
    )
    encoder_adapter = _save_dense_adapter(
        tmp_path / "encoder", hidden_size, [encoder_dense]
    )

    with pytest.raises(Exception, match="bert.pooler.dense"):
        llm.llm_engine.add_lora(LoRARequest("pooler", 1, pooler_adapter))
    assert llm.llm_engine.add_lora(LoRARequest("encoder", 2, encoder_adapter))
    assert set(llm.llm_engine.list_loras()) == {2}

    del llm
    cleanup_dist_env_and_memory()


def _test_batched_loras(
    qwen3_guard_star_trek_lora_files: str,
    qwen3_guard_new_zealand_lora_files: str,
    tp_size: int,
) -> None:
    names = ["star_trek", "new_zealand"]
    prompts = [PROMPTS[name] for name in names]
    adapter_paths = [
        qwen3_guard_star_trek_lora_files,
        qwen3_guard_new_zealand_lora_files,
    ]
    requests = [
        LoRARequest(name, index, adapter_path)
        for index, (name, adapter_path) in enumerate(zip(names, adapter_paths), start=1)
    ]

    llm = LLM(
        model=MODEL_NAME,
        runner="pooling",
        convert="classify",
        hf_overrides={"num_labels": 1},
        dtype="float16",
        enable_lora=True,
        max_loras=2,
        max_cpu_loras=2,
        max_lora_rank=16,
        max_lora_cls_labels=2,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=tp_size,
    )
    actual = _classify_logits(llm, prompts, requests)

    torch.testing.assert_close(
        actual,
        torch.tensor([EXPECTED_OUT[name] for name in names]),
        atol=2e-2,
        rtol=2e-2,
    )

    del llm
    cleanup_dist_env_and_memory()


def test_batched_loras(
    qwen3_guard_star_trek_lora_files: str,
    qwen3_guard_new_zealand_lora_files: str,
) -> None:
    _test_batched_loras(
        qwen3_guard_star_trek_lora_files,
        qwen3_guard_new_zealand_lora_files,
        tp_size=1,
    )


@multi_gpu_test(num_gpus=2)
def test_batched_loras_tp(
    qwen3_guard_star_trek_lora_files: str,
    qwen3_guard_new_zealand_lora_files: str,
) -> None:
    _test_batched_loras(
        qwen3_guard_star_trek_lora_files,
        qwen3_guard_new_zealand_lora_files,
        tp_size=2,
    )
