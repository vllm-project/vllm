# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates async reinforcement learning using vLLM and Ray,
with native weight syncing APIs at engine instance.

The script separates training and inference workloads onto distinct GPUs
so that Ray can manage process placement and inter-process communication.
A Hugging Face Transformer model occupies one GPU for training, whereas a
2x tensor-parallel vLLM inference engine occupies two GPUs.

The example performs the following steps:
* Load the training model on one gpu (scheduled via ray)
* Initialize the inference model with dummy weights across
  two gpus using vLLM's tensor parallelism and Ray placement groups.
* Generate gibberish from a list of prompts using the randomly initialized
  inference engine.
* Pause generation once generation completes for one sequence
* Update the weights of the training model and broadcast the updated weights
  to the inference engine by using a Ray collective RPC group.
* Resume generation and print out the results

This example assumes a single-node cluster with three GPUs, but Ray
supports multi-node clusters. vLLM expects the GPUs are only used for vLLM
workloads. Residual GPU activity interferes with vLLM memory profiling and
causes unexpected behavior.
"""

import asyncio
import uuid
from dataclasses import asdict

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import vllm
from vllm import SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
    NCCLWeightTransferInitInfo,
    NCCLWeightTransferUpdateInfo,
)
from vllm.utils.network_utils import get_ip, get_open_port
from vllm.v1.executor import Executor

MODEL_NAME_V1 = "Qwen/Qwen3-1.7B-Base"
MODEL_NAME_V2 = "Qwen/Qwen3-1.7B"
PAUSE_TOKEN_THRESHOLD = 10


class MyLLM(vllm.AsyncLLMEngine):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, **kwargs):
        engine_args = vllm.AsyncEngineArgs(**kwargs)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        super().__init__(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
        )
        self._generation_paused = False
        self._request_pause_flag = False

    async def do_generate(
        self, prompt_token_ids: list[int], sampling_params: vllm.SamplingParams
    ) -> tuple[vllm.RequestOutput, int]:
        """Generate a single request, setting the request pause flag once the
        token count reaches the threshold.

        Returns (output, pause_token_index). pause_token_index is the number
        of tokens generated before the weight change, or -1 if no pause.
        """
        pause_token_index = -1
        prev_token_count = 0
        async for request_output in self.generate(
            {"prompt_token_ids": prompt_token_ids},
            sampling_params,
            request_id=str(uuid.uuid4()),
        ):
            output = request_output
            cur_token_count = len(output.outputs[0].token_ids)
            if (
                cur_token_count >= PAUSE_TOKEN_THRESHOLD
                and not self._request_pause_flag
            ):
                self._request_pause_flag = True
            if self._generation_paused and pause_token_index == -1:
                pause_token_index = prev_token_count
            prev_token_count = cur_token_count
        return output, pause_token_index

    async def pause_after_n_tokens(self):
        """Wait for any request to set the pause flag, then pause."""
        while not self._request_pause_flag:
            await asyncio.sleep(0)
        await super().pause_generation(mode="keep")
        await asyncio.sleep(0.2)
        self._generation_paused = True


@ray.remote(num_gpus=1)
class TrainModel:
    """Ray actor that wraps the training model on a dedicated GPU."""

    def __init__(self, model_name: str):
        from vllm.model_executor.layers.batch_invariant import (
            init_batch_invariance,
        )
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        # need to init all env vars for batch invariance which affect nccl ops
        init_batch_invariance(AttentionBackendEnum.FLASH_ATTN)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        ).to("cuda:0")
        self.port = get_open_port()
        self.master_address = get_ip()

    def get_master_address_and_port(self):
        return self.master_address, self.port

    def get_weight_metadata(self):
        """Return weight names, dtypes, and shapes for weight transfer."""
        names = []
        dtype_names = []
        shapes = []
        for name, p in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))
        return names, dtype_names, shapes

    def init_weight_transfer_group(self, world_size):
        """Initialize the NCCL process group for weight transfer."""
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=self.master_address,
                master_port=self.port,
                world_size=world_size,
            ),
        )

    def broadcast_weights(self, packed: bool = True):
        """Broadcast weights to the inference engine."""
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=self.model.named_parameters(),
            group=self.model_update_group,
            packed=packed,
        )

    @torch.inference_mode()
    def generate(self, token_ids: list[int], max_new_tokens: int) -> list[int]:
        """Greedy-decode max_new_tokens from the given context."""
        input_ids = torch.tensor([token_ids], device="cuda:0")
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        new_token_ids = output[0, len(token_ids) :].tolist()
        return new_token_ids


ray.init(
    runtime_env={
        "env_vars": {
            # enable batch invariance for deterministic outputs
            "VLLM_BATCH_INVARIANT": "1",
            # prevent ray from setting CUDA_VISIBLE_DEVICES
            "RAY_EXPERIMENTAL_NOSET_CUDA_ENV_VAR": "1",
        }
    }
)

# Launch the training model actor. Ray's resource scheduler will allocate
# 1 GPU (via num_gpus=1 in the decorator), ensuring pg_inference gets different GPUs.
train_model = TrainModel.remote(MODEL_NAME_V2)

# Launch the vLLM inference engine. The `enforce_eager` flag reduces
# start-up latency.
# With data_parallel_backend="ray", vLLM's CoreEngineActorManager creates
# its own placement groups internally for each DP rank, so we must NOT
# create an outer placement group (it would reserve GPUs and hide them
# from the internal DP resource check).
llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
)(MyLLM).remote(
    model=MODEL_NAME_V1,
    enforce_eager=True,
    max_model_len=8192,
    distributed_executor_backend="ray",
    attention_backend="FLASH_ATTN",
    gpu_memory_utilization=0.75,
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
)

PROMPTS = [
    "The president of the United States is",
    "The capital of France is",
    "The largest ocean on Earth is",
    "The speed of light in a vacuum is",
    "The chemical formula for water is",
    "The tallest mountain in the world is",
    "The first person to walk on the moon was",
    "The Great Wall of China was built to",
    "Photosynthesis is the process by which",
    "The theory of general relativity was proposed by",
    "The boiling point of water at sea level is",
    "The largest planet in our solar system is",
    "DNA stands for deoxyribonucleic acid and it",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_V1)
batch_prompt_token_ids = [
    tokenizer.encode(prompt, add_special_tokens=False) for prompt in PROMPTS
]


# Set up the communication channel between the training process and the
# inference engine.
master_address, master_port = ray.get(train_model.get_master_address_and_port.remote())

world_size = 2  # 1 trainer + 1 inference worker
inference_handle = llm.init_weight_transfer_engine.remote(
    WeightTransferInitRequest(
        init_info=asdict(
            NCCLWeightTransferInitInfo(
                master_address=master_address,
                master_port=master_port,
                rank_offset=1,
                world_size=world_size,
            )
        )
    )
)

# Initialize weight transfer group on both the training actor and inference engine
train_handle = train_model.init_weight_transfer_group.remote(world_size)
ray.get([train_handle, inference_handle])


N_NEW_TOKENS = 100

# Collect weight metadata once
names, dtype_names, shapes = ray.get(train_model.get_weight_metadata.remote())

# ── Phase 1: concurrent requests with weight sync ───────────────────
print(f"\n{'=' * 50}")
print(f"Prompts ({len(PROMPTS)}):")
for p in PROMPTS:
    print(f"  - {p!r}")
print(f"{'=' * 50}")

sampling_params = SamplingParams(
    temperature=0, max_tokens=PAUSE_TOKEN_THRESHOLD + N_NEW_TOKENS
)

gen_futures = [
    llm.do_generate.remote(ptids, sampling_params) for ptids in batch_prompt_token_ids
]

ray.get(llm.pause_after_n_tokens.remote())

inference_handle = llm.update_weights.remote(
    WeightTransferUpdateRequest(
        update_info=asdict(
            NCCLWeightTransferUpdateInfo(
                names=names,
                dtype_names=dtype_names,
                shapes=shapes,
                packed=True,
            )
        )
    )
)
train_handle = train_model.broadcast_weights.remote(packed=True)
ray.get([train_handle, inference_handle])

ray.get(llm.resume_generation.remote())
results = ray.get(gen_futures)

for i, (output, pause_idx) in enumerate(results):
    all_token_ids = list(output.outputs[0].token_ids)
    before_text = tokenizer.decode(all_token_ids[:pause_idx])
    after_text = tokenizer.decode(all_token_ids[pause_idx:])
    print(f"\n  Request {i} ({PROMPTS[i]!r}):")
    print(f"    Old weights ({pause_idx} tokens): {before_text!r}")
    n_after = len(all_token_ids) - pause_idx
    print(f"    New weights ({n_after} tokens): {after_text!r}")

# ── Phase 2: validate with a fresh V2 vLLM instance ────────────────
print(f"\n{'=' * 50}")
print("VALIDATION: comparing weight-synced vLLM with fresh V2 instance")
print(f"{'=' * 50}")

ray.get(llm.shutdown.remote())
ray.kill(llm)
ray.kill(train_model)

llm_v2 = ray.remote(
    num_cpus=0,
    num_gpus=0,
)(MyLLM).remote(
    model=MODEL_NAME_V2,
    enforce_eager=True,
    max_model_len=8192,
    gpu_memory_utilization=0.75,
    distributed_executor_backend="ray",
    attention_backend="FLASH_ATTN",
)

val_futures = [
    llm_v2.do_generate.remote(
        list(output.prompt_token_ids) + list(output.outputs[0].token_ids)[:pause_idx],
        SamplingParams(
            temperature=0, max_tokens=len(output.outputs[0].token_ids) - pause_idx
        ),
    )
    for output, pause_idx in results
]
val_results = ray.get(val_futures)

all_pass = True
for i, ((output, pause_idx), (val_output, _)) in enumerate(zip(results, val_results)):
    expected = list(output.outputs[0].token_ids)[pause_idx:]
    actual = list(val_output.outputs[0].token_ids)
    match = actual == expected

    if match:
        print(f"  [PASS] {PROMPTS[i]!r}")
    else:
        all_pass = False
        print(f"  [FAIL] {PROMPTS[i]!r}")
        print(f"         weight-synced vLLM: {tokenizer.decode(expected)!r}")
        print(f"         V2 vLLM:           {tokenizer.decode(actual)!r}")
        for j, (e, a) in enumerate(zip(expected, actual)):
            if e != a:
                print(
                    f"         first divergence at output token {j}: "
                    f"expected {e} ({tokenizer.decode([e])!r}) vs "
                    f"actual {a} ({tokenizer.decode([a])!r})"
                )
                break

ray.get(llm_v2.shutdown.remote())
ray.kill(llm_v2)
assert all_pass, "Some prompts failed validation, see above for details"
print("=" * 50)
