# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense NCCL weight transfer from two disjoint trainer sources.

The example uses four GPUs:

* GPUs 0-1 run two trainer actors. Each actor exposes the parameters owned by
  one logical pipeline stage.
* GPUs 2-3 run one tensor-parallel vLLM engine.

Both trainer sources join the weight-transfer communicator. Source rank 0
drives the control plane, while each source broadcasts its own manifest. The
two manifests are applied within one start/finish lifecycle.

For simplicity, both trainer actors load the small model before filtering their
parameter streams. A real pipeline-parallel trainer would already hold only its
stage-local parameters; the transfer behavior is the same.

This example was written for four GPUs with at least 8 GiB each.
"""

import os

import ray
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer import (
    ParamMeta,
    RayVLLMWeightSyncClient,
    WeightSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo
from vllm.utils.network_utils import get_ip, get_open_port

MODEL_NAME = "Qwen/Qwen3-0.6B"
NUM_SOURCES = 2
INFERENCE_TP_SIZE = 2


class StageSource(WeightSource):
    """Expose the checkpoint parameters assigned to one logical stage."""

    def __init__(
        self,
        model: torch.nn.Module,
        stage_rank: int,
        num_stages: int,
        num_layers: int,
    ) -> None:
        self.model = model
        self.stage_rank = stage_rank
        self.num_stages = num_stages
        self.num_layers = num_layers

    def _owner(self, name: str) -> int:
        parts = name.split(".")
        for index, part in enumerate(parts[:-1]):
            if part == "layers" and parts[index + 1].isdigit():
                layer = int(parts[index + 1])
                return min(
                    layer * self.num_stages // self.num_layers, self.num_stages - 1
                )
        return 0 if "embed_tokens" in name else self.num_stages - 1

    def metadata(self) -> list[ParamMeta]:
        return [
            ParamMeta(name, param.dtype, tuple(param.shape))
            for name, param in self.model.named_parameters()
        ]

    def held_names(self) -> list[str]:
        return [
            name
            for name, _ in self.model.named_parameters()
            if self._owner(name) == self.stage_rank
        ]

    def __iter__(self):
        params = dict(self.model.named_parameters())
        for group in self.groups():
            for name in group:
                param = params[name] if self._owner(name) == self.stage_rank else None
                yield name, param


class MyLLM(LLM):
    """Allow the vLLM Ray executor to allocate its own worker GPUs."""

    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)

    def ready(self):
        return True


@ray.remote(num_cpus=1, num_gpus=1)
class TrainerSource:
    def __init__(self, model_path: str, source_rank: int):
        self.source_rank = source_rank
        torch.accelerator.set_device_index(0)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).cuda()
        self.engine = None

    def ready(self):
        return self.source_rank

    def setup_engine(
        self,
        llm_handle,
        master_address: str,
        master_port: int,
        coordination_port: int,
        world_size: int,
    ):
        source = StageSource(
            self.model,
            self.source_rank,
            NUM_SOURCES,
            self.model.config.num_hidden_layers,
        )
        self.engine = WeightTransferTrainerFactory.trainer_init(
            init_info=NCCLTrainerInitInfo(
                master_address=master_address,
                master_port=master_port,
                coordination_port=coordination_port,
                world_size=world_size,
                num_source_ranks=NUM_SOURCES,
                source_rank=self.source_rank,
                rank=self.source_rank,
                packed=True,
            ),
            client=RayVLLMWeightSyncClient(llm_handle),
            source=source,
        )

    def send_weights(self):
        self.engine.send_weights()


def generate(llm, prompts: list[str]) -> list[list[int]]:
    outputs = ray.get(
        llm.generate.remote(
            prompts,
            SamplingParams(temperature=0, max_tokens=16),
        )
    )
    return [output.outputs[0].token_ids for output in outputs]


def main():
    ray.init()
    model_path = snapshot_download(MODEL_NAME)

    sources = [
        TrainerSource.remote(model_path, source_rank)
        for source_rank in range(NUM_SOURCES)
    ]
    ray.get([source.ready.remote() for source in sources])

    llm = ray.remote(num_cpus=0, num_gpus=0)(MyLLM).remote(
        model=model_path,
        enforce_eager=True,
        tensor_parallel_size=INFERENCE_TP_SIZE,
        distributed_executor_backend="ray",
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
        load_format="dummy",
    )
    ray.get(llm.ready.remote())
    num_workers = ray.get(llm.get_world_size.remote())

    prompts = ["The capital of France is", "The future of AI is"]
    before = generate(llm, prompts)

    master_address = get_ip()
    master_port = get_open_port()
    coordination_port = get_open_port()
    world_size = NUM_SOURCES + num_workers
    ray.get(
        [
            source.setup_engine.remote(
                llm,
                master_address,
                master_port,
                coordination_port,
                world_size,
            )
            for source in sources
        ]
    )

    ray.get([source.send_weights.remote() for source in sources])
    after = generate(llm, prompts)
    ray.get([source.send_weights.remote() for source in sources])
    repeated = generate(llm, prompts)

    assert after != before, "generation did not change after weight transfer"
    assert repeated == after, "repeating the same transfer changed generation"
    print("Multi-source NCCL weight transfer completed successfully.")


if __name__ == "__main__":
    main()
