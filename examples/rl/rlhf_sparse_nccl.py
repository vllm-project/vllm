# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates dense-vs-sparse NCCL weight syncing with a real model.

This example mirrors the validation story used for the sparse NCCL MVP:
both the dense update path and the sparse patch path start from the same real
checkpoint and apply the same deterministic trainer-side patch. The script then
checks that greedy 1-token outputs match between the dense and sparse vLLM
engines after the update.

The example performs the following steps:
* Load a training model on one GPU via a Ray actor.
* Launch a vLLM engine with the same real model on a second GPU.
* Verify trainer vs vLLM baseline agreement before any update.
* Apply a deterministic patch to ``model.embed_tokens.weight`` on the trainer.
* Run a dense NCCL update into a fresh vLLM engine and collect post-update
  outputs.
* Reset the trainer back to the baseline checkpoint.
* Apply the same deterministic patch again.
* Run a sparse NCCL update into another fresh vLLM engine and collect
  post-update outputs.
* Compare dense vs sparse baseline outputs, dense vs sparse post-update
  outputs, estimated payload sizes, and trainer-side send times.

Current sparse weight transfer MVP limitations:
* ``TP=1`` and ``PP=1`` only
* sparse updates use runtime/kernel-format parameter names
* sparse updates are not composable with checkpoint-format or packed updates

This example assumes a single-node cluster with two GPUs.
"""

import hashlib
import os
import time
from collections.abc import Sequence

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.config import NCCLWeightTransferConfig, WeightTransferConfig
from vllm.distributed.weight_transfer.nccl_common import (
    NCCLTrainerInitInfo,
    trainer_init,
)
from vllm.distributed.weight_transfer.sparse_nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    SparseNCCLWeightTransferEngine,
    SparseWeightPatch,
)
from vllm.utils.network_utils import get_ip, get_open_port

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PATCHED_PARAM_NAME = "model.embed_tokens.weight"
MAX_PATCH_ROWS = 32
PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=1)


class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0"
        super().__init__(*args, **kwargs)


@ray.remote(num_gpus=1)
class TrainModel:
    """Ray actor that owns the trainer-side model and deterministic patch state."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = None
        self.patched_param = None
        self.pending_sparse_patches: list[SparseWeightPatch] | None = None
        self.model_update_group = None
        self.master_address = get_ip()
        self.port = get_open_port()
        self.reset_model()

    def reset_model(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
        ).to("cuda:0")
        self.model.eval()

        try:
            self.patched_param = self.model.get_parameter(PATCHED_PARAM_NAME)
        except AttributeError as exc:
            raise RuntimeError(
                f"Expected trainer model to expose `{PATCHED_PARAM_NAME}`"
            ) from exc

        self.pending_sparse_patches = None

    def create_rendezvous(self) -> tuple[str, int]:
        self.port = get_open_port()
        return self.master_address, self.port

    def init_weight_transfer_group(self, world_size: int) -> None:
        self.model_update_group = trainer_init(
            NCCLTrainerInitInfo(
                master_address=self.master_address,
                master_port=self.port,
                world_size=world_size,
            )
        )

    def get_dense_update_info(self) -> tuple[dict, int]:
        names = []
        dtype_names = []
        shapes = []
        payload_bytes = 0
        for name, param in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(param.dtype).split(".")[-1])
            shapes.append(list(param.shape))
            payload_bytes += param.numel() * param.element_size()

        return (
            dict(
                names=names,
                dtype_names=dtype_names,
                shapes=shapes,
            ),
            payload_bytes,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int = 1,
    ) -> list[dict[str, object]]:
        generations = []
        for prompt in prompts:
            model_inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
            output = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_token_ids = output[0, model_inputs["input_ids"].shape[1] :].tolist()
            generations.append(
                {
                    "token_ids": new_token_ids,
                    "text": self.tokenizer.decode(
                        new_token_ids,
                        skip_special_tokens=False,
                    ),
                }
            )
        return generations

    def prepare_sparse_patch(
        self,
        prompts: Sequence[str],
        max_patch_rows: int = MAX_PATCH_ROWS,
    ) -> tuple[dict[str, object], list[int], str, int]:
        selected_token_ids: list[int] = []
        special_ids = set(self.tokenizer.all_special_ids)
        for prompt in prompts:
            token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            for token_id in token_ids:
                if token_id in special_ids or token_id in selected_token_ids:
                    continue
                selected_token_ids.append(token_id)
                if len(selected_token_ids) == max_patch_rows:
                    break
            if len(selected_token_ids) == max_patch_rows:
                break

        if not selected_token_ids:
            raise ValueError("Could not derive any non-special token IDs to patch")

        vocab_size = self.patched_param.shape[0]
        next_token_id = selected_token_ids[-1]
        while len(selected_token_ids) < max_patch_rows:
            next_token_id = (next_token_id + 1) % vocab_size
            if next_token_id in special_ids or next_token_id in selected_token_ids:
                continue
            selected_token_ids.append(next_token_id)

        row_ids = torch.tensor(
            selected_token_ids,
            device=self.patched_param.device,
            dtype=torch.long,
        )
        hidden_size = self.patched_param.shape[1]
        column_offsets = torch.arange(
            hidden_size,
            device=self.patched_param.device,
            dtype=torch.long,
        )

        with torch.no_grad():
            # Rotate the selected embedding rows instead of zeroing them so the
            # patch remains deterministic while avoiding a degenerate collapse
            # to the same special token after the update.
            replacement_rows = self.patched_param[row_ids].roll(shifts=1, dims=0)
            self.patched_param[row_ids] = replacement_rows

        flat_indices = (
            row_ids.unsqueeze(1).mul(hidden_size).add(column_offsets).reshape(-1)
        )
        flat_values = self.patched_param[row_ids].reshape(-1).contiguous()
        self.pending_sparse_patches = [
            SparseWeightPatch(
                name=PATCHED_PARAM_NAME,
                indices=flat_indices.to(torch.int32),
                values=flat_values,
            )
        ]
        patch_digest = hashlib.sha256(
            self.pending_sparse_patches[0].indices.cpu().numpy().tobytes()
            + self.pending_sparse_patches[0]
            .values.detach()
            .float()
            .cpu()
            .numpy()
            .tobytes()
        ).hexdigest()

        sparse_payload_bytes = (
            flat_indices.numel() * torch.tensor([], dtype=torch.int32).element_size()
            + flat_values.numel() * flat_values.element_size()
        )
        update_info = dict(
            names=[PATCHED_PARAM_NAME],
            dtype_names=[str(self.patched_param.dtype).split(".")[-1]],
            shapes=[list(self.patched_param.shape)],
            num_updates_list=[flat_indices.numel()],
        )
        return update_info, selected_token_ids, patch_digest, sparse_payload_bytes

    def broadcast_weights(self) -> float:
        if self.model_update_group is None:
            raise RuntimeError("Weight transfer group is not initialized")

        # Dense baseline: simple unpacked per-tensor broadcast (the inference
        # side is configured with packed=False).
        start = time.perf_counter()
        stream = torch.cuda.current_stream()
        for _, tensor in self.model.named_parameters():
            self.model_update_group.broadcast(tensor, src=0, stream=stream)
        torch.accelerator.synchronize()
        return (time.perf_counter() - start) * 1000.0

    def broadcast_pending_sparse_patch(self) -> float:
        if self.model_update_group is None:
            raise RuntimeError("Weight transfer group is not initialized")
        if self.pending_sparse_patches is None:
            raise RuntimeError("Sparse patch has not been prepared")

        start = time.perf_counter()
        SparseNCCLWeightTransferEngine.trainer_send_weights(
            iter(self.pending_sparse_patches),
            NCCLTrainerSendWeightsArgs(group=self.model_update_group),
        )
        torch.accelerator.synchronize()
        self.pending_sparse_patches = None
        return (time.perf_counter() - start) * 1000.0


def launch_llm(
    scheduling_inference: PlacementGroupSchedulingStrategy,
    backend: str = "nccl",
):
    # Dense NCCL reads `packed` from its config (unpacked here to match the
    # trainer's simple per-tensor broadcast); sparse carries no wire params.
    if backend == "nccl":
        wt_config: WeightTransferConfig = NCCLWeightTransferConfig(packed=False)
    else:
        wt_config = WeightTransferConfig(backend=backend)
    return ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=scheduling_inference,
    )(MyLLM).remote(
        model=MODEL_NAME,
        enforce_eager=True,
        tensor_parallel_size=1,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.7,
        weight_transfer_config=wt_config,
    )


def collect_vllm_generations(llm_handle) -> list[dict[str, object]]:
    outputs = ray.get(llm_handle.generate.remote(PROMPTS, SAMPLING_PARAMS))
    generations = []
    for output in outputs:
        generations.append(
            {
                "token_ids": output.outputs[0].token_ids,
                "text": output.outputs[0].text,
            }
        )
    return generations


def token_sequences_match(
    left: Sequence[dict[str, object]],
    right: Sequence[dict[str, object]],
) -> bool:
    return [item["token_ids"] for item in left] == [item["token_ids"] for item in right]


def print_generations(label: str, prompts: Sequence[str], generations) -> None:
    print(f"\n{label}")
    print("-" * 50)
    for prompt, generation in zip(prompts, generations):
        print(f"Prompt: {prompt!r}")
        print(f"Token IDs: {generation['token_ids']}")
        print(f"Text: {generation['text']!r}")
        print("-" * 50)


def run_dense_phase(
    train_model,
    scheduling_inference: PlacementGroupSchedulingStrategy,
) -> dict[str, object]:
    ray.get(train_model.reset_model.remote())
    llm = launch_llm(scheduling_inference, backend="nccl")
    try:
        dense_before = collect_vllm_generations(llm)

        ray.get(llm.sleep.remote(level=0))
        master_address, master_port = ray.get(train_model.create_rendezvous.remote())
        world_size = ray.get(llm.get_world_size.remote()) + 1
        inference_init = llm.init_weight_transfer_engine.remote(
            dict(
                init_info=dict(
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=1,
                    world_size=world_size,
                )
            )
        )
        trainer_init = train_model.init_weight_transfer_group.remote(world_size)
        ray.get([trainer_init, inference_init])
        ray.get(llm.start_weight_update.remote())

        dense_update_info, dense_payload_bytes = ray.get(
            train_model.get_dense_update_info.remote()
        )
        _, selected_token_ids, patch_digest, _ = ray.get(
            train_model.prepare_sparse_patch.remote(PROMPTS)
        )

        inference_update = llm.update_weights.remote(
            dict(update_info=dense_update_info)
        )
        dense_send_ms, _ = ray.get(
            [
                train_model.broadcast_weights.remote(),
                inference_update,
            ]
        )
        ray.get(llm.finish_weight_update.remote())
        ray.get(llm.wake_up.remote(tags=["scheduling"]))

        dense_after = collect_vllm_generations(llm)

        return {
            "dense_before": dense_before,
            "dense_after": dense_after,
            "selected_token_ids": selected_token_ids,
            "patch_digest": patch_digest,
            "dense_payload_bytes": dense_payload_bytes,
            "dense_send_ms": dense_send_ms,
        }
    finally:
        ray.kill(llm)


def run_sparse_phase(
    train_model,
    scheduling_inference: PlacementGroupSchedulingStrategy,
) -> dict[str, object]:
    ray.get(train_model.reset_model.remote())
    llm = launch_llm(scheduling_inference, backend="sparse_nccl")
    try:
        sparse_before = collect_vllm_generations(llm)

        ray.get(llm.sleep.remote(level=0))
        master_address, master_port = ray.get(train_model.create_rendezvous.remote())
        world_size = ray.get(llm.get_world_size.remote()) + 1
        inference_init = llm.init_weight_transfer_engine.remote(
            dict(
                init_info=dict(
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=1,
                    world_size=world_size,
                )
            )
        )
        trainer_init = train_model.init_weight_transfer_group.remote(world_size)
        ray.get([trainer_init, inference_init])
        ray.get(llm.start_weight_update.remote())

        sparse_update_info, selected_token_ids, patch_digest, sparse_payload_bytes = (
            ray.get(train_model.prepare_sparse_patch.remote(PROMPTS))
        )

        inference_update = llm.update_weights.remote(
            dict(update_info=sparse_update_info)
        )
        sparse_send_ms, _ = ray.get(
            [
                train_model.broadcast_pending_sparse_patch.remote(),
                inference_update,
            ]
        )
        ray.get(llm.finish_weight_update.remote())
        ray.get(llm.wake_up.remote(tags=["scheduling"]))

        sparse_after = collect_vllm_generations(llm)

        return {
            "sparse_before": sparse_before,
            "sparse_after": sparse_after,
            "selected_token_ids": selected_token_ids,
            "patch_digest": patch_digest,
            "sparse_payload_bytes": sparse_payload_bytes,
            "sparse_send_ms": sparse_send_ms,
        }
    finally:
        ray.kill(llm)


ray.init()

try:
    train_model = TrainModel.remote(MODEL_NAME)

    pg_inference = placement_group([{"GPU": 1, "CPU": 0}])
    ray.get(pg_inference.ready())
    scheduling_inference = PlacementGroupSchedulingStrategy(
        placement_group=pg_inference,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=0,
    )

    dense_results = run_dense_phase(train_model, scheduling_inference)
    sparse_results = run_sparse_phase(train_model, scheduling_inference)

    baseline_equal = token_sequences_match(
        dense_results["dense_before"],
        sparse_results["sparse_before"],
    )
    patch_selection_equal = (
        dense_results["selected_token_ids"] == sparse_results["selected_token_ids"]
    )
    patch_digest_equal = dense_results["patch_digest"] == sparse_results["patch_digest"]
    after_equal = token_sequences_match(
        dense_results["dense_after"],
        sparse_results["sparse_after"],
    )
    any_output_changed = any(
        before["token_ids"] != after["token_ids"]
        for before, after in zip(
            dense_results["dense_before"],
            dense_results["dense_after"],
        )
    )
    dense_payload_mb = dense_results["dense_payload_bytes"] / (1024 * 1024)
    sparse_payload_mb = sparse_results["sparse_payload_bytes"] / (1024 * 1024)

    print_generations(
        "Dense baseline outputs",
        PROMPTS,
        dense_results["dense_before"],
    )
    print_generations(
        "Sparse baseline outputs", PROMPTS, sparse_results["sparse_before"]
    )
    print_generations(
        "Dense outputs after update", PROMPTS, dense_results["dense_after"]
    )
    print_generations(
        "Sparse outputs after update",
        PROMPTS,
        sparse_results["sparse_after"],
    )

    print(f"patched_token_ids = {dense_results['selected_token_ids']}")
    print(f"patch_selection_equal = {patch_selection_equal}")
    print(f"dense_patch_digest = {dense_results['patch_digest']}")
    print(f"sparse_patch_digest = {sparse_results['patch_digest']}")
    print(f"patch_digest_equal = {patch_digest_equal}")
    print(f"baseline_equal = {baseline_equal}")
    print(f"after_equal = {after_equal}")
    print(f"any_output_changed = {any_output_changed}")
    print(f"dense_payload_mb = {dense_payload_mb:.2f}")
    print(f"sparse_payload_mb = {sparse_payload_mb:.2f}")
    print(f"dense_send_ms = {dense_results['dense_send_ms']:.2f}")
    print(f"sparse_send_ms = {sparse_results['sparse_send_ms']:.2f}")

    if not baseline_equal:
        raise RuntimeError(
            "Dense and sparse phases did not start from the same baseline"
        )
    if not patch_selection_equal:
        raise RuntimeError("Dense and sparse phases used different sparse patches")
    if not patch_digest_equal:
        raise RuntimeError("Dense and sparse phases produced different patch values")
    if not after_equal:
        raise RuntimeError("Dense and sparse updates produced different outputs")
    if not any_output_changed:
        raise RuntimeError("Patch did not change the observed outputs")
finally:
    ray.shutdown()
