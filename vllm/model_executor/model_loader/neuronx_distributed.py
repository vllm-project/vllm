# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for selecting and loading Neuron models in
neuronx-distributed-inference framework."""
# Disabling yapf because yapf and isort have conflicts for the below imports
# yapf: disable
import copy
import hashlib
import importlib
import multiprocessing
import os
import shutil
from typing import Optional

import torch
import torch.nn as nn
from neuronx_distributed_inference.models.config import (
    FusedSpecNeuronConfig, OnDeviceSamplingConfig)
from neuronx_distributed_inference.models.mllama.utils import (
    create_vision_mask)
from neuronx_distributed_inference.modules.lora_serving import (
    LoraServingConfig)
from neuronx_distributed_inference.utils.hf_adapter import (
    load_pretrained_config)
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from vllm.config import (ModelConfig, ParallelConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import (CompletionSequenceGroupOutput, Logprob,
                           SequenceOutput)

# yapf: enable
logger = init_logger(__name__)

TORCH_DTYPE_TO_NEURON_AMP = {
    "auto": "float32",
    "half": "float16",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float": "float32",
    "float32": "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}

# Models supported by Neuronx distributed for inference.
_NEURON_SUPPORTED_MODELS: dict[str, tuple[str, str]] = {
    "LlamaForCausalLM":
    ("neuronx_distributed_inference.models.llama.modeling_llama",
     "NeuronLlamaForCausalLM"),
    "MistralForCausalLM":
    ("neuronx_distributed_inference.models.llama.modeling_llama",
     "NeuronLlamaForCausalLM"),
    "DbrxForCausalLM":
    ("neuronx_distributed_inference.models.dbrx.modeling_dbrx",
     "NeuronDbrxForCausalLM"),
    "MixtralForCausalLM":
    ("neuronx_distributed_inference.models.mixtral.modeling_mixtral",
     "NeuronMixtralForCausalLM"),
    "MllamaForConditionalGeneration":
    ("neuronx_distributed_inference.models.mllama.modeling_mllama",
     "NeuronMllamaForCausalLM"),
}


class NeuronCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                logits_as_input=True)
        self.sampler = Sampler()

        # Lazy initialized
        self.model: nn.Module

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                input_block_ids: torch.Tensor,
                sampling_params: torch.Tensor,
                prev_hidden: Optional[torch.Tensor] = None,
                adapter_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # sort block ids sequentially for perf/neuron support reasons
        sorted_input_block_ids, sorted_indices = torch.sort(input_block_ids)
        input_ids = torch.index_select(input_ids, 0, sorted_indices)
        positions = torch.index_select(positions, 0, sorted_indices)
        sampling_params = torch.index_select(sampling_params, 0,
                                             sorted_indices)
        output = self.model(input_ids,
                            attention_mask=None,
                            position_ids=positions,
                            seq_ids=sorted_input_block_ids,
                            sampling_params=sampling_params,
                            prev_hidden=prev_hidden,
                            adapter_ids=adapter_ids)
        # on-device sampling
        if self.config.neuron_config.on_device_sampling_config:
            output = output.hidden_states
        else:
            output = output.logits[:, -1, :]

        restored_indices = torch.argsort(sorted_indices)
        if input_block_ids.shape[0] != 1:
            output = torch.index_select(output, 0, restored_indices)

        return output

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        # on-device sampling
        if self.config.neuron_config.on_device_sampling_config:
            batch_size = logits.shape
            seq_ids = [
                seq_id for sg in sampling_metadata.seq_groups
                for seq_id in sg.seq_ids
            ]
            assert len(seq_ids) == list(batch_size)[0], "batch size mismatch"
            # Organize input tensors by step instead of by sequence.
            accepted_token_ids_by_step = logits.flatten()
            accepted_token_ids_by_step = accepted_token_ids_by_step.tolist()

            step_output_token_ids = []
            for i, seq_id in enumerate(seq_ids):
                token_id = accepted_token_ids_by_step[i]
                step_output_token_ids.append(
                    CompletionSequenceGroupOutput(samples=[
                        SequenceOutput(parent_seq_id=seq_id,
                                       output_token=token_id,
                                       logprobs={token_id: Logprob(token_id)})
                    ],
                                                  prompt_logprobs=None))
            return SamplerOutput(outputs=step_output_token_ids)
        else:
            return self.sampler(logits, sampling_metadata)

    def load_weights(self, model_name_or_path: str, **kwargs):
        arch = _get_model_architecture(self.config)
        neuronx_module_path, neuronx_model_cls_name = (
            _NEURON_SUPPORTED_MODELS[arch])
        neuronx_module = importlib.import_module(neuronx_module_path)
        neuronx_model_cls = getattr(neuronx_module, neuronx_model_cls_name)
        neuron_config = neuronx_model_cls.get_neuron_config_cls()(
            **kwargs['neuron_config'])
        self.config.neuron_config = neuron_config
        config = neuronx_model_cls.get_config_cls()(
            neuron_config,
            load_config=load_pretrained_config(model_name_or_path))
        hashed_config = hashlib.md5(config.to_json_string().encode('utf-8'),
                                    usedforsecurity=False).hexdigest()
        if os.getenv("NEURON_COMPILED_ARTIFACTS") is not None:
            compiled_model_path = os.getenv("NEURON_COMPILED_ARTIFACTS")
        elif os.path.exists(model_name_or_path):
            compiled_model_path = os.path.join(model_name_or_path,
                                               "neuron-compiled-artifacts",
                                               hashed_config)
            shutil.rmtree(compiled_model_path, ignore_errors=True)
        else:
            compiled_model_path = os.path.join("local-models",
                                               model_name_or_path,
                                               "neuron-compiled-artifacts",
                                               hashed_config)
            shutil.rmtree(compiled_model_path, ignore_errors=True)
        try:
            self.model = neuronx_model_cls(compiled_model_path)
            override_neuron_config = kwargs["override_neuron_config"]
            for k, v in override_neuron_config.items():
                setattr(self.model.config.neuron_config, k, v)
            self.model.load(compiled_model_path)
            return
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Exception: %s", e)
            logger.warning("Failed to load the model from %s, Recompiling...",
                           compiled_model_path)
        if not os.path.exists(model_name_or_path):
            hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            saved_path = os.path.join("local-models", model_name_or_path)
            hf_model.save_pretrained(saved_path)
            model_name_or_path = saved_path
        self.model = neuronx_model_cls(model_name_or_path, config)
        self.model.compile(compiled_model_path)
        self.model.load(compiled_model_path)


class NeuronMllamaForCausalLM(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 on_device_sampling_disabled: bool = False) -> None:
        super().__init__()
        # has_image is the only multimodal input that is used in
        # token-generation
        # This is a cache (on CPU) that saves has_image data per sequence id
        # The number of entries in this cache is <= Batch-Size
        self.has_image_cache: dict[int, torch.Tensor] = {}
        self.config = config
        self.logits_processor = LogitsProcessor(
            config.get_text_config().vocab_size, logits_as_input=True)

        self.on_device_sampling_disabled = on_device_sampling_disabled
        if self.on_device_sampling_disabled:
            # Use default sampler
            self.sampler = Sampler()

        # Lazy initialized
        self.model: nn.Module
        self.is_reorder_needed: bool = True

    def read_from_has_image_cache(self, seq_ids: torch.Tensor):
        has_image_list = []
        for index in range(len(seq_ids)):
            seq_id = seq_ids[index].item()
            if seq_id in self.has_image_cache:
                has_image_list.append(self.has_image_cache[seq_id])
            else:
                has_image_list.append(torch.tensor([0]))
        return torch.tensor(has_image_list)

    def write_to_has_image_cache(self, seq_ids: torch.Tensor,
                                 has_image: torch.Tensor):
        for index in range(len(seq_ids)):
            seq_id = seq_ids[index].item()
            if index < len(has_image):
                self.has_image_cache[seq_id] = has_image[index]
            else:
                self.has_image_cache[seq_id] = torch.zeros(1)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                seq_ids: torch.Tensor, pixel_values: torch.Tensor,
                aspect_ratios: torch.Tensor, num_chunks: torch.Tensor,
                has_image: torch.Tensor, sampling_params) -> torch.Tensor:

        # We update the has_image cache during prefill
        # and read the has_image cache during decode
        if input_ids.shape[-1] > 1:  # prefill
            self.write_to_has_image_cache(seq_ids, has_image)
        else:
            has_image = self.read_from_has_image_cache(seq_ids)
            bs = input_ids.shape[0]
            num_chunks = torch.zeros((bs, 1))
            aspect_ratios = torch.zeros((bs, 1, 2))

        input_block_ids = seq_ids
        origin_input_block_ids = seq_ids
        if self.is_reorder_needed:
            # sort block ids sequentially for perf/neuron support reasons
            input_block_ids, sorted_indices = torch.sort(input_block_ids)
            input_ids = torch.index_select(input_ids, 0, sorted_indices)
            positions = torch.index_select(positions, 0, sorted_indices)
            sampling_params = torch.index_select(sampling_params, 0,
                                                 sorted_indices)
            pixel_values = torch.index_select(pixel_values, 0, sorted_indices)
            aspect_ratios = torch.index_select(aspect_ratios, 0,
                                               sorted_indices)
            num_chunks = torch.index_select(num_chunks, 0, sorted_indices)
            has_image = torch.index_select(has_image, 0, sorted_indices)

        self.vision_mask = create_vision_mask(input_ids, self.vision_token_id)
        output = self.model(
            input_ids.to(torch.int32),
            attention_mask=None,
            position_ids=positions.to(torch.int32),
            seq_ids=seq_ids.flatten().to(torch.int32),
            pixel_values=pixel_values.to(
                self.config.vision_config.torch_dtype),
            aspect_ratios=aspect_ratios.to(torch.int32),
            vision_mask=self.vision_mask.to(torch.int32),
            sampling_params=sampling_params,
            num_chunks=num_chunks.to(torch.int32),
            has_image=has_image.to(torch.int32),
        )
        if self.config.neuron_config.on_device_sampling_config:
            output = output.hidden_states
        else:
            output = output.logits[:, -1, :]

        if self.is_reorder_needed and origin_input_block_ids.shape[0] != 1:
            restored_indices = torch.argsort(sorted_indices)
            output = torch.index_select(output, 0, restored_indices)
        return output

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(self, hidden_states, sampling_metadata):
        if not self.on_device_sampling_disabled:
            with torch.profiler.record_function("sample"):
                hidden_states = hidden_states.flatten()
                res = []
                sample_idx = 0
                for seq_group in sampling_metadata.seq_groups:
                    seq_ids = seq_group.seq_ids
                    samples = []
                    for seq_id in seq_ids:
                        token_id = hidden_states[sample_idx].item()
                        samples.append(
                            SequenceOutput(
                                parent_seq_id=seq_id,
                                output_token=token_id,
                                logprobs={token_id: Logprob(token_id)}))
                        sample_idx += 1
                    res.append(
                        CompletionSequenceGroupOutput(samples=samples,
                                                      prompt_logprobs=None))
                next_tokens = SamplerOutput(outputs=res)
        else:
            next_tokens = self.sampler(None, hidden_states, sampling_metadata)
        return next_tokens

    def load_weights(self, model_name_or_path: str, **kwargs):
        arch = _get_model_architecture(self.config)
        neuronx_module_path, neuronx_model_cls_name = (
            _NEURON_SUPPORTED_MODELS[arch])
        neuronx_module = importlib.import_module(neuronx_module_path)
        neuronx_model_cls = getattr(neuronx_module, neuronx_model_cls_name)
        neuron_config = neuronx_model_cls.get_neuron_config_cls()(
            **kwargs['neuron_config'])
        self.config.neuron_config = neuron_config
        logger.info("neuron_config buckets: %s",
                    self.config.neuron_config.buckets)
        config = neuronx_model_cls.get_config_cls()(
            neuron_config,
            load_config=load_pretrained_config(model_name_or_path))
        hashed_config = hashlib.md5(config.to_json_string().encode('utf-8'),
                                    usedforsecurity=False).hexdigest()
        if os.getenv("NEURON_COMPILED_ARTIFACTS") is not None:
            compiled_model_path = os.getenv("NEURON_COMPILED_ARTIFACTS")
        elif os.path.exists(model_name_or_path):
            compiled_model_path = os.path.join(model_name_or_path,
                                               "neuron-compiled-artifacts",
                                               hashed_config)
        else:
            compiled_model_path = os.path.join("local-models",
                                               model_name_or_path,
                                               "neuron-compiled-artifacts",
                                               hashed_config)
        try:
            self.model = neuronx_model_cls(compiled_model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.vision_token_id = tokenizer(
                "<|image|>", add_special_tokens=False).input_ids[0]
            self.model.load(compiled_model_path)
            return
        except (FileNotFoundError, ValueError):
            logger.warning("Failed to load the model from %s, Recompiling...",
                           compiled_model_path)
        if not os.path.exists(model_name_or_path):
            hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            saved_path = os.path.join("local-models", model_name_or_path)
            hf_model.save_pretrained(saved_path)
            model_name_or_path = saved_path
        self.model = neuronx_model_cls(model_name_or_path, config)

        logger.info("\nCompiling and saving model to %s", model_name_or_path)

        p = multiprocessing.Process(target=compile_model,
                                    args=(self, compiled_model_path))
        p.start()
        p.join()

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.save_pretrained(compiled_model_path)
        logger.info("Successfully compiled and saved the model in %s",
                    compiled_model_path)

        # Read "<|image|>" token_id from the tokenizer
        self.vision_token_id = tokenizer("<|image|>",
                                         add_special_tokens=False).input_ids[0]
        logger.info("\nLoading model from compiled checkpoint...")
        self.model.load(compiled_model_path)


def compile_model(neuron_model, traced_model_path):
    neuron_model.model.compile(traced_model_path)


class NeuronSpeculationCausalLM(nn.Module):
    """A Neuron-optimized causal language model with speculative decoding."""

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                logits_as_input=True)
        # Lazy initialized
        self.model: nn.Module

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_block_ids: torch.Tensor,
        sampling_params: torch.Tensor,
    ) -> torch.Tensor:
        # sort block ids sequentially for perf/neuron support reasons
        sorted_input_block_ids, sorted_indices = torch.sort(input_block_ids)
        input_ids = torch.index_select(input_ids, 0, sorted_indices)
        positions = torch.index_select(positions, 0, sorted_indices)
        sampling_params = torch.index_select(sampling_params, 0,
                                             sorted_indices)

        output = self.model(input_ids,
                            attention_mask=None,
                            position_ids=positions,
                            seq_ids=sorted_input_block_ids,
                            sampling_params=sampling_params)
        restored_indices = torch.argsort(sorted_indices)

        # CTX encoding
        if (positions[:, 0]).sum().item() == 0:
            output = output.fused_outputs[0][:, 0:1]
            if input_block_ids.shape[0] != 1:
                output = torch.index_select(output, 0, restored_indices)
            return output

        # Fused Spec (Generation)
        accepted_tokens_with_padding = output.fused_outputs[0]
        next_pos_ids = output.fused_outputs[-1]
        generated_token_counts = next_pos_ids - positions

        assert torch.any(generated_token_counts == 0).item() is False, \
            "NxDI model generated no output for one or more sequences."

        batch_size, steps = accepted_tokens_with_padding.shape
        mask = torch.arange(steps).expand(batch_size,
                                          -1) >= generated_token_counts
        accepted_tokens_with_padding[mask] = -1

        if input_block_ids.shape[0] != 1:
            accepted_tokens_with_padding = torch.index_select(
                accepted_tokens_with_padding, 0, restored_indices)

        return accepted_tokens_with_padding

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[list[SamplerOutput]]:
        batch_size, num_steps = logits.shape
        seq_ids = [
            seq_id for sg in sampling_metadata.seq_groups
            for seq_id in sg.seq_ids
        ]
        # Organize input tensors by step instead of by sequence.
        accepted_token_ids_by_step = logits.transpose(0, 1)
        accepted_token_ids_by_step = accepted_token_ids_by_step.tolist()

        sampler_output_list = []
        for step_index in range(num_steps):
            if all(token_id == -1
                   for token_id in accepted_token_ids_by_step[step_index]):
                break
            step_output_token_ids = []
            for sequence_index in range(batch_size):
                token_id = accepted_token_ids_by_step[step_index][
                    sequence_index]
                step_output_token_ids.append(
                    CompletionSequenceGroupOutput(samples=[
                        SequenceOutput(parent_seq_id=seq_ids[sequence_index],
                                       output_token=token_id,
                                       logprobs={token_id: Logprob(token_id)})
                    ],
                                                  prompt_logprobs=None))
            sampler_output_list.append(
                SamplerOutput(outputs=step_output_token_ids))
        return sampler_output_list

    def load_weights(self, model_name_or_path: str,
                     draft_model_name_or_path: str, **kwargs):
        arch = _get_model_architecture(self.config)
        neuronx_module_path, neuronx_model_cls_name = (
            _NEURON_SUPPORTED_MODELS[arch])
        neuronx_module = importlib.import_module(neuronx_module_path)
        neuronx_model_cls = getattr(neuronx_module, neuronx_model_cls_name)
        neuron_config = neuronx_model_cls.get_neuron_config_cls()(
            **kwargs['neuron_config'])
        config = neuronx_model_cls.get_config_cls()(
            neuron_config,
            load_config=load_pretrained_config(model_name_or_path))

        draft_neuron_config = copy.deepcopy(config.neuron_config)
        if not config.neuron_config.enable_eagle_speculation:
            draft_neuron_config.speculation_length = 0
        draft_neuron_config.trace_tokengen_model = True
        draft_neuron_config.enable_fused_speculation = False
        if getattr(config.neuron_config, "draft_model_modules_to_not_convert",
                   None):
            draft_neuron_config.modules_to_not_convert = (
                draft_neuron_config.draft_model_modules_to_not_convert)
        if config.neuron_config.enable_eagle_speculation:
            draft_neuron_config.is_eagle_draft = True
            draft_neuron_config.sequence_parallel_enabled = False
        draft_config = neuronx_model_cls.get_config_cls()(
            draft_neuron_config,
            load_config=load_pretrained_config(draft_model_name_or_path))
        fused_spec_config = (FusedSpecNeuronConfig(
            neuronx_model_cls._model_cls,
            draft_config=draft_config,
            draft_model_path=draft_model_name_or_path))
        config.fused_spec_config = fused_spec_config
        self.config.neuron_config = neuron_config

        hashed_config = hashlib.md5(config.to_json_string().encode('utf-8'),
                                    usedforsecurity=False).hexdigest()
        if os.getenv("NEURON_COMPILED_ARTIFACTS") is not None:
            compiled_model_path = os.getenv("NEURON_COMPILED_ARTIFACTS")
        elif os.path.exists(model_name_or_path):
            compiled_model_path = os.path.join(model_name_or_path,
                                               "neuron-compiled-artifacts",
                                               hashed_config)
            shutil.rmtree(compiled_model_path, ignore_errors=True)
        else:
            compiled_model_path = os.path.join("local-models",
                                               model_name_or_path,
                                               "neuron-compiled-artifacts",
                                               hashed_config)
            shutil.rmtree(compiled_model_path, ignore_errors=True)
        try:
            self.model = neuronx_model_cls(compiled_model_path)
            override_neuron_config = kwargs["override_neuron_config"]
            for k, v in override_neuron_config.items():
                setattr(self.model.config.neuron_config, k, v)
            self.model.load(compiled_model_path)
            return
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Exception: %s", e)
            logger.warning("Failed to load the model from %s Recompiling...",
                           compiled_model_path)
        if not os.path.exists(model_name_or_path):
            hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            saved_path = os.path.join("local-models", model_name_or_path)
            hf_model.save_pretrained(saved_path)
            model_name_or_path = saved_path
        if not os.path.exists(draft_model_name_or_path):
            if draft_model_name_or_path != model_name_or_path:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    draft_model_name_or_path)
                saved_path = os.path.join("local-models",
                                          draft_model_name_or_path)
                hf_model.save_pretrained(saved_path)
                draft_model_name_or_path = saved_path
            else:
                draft_model_name_or_path = model_name_or_path
            config.fused_spec_config.draft_model_path = draft_model_name_or_path
        self.model = neuronx_model_cls(model_name_or_path, config)
        self.model.compile(compiled_model_path)
        self.model.load(compiled_model_path)


def _get_model_architecture(config: PretrainedConfig) -> str:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _NEURON_SUPPORTED_MODELS:
            return arch
    raise ValueError(
        f"Model architectures {architectures} are not supported on Neuron "
        f"for now. Supported architectures: "
        f"{list(_NEURON_SUPPORTED_MODELS.keys())}")


def _get_default_neuron_config(model_config: ModelConfig,
                               parallel_config: ParallelConfig,
                               scheduler_config: SchedulerConfig,
                               lora_serving_config: LoraServingConfig):
    """Generate a neuron config based on vllm config args."""
    on_device_sampling_config = OnDeviceSamplingConfig(dynamic=True,
                                                       deterministic=False)
    batch_size = scheduler_config.max_num_seqs

    neuron_config = dict(
        tp_degree=parallel_config.tensor_parallel_size,
        ctx_batch_size=1,
        batch_size=batch_size,
        max_context_length=scheduler_config.max_model_len,
        seq_len=scheduler_config.max_model_len,
        enable_bucketing=True,
        is_continuous_batching=True,
        quantized=False,
        torch_dtype=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
        padding_side="right",
        on_device_sampling_config=on_device_sampling_config,
        sequence_parallel_enabled=True,
        lora_serving_config=lora_serving_config)
    return neuron_config


def _get_default_speculation_config(model_config: ModelConfig,
                                    parallel_config: ParallelConfig,
                                    scheduler_config: SchedulerConfig,
                                    speculation_config: SpeculativeConfig):
    """Generate a neuron config for speculative decoding based on vllm config
    args."""
    neuron_config = dict(
        tp_degree=parallel_config.tensor_parallel_size,
        ctx_batch_size=1,
        batch_size=scheduler_config.max_num_seqs,
        max_context_length=scheduler_config.max_model_len,
        seq_len=scheduler_config.max_model_len,
        speculation_length=speculation_config.num_speculative_tokens,
        trace_tokengen_model=False,
        enable_fused_speculation=True,
        enable_bucketing=True,
        is_continuous_batching=True,
        quantized=False,
        torch_dtype=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
        on_device_sampling_config=dict(
            top_k=1,
            do_sample=False,
        ))
    return neuron_config


def _get_neuron_config_after_override(default_neuron_config,
                                      overridden_neuron_config):
    """Update default neuron config values with override args"""
    overridden_neuron_config = overridden_neuron_config or {}
    default_neuron_config.update(overridden_neuron_config)
    return default_neuron_config


def get_neuron_model(model_config: ModelConfig,
                     parallel_config: ParallelConfig,
                     scheduler_config: SchedulerConfig,
                     lora_serving_config: LoraServingConfig) -> nn.Module:
    """Initializes a neuron-optimized model for inference."""
    model_arch = _get_model_architecture(model_config.hf_config)
    if model_arch == "MllamaForConditionalGeneration":
        model = NeuronMllamaForCausalLM(model_config.hf_config)
    else:
        model = NeuronCausalLM(model_config.hf_config)
    default_neuron_config_args = _get_default_neuron_config(
        model_config, parallel_config, scheduler_config, lora_serving_config)
    neuron_config = _get_neuron_config_after_override(
        default_neuron_config_args, model_config.override_neuron_config)

    override_neuron_config = model_config.override_neuron_config
    model.load_weights(model_config.model,
                       neuron_config=neuron_config,
                       override_neuron_config=override_neuron_config)
    return model.eval()


def get_neuron_speculation_model(model_config: ModelConfig,
                                 parallel_config: ParallelConfig,
                                 scheduler_config: SchedulerConfig,
                                 speculation_config: SpeculativeConfig):
    """Initializes a neuron-optimized speculation model for inference.
    
    This model handles speculation using both a draft model and an EAGLE draft. 
    """
    model = NeuronSpeculationCausalLM(model_config.hf_config)
    default_neuron_config_args = _get_default_speculation_config(
        model_config, parallel_config, scheduler_config, speculation_config)
    neuron_config = _get_neuron_config_after_override(
        default_neuron_config_args, model_config.override_neuron_config)

    override_neuron_config = model_config.override_neuron_config
    model.load_weights(model_config.model,
                       speculation_config.draft_model_config.model,
                       neuron_config=neuron_config,
                       override_neuron_config=override_neuron_config)
    return model.eval()
