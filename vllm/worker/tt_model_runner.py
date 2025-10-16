# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn
from transformers import TopPLogitsWarper

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.tt_loader import TTModelLoader
from vllm.model_executor.models import supports_multimodal
from vllm.platforms.tt import TTPlatform
from vllm.sequence import (CompletionSequenceGroupOutput, IntermediateTensors,
                           Logprob, SequenceGroupMetadata, SequenceOutput)
from vllm.utils import make_tensor_with_pad
from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase

logger = init_logger(__name__)


@dataclass(frozen=True)
class TTSamplingParams:
    """
    Used by TTModelInput.
    """
    temperature: Union[float, list[float]]
    top_k: Union[int, list[int]]
    top_p: Union[float, list[float]]


@dataclass(frozen=True)
class TTModelInput(ModelRunnerInputBase):
    """
    Used by the TTModelRunner.
    """
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    prompt_lens: Optional[List[int]]
    seq_groups: List[int]
    block_tables: torch.Tensor
    unpadded_batch_size: Union[int, List[int]]  # List is used for DP in V1
    tt_sampling_params: Union[Optional[TTSamplingParams], List[
        Optional[TTSamplingParams]]]  # List is used for DP in V1
    sampling_params_list: Optional[List[Any]]
    compat_sampling_used: bool
    sampling_metadata: Optional["SamplingMetadata"]
    multi_modal_kwargs: Dict[str, Any]
    cross_block_tables: torch.Tensor
    is_first_multi_step: bool = True
    is_last_step: bool = True
    async_callback: Optional[Callable] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "prompt_lens": self.prompt_lens,
            "seq_groups": self.seq_groups,
            "block_tables": self.block_tables,
            "unpadded_batch_size": self.unpadded_batch_size,
            "tt_sampling_params": self.tt_sampling_params,
            "sampling_params_list": self.sampling_params_list,
            "compat_sampling_used": self.compat_sampling_used,
            "sampling_metadata": self.sampling_metadata,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "cross_block_tables": self.cross_block_tables,
            "is_first_multi_step": self.is_first_multi_step,
            "is_last_step": self.is_last_step,
        }

        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type["TTModelInput"],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "TTModelInput":
        return cls(**tensor_dict)


def top_pk_logits_efficient(logits,
                            p=0.9,
                            k=10,
                            temperature=1.0,
                            return_probs=False):
    # Do not keep the entire vocab size after top k.
    # Instead, keep the k size tensor and record the associated indices.
    if k < 1:  # no top-k sampling if set to -1 or 0
        top_k_values, top_k_indices = logits, torch.arange(
            logits.shape[-1]).unsqueeze(0).repeat(logits.shape[0], 1)
    else:
        top_k_values, top_k_indices = torch.topk(logits, k=k)
    top_p_values = TopPLogitsWarper(top_p=p)(None, top_k_values)
    probs = F.softmax(top_p_values / temperature, dim=-1)
    probs = torch.nan_to_num(
        probs)  # convert nan to num to prevent error in multinomial
    top_k_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)
    if return_probs:
        return token, (probs, top_k_indices)
    else:
        return token


def sample_tokens(logits, tt_sampling_params: TTSamplingParams):
    if tt_sampling_params.temperature == 0:  # greedy decoding
        return torch.argmax(logits, dim=-1)
    else:  # top-k top-p sampling
        return top_pk_logits_efficient(
            logits,
            p=tt_sampling_params.top_p,
            k=tt_sampling_params.top_k,
            temperature=tt_sampling_params.temperature)


class TTModelRunner(ModelRunnerBase[TTModelInput]):

    def __init__(
        self,
        vllm_config: VllmConfig,
        trace_mode: bool = True,
    ):
        ModelRunnerBase.__init__(self, vllm_config=vllm_config)

        # Because of multiprocessing, the config-dependent
        # class attributes might not have been set in this process,
        # so we need to call this again.
        TTPlatform.check_and_update_config(vllm_config)

        # Currently, TT worker doesn't support chunked prefill.
        assert self.scheduler_config.chunked_prefill_enabled is False

        self.block_size = self.cache_config.block_size

        # whether to use ttnn tracing for model execution
        self.trace_mode = trace_mode
        self.sample_on_device_mode = TTPlatform.sample_on_device_mode
        logger.info(
            "TTModelRunner: trace_mode=%s, sample_on_device_mode=%s",
            self.trace_mode,
            self.sample_on_device_mode,
        )

        self.cached_step_outputs: List[torch.Tensor] = [
        ]  # Only used for multi-step execution

        self.request_specific_rope = 'Qwen2.5-VL' in self.model_config.model
        if self.model_config.is_encoder_decoder or self.request_specific_rope:
            assert (
                self.model_config.is_encoder_decoder
                and self.request_specific_rope
            ) is False, (
                "a model cannot be encoder-decoder and request-specific rope")
            # seq_id -> cached_req_data
            self.cached_req_data: Dict[int, Dict[str, Any]] = {}
            self.previous_seq_ids: Set[int] = set()

        # Detect if the model has "mrope" rope_scaling type.
        # mrope requires keep "rope_deltas" between prompt and decoding phases.
        if self.model_config.uses_mrope:
            assert ("TTModelRunner does not currently support models with "
                    "mrope rope_scaling")

        if TTPlatform.compat_sampling_possible:
            vocab_size = self.model_config.get_vocab_size()
            self.logits_processor = LogitsProcessor(vocab_size,
                                                    logits_as_input=True)
            # We are relying on having our logits shaped correctly,
            # as if they came from a regular vLLM model
            # and then got trimmed by the LogitsProcessor.
            # If we add prompt_logprobs or chunked prefill,
            # we need to fully match the relevant parts of
            # SamplingMetadata.selected_token_indices logic.
            self.sampler = get_sampler()

    def load_model(self) -> None:
        # Note: using custom TT loader
        # instead of selecting from default vllm loaders
        loader = TTModelLoader(self.load_config)

        self.model = loader.load_model(vllm_config=self.vllm_config,
                                       model_config=self.model_config)
        if self.model_config.is_encoder_decoder:
            self.max_cross_blocks = (self.model.max_cross_attn_tokens //
                                     self.cache_config.block_size)

        is_dp = (self.model_config.override_tt_config
                 and self.model_config.override_tt_config.get(
                     "data_parallel", 1) > 1)

        # Detect if the model is a TG Llama to use DP KV cache
        # vLLM doesn't know which blocks correspond to which DP device pool so
        # may allocate non-local blocks to a user. To avoid bad output because
        # of this, we maintain a seq_id_to_batch_slot mapping so that we can
        # place the users on the correct devices. This requires passing seq_id
        # and finished requests to the generator.
        # TODO: Extend this to support other DP models

        if ("Llama" in self.model_config.model
                and "70B" in self.model_config.model
                and self.device_config.num_devices == 32) or is_dp:
            self.dp_kv_cache = True
        else:
            self.dp_kv_cache = False

        self.async_torch_proc = self.sample_on_device_mode is not None

        if self.dp_kv_cache:
            # Map request id strs to seq group ids
            self.req_id_to_seq_id: Dict[str, int] = {}
            self.empty_slots = list(range(self.scheduler_config.max_num_seqs))
            self.seq_groups_to_batch_slot: Dict[int, int] = {}
            if self.async_torch_proc:
                self.cached_read_events: List[List[Any]] = [
                ]  # Only used for multi-step execution
                self.perm_table_tensor: List[torch.Tensor] = []

    def get_model(self) -> nn.Module:
        return self.model

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> TTModelInput:
        return TTModelInput.from_broadcasted_tensor_dict(tensor_dict, )

    def recover_orphaned_slots(self, orphaned_seq_ids: List[int]):
        for seq_id in orphaned_seq_ids:
            slot = self.seq_groups_to_batch_slot[seq_id]
            if slot not in self.empty_slots:
                self.empty_slots.append(slot)
                logger.warning(
                    "SLOT_DEBUG: Recovered slot %s from orphaned seq %s", slot,
                    seq_id)
            else:
                logger.warning(
                    "SLOT_DEBUG: Orphaned slot %s already in "
                    "empty_slots, from seq %s", slot, seq_id)
            del self.seq_groups_to_batch_slot[seq_id]
            # Clean up req_id_to_seq_id mapping for orphaned sequences
            req_ids_to_remove = [
                req_id
                for req_id, mapped_seq_id in self.req_id_to_seq_id.items()
                if mapped_seq_id == seq_id
            ]
            for req_id in req_ids_to_remove:
                del self.req_id_to_seq_id[req_id]

        # sort empty_slots for consistency
        self.empty_slots.sort()

    def prepare_model_input(
            self,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            virtual_engine: int = 0,
            finished_requests_ids: Optional[List[str]] = None) -> TTModelInput:

        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[
            0].is_prompt  # prefill if True, otherwise decode
        assert all(
            x.is_prompt == is_prompt for x in seq_group_metadata_list
        ), "Currently only supporting all prefills or all decodes in seq group"

        unpadded_batch_size = len(seq_group_metadata_list)
        assert unpadded_batch_size > 0

        input_tokens_list: List[int] = []
        input_positions_list: List[int] = []
        seq_lens: List[int] = []
        block_tables_list: List[List[int]] = []
        seq_groups_list: List[int] = []
        sampling_params_list = []
        top_pk_sampling_params: Dict[str, Any] = {}
        multi_modal_kwargs: Dict[str, Any] = {}
        if supports_multimodal(self.model) and is_prompt:
            multi_modal_kwargs = {"images": []}
        cross_block_tables_list: List[List[int]] = []

        # create seq_groups_list before any cleanup to active batch slots
        for seq_group_metadata in seq_group_metadata_list:
            _seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(_seq_ids) == 1, (
                "Currently only supporting one sequence per request group")
            seq_groups_list.append(_seq_ids[0])

        if self.dp_kv_cache and finished_requests_ids is not None:
            # Delete finished requests from req_id_to_seq_id
            finished_requests_seq_ids = []
            for req_id in finished_requests_ids:
                # Only delete if the request was added in the first place
                if req_id in self.req_id_to_seq_id:
                    finished_requests_seq_ids.append(
                        self.req_id_to_seq_id[req_id])
                    del self.req_id_to_seq_id[req_id]

        # Compat sampling is off by default, and enabled only on request
        # or if any of the requests in the batch require it
        compat_sampling_used = False
        if TTPlatform.always_compat_sampling:
            compat_sampling_used = True
        else:
            for seq_group_metadata in seq_group_metadata_list:
                sampling_params = seq_group_metadata.sampling_params
                if TTPlatform.compat_sampling_required(sampling_params):
                    compat_sampling_used = True
                    break

        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            seq_id = seq_ids[0]
            if self.dp_kv_cache:
                # Add new request id to req_id_to_seq_id
                self.req_id_to_seq_id[seq_group_metadata.request_id] = seq_id

            multi_modal_data = seq_group_metadata.multi_modal_data
            seq_data = seq_group_metadata.seq_data[seq_id]

            seq_lens.append(seq_data.get_len())

            if is_prompt:
                # tokens
                prompt_tokens = seq_data.get_token_ids()
                input_tokens_list.append(prompt_tokens)
            else:
                # tokens
                generation_token = seq_data.get_last_token_id()
                input_tokens_list.append(generation_token)

                # positions
                position = seq_data.get_len() - 1
                input_positions_list.append(position)

            block_table = seq_group_metadata.block_tables[seq_id]
            block_tables_list.append(block_table)

            # Multi-modal data
            # TODO: Replace with multi_modal_input_mapper
            # (used by CPU/GPU model runners) once TT models
            # no longer require raw PIL images
            if supports_multimodal(self.model) and is_prompt:
                if (multi_modal_data := seq_group_metadata.multi_modal_data):
                    assert "image" in multi_modal_data, (
                        "Currently only supporting image multi-modal inputs")
                    image = multi_modal_data[
                        "image"]  # this is of type PIL.Image.Image
                    multi_modal_kwargs["images"].append(image)
                else:
                    multi_modal_kwargs["images"].append(None)

            # Encoder-decoder data
            # (currently only supporting cross attention metadata
            # and not additional encoder data)
            if self.model_config.is_encoder_decoder:
                cross_block_table = seq_group_metadata.cross_block_table
                cross_block_tables_list.append(cross_block_table)

            sampling_params = seq_group_metadata.sampling_params
            if compat_sampling_used:
                sampling_params_list.append(sampling_params)
            elif TTPlatform.non_greedy_decoding_on_device:
                # non-uniform sampling

                # initializing an empty list for each value on first iter
                # fill values after first iter
                for key in ["temperature", "top_k", "top_p"]:
                    top_pk_sampling_params.setdefault(key, []).append(
                        getattr(sampling_params, key))
            else:
                # uniform sampling
                if len(top_pk_sampling_params) == 0:
                    top_pk_sampling_params[
                        "temperature"] = sampling_params.temperature
                    top_pk_sampling_params["top_k"] = sampling_params.top_k
                    top_pk_sampling_params["top_p"] = sampling_params.top_p
                else:
                    if (top_pk_sampling_params["temperature"]
                            != sampling_params.temperature):
                        logger.warning(
                            "Currently only supporting same temperature for"
                            "all sequences in batch, falling back to first "
                            "sequence's temperature (%s)",
                            top_pk_sampling_params['temperature'])
                    if top_pk_sampling_params[
                            "top_k"] != sampling_params.top_k:
                        logger.warning(
                            "Currently only supporting same top_k"
                            "for all sequences in batch, "
                            "falling back to first sequence's top_k (%s)",
                            top_pk_sampling_params['top_k'])
                    if top_pk_sampling_params[
                            "top_p"] != sampling_params.top_p:
                        logger.warning(
                            "Currently only supporting same top_p"
                            "for all sequences in batch, "
                            "falling back to first sequence's top_p (%s)",
                            top_pk_sampling_params['top_p'])

        if compat_sampling_used:
            # seq_lens means how many tokens are in the sequence in total,
            # query lens means how many tokens are newly being processed,
            # and are contained in the output logits.
            if is_prompt:
                query_lens = [x for x in seq_lens]
            else:
                query_lens = [1 for x in seq_lens]
            generators = self.get_generators(finished_requests_ids)
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list,
                seq_lens,
                query_lens,
                "cpu",
                pin_memory=False,
                generators=generators)
            tt_sampling_params = None
        else:
            sampling_metadata = None
            tt_sampling_params = TTSamplingParams(
                temperature=top_pk_sampling_params["temperature"],
                top_k=top_pk_sampling_params["top_k"],
                top_p=top_pk_sampling_params["top_p"])

        # Remove cached encoder-decoder data
        # for any seq ids that are not in the current batch
        # (assume they were either finished or preempted)
        if ((self.model_config.is_encoder_decoder
             or self.request_specific_rope) and not is_prompt
                and self.cached_req_data):
            seq_ids_to_del = []
            for seq_id in self.cached_req_data:
                if seq_id not in seq_groups_list:
                    seq_ids_to_del.append(seq_id)
            for seq_id in seq_ids_to_del:
                del self.cached_req_data[seq_id]

        # Convert lists to tensors and add padding

        block_tables = make_tensor_with_pad(block_tables_list,
                                            dtype=torch.int32,
                                            device="cpu",
                                            pad=0)
        if self.model_config.is_encoder_decoder:
            cross_block_tables = make_tensor_with_pad(cross_block_tables_list,
                                                      dtype=torch.int32,
                                                      device="cpu",
                                                      pad=0)
        else:
            cross_block_tables = None
        if is_prompt:
            input_tokens = make_tensor_with_pad(input_tokens_list,
                                                dtype=torch.int32,
                                                device="cpu",
                                                pad=0)
            input_positions = 0
            prompt_lens = seq_lens
        else:
            input_tokens = torch.tensor(input_tokens_list,
                                        dtype=torch.int32,
                                        device="cpu").view(-1, 1)
            input_positions = torch.tensor(input_positions_list,
                                           dtype=torch.int32,
                                           device="cpu")
            prompt_lens = None

            # TODO: Remove once TT models can support arbitrary batch sizes
            # Pad batch to max_num_seqs
            if input_tokens.shape[0] < self.scheduler_config.max_num_seqs:
                batch_pad_len = self.scheduler_config.max_num_seqs - \
                    input_tokens.shape[0]
                input_tokens = torch.cat([
                    input_tokens,
                    torch.zeros(batch_pad_len,
                                1,
                                dtype=torch.int32,
                                device="cpu")
                ])
                input_positions = torch.cat([
                    input_positions,
                    torch.ones(batch_pad_len, dtype=torch.int32, device="cpu")
                    * -1  # Pad with -1 to indicate no position
                ])
                block_tables = torch.cat([
                    block_tables,
                    torch.zeros(batch_pad_len,
                                block_tables.shape[1],
                                dtype=torch.int32,
                                device="cpu")
                ])
                if self.model_config.is_encoder_decoder:
                    cross_block_tables = torch.cat([
                        cross_block_tables,
                        torch.zeros(batch_pad_len,
                                    cross_block_tables.shape[1],
                                    dtype=torch.int32,
                                    device="cpu")
                    ])

            # Pad block_tables to max num blocks
            # so ttnn tracing can work (requires constant shape)
            if self.trace_mode:
                block_tables = torch.cat([
                    block_tables,
                    torch.zeros(block_tables.shape[0],
                                self.cache_config.num_gpu_blocks -
                                block_tables.shape[1],
                                dtype=torch.int32,
                                device="cpu")
                ],
                                         dim=1)
                if self.model_config.is_encoder_decoder:
                    # Note for vision models: the number of cross blocks
                    # may change if the number of image tiles changes
                    # or if prompts are text-only
                    cross_block_tables = torch.cat([
                        cross_block_tables,
                        torch.zeros(cross_block_tables.shape[0],
                                    self.max_cross_blocks -
                                    cross_block_tables.shape[1],
                                    dtype=torch.int32,
                                    device="cpu")
                    ],
                                                   dim=1)

        if self.dp_kv_cache:
            # Clean up finished requests
            if finished_requests_ids:
                for seq_id in finished_requests_seq_ids:
                    if seq_id in self.seq_groups_to_batch_slot:
                        empty_batch_slot = (
                            self.seq_groups_to_batch_slot[seq_id])
                        # Only add to empty_slots if not already present
                        # (prevent duplicates)
                        if empty_batch_slot not in self.empty_slots:
                            self.empty_slots.append(empty_batch_slot)
                        else:
                            logger.warning(
                                "SLOT_DEBUG: Slot %s from seq %s already in "
                                "empty_slots", empty_batch_slot, seq_id)
                        del self.seq_groups_to_batch_slot[seq_id]
                    else:
                        logger.warning(
                            "SLOT_DEBUG: Finished seq %s not found in "
                            "seq_groups_to_batch_slot", seq_id)

            # Clean up disappeared sequences (preempted/swapped out)
            # ONLY during DECODE batches: all active sequences should be
            # present, so any missing sequence is truly gone
            # (preempted/swapped/finished).
            # During PREFILL batches: NEVER clean up based on "not in
            # batch" because we can't distinguish between active decode
            # sequences and truly gone sequences.
            # Only finished_requests_ids is a reliable signal for cleanup
            # during prefill.
            if not is_prompt:
                # Decode batch: clean up sequences not in this batch
                orphaned_seq_ids = []
                current_batch_seq_ids = set(seq_groups_list)
                for seq_id in list(self.seq_groups_to_batch_slot.keys()):
                    if seq_id not in current_batch_seq_ids:
                        orphaned_seq_ids.append(seq_id)

                if orphaned_seq_ids:
                    logger.info(
                        "SLOT_DEBUG: Detected %s sequences not in decode "
                        "batch (preempted/swapped): %s", len(orphaned_seq_ids),
                        orphaned_seq_ids)
                    logger.info(
                        "SLOT_DEBUG: Before cleanup - empty_slots "
                        "(len=%s)=%s", len(self.empty_slots), self.empty_slots)
                    # Free their slots immediately
                    self.recover_orphaned_slots(orphaned_seq_ids)
                    logger.info(
                        "SLOT_DEBUG: After cleanup - empty_slots "
                        "(len=%s)=%s", len(self.empty_slots), self.empty_slots)
                    logger.info(
                        "SLOT_DEBUG: After cleanup - "
                        "seq_groups_to_batch_slot=%s",
                        self.seq_groups_to_batch_slot)
            elif is_prompt and len(self.empty_slots) < unpadded_batch_size:
                # Prefill batch without enough slots: this is a scheduler
                # bug. Log an error but don't forcibly clean up - we can't
                # safely determine which sequences are truly gone vs just
                # not in this prefill batch
                logger.error(
                    "SLOT_DEBUG: Prefill batch needs %s slots but only "
                    "%s available. This indicates a scheduler issue - "
                    "sequences may not be properly freed. Active slots: "
                    "%s, seq_groups_to_batch_slot=%s", unpadded_batch_size,
                    len(self.empty_slots), len(self.seq_groups_to_batch_slot),
                    self.seq_groups_to_batch_slot)

        return TTModelInput(input_tokens=input_tokens,
                            input_positions=input_positions,
                            prompt_lens=prompt_lens,
                            seq_groups=seq_groups_list,
                            block_tables=block_tables,
                            unpadded_batch_size=unpadded_batch_size,
                            tt_sampling_params=tt_sampling_params,
                            sampling_params_list=sampling_params_list,
                            compat_sampling_used=compat_sampling_used,
                            sampling_metadata=sampling_metadata,
                            multi_modal_kwargs=multi_modal_kwargs,
                            cross_block_tables=cross_block_tables)

    @torch.no_grad()
    def execute_model(
        self,
        model_input: TTModelInput,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        is_decode = model_input.prompt_lens is None

        # Note on async_out_proc + multi-step: for gpu/tpu, the N steps are
        # enqueued on device and the last step will trigger the output
        # processor for all outputs but the last. Currently for TT,
        # the inputs/outputs of each step are transferred between host/device,
        # and async_out_proc will trigger the output processor for step (i)
        # on host while device is executing step (i+1).
        use_async_out_proc = model_input.async_callback is not None

        if not is_decode:
            assert num_steps == 1, "Num steps must be 1 for prefill"
        # always true if not using multi-step
        if model_input.is_first_multi_step:
            # This is a queue of torch tensor with step outputs
            # - full sampler outputs for compat mode,
            # ttnn tensors in flight for async_torch_proc,
            # or torch tensors otherwise.
            # If we do async output processing,
            # the queue is consumed by _send_prev_step_async_out,
            # except the last step
            # If not, we consume the whole queue after executing the last step.
            self.cached_step_outputs = []
            if is_decode:
                self.cached_read_events = []
            for i in range(num_steps):
                next_token_ids = self._execute_model_single_step(
                    model_input,
                    kv_caches,
                    is_decode,
                    use_async_out_proc,
                    step_idx=i)
                if is_decode and self.async_torch_proc:
                    next_token_ids, read_event = next_token_ids
                    self.cached_read_events.append(read_event)
                self.cached_step_outputs.append(next_token_ids)
                if (i < num_steps - 1 and not self.sample_on_device_mode):
                    if model_input.compat_sampling_used:
                        # For now, this will only get called
                        # if we explicitly enable compat sampling
                        # Most cases where we want to use compat sampling
                        # are not compatible with multistep
                        next_token_ids = self._get_next_token_ids_from_sampler_output(  #noqa: E501
                            next_token_ids)
                    # Prepare the inputs for the next step
                    new_input_tokens = next_token_ids.unsqueeze(dim=1).int()
                    if new_input_tokens.shape[
                            0] < self.scheduler_config.max_num_seqs:
                        # Pad batch to max_num_seqs
                        batch_pad_len = model_input.input_tokens.shape[
                            0] - new_input_tokens.shape[0]
                        new_input_tokens = torch.cat([
                            new_input_tokens,
                            torch.zeros(batch_pad_len,
                                        1,
                                        dtype=torch.int32,
                                        device="cpu")
                        ])

                    # Update input positions for all
                    # except those that are -1 (padding)
                    new_input_positions = torch.where(
                        model_input.input_positions == -1,
                        model_input.input_positions,
                        model_input.input_positions + 1)

                    model_input = dataclasses.replace(
                        model_input,
                        input_tokens=new_input_tokens,
                        input_positions=new_input_positions)

            if use_async_out_proc:
                assert model_input.async_callback is not None
                model_input.async_callback()  # trigger output processor

        sampler_outputs = []  # no outputs unless last step
        if model_input.is_last_step:  # always true if not using multi-step
            num_outputs = len(self.cached_step_outputs)
            if use_async_out_proc:
                # The queue should be getting consumed by
                # _send_prev_step_async_out.
                # The last step should have 1 output unless we have
                # scheduled less than self.scheduler_config.num_lookahead_slots
                # + 1 steps in which case there will be 0 outputs.
                assert num_outputs <= 1, (
                    "Last step should have at most one output")
            for i in range(num_outputs):
                if model_input.compat_sampling_used:
                    sampler_output = self.cached_step_outputs.pop(0)
                else:
                    next_token_ids = self.cached_step_outputs.pop(0)
                    if is_decode and self.async_torch_proc:
                        next_token_ids = self._complete_torch_async_proc(
                            next_token_ids)
                    # TODO: sync read back from device
                    # once model can keep executing steps on device
                    sampler_output = self._make_sampler_output(
                        next_token_ids, model_input.seq_groups)
                sampler_outputs.append(sampler_output)

        return sampler_outputs

    def _complete_torch_async_proc(self, next_token_ids):
        read_events = self.cached_read_events.pop(0)
        for event in read_events:
            ttnn.event_synchronize(event)
        next_token_ids = self.model.process_decode_output_host(
            next_token_ids, is_tokens=(self.sample_on_device_mode is not None))
        if self.dp_kv_cache:
            # permute the tt_out
            next_token_ids = next_token_ids[self.perm_table_tensor.pop(0)]
        return next_token_ids

    def _send_async_out(self, sampler_output, async_callback,
                        is_first_step_output):
        ctx = async_callback.keywords["ctx"]
        ctx.append_output(outputs=[sampler_output],
                          seq_group_metadata_list=ctx.seq_group_metadata_list,
                          scheduler_outputs=ctx.scheduler_outputs,
                          is_async=False,
                          is_last_step=False,
                          is_first_step_output=is_first_step_output)
        async_callback()  # trigger output processor

    def _make_sampler_output(
        self,
        next_token_ids: List[int],
        seq_groups: List[int],
    ) -> SamplerOutput:
        # Minimal code to construct the sampler outputs,
        # based on tpu_model_runner.py
        # TT backend does not support the advanced sampling parameters
        # such as logprobs.
        zero_logprob = Logprob(0.0)
        sampler_outputs = []
        for batch_idx, seq_id in enumerate(seq_groups):
            next_token_id = int(next_token_ids[batch_idx])
            seq_outputs = [
                SequenceOutput(seq_id, next_token_id,
                               {next_token_id: zero_logprob})
            ]
            sampler_outputs.append(
                CompletionSequenceGroupOutput(seq_outputs, None))
        return SamplerOutput(sampler_outputs)

    def _send_prev_step_async_out(self, model_input: TTModelInput, step_idx):
        if step_idx > 0:
            step_output = self.cached_step_outputs.pop(0)
            if model_input.compat_sampling_used:
                sampler_output = step_output
            else:
                next_token_ids = step_output
                if self.async_torch_proc:
                    next_token_ids = self._complete_torch_async_proc(
                        next_token_ids)
                # TODO: sync read back from device
                # once model can keep executing steps on device
                sampler_output = self._make_sampler_output(
                    next_token_ids, model_input.seq_groups)
            self._send_async_out(sampler_output,
                                 model_input.async_callback,
                                 is_first_step_output=(step_idx == 1))
        else:
            # trigger output processor in case last step was prefill
            assert model_input.async_callback is not None
            model_input.async_callback()

    def _execute_model_single_step(self,
                                   model_input: TTModelInput,
                                   kv_caches: List[torch.Tensor],
                                   is_decode,
                                   use_async_out_proc=False,
                                   step_idx=0):
        execute_model_kwargs = {
            "tokens": model_input.input_tokens,
            "page_table": model_input.block_tables,
            "kv_cache": kv_caches,
            **(model_input.multi_modal_kwargs or {}),
        }
        if not is_decode:
            execute_model_kwargs["prompt_lens"] = model_input.prompt_lens
        else:
            execute_model_kwargs["start_pos"] = model_input.input_positions

        if self.sample_on_device_mode == "all" or (
                self.sample_on_device_mode == "decode_only" and is_decode):
            execute_model_kwargs[
                "sampling_params"] = model_input.tt_sampling_params

        if model_input.cross_block_tables is not None:
            execute_model_kwargs[
                "cross_page_table"] = model_input.cross_block_tables

        assert isinstance(model_input.unpadded_batch_size,
                          int), ("unpadded_batch_size must be an int")

        if not is_decode:
            if self.dp_kv_cache:
                slots_to_allocate = self.empty_slots[:model_input.
                                                     unpadded_batch_size]
                execute_model_kwargs["empty_slots"] = slots_to_allocate

            outputs = self.model.prefill_forward(**execute_model_kwargs)

            if self.dp_kv_cache:
                # update the batch slot table
                recently_filled_slots = self.empty_slots[:model_input.
                                                         unpadded_batch_size]
                self.empty_slots = self.empty_slots[model_input.
                                                    unpadded_batch_size:]

                # iterate through recently_filled_slots slice
                for i, s in enumerate(model_input.seq_groups):
                    self.seq_groups_to_batch_slot[s] = recently_filled_slots[i]

            if self.model_config.is_encoder_decoder:
                # Save encoder-decoder data for use in subsequent decode steps
                # (may need to be updated for future models)
                tt_out, prefill_cross_attention_masks, \
                prefill_full_text_row_masked_out_mask, \
                decode_cross_attention_masks, \
                 decode_full_text_row_masked_out_mask = outputs

                for i, seq_id in enumerate(model_input.seq_groups):
                    enc_dec_data = {
                        "prefill_cross_attention_masks":
                        prefill_cross_attention_masks[i],
                        "prefill_full_text_row_masked_out_mask":
                        prefill_full_text_row_masked_out_mask[i],
                        "decode_cross_attention_masks":
                        decode_cross_attention_masks[i],
                        "decode_full_text_row_masked_out_mask":
                        decode_full_text_row_masked_out_mask[i]
                    }
                    self.cached_req_data[seq_id] = enc_dec_data
            elif self.request_specific_rope:
                tt_out, rot_mats = outputs
                # tt_out: [batch_size, seq_len, vocab_size];
                # rot_mats: List[[batch_size, 1, seq_len, head_dim]]
                for i, seq_id in enumerate(model_input.seq_groups):
                    self.cached_req_data[seq_id] = {
                        "rot_mats": (
                            # cos: [1, 1, seq_len, head_dim]
                            rot_mats[0][i:i + 1],
                            # sin: [1, 1, seq_len, head_dim]
                            rot_mats[1][i:i + 1],
                        )
                    }
            else:
                # [ batch_size] if sampling on device
                # [ batch_size, len, vocab_size] if not sampling on device
                # the logits are not guaranteed to be for the whole sequence,
                # usually only last token.
                tt_out = outputs
        else:  #decode
            if self.model_config.is_encoder_decoder:
                assert self.cached_req_data

                # Use encoder-decoder data from prefill step
                prefill_cross_attention_masks = [
                    self.cached_req_data[seq_id]
                    ["prefill_cross_attention_masks"]
                    for seq_id in model_input.seq_groups
                ]
                prefill_full_text_row_masked_out_mask = [
                    self.cached_req_data[seq_id]
                    ["prefill_full_text_row_masked_out_mask"]
                    for seq_id in model_input.seq_groups
                ]
                decode_cross_attention_masks = [
                    self.cached_req_data[seq_id]
                    ["decode_cross_attention_masks"]
                    for seq_id in model_input.seq_groups
                ]
                decode_full_text_row_masked_out_mask = [
                    self.cached_req_data[seq_id]
                    ["decode_full_text_row_masked_out_mask"]
                    for seq_id in model_input.seq_groups
                ]
                enc_dec_kwargs = {
                    "prefill_cross_attention_masks":
                    prefill_cross_attention_masks,
                    "prefill_full_text_row_masked_out_mask":
                    prefill_full_text_row_masked_out_mask,
                    "decode_cross_attention_masks":
                    decode_cross_attention_masks,
                    "decode_full_text_row_masked_out_mask":
                    decode_full_text_row_masked_out_mask
                }
            elif self.request_specific_rope:
                if any(seq_id not in self.previous_seq_ids
                       for seq_id in model_input.seq_groups):
                    enc_dec_kwargs = {
                        "rot_mats_all_users": [
                            self.cached_req_data[seq_id]["rot_mats"]
                            for seq_id in model_input.seq_groups
                        ]
                    }
                else:
                    enc_dec_kwargs = {"rot_mats_all_users": None}
                self.previous_seq_ids = set(model_input.seq_groups)
            else:
                enc_dec_kwargs = {}

            if self.dp_kv_cache:
                # Calculate perm_table_tensor:
                # perm_table_tensor[new_idx] = current_slot_idx
                active_slots = [
                    self.seq_groups_to_batch_slot[s]
                    for s in model_input.seq_groups
                ]

                perm_table_tensor = torch.as_tensor(
                    active_slots + self.empty_slots,
                    dtype=torch.long,
                )
                if self.async_torch_proc:
                    self.perm_table_tensor.append(perm_table_tensor)

                # Calculate inverse_perm_indices:
                # inverse_perm_indices[current_slot_idx] = new_idx
                inverse_perm_indices = torch.empty_like(perm_table_tensor)
                inverse_perm_indices[perm_table_tensor] = torch.arange(
                    perm_table_tensor.size(0),
                    dtype=torch.long,
                )

                # permute the start_pos, tokens, and page_table
                execute_model_kwargs["start_pos"] = execute_model_kwargs[
                    "start_pos"][inverse_perm_indices]
                execute_model_kwargs["tokens"] = execute_model_kwargs[
                    "tokens"][inverse_perm_indices, :]
                execute_model_kwargs["page_table"] = execute_model_kwargs[
                    "page_table"][inverse_perm_indices, :]

            tt_out = self.model.decode_forward(**execute_model_kwargs,
                                               **enc_dec_kwargs,
                                               enable_trace=self.trace_mode,
                                               read_from_device=False)
            if use_async_out_proc:
                # trigger output processor on host while device is executing
                # next step
                self._send_prev_step_async_out(model_input, step_idx)
            if self.async_torch_proc:
                tt_out, read_event = self.model.read_decode_output(
                    tt_out, async_read=True)
            else:
                # outputs ttnn host tensors
                tt_out = self.model.read_decode_output(tt_out)
                # outputs torch tensor
                tt_out = self.model.process_decode_output_host(
                    tt_out, is_tokens=(self.sample_on_device_mode is not None))
            if self.dp_kv_cache and not self.async_torch_proc:
                tt_out = tt_out[perm_table_tensor]

        if model_input.compat_sampling_used:
            tt_logits = tt_out[:model_input.unpadded_batch_size,
                               -1, :]  # [unpadded batch, vocab]
            #This is coincidentally the same shape as the logits
            # we would get from a regular vllm model,
            # assuming we have no prompt logprobs, and one sequence per group.

            # Apply logits processing (including structured output filtering!)
            filtered_logits = self.logits_processor(
                lm_head=None,  # Ignored in our subclass
                hidden_states=tt_logits,  # Pass pre-computed logits
                sampling_metadata=model_input.sampling_metadata)

            # Sample tokens using standard vLLM sampler
            sampler_output = self.sampler(
                logits=filtered_logits,
                sampling_metadata=model_input.sampling_metadata)

            return sampler_output
        else:
            if not self.sample_on_device_mode or (self.sample_on_device_mode
                                                  == "decode_only"
                                                  and not is_decode):
                # unpadded batch, vocab of last token
                next_logits = tt_out[:model_input.unpadded_batch_size, -1, :]
                assert model_input.tt_sampling_params is not None
                assert isinstance(
                    model_input.tt_sampling_params, TTSamplingParams), (
                        "tt_sampling_params must be a TTSamplingParams")
                next_token_ids = sample_tokens(next_logits,
                                               model_input.tt_sampling_params)
            else:  # sample on device
                if self.async_torch_proc:
                    # do not slice as this may be mid-transfer to host
                    next_token_ids = tt_out
                else:
                    next_token_ids = tt_out[:model_input.unpadded_batch_size]
            if is_decode and self.async_torch_proc:
                # async torch proc only works in decode
                return tt_out, read_event
            else:
                return next_token_ids

    def _get_next_token_ids_from_sampler_output(
            self, sampler_output: SamplerOutput) -> torch.Tensor:
        """Extract next token IDs from sampler output."""
        next_token_ids = []
        for seq_group_output in sampler_output.outputs:
            for seq_output in seq_group_output.samples:
                next_token_ids.append(seq_output.output_token)
        return torch.tensor(next_token_ids, dtype=torch.int32, device="cpu")
