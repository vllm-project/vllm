# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Type, Union)
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import (CompletionSequenceGroupOutput, IntermediateTensors,
                           Logprob, SequenceGroupMetadata, SequenceOutput)
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase,
    _add_attn_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)

# Here we utilize the behavior that out-of-bound index is ignored.
# FIXME(woosuk): Find a more reliable way to prevent possible bugs.
_PAD_SLOT_ID = 1_000_000_000
# FIXME(woosuk): Temporarily disabled top-p sampling since it's too slow.
_ENABLE_TOP_P = False
# FIXME(woosuk): A temporary hack to support `n > 1`.
# This can significantly affect the performance if too large.
_MAX_NUM_SAMPLES = 128


class ExecutionMode(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()
    PREFIX_PREFILL = enum.auto()

    def is_prefill(self) -> bool:
        return self in (ExecutionMode.PREFILL, ExecutionMode.PREFIX_PREFILL)


@dataclass(frozen=True)
class ModelInputForTPU(ModelRunnerInputBase):
    token_ids: torch.Tensor
    position_ids: torch.Tensor
    attn_metadata: AttentionMetadata
    input_lens: torch.Tensor
    t: torch.Tensor
    p: torch.Tensor
    num_samples: int
    n: List[int]
    seq_groups: List[List[int]]
    is_first_multi_step: bool = True
    is_last_step: bool = True
    virtual_engine: int = 0
    async_callback: Optional[Callable] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = {
            "token_ids": self.token_ids,
            "position_ids": self.position_ids,
            "input_lens": self.input_lens,
            "t": self.t,
            "p": self.p,
            "num_samples": self.num_samples,
            "n": self.n,
            "seq_groups": self.seq_groups,
            "is_first_multi_step": self.is_first_multi_step,
            "is_last_step": self.is_last_step,
            "virtual_engine": self.virtual_engine,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type["ModelInputForTPU"],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForTPU":
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


class TPUModelRunner(ModelRunnerBase[ModelInputForTPU]):

    def __init__(
        self,
        vllm_config: VllmConfig,
        is_driver_worker: bool = False,
    ):
        ModelRunnerBase.__init__(self, vllm_config=vllm_config)
        self.is_driver_worker = is_driver_worker

        self.block_size = self.cache_config.block_size
        self.max_num_blocks_per_seq = (self.model_config.max_model_len //
                                       self.block_size)
        self.block_tables = np.zeros(
            (self.scheduler_config.max_num_seqs, self.max_num_blocks_per_seq),
            dtype=np.int32)
        self.attn_backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.cache_config.cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
            False,
        )
        self.cached_step_outputs: List[torch.Tensor] = []

        smem_size = 512 * 1024
        block_table_size = 4 * self.block_tables.size
        if block_table_size >= smem_size:
            logger.warning(
                "The max_model_len (%d) is too large. This may degrade the "
                "performance due to the insufficient smem size. Consider "
                "setting --max-model-len to a smaller value, like %d.",
                self.model_config.max_model_len,
                self.model_config.max_model_len /
                (block_table_size / smem_size))

    def load_model(self) -> None:
        self.device = self.device_config.device

        # NOTE(woosuk): While the executor assigns the TP ranks to the worker
        # process, the ranks can be different from the ranks internally assigned
        # by the xm runtime. Therefore, there is a mismatch in the rank
        # assignment between the gloo (cpu) runtime and the xm (tpu) runtime.
        # This is not a problem in linear layers because all-reduce is
        # rank-agnostic. However, it matters for all-gather as the ranks
        # determine the order of concatenating the output tensors.
        # As a workaround, we use the xm's rank assignment only when loading
        # the embedding weights.
        xm_tp_rank = xr.global_ordinal()
        with patch(
                "vllm.model_executor.layers.vocab_parallel_embedding."
                "get_tensor_model_parallel_rank",
                return_value=xm_tp_rank):
            model = get_model(vllm_config=self.vllm_config)
        model = model.eval()
        xm.wait_device_ops()
        model = ModelWrapper(model)
        self.model = torch.compile(model,
                                   backend="openxla",
                                   fullgraph=True,
                                   dynamic=False)

    def get_model(self) -> nn.Module:
        return self.model.model

    def _dummy_run(
        self,
        batch_size: int,
        seq_len: int,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        exec_mode: ExecutionMode,
    ) -> None:
        exec_mode = ExecutionMode(exec_mode)
        if exec_mode.is_prefill():
            seq_len = (seq_len + 15) // 16 * 16
            token_ids = torch.zeros((batch_size, seq_len),
                                    dtype=torch.int32,
                                    device=self.device)
            position_ids = torch.zeros((batch_size, seq_len),
                                       dtype=torch.int32,
                                       device=self.device)
            slot_mapping = torch.zeros((batch_size, seq_len),
                                       dtype=torch.int64,
                                       device=self.device)
            input_lens = torch.ones((batch_size, ),
                                    dtype=torch.int32,
                                    device=self.device)
            if exec_mode == ExecutionMode.PREFILL:
                attn_metadata = self.attn_backend.make_metadata(
                    num_prefills=batch_size,
                    num_prefill_tokens=batch_size * seq_len,
                    num_decode_tokens=0,
                    slot_mapping=slot_mapping,
                    multi_modal_placeholder_index_maps=None,
                    enable_kv_scales_calculation=False,
                    block_tables=None,
                    context_lens=None,
                    effective_query_lens=None,
                )
            else:
                context_lens = torch.ones((batch_size, ),
                                          dtype=torch.int32,
                                          device=self.device)
                block_tables = torch.tensor(self.block_tables[:batch_size],
                                            dtype=torch.int32,
                                            device=self.device)
                effective_query_lens = torch.ones_like(context_lens)
                attn_metadata = self.attn_backend.make_metadata(
                    num_prefills=batch_size,
                    num_prefill_tokens=batch_size * seq_len,
                    num_decode_tokens=0,
                    slot_mapping=slot_mapping,
                    multi_modal_placeholder_index_maps=None,
                    enable_kv_scales_calculation=False,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    effective_query_lens=effective_query_lens,
                )
        else:
            assert seq_len == 1
            token_ids = torch.zeros((batch_size, seq_len),
                                    dtype=torch.int32,
                                    device=self.device)
            position_ids = torch.zeros((batch_size, seq_len),
                                       dtype=torch.int32,
                                       device=self.device)
            slot_mapping = torch.zeros((batch_size, seq_len),
                                       dtype=torch.int64,
                                       device=self.device)
            block_tables = torch.zeros(
                (batch_size, self.max_num_blocks_per_seq),
                dtype=torch.int32,
                device=self.device)
            context_lens = torch.ones((batch_size, ),
                                      dtype=torch.int32,
                                      device=self.device)
            input_lens = torch.ones((batch_size, ),
                                    dtype=torch.int32,
                                    device=self.device)
            attn_metadata = self.attn_backend.make_metadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decode_tokens=batch_size * seq_len,
                slot_mapping=slot_mapping,
                multi_modal_placeholder_index_maps=None,
                enable_kv_scales_calculation=False,
                block_tables=block_tables,
                context_lens=context_lens,
            )
        t = torch.ones((batch_size, ), dtype=torch.float32, device=self.device)
        p = torch.ones((batch_size, ), dtype=torch.float32, device=self.device)
        num_samples = _MAX_NUM_SAMPLES if exec_mode.is_prefill() else 1

        # NOTE(woosuk): There are two stages of compilation: torch.compile and
        # XLA compilation. Using `mark_dynamic` can reduce the torch.compile
        # overhead by reusing the FX graph for different shapes.
        # However, the XLA graph will still require static shapes and needs to
        # be re-compiled for every different shapes. This overhead is inevitable
        # in the first run, but can be skipped afterwards as we cache the XLA
        # graphs in the disk (VLLM_XLA_CACHE_PATH).
        if exec_mode.is_prefill():
            # Prefll
            torch._dynamo.mark_dynamic(token_ids, 1)
            torch._dynamo.mark_dynamic(position_ids, 1)
            torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 1)
        else:
            # Decode
            torch._dynamo.mark_dynamic(token_ids, 0)
            torch._dynamo.mark_dynamic(position_ids, 0)
            torch._dynamo.mark_dynamic(input_lens, 0)
            torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 0)
            torch._dynamo.mark_dynamic(attn_metadata.context_lens, 0)
            torch._dynamo.mark_dynamic(attn_metadata.block_tables, 0)
            torch._dynamo.mark_dynamic(t, 0)
            torch._dynamo.mark_dynamic(p, 0)
        # Dummy run.
        with set_forward_context(attn_metadata, self.vllm_config, 0):
            self.model(token_ids, position_ids, input_lens, t, p, num_samples,
                       kv_caches)

    def warmup_model(
        self,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        # Prefill
        logger.info("Compiling the model with different input shapes...")
        start = time.time()
        for batch_size in [1]:
            seq_len = 16
            while seq_len <= self.model_config.max_model_len:
                self._dummy_run(batch_size,
                                seq_len,
                                kv_caches,
                                exec_mode=ExecutionMode.PREFILL)
                xm.wait_device_ops()
                logger.info("batch_size: %d, seq_len: %d", batch_size, seq_len)
                num_tokens = batch_size * seq_len
                if num_tokens >= self.scheduler_config.max_num_batched_tokens:
                    break
                seq_len = seq_len * 2

        end = time.time()
        logger.info("Compilation for prefill done in %.2f s.", end - start)

        # Prefix prefill
        if self.cache_config.enable_prefix_caching:
            logger.info("Compiling the model with different input shapes for "
                        "prefix prefill...")
            start = time.time()
            for batch_size in [1]:
                seq_len = 16
                while seq_len <= self.model_config.max_model_len:
                    self._dummy_run(batch_size,
                                    seq_len,
                                    kv_caches,
                                    exec_mode=ExecutionMode.PREFIX_PREFILL)
                    xm.wait_device_ops()
                    logger.info("batch_size: %d, seq_len: %d", batch_size,
                                seq_len)
                    num_tokens = batch_size * seq_len
                    if (num_tokens
                            >= self.scheduler_config.max_num_batched_tokens):
                        break
                    seq_len = seq_len * 2
            end = time.time()
            logger.info("Compilation for prefix prefill done in %.2f s.",
                        end - start)

        # Decode
        start = time.time()
        seq_len = 1
        batch_size = 8  # Must be in sync with _get_padded_batch_size()
        while True:
            self._dummy_run(batch_size,
                            seq_len,
                            kv_caches,
                            exec_mode=ExecutionMode.DECODE)
            xm.wait_device_ops()
            logger.info("batch_size: %d, seq_len: %d", batch_size, seq_len)

            if batch_size >= self.scheduler_config.max_num_seqs:
                break
            batch_size = batch_size + 16 if batch_size >= 16 else batch_size * 2

        end = time.time()
        logger.info("Compilation for decode done in %.2f s.", end - start)

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        prompt_lens: List[int] = []
        context_lens: List[int] = []
        slot_mapping: List[int] = []

        for batch_idx, seq_group_metadata in enumerate(
                seq_group_metadata_list):
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            # Could include output tokens when a request is preempted.
            prompt_tokens = seq_data.get_token_ids()
            seq_len = len(prompt_tokens)

            num_computed_blocks = len(seq_group_metadata.computed_block_nums)
            num_computed_tokens = num_computed_blocks * self.block_size
            if num_computed_tokens > 0:
                prompt_tokens = prompt_tokens[num_computed_tokens:]
                context_lens.append(seq_len)
            else:
                context_lens.append(0)

            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.extend(prompt_tokens)
            input_positions.extend(range(num_computed_tokens, seq_len))

            assert seq_group_metadata.block_tables is not None
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(num_computed_tokens, seq_len):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)
            if num_computed_tokens > 0:
                self.block_tables[batch_idx, :len(block_table)] = block_table

            # Add paddings to EACH prompt to the smallest power of 2 that is
            # greater than or equal to the prompt length.
            # We pad the seq_len to reduce the compilation overhead.
            # We execute each prompt individually (i.e., with batch_size 1)
            # because the FlashAttention kernel does not support ragged inputs.
            # TODO(woosuk): Use SplashAttention to support ragged inputs.
            padded_prompt_len = _get_padded_prefill_len(prompt_len)
            num_paddings = padded_prompt_len - prompt_len
            input_tokens += [0] * num_paddings
            input_positions += [0] * num_paddings
            slot_mapping += [_PAD_SLOT_ID] * num_paddings

        assert len(prompt_lens) > 0
        num_prefills = len(prompt_lens)
        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.int32,
                                    device="cpu")
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.int32,
                                       device="cpu")
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.int64,
                                    device="cpu")
        prompt_lens = torch.tensor(prompt_lens,
                                   dtype=torch.int32,
                                   device="cpu")
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int32,
                                    device="cpu")
        block_tables = torch.tensor(self.block_tables[:num_prefills],
                                    dtype=torch.int32,
                                    device="cpu")
        attn_metadata = self.attn_backend.make_metadata(
            num_prefills=num_prefills,
            num_prefill_tokens=0,  # NOTE: This is not used.
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            block_tables=block_tables,
            context_lens=context_lens,
            effective_query_lens=prompt_lens,
        )
        return input_tokens, input_positions, attn_metadata, prompt_lens

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        context_lens: List[int] = []

        batch_idx = 0
        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append([position])
                context_lens.append(seq_len)

                assert seq_group_metadata.block_tables is not None
                block_table = seq_group_metadata.block_tables[seq_id]
                self.block_tables[batch_idx, :len(block_table)] = block_table
                batch_idx += 1

                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append([slot])

        batch_size = _get_padded_batch_size(batch_idx)
        num_paddings = batch_size - batch_idx
        input_tokens = input_tokens + [[0]] * num_paddings
        input_positions = input_positions + [[0]] * num_paddings
        slot_mapping = slot_mapping + [[_PAD_SLOT_ID]] * num_paddings
        context_lens = context_lens + [0] * num_paddings

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.int32,
                                    device="cpu")
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.int32,
                                       device="cpu")
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.int64,
                                    device="cpu")
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int32,
                                    device="cpu")
        block_tables = torch.tensor(self.block_tables[:batch_size],
                                    dtype=torch.int32,
                                    device="cpu")
        input_lens = torch.tensor([1] * batch_size,
                                  dtype=torch.int32,
                                  device="cpu")
        attn_metadata = self.attn_backend.make_metadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=batch_size,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            block_tables=block_tables,
            context_lens=context_lens,
        )
        return input_tokens, input_positions, attn_metadata, input_lens

    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        padded_batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        assert len(seq_group_metadata_list) > 0
        t = []
        p = []
        n = []
        for seq_group_metadata in seq_group_metadata_list:
            sampling_params = seq_group_metadata.sampling_params
            t.append(sampling_params.temperature)
            if sampling_params.top_p != 1 and not _ENABLE_TOP_P:
                raise NotImplementedError(
                    "Top-p sampling is currently disabled for the TPU backend "
                    "due to performance issues.")
            p.append(sampling_params.top_p)
            if sampling_params.top_k > 0:
                raise NotImplementedError(
                    "Top-k sampling is currently disabled for the TPU backend "
                    "due to performance issues.")
            if sampling_params.n > _MAX_NUM_SAMPLES:
                raise NotImplementedError(
                    f"Best of > {_MAX_NUM_SAMPLES} is not supported by the TPU "
                    "backend.")
            n.append(sampling_params.n)
            if sampling_params.logprobs is not None:
                raise NotImplementedError(
                    "logprobs is not currently supported by the TPU backend.")
            if sampling_params.prompt_logprobs is not None:
                raise NotImplementedError(
                    "prompt_logprobs is not currently supported by the TPU "
                    "backend.")

            # Repeat the sampling params if the seq group has multiple seqs.
            num_seqs = len(seq_group_metadata.seq_data)
            t += [t[-1]] * (num_seqs - 1)
            p += [p[-1]] * (num_seqs - 1)
            n += [n[-1]] * (num_seqs - 1)

        num_paddings = padded_batch_size - len(t)
        t += [1.0] * num_paddings
        p += [1.0] * num_paddings

        t = torch.tensor(t, dtype=torch.float32, device="cpu")
        p = torch.tensor(p, dtype=torch.float32, device="cpu")
        return t, p, n

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> ModelInputForTPU:
        del finished_requests_ids  # Unused.
        assert virtual_engine == 0
        assert len(seq_group_metadata_list) > 0
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        if is_prompt:
            inputs = self._prepare_prompt(seq_group_metadata_list)
        else:
            inputs = self._prepare_decode(seq_group_metadata_list)
        input_tokens, input_positions, attn_metadata, input_lens = inputs
        padded_batch_size = input_tokens.shape[0]
        t, p, n = self._prepare_sample(seq_group_metadata_list,
                                       padded_batch_size)
        num_samples = _MAX_NUM_SAMPLES if is_prompt else 1

        seq_groups = [
            list(metadata.seq_data.keys())
            for metadata in seq_group_metadata_list
        ]
        return ModelInputForTPU(input_tokens, input_positions, attn_metadata,
                                input_lens, t, p, num_samples, n, seq_groups)

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> ModelInputForTPU:
        model_input = ModelInputForTPU.from_broadcasted_tensor_dict(
            tensor_dict, attn_backend=self.attn_backend)
        return model_input

    @torch.no_grad()
    def execute_model(
        self,
        model_input: ModelInputForTPU,
        kv_caches: Optional[List[Any]],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> List[SamplerOutput]:
        assert intermediate_tensors is None
        if not model_input.is_first_multi_step:
            if not model_input.is_last_step:
                return []

            use_async_out_proc = model_input.async_callback is not None
            sampler_outputs = []
            num_outputs = len(self.cached_step_outputs)
            for i in range(num_outputs):
                next_token_ids = self.cached_step_outputs.pop(0)
                next_token_ids = next_token_ids.cpu().tolist()
                sampler_output = _make_decode_output(next_token_ids,
                                                     model_input.seq_groups)
                sampler_outputs.append(sampler_output)

                if i < num_outputs - 1 and use_async_out_proc:
                    assert model_input.async_callback is not None
                    ctx = model_input.async_callback.keywords[  # type: ignore
                        "ctx"]
                    ctx.append_output(
                        outputs=[sampler_output],
                        seq_group_metadata_list=ctx.seq_group_metadata_list,
                        scheduler_outputs=ctx.scheduler_outputs,
                        is_async=False,
                        is_last_step=False,
                        is_first_step_output=i == 0)
                    model_input.async_callback()
            if use_async_out_proc:
                return [sampler_outputs[-1]]
            else:
                return sampler_outputs

        is_prompt = model_input.attn_metadata.num_prefills > 0
        if is_prompt:
            assert num_steps == 1
            # NOTE(woosuk): Since the FlashAttention kernel does not support
            # ragged inputs, we split the prompts into different batches and
            # process them separately. This is a temporary hack that should be
            # optimized by using SplashAttention.
            orig_slot_mapping = model_input.attn_metadata.slot_mapping
            orig_block_tables = model_input.attn_metadata.block_tables
            orig_context_lens = model_input.attn_metadata.context_lens
            orig_effective_query_lens = \
                model_input.attn_metadata.effective_query_lens
            batch_size = model_input.input_lens.shape[0]
            start_idx = 0
            next_token_ids = []
            for i in range(batch_size):
                # Get the actual prefill_len.
                prefill_len = model_input.input_lens[i:i + 1].item()
                prefill_len = _get_padded_prefill_len(prefill_len)
                end_idx = start_idx + prefill_len

                token_ids = model_input.token_ids[None, start_idx:end_idx].to(
                    self.device)
                position_ids = model_input.position_ids[None,
                                                        start_idx:end_idx].to(
                                                            self.device)
                attn_metadata = model_input.attn_metadata
                attn_metadata.num_prefills = 1
                attn_metadata.slot_mapping = orig_slot_mapping[
                    None, start_idx:end_idx].to(self.device)
                if orig_context_lens[i].item() > 0:
                    attn_metadata.context_lens = orig_context_lens[i:i + 1].to(
                        self.device)
                    attn_metadata.block_tables = orig_block_tables[
                        i].unsqueeze(0).to(self.device)
                    attn_metadata.effective_query_lens = \
                        orig_effective_query_lens[i:i + 1].to(self.device)
                else:
                    attn_metadata.context_lens = None
                    attn_metadata.block_tables = None
                    attn_metadata.effective_query_lens = None
                input_lens = model_input.input_lens[i:i + 1].to(self.device)
                t = model_input.t[i:i + 1].to(self.device)
                p = model_input.p[i:i + 1].to(self.device)
                with set_forward_context(model_input.attn_metadata,
                                         self.vllm_config,
                                         model_input.virtual_engine):
                    output_token_ids = self.model(token_ids, position_ids,
                                                  input_lens, t, p,
                                                  model_input.num_samples,
                                                  kv_caches)
                next_token_ids.append(output_token_ids[0])
                start_idx = end_idx

            if model_input.async_callback is not None:
                model_input.async_callback()
            # Retrieve the outputs to CPU.
            next_token_ids = [
                output_token_ids.cpu().tolist()
                for output_token_ids in next_token_ids
            ]

            # NOTE(woosuk): Minimal code to construct the sampler outputs.
            # The TPU backend does not reuse the sampler, since the TPU backend
            # does not support advanced sampling parameters such as logprobs.
            zero_logprob = Logprob(0.0)
            sampler_outputs = []
            for i, seq_group in enumerate(model_input.seq_groups):
                seq_ids = seq_group
                assert len(seq_ids) == 1
                seq_id = seq_ids[0]
                seq_outputs = []
                for j in range(model_input.n[i]):
                    next_token_id = next_token_ids[i][j]
                    seq_outputs.append(
                        SequenceOutput(seq_id, next_token_id,
                                       {next_token_id: zero_logprob}))
                sampler_outputs.append(
                    CompletionSequenceGroupOutput(seq_outputs, None))
            return [SamplerOutput(sampler_outputs)]
        else:
            token_ids = model_input.token_ids.to(self.device)
            position_ids = model_input.position_ids.to(self.device)
            attn_metadata = model_input.attn_metadata
            attn_metadata.slot_mapping = attn_metadata.slot_mapping.to(
                self.device)
            attn_metadata.block_tables = attn_metadata.block_tables.to(
                self.device)
            attn_metadata.context_lens = attn_metadata.context_lens.to(
                self.device)
            t = model_input.t.to(self.device)
            p = model_input.p.to(self.device)
            input_lens = model_input.input_lens.to(self.device)
            for i in range(num_steps):
                slot_mapping = attn_metadata.slot_mapping
                with set_forward_context(model_input.attn_metadata,
                                         self.vllm_config,
                                         model_input.virtual_engine):
                    output_token_ids = self.model(token_ids, position_ids,
                                                  input_lens, t, p,
                                                  model_input.num_samples,
                                                  kv_caches)
                self.cached_step_outputs.append(output_token_ids)

                if i < num_steps - 1:
                    # Prepare the inputs for the next step.
                    token_ids = output_token_ids.unsqueeze(dim=1).int()
                    position_ids = position_ids + 1
                    attn_metadata.context_lens = attn_metadata.context_lens + 1

                    block_tables = attn_metadata.block_tables
                    block_number = block_tables.gather(
                        1,
                        position_ids.long() // self.block_size)
                    block_offset = position_ids % self.block_size

                    is_padding = slot_mapping == _PAD_SLOT_ID
                    slot_mapping = block_number * self.block_size + block_offset
                    slot_mapping = slot_mapping.long()
                    slot_mapping = torch.where(is_padding, _PAD_SLOT_ID,
                                               slot_mapping)
                    attn_metadata.slot_mapping = slot_mapping

            if model_input.async_callback is not None:
                model_input.async_callback()

            if num_steps > 1:
                return []
            # Retrieve the outputs to CPU.
            next_token_ids = self.cached_step_outputs.pop(0)
            next_token_ids = next_token_ids.cpu().tolist()
            sampler_output = _make_decode_output(next_token_ids,
                                                 model_input.seq_groups)
            return [sampler_output]


class ModelWrapper(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        input_lens: torch.Tensor,
        t: torch.Tensor,
        p: torch.Tensor,
        num_samples: int,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Executes the forward pass of the model and samples the next token.

        Args:
            token_ids: The input token IDs of shape [batch_size, seq_len].
            position_ids: The input position IDs of shape [batch_size, seq_len].
            input_lens: The actual input lengths of shape [batch_size].
            t: The sampling temperature of shape [batch_size].
            p: The top-p probability of shape [batch_size].
            num_samples: Number of samples to draw from each logits vector.
            kv_caches: The key and value caches. They can be None during the
                memory profiling at initialization.
        """
        batch_size, seq_len = token_ids.shape
        # Calculate the positions to sample from.
        start_indices = torch.arange(
            batch_size, dtype=torch.int32, device=input_lens.device) * seq_len
        logits_indices = start_indices + input_lens - 1
        attn_metadata = get_forward_context().attn_metadata

        # FIXME(woosuk): This is a temporary hack to avoid using the existing
        # sampler and sampling metadata.
        sampling_metadata = SamplingMetadata(
            seq_groups=[],
            selected_token_indices=logits_indices,
            categorized_sample_indices={},
            num_prompts=attn_metadata.num_prefills,
        )

        # Skip this in memory profiling at initialization.
        if kv_caches[0][0].numel() > 0:
            # index_copy_(slot_mapping) only works when the inserted dimension
            # is 0. However, the KV cache in the Pallas backend has the shape
            # [num_kv_heads, num_blocks, block_size, head_size]. To make it
            # work, we need to flatten the first three dimensions and modify
            # the slot_mapping accordingly.
            num_kv_heads, num_blocks, block_size, _ = kv_caches[0][0].shape
            slot_mapping = attn_metadata.slot_mapping
            slot_mapping = slot_mapping.flatten()
            head_indices = torch.arange(0,
                                        num_kv_heads,
                                        device=slot_mapping.device,
                                        dtype=slot_mapping.dtype)
            head_indices *= block_size * num_blocks
            slot_mapping = slot_mapping.repeat_interleave(num_kv_heads).view(
                -1, num_kv_heads)
            slot_mapping = slot_mapping + head_indices.view(1, -1)
            slot_mapping = slot_mapping.flatten()
            attn_metadata.slot_mapping = slot_mapping

        hidden_states = self.model(token_ids, position_ids)
        hidden_states = hidden_states.flatten(0, 1)
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Argmax sampling.
        argmax_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        argmax_token_ids = argmax_token_ids.repeat(1, num_samples)

        # Zero temperature means greedy decoding. Avoid division by zero.
        nonzero_t = torch.where(t != 0, t, 1.0)
        logits = logits / nonzero_t.unsqueeze(dim=1)
        if _ENABLE_TOP_P:
            logits = _apply_top_p(logits, p.unsqueeze(dim=1))

        # Random sampling.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        sampled_token_ids = torch.multinomial(probs,
                                              num_samples,
                                              replacement=True)
        if num_samples == 1:
            argmax_token_ids = argmax_token_ids.squeeze(dim=-1)
            sampled_token_ids = sampled_token_ids.squeeze(dim=-1)
        next_token_ids = torch.where(t != 0, sampled_token_ids,
                                     argmax_token_ids)
        return next_token_ids


def _get_padded_prefill_len(x: int) -> int:
    # NOTE(woosuk): The pallas FlashAttention kernel requires the sequence
    # length to be a multiple of 16. We pad the prompt length to the nearest
    # multiple of 16. This is also good for performance.
    if x <= 16:
        return 16
    return 1 << (x - 1).bit_length()


def _get_padded_batch_size(batch_size: int) -> int:
    # The GMM Pallas kernel requires num_tokens * topk to be a multiple of 16.
    # To meet this requirement in the simplest way, we set the minimal batch
    # size to 8.
    if batch_size <= 8:
        return 8
    else:
        return ((batch_size + 15) // 16) * 16


def _apply_top_p(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    logits_sorted = torch.sort(logits, dim=-1, descending=True).values
    sorted_cum_probs = torch.cumsum(logits_sorted.softmax(dim=-1), dim=-1)
    cutoff_index = torch.sum(sorted_cum_probs < p, dim=-1, keepdim=True)
    cutoff_logit = torch.gather(logits_sorted, -1, cutoff_index)
    logits = logits.masked_fill_(logits < cutoff_logit, -float("inf"))
    return logits


def _make_decode_output(
    next_token_ids: List[int],
    seq_groups: List[List[int]],
) -> SamplerOutput:
    zero_logprob = Logprob(0.0)
    sampler_outputs = []
    batch_idx = 0
    for seq_group in seq_groups:
        seq_ids = seq_group
        seq_outputs = []
        for seq_id in seq_ids:
            next_token_id = next_token_ids[batch_idx]
            seq_outputs.append(
                SequenceOutput(seq_id, next_token_id,
                               {next_token_id: zero_logprob}))
            batch_idx += 1
        sampler_outputs.append(CompletionSequenceGroupOutput(
            seq_outputs, None))
    return SamplerOutput(sampler_outputs)
