import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm

from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VisionLanguageConfig)
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import (CompletionSequenceGroupOutput, Logprob,
                           SamplerOutput, SequenceGroupMetadata,
                           SequenceOutput)
from vllm.utils import make_tensor_with_pad

logger = init_logger(__name__)

_PAD_SLOT_ID = 0  # FIXME(woosuk)
# FIXME(woosuk): Temporarily disabled top-p sampling since it's too slow.
_ENABLE_TOP_P = False
# FIXME(woosuk): A temporary hack to support `n > 1`.
# This can significantly affect the performance if too large.
_MAX_NUM_SAMPLES = 128


class TPUModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        vision_language_config: Optional[VisionLanguageConfig] = None,
        is_driver_worker: bool = False,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.vision_language_config = vision_language_config
        self.is_driver_worker = is_driver_worker

        self.block_size = self.cache_config.block_size
        self.max_num_blocks_per_seq = (self.model_config.max_model_len //
                                       self.block_size)
        self.block_tables = np.zeros(
            (self.scheduler_config.max_num_seqs, self.max_num_blocks_per_seq),
            dtype=np.int32)
        self.attn_backend = get_attn_backend(
            self.model_config.get_num_attention_heads(self.parallel_config),
            self.model_config.get_head_size(),
            self.model_config.get_num_kv_heads(self.parallel_config),
            self.model_config.get_sliding_window(),
            self.model_config.dtype,
            self.cache_config.cache_dtype,
            self.block_size,
            False,
        )

    def load_model(self) -> None:
        self.device = self.device_config.device

        model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=self.device_config,
            parallel_config=self.parallel_config,
            cache_config=self.cache_config,
            scheduler_config=self.scheduler_config,
            vision_language_config=self.vision_language_config,
            lora_config=None,
        )
        xm.wait_device_ops()

        model = ModelWrapper(model)
        self.model = torch.compile(model, backend="openxla", fullgraph=True)

    def _dummy_run(
        self,
        batch_size: int,
        seq_len: int,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        is_prompt: bool,
    ) -> None:
        if is_prompt:
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
            attn_metadata = self.attn_backend.make_metadata(
                num_prefills=batch_size,
                num_prefill_tokens=batch_size * seq_len,
                num_decode_tokens=0,
                slot_mapping=slot_mapping,
                block_tables=None,
                context_lens=None,
            )
            input_lens = torch.ones((batch_size, ),
                                    dtype=torch.int32,
                                    device=self.device)
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
                block_tables=block_tables,
                context_lens=context_lens,
            )
        t = torch.ones((batch_size, ), dtype=torch.float32, device=self.device)
        p = torch.ones((batch_size, ), dtype=torch.float32, device=self.device)

        # Dummy run.
        num_samples = _MAX_NUM_SAMPLES if is_prompt else 1
        self.model(token_ids, position_ids, kv_caches, attn_metadata,
                   input_lens, t, p, num_samples)

    def warmup_model(
        self,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        # Prefill
        logger.info("Compiling the model with different input shapes...")
        start = time.time()
        for batch_size in [1]:
            seq_len = 16
            while True:
                self._dummy_run(batch_size, seq_len, kv_caches, is_prompt=True)
                xm.wait_device_ops()
                logger.info("batch_size: %d, seq_len: %d", batch_size, seq_len)

                if seq_len >= self.model_config.max_model_len:
                    break
                num_tokens = batch_size * seq_len
                if num_tokens >= self.scheduler_config.max_num_batched_tokens:
                    break
                seq_len = seq_len * 2

        end = time.time()
        logger.info("Compilation for prefill done in %.2f s.", end - start)

        # Decode
        start = time.time()
        seq_len = 1
        batch_size = 1
        while True:
            self._dummy_run(batch_size, seq_len, kv_caches, is_prompt=False)
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
    ):
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        prompt_lens: List[int] = []
        slot_mapping: List[List[int]] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            # Could include output tokens when a request is preempted.
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.append(prompt_tokens)
            input_positions.append(list(range(prompt_len)))

            assert seq_group_metadata.block_tables is not None
            block_table = seq_group_metadata.block_tables[seq_id]
            slot_mapping.append([])
            for i in range(prompt_len):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping[-1].append(slot)

        assert len(prompt_lens) > 0
        num_prefills = len(prompt_lens)
        num_prefill_tokens = sum(prompt_lens)

        # Add paddings to make the shape [batch_size, max_prompt_len] where
        # max_prompt_len is smallest power of 2 that is greater than or equal
        # to the maximum prompt length.
        # We need the 2D input shape because the Pallas FlashAttention kernel
        # does not support packed 1D inputs.
        # We pad the seq_len to powers of 2 to reduce the compilation overhead.
        max_prompt_len = _get_padded_prefill_len(max(prompt_lens))
        input_tokens = make_tensor_with_pad(input_tokens,
                                            max_prompt_len,
                                            pad=0,
                                            dtype=torch.int32,
                                            device=self.device)
        input_positions = make_tensor_with_pad(input_positions,
                                               max_prompt_len,
                                               pad=0,
                                               dtype=torch.int32,
                                               device=self.device)
        slot_mapping = make_tensor_with_pad(slot_mapping,
                                            max_prompt_len,
                                            pad=_PAD_SLOT_ID,
                                            dtype=torch.int64,
                                            device=self.device)
        prompt_lens = torch.tensor(prompt_lens,
                                   dtype=torch.int32,
                                   device=self.device)
        attn_metadata = self.attn_backend.make_metadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,  # NOTE: This is not used.
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            block_tables=None,
            context_lens=None,
        )
        return input_tokens, input_positions, attn_metadata, prompt_lens

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ):
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
                                    device=self.device)
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.int32,
                                       device=self.device)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.int64,
                                    device=self.device)
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int32,
                                    device=self.device)
        block_tables = torch.tensor(self.block_tables[:batch_size],
                                    dtype=torch.int32,
                                    device=self.device)
        input_lens = torch.tensor([1] * batch_size,
                                  dtype=torch.int32,
                                  device=self.device)
        attn_metadata = self.attn_backend.make_metadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=batch_size,
            slot_mapping=slot_mapping,
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
        best_of = []
        for seq_group_metadata in seq_group_metadata_list:
            sampling_params = seq_group_metadata.sampling_params
            # NOTE(woosuk): Here we mimic argmax sampling by applying a very
            # low temperature. This is not accurate.
            t.append(sampling_params.temperature
                     if sampling_params.temperature >= 1e-5 else 1e-5)
            if sampling_params.top_p != 1 and not _ENABLE_TOP_P:
                raise NotImplementedError(
                    "Top-p sampling is currently disabled for the TPU backend "
                    "due to performance issues.")
            p.append(sampling_params.top_p)
            if sampling_params.top_k != -1:
                raise NotImplementedError(
                    "Top-k sampling is currently disabled for the TPU backend "
                    "due to performance issues.")
            if sampling_params.best_of > _MAX_NUM_SAMPLES:
                raise NotImplementedError(
                    f"Best of > {_MAX_NUM_SAMPLES} is not supported by the TPU "
                    "backend.")
            best_of.append(sampling_params.best_of)
            if sampling_params.use_beam_search:
                raise NotImplementedError(
                    "Beam search is not supported by the TPU backend.")
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
            best_of += [best_of[-1]] * (num_seqs - 1)

        num_paddings = padded_batch_size - len(t)
        t += [1.0] * num_paddings
        p += [1.0] * num_paddings

        t = torch.tensor(t, dtype=torch.float32, device=self.device)
        p = torch.tensor(p, dtype=torch.float32, device=self.device)
        return t, p, best_of

    def _execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[CompletionSequenceGroupOutput]:
        # Prepare inputs.
        assert len(seq_group_metadata_list) > 0
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        if is_prompt:
            inputs = self._prepare_prompt(seq_group_metadata_list)
        else:
            inputs = self._prepare_decode(seq_group_metadata_list)
        padded_batch_size = inputs[0].shape[0]
        t, p, best_of = self._prepare_sample(seq_group_metadata_list,
                                             padded_batch_size)
        num_samples = _MAX_NUM_SAMPLES if is_prompt else 1

        # Execute the model.
        next_token_ids = self.model(inputs[0], inputs[1], kv_caches,
                                    *inputs[2:], t, p, num_samples)
        # Retrieve the outputs to CPU.
        next_token_ids = next_token_ids.cpu().tolist()

        # NOTE(woosuk): Minimal code to construct the sampler outputs.
        # The TPU backend does not reuse the sampler, since the TPU backend
        # does not support the advanced sampling parameters such as logprobs.
        zero_logprob = Logprob(0.0)
        batch_idx = 0
        sampler_outputs = []
        for seq_group_metadata in seq_group_metadata_list:
            seq_outputs = []
            seq_ids = list(seq_group_metadata.seq_data.keys())
            if is_prompt:
                assert len(seq_ids) == 1
                seq_id = seq_ids[0]
                for i in range(best_of[batch_idx]):
                    next_token_id = next_token_ids[batch_idx][i]
                    seq_outputs.append(
                        SequenceOutput(seq_id, next_token_id,
                                       {next_token_id: zero_logprob}))
                batch_idx += 1
            else:
                for seq_id in seq_ids:
                    next_token_id = next_token_ids[batch_idx][0]
                    seq_outputs.append(
                        SequenceOutput(seq_id, next_token_id,
                                       {next_token_id: zero_logprob}))
                    batch_idx += 1
            sampler_outputs.append(
                CompletionSequenceGroupOutput(seq_outputs, None))
        return sampler_outputs

    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> SamplerOutput:
        assert seq_group_metadata_list is not None
        assert len(seq_group_metadata_list) > 0
        if seq_group_metadata_list[0].is_prompt:
            # NOTE(woosuk): To reduce the compilation time, we only compile the
            # prefill inputs with batch size 1. Because the scheduler is not
            # aware of this limitation, we need to handle batch size > 1
            # internally by calling the model multiple times and concatenating
            # the outputs.
            # FIXME(woosuk): This is a temporary hack to not change the existing
            # scheduler. We need to fix this in the future.
            sampler_outputs = []
            for seq_group_metadata in seq_group_metadata_list:
                sampler_outputs += self._execute_model([seq_group_metadata],
                                                       kv_caches)
        else:
            sampler_outputs = self._execute_model(seq_group_metadata_list,
                                                  kv_caches)
        return SamplerOutput(sampler_outputs)


class ModelWrapper(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
        attn_metadata: AttentionMetadata,
        input_lens: torch.Tensor,
        t: torch.Tensor,
        p: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Executes the forward pass of the model and samples the next token.

        Args:
            token_ids: The input token IDs of shape [batch_size, seq_len].
            position_ids: The input position IDs of shape [batch_size, seq_len].
            kv_caches: The key and value caches. They can be None during the
                memory profiling at initialization.
            attn_metadata: The Pallas attention metadata.
            input_lens: The actual input lengths of shape [batch_size].
            t: The sampling temperature of shape [batch_size].
            p: The top-p probability of shape [batch_size].
        """
        batch_size, seq_len = token_ids.shape
        # Calculate the positions to sample from.
        base_indicies = torch.arange(
            batch_size, dtype=torch.int32, device=input_lens.device) * seq_len
        logits_indices = base_indicies + input_lens - 1

        # FIXME(woosuk): This is a temporary hack to avoid using the existing
        # sampler and sampling metadata.
        sampling_metadata = SamplingMetadata(
            seq_groups=[],
            selected_token_indices=logits_indices,
            categorized_sample_indices={},
            num_prompts=attn_metadata.num_prefills,
        )

        # Skip this in memory profiling at initialization.
        if kv_caches[0][0] is not None:
            # index_copy_(slot_mapping) only works when the inserted dimension
            # is 0. However, the KV cache in the Pallas backend has the shape
            # [num_kv_heads, num_blocks, block_size, head_size]. To make it
            # work, we need to flatten the first three dimensions and modify
            # the slot_mapping accordingly.
            num_kv_heads, num_blocks, block_size, _ = kv_caches[0][0].shape
            slot_mapping = attn_metadata.slot_mapping
            slot_mapping = slot_mapping.flatten()
            head_indicies = torch.arange(0,
                                         num_kv_heads,
                                         device=slot_mapping.device,
                                         dtype=slot_mapping.dtype)
            head_indicies *= block_size * num_blocks
            slot_mapping = slot_mapping.repeat_interleave(num_kv_heads).view(
                -1, num_kv_heads)
            slot_mapping = slot_mapping + head_indicies.view(1, -1)
            slot_mapping = slot_mapping.flatten()
            attn_metadata.slot_mapping = slot_mapping

        hidden_states = self.model(
            token_ids,
            position_ids,
            kv_caches,
            attn_metadata,
        )
        hidden_states = hidden_states.flatten(0, 1)
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        logits = logits / t.unsqueeze(dim=1)
        if _ENABLE_TOP_P:
            logits = _apply_top_p(logits, p.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        next_token_ids = torch.multinomial(probs,
                                           num_samples,
                                           replacement=True)
        return next_token_ids


def _get_padded_prefill_len(x: int) -> int:
    # NOTE(woosuk): The pallas FlashAttention kernel requires the sequence
    # length to be a multiple of 16. We pad the prompt length to the nearest
    # multiple of 16. This is also good for performance.
    if x <= 16:
        return 16
    return 1 << (x - 1).bit_length()


def _get_padded_batch_size(batch_size: int) -> int:
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    elif batch_size <= 8:
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
