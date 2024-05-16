import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp

from vllm.attention import get_attn_backend
from vllm.config import (DeviceConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VisionLanguageConfig)
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.utils import pad_to_max_length

logger = init_logger(__name__)

_PAD_SLOT_ID = 0  # FIXME(woosuk)
_MAX_NUM_SEQS = 256
_MAX_NUM_BLOCKS_PER_SEQ = 8192 // 16


class TPUModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        vision_language_config: Optional[VisionLanguageConfig],
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.vision_language_config = vision_language_config

        if model_config is not None and model_config.get_sliding_window():
            logger.warning("Sliding window is not supported on TPU. "
                           "The model will run without sliding window.")
        self.model = None
        self.block_size = None
        # FIXME(woosuk)
        self.block_tables = np.zeros((_MAX_NUM_SEQS, _MAX_NUM_BLOCKS_PER_SEQ),
                                     dtype=np.int32)
        self.device = None
        self.attn_backend = get_attn_backend(torch.bfloat16)

    def load_model(self) -> None:
        self.device = self.device_config.device

        from vllm.model_executor.models.tpu.gemma import GemmaForCausalLM
        model_arch = self.model_config.hf_config.architectures[0]
        if model_arch != "GemmaForCausalLM":
            raise NotImplementedError("Currently, only Gemma is supported. "
                                      f"Got {model_arch}.")

        model = GemmaForCausalLM.from_pretrained(
            self.model_config.model, config=self.model_config.hf_config)
        model = model.to(self.device)
        model = ModelWrapper(model)
        self.model = torch.compile(model, backend="openxla", fullgraph=True)

    def warmup_model(
        self,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        torch._dynamo.config.cache_size_limit = 128
        # Prefill
        logger.info("Compiling the model with different input shapes...")
        start = time.time()
        for batch_size in [1]:
            for seq_len in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
                if batch_size * seq_len > 8192:
                    continue
                token_ids = torch.zeros((batch_size, seq_len),
                                        dtype=torch.int32,
                                        device=self.device)
                position_ids = torch.zeros((batch_size, seq_len),
                                           dtype=torch.int32,
                                           device=self.device)
                slot_mapping = torch.zeros((batch_size, seq_len),
                                           dtype=torch.int64,
                                           device=self.device)
                block_tables = None
                context_lens = None
                attn_metadata = self.attn_backend.make_metadata(
                    num_prefills=0,
                    num_prefill_tokens=0,
                    num_decode_tokens=0,
                    prefill_metadata=None,
                    decode_metadata=None,
                    kv_cache_dtype=None,
                    slot_mapping=slot_mapping,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    is_prompt=True,
                )
                input_lens = torch.ones((batch_size, ),
                                        dtype=torch.int32,
                                        device=self.device)
                xm.mark_step()

                # Dummy run.
                self.model(token_ids, position_ids, kv_caches, attn_metadata,
                           input_lens)
                xm.mark_step()
                xm.wait_device_ops()
                logger.info(f"batch_size: {batch_size}, seq_len: {seq_len}")

        end = time.time()
        logger.info(f"Compilation for prefill done in {(end - start):.2f} s.")

        # Decode
        start = time.time()
        for batch_size in [1, 2, 4, 8] + [16 * i for i in range(1, 17)]:
            seq_len = 1
            token_ids = torch.zeros((batch_size, seq_len),
                                    dtype=torch.int32,
                                    device=self.device)
            position_ids = torch.zeros((batch_size, seq_len),
                                       dtype=torch.int32,
                                       device=self.device)
            slot_mapping = torch.zeros((batch_size, seq_len),
                                       dtype=torch.int64,
                                       device=self.device)
            block_tables = torch.zeros((batch_size, _MAX_NUM_BLOCKS_PER_SEQ),
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
                num_decode_tokens=0,
                prefill_metadata=None,
                decode_metadata=None,
                kv_cache_dtype=None,
                slot_mapping=slot_mapping,
                block_tables=block_tables,
                context_lens=context_lens,
                is_prompt=False,
            )
            xm.mark_step()

            # Dummy run.
            self.model(token_ids, position_ids, kv_caches, attn_metadata,
                       input_lens)
            xm.mark_step()
            xm.wait_device_ops()
            logger.info(f"batch_size: {batch_size}, seq_len: {seq_len}")

        end = time.time()
        logger.info(f"Compilation for decode done in {(end - start):.2f} s.")

        # self.server = xp.start_server(9012)
        # # Update to your own gs bucket address if you want to profile
        # profile_logdir = "gs://tpu-pytorch/tmp/woosuk/"
        # xp.trace_detached('localhost:9012', profile_logdir, duration_ms=120 * 1000)

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
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.append(prompt_tokens)
            input_positions.append(list(range(prompt_len)))

            assert seq_group_metadata.block_tables is not None
            block_table = seq_group_metadata.block_tables[seq_id]
            slot_mapping.append([])
            for i in range(prompt_len):
                block_number = block_table[i //
                                           self.block_size]  # type: ignore
                block_offset = i % self.block_size  # type: ignore
                slot = block_number * self.block_size + block_offset
                slot_mapping[-1].append(slot)

        max_prompt_len = max(prompt_lens)
        assert max_prompt_len > 0
        max_prompt_len = _get_padded_prefill_len(max_prompt_len)

        input_tokens = _make_array_with_pad(input_tokens,
                                            max_prompt_len,
                                            pad=0,
                                            dtype=torch.int32,
                                            device=self.device)
        input_positions = _make_array_with_pad(input_positions,
                                               max_prompt_len,
                                               pad=0,
                                               dtype=torch.int32,
                                               device=self.device)
        slot_mapping = _make_array_with_pad(slot_mapping,
                                            max_prompt_len,
                                            pad=_PAD_SLOT_ID,
                                            dtype=torch.int64,
                                            device=self.device)
        prompt_lens = torch.tensor(prompt_lens,
                                   dtype=torch.int32,
                                   device=self.device)
        attn_metadata = self.attn_backend.make_metadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=0,
            prefill_metadata=None,
            decode_metadata=None,
            kv_cache_dtype=None,
            slot_mapping=slot_mapping,
            block_tables=None,
            context_lens=None,
            is_prompt=True,
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
        num_seq_groups = len(seq_group_metadata_list)
        batch_size = _get_padded_batch_size(num_seq_groups)

        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
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
                self.block_tables[i, :len(block_table)] = block_table

                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append([slot])

        num_paddings = batch_size - num_seq_groups
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
            num_decode_tokens=0,
            prefill_metadata=None,
            decode_metadata=None,
            kv_cache_dtype=None,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            is_prompt=False,
        )
        return input_tokens, input_positions, attn_metadata, input_lens

    def prepare_inputs(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ):
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            return self._prepare_prompt(seq_group_metadata_list)
        else:
            return self._prepare_decode(seq_group_metadata_list)

    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[List[SamplerOutput]]:
        from vllm.sequence import SequenceOutput, SequenceGroupOutput, Logprob

        start = time.time()
        inputs = self.prepare_inputs(seq_group_metadata_list)
        end = time.time()
        # phase = "prompt" if inputs[2].is_prompt else "decode"
        # batch_size, seq_len = inputs[0].shape
        # print(f"{phase} inputs: batch_size={batch_size}, seq_len={seq_len}")
        # print(f"prepare_inputs(): {(end - start) * 1000:.2f} ms")

        start = time.time()
        next_token_ids = self.model(inputs[0], inputs[1], kv_caches, inputs[2],
                                    inputs[3])
        end = time.time()
        # print(f"model(): {(end - start) * 1000:.2f} ms")

        start = time.time()
        next_token_ids = next_token_ids.cpu()
        end = time.time()
        # print(f".cpu(): {(end - start) * 1000:.2f} ms")

        next_token_ids = next_token_ids.tolist()
        i = 0

        sampler_outputs = []
        for seq_group_metadata in seq_group_metadata_list:
            seq_outputs = []

            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                next_token_id = next_token_ids[i]
                seq_outputs.append(
                    SequenceOutput(seq_id, next_token_id,
                                   {next_token_id: Logprob(0.0)}))
                i += 1

            sampler_outputs.append(SequenceGroupOutput(seq_outputs, None))
        return [SamplerOutput(sampler_outputs)]


class ModelWrapper(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        attn_metadata: Any,
        input_lens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        base_indicies = torch.arange(
            batch_size, dtype=torch.int32, device=input_lens.device) * seq_len
        logits_indices = base_indicies + input_lens - 1

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
        logits = self.model.compute_logits(hidden_states, logits_indices)
        # TODO(woosuk): Support sampling with temperature and top_p.
        next_token_ids = torch.argmax(logits, axis=-1)
        return next_token_ids


def _make_array_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    padded_x = [pad_to_max_length(x_i, max_len, pad) for x_i in x]
    return torch.tensor(padded_x, dtype=dtype, device=device)


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
