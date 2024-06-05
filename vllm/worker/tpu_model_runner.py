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
from vllm.sequence import (CompletionSequenceGroupOutput, SamplerOutput,
                           SequenceGroupMetadata, SequenceOutput, Logprob)
from vllm.utils import make_tensor_with_pad

logger = init_logger(__name__)

_PAD_SLOT_ID = 0  # FIXME(woosuk)


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
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.vision_language_config = vision_language_config

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

        xm.mark_step()
        # Dummy run.
        self.model(token_ids, position_ids, kv_caches, attn_metadata,
                   input_lens, t, p)
        xm.mark_step()

    def warmup_model(
        self,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        # Prefill
        logger.info("Compiling the model with different input shapes...")
        start = time.time()
        for batch_size in [1]:
            for seq_len in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
                if batch_size * seq_len > 8192:
                    continue
                self._dummy_run(batch_size, seq_len, kv_caches, is_prompt=True)
                xm.wait_device_ops()
                logger.info("batch_size: %d, seq_len: %d", batch_size, seq_len)

        end = time.time()
        logger.info("Compilation for prefill done in %.2f s.", end - start)

        # Decode
        start = time.time()
        for batch_size in [1, 2, 4, 8] + [16 * i for i in range(1, 17)]:
            seq_len = 1
            self._dummy_run(batch_size, seq_len, kv_caches, is_prompt=False)
            xm.wait_device_ops()
            logger.info("batch_size: %d, seq_len: %d", batch_size, seq_len)

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

        assert len(prompt_lens) > 0
        num_prefills = len(prompt_lens)
        num_prefill_tokens = sum(prompt_lens)
        max_prompt_len = max(prompt_lens)
        max_prompt_len = _get_padded_prefill_len(max_prompt_len)

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
            num_decode_tokens=batch_size,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
        )
        return input_tokens, input_positions, attn_metadata, input_lens

    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        t = []
        p = []
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.sampling_params is not None
            sampling_params = seq_group_metadata.sampling_params

            t.append(sampling_params.temperature
                     if sampling_params.temperature >= 1e-5 else 1e-5)
            p.append(sampling_params.top_p)

        t = torch.tensor(t, dtype=torch.float32, device=self.device)
        p = torch.tensor(p, dtype=torch.float32, device=self.device)
        return t, p

    def prepare_inputs(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ):
        assert seq_group_metadata_list is not None
        assert len(seq_group_metadata_list) > 0
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        if is_prompt:
            inputs = self._prepare_prompt(seq_group_metadata_list)
        else:
            inputs = self._prepare_decode(seq_group_metadata_list)
        sample_inputs = self._prepare_sample(seq_group_metadata_list)
        return inputs + sample_inputs

    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[SamplerOutput]:
        assert seq_group_metadata_list is not None

        start = time.time()
        inputs = self.prepare_inputs(seq_group_metadata_list)
        end = time.time()
        # phase = "prompt" if inputs[2].is_prompt else "decode"
        # batch_size, seq_len = inputs[0].shape
        # print(f"{phase} inputs: batch_size={batch_size}, seq_len={seq_len}")
        # print(f"prepare_inputs(): {(end - start) * 1000:.2f} ms")

        start = time.time()
        next_token_ids = self.model(inputs[0], inputs[1], kv_caches,
                                    *inputs[2:])
        end = time.time()
        # print(f"model(): {(end - start) * 1000:.2f} ms")

        start = time.time()
        next_token_ids = next_token_ids.cpu().tolist()
        end = time.time()
        # print(f".cpu(): {(end - start) * 1000:.2f} ms")

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
            sampler_outputs.append(
                CompletionSequenceGroupOutput(seq_outputs, None))
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
    ) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        base_indicies = torch.arange(
            batch_size, dtype=torch.int32, device=input_lens.device) * seq_len
        logits_indices = base_indicies + input_lens - 1

        if kv_caches[0][0] is not None:
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

        logits = logits.div_(t.unsqueeze(dim=1))
        # TODO(woosuk): Support top-p sampling.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        # FIXME(woosuk): best_of > 1 is not supported.
        next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(dim=1)
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
