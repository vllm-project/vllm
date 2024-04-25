import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from vllm.config import (DeviceConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VisionLanguageConfig)
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.utils import pad_to_max_length

# DELETE
from jax_smi import initialise_tracking
initialise_tracking()

logger = init_logger(__name__)

_PAD_SLOT_ID = -1
_MAX_NUM_SEQS = 256


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
        self.compiled_fn = jax.jit(self._execute_step, donate_argnums=(7,))
        # FIXME(woosuk)
        self.block_tables = np.zeros((_MAX_NUM_SEQS, 512), dtype=np.int32)

    def load_model(self) -> None: 
        from huggingface_hub import snapshot_download

        from vllm.model_executor.models.jax.gemma import Transformer

        assert self.model_config.hf_config.model_type == "gemma"
        self.model = Transformer(self.model_config.hf_config)

        model_name = "google/gemma-7b-flax"
        model_dir = snapshot_download(model_name)
        params = load_and_format_params(model_dir + "/7b/")["transformer"]
        self.params = {"params": params}

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
                                            dtype=jnp.int32)
        input_positions = _make_array_with_pad(input_positions,
                                               max_prompt_len,
                                               pad=0,
                                               dtype=jnp.int32)
        slot_mapping = _make_array_with_pad(slot_mapping,
                                           max_prompt_len,
                                           pad=_PAD_SLOT_ID,
                                           dtype=jnp.int32)
        prompt_lens = jnp.asarray(prompt_lens, dtype=jnp.int32)
        return input_tokens, input_positions, slot_mapping, None, None, prompt_lens

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

        input_tokens = jnp.asarray(input_tokens, dtype=jnp.int32)
        input_positions = jnp.asarray(input_positions, dtype=jnp.int32)
        slot_mapping = jnp.asarray(slot_mapping, dtype=jnp.int32)
        context_lens = jnp.asarray(context_lens, dtype=jnp.int32)

        block_tables = jnp.asarray(self.block_tables[:batch_size], dtype=jnp.int32)
        input_lens = jnp.asarray([1] * batch_size, dtype=jnp.int32)
        return (input_tokens, input_positions, slot_mapping, block_tables,
                context_lens, input_lens)

    def prepare_input_arrays(
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

    def _execute_step(
        self,
        params: Dict[str, Any],
        token_ids: jax.Array,
        position_ids: jax.Array,
        slot_mapping: jax.Array,
        block_tables: Optional[jax.Array],
        context_lens: Optional[jax.Array],
        input_lens: jax.Array,
        kv_caches: List[jax.Array],
    ) -> tuple[jax.Array, List[jax.Array]]:
        batch_size, seq_len = token_ids.shape
        base_indicies = jnp.arange(batch_size, dtype=jnp.int32) * seq_len
        logits_indices = base_indicies + input_lens - 1

        logits, new_kv_caches = self.model.apply(
            params,
            token_ids,
            position_ids,
            slot_mapping,
            block_tables,
            context_lens,
            kv_caches,
            logits_indices,
        )
        # TODO(woosuk): Support sampling with temperature and top_p.
        next_token_ids = jnp.argmax(logits, axis=-1)
        return next_token_ids, new_kv_caches

    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[jax.Array],
    ) -> Tuple[Optional[SamplerOutput], List[jax.Array]]:
        from vllm.sequence import SequenceOutput, SequenceGroupOutput, Logprob

        start = time.time()
        inputs = self.prepare_input_arrays(seq_group_metadata_list)
        end = time.time()
        print(f"prepare_input_arrays: {(end - start) * 1000:.2f} ms")

        start = time.time()
        next_token_ids, new_kv_caches = self.compiled_fn(self.params, *inputs, kv_caches)
        next_token_ids.block_until_ready()
        end = time.time()
        print(f"compiled_fn: {(end - start) * 1000:.2f} ms")

        start = time.time()
        next_token_ids = jax.device_put(next_token_ids, jax.devices("cpu")[0])
        end = time.time()
        print(f"jax.device_put: {(end - start) * 1000:.2f} ms")

        next_token_ids = next_token_ids.tolist()
        i = 0

        sampler_outputs = []
        for seq_group_metadata in seq_group_metadata_list:
            seq_outputs = []

            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                next_token_id = next_token_ids[i]
                seq_outputs.append(SequenceOutput(seq_id, next_token_id, {next_token_id: Logprob(0.0)}))
                i += 1

            sampler_outputs.append(SequenceGroupOutput(seq_outputs, None))
        return SamplerOutput(sampler_outputs), new_kv_caches


def _make_array_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: jnp.dtype,
) -> jax.Array:
    padded_x = [pad_to_max_length(x_i, max_len, pad) for x_i in x]
    return jnp.asarray(padded_x, dtype)


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
    else:
        return ((batch_size + 7) // 8) * 8


import functools
from typing import Any, Mapping

import orbax.checkpoint

Params = Mapping[str, Any]


def load_and_format_params(path: str) -> Params:
  """Loads parameters and formats them for compatibility."""
  params = load_params(path)
  param_state = jax.tree_util.tree_map(jnp.array, params)
  remapped_params = param_remapper(param_state)
  nested_params = nest_params(remapped_params)
  return nested_params


@functools.cache
def load_params(path: str) -> Params:
  """Loads parameters from a checkpoint path."""
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  params = checkpointer.restore(path)
  return params


def param_remapper(orig_params: Params) -> Params:
  """Remaps params to new module layout.

  This is needed here because the model definition  does not have a separate
  `mlp` module.

  Args:
    orig_params: original dict of parameters in Gemma format.

  Returns:
    dict of params with different names.
  """
  new_params = {}
  for k, v in orig_params.items():
    if 'mlp/' in k:
      layer_name, param = k.rsplit('/', maxsplit=1)
      if layer_name not in new_params:
        new_params[layer_name] = {}
      if 'w' in v:
        new_params[layer_name][param] = v['w']
    else:
      new_params[k] = v
  return new_params


def nest_params(params: Params) -> Params:
  """Nests params as a dict of dicts rather than a flat dict."""
  nested_params = {}
  for path, param in params.items():
    *path, leaf = path.split('/')
    subdict = nested_params
    for key in path:
      subdict = subdict.setdefault(key, {})
    subdict[leaf] = param
  return nested_params
