# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV cache helper for store.
"""
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import CancelledError, Future
from typing import Optional, cast

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.attention.selector import backend_name_to_enum, get_attn_backend
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import _Backend
from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)


class model_aware_kv_ops_helper:

    def __init__(self, vllm_config: VllmConfig):
        self.is_deepseek_mla = vllm_config.model_config.is_deepseek_mla
        self.use_mla_opt = not envs.VLLM_MLA_DISABLE
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        backend = get_attn_backend(self.model_config.get_head_size(),
                                   self.model_config.dtype,
                                   self.cache_config.cache_dtype,
                                   self.cache_config.block_size,
                                   self.model_config.is_attention_free,
                                   use_mla=self.model_config.use_mla)
        self.backend_name = backend.get_name()
        attn_backend = backend_name_to_enum(self.backend_name)
        self._use_xformers = attn_backend == _Backend.XFORMERS
        self._use_torch_sdpa = attn_backend == _Backend.TORCH_SDPA
        self._use_rocm_flash = attn_backend == _Backend.ROCM_FLASH
        self._use_rocm_aiter = attn_backend == _Backend.ROCM_AITER_MLA
        logger.debug("Detected attention backend %s", self.backend_name)
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size

    def get_model_args(self, model_executable: torch.nn.Module):

        model_config = model_executable.model.config
        self.model_executable = model_executable
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads

        # Deepseek's MLA (Multi-head Latent Attention) uses two different
        # kv_cache shapes based on whether VLLM_MLA_DISABLE is set to 0.
        # When VLLM_MLA_DISABLE=0 (default), forward absorb is applied,
        # resulting in a kv_cache shape of [num_blks, blk_size, 1,
        # kv_lora_rank + qk_rope_head_dim].
        # When VLLM_MLA_DISABLE=1, standard FA is used instead, leading
        # to a kv_cache shape of [2, num_blks, blk_size,
        # num_key_value_heads / tp, qk_nope_head_dim + qk_rope_head_dim].
        # For more details, see vllm/attention/backends/mla/common.py.
        if self.is_deepseek_mla and self.use_mla_opt:
            head_size = model_config.kv_lora_rank + \
                model_config.qk_rope_head_dim
            num_heads = 1
        elif self.is_deepseek_mla and not self.use_mla_opt:
            head_size = model_config.qk_nope_head_dim + \
                model_config.qk_rope_head_dim
        else:
            head_size = getattr(model_config, "head_dim", None)
            if head_size is None:
                head_size = int(hidden_size // num_attention_heads)

        return num_heads, head_size

    def is_3D_kvcache(self):
        return self._use_xformers or self._use_torch_sdpa or \
            self._use_rocm_flash or self._use_rocm_aiter

    def get_kv_from_cache(self, kv_cache, num_heads, head_size):
        """Return key and value tensors in a flattened view.

        The layout of ``kv_cache`` depends on the attention backend.
        Currently there are 3 kinds of kv_cache tensor layout.
        1) 4-D kv cache corresponds to the tensor layout of shape
        ``[2, num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]``.
        2) 3-D kv cache corresponds to tensor layout of shape
        ``[2, num_blocks, block_size * num_heads * head_size]``.
        We do pre-process the kv cache by first splitting the cache
        and then reshape so that the first dimension indexes tokens.
        3) 5-D kv cache corresponds to the tensor layout of shape
        ``[2, num_blocks, block_size, num_heads, head_size]``.
        """
        if self.is_deepseek_mla and self.use_mla_opt:
            key_cache = kv_cache.reshape(-1, num_heads, head_size)
            value_cache = kv_cache.reshape(-1, num_heads, head_size)
        elif self.is_3D_kvcache():
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, num_heads, head_size)

            # Do permutation on key_cache and value_cache according
            # to the following shape:
            # key_cache:
            # (num_blocks, num_kv_heads, head_size // x, block_size, x)
            # value_cache:
            # (num_blocks, num_kv_heads, head_size, block_size)
            key_cache = key_cache.permute(0, 3, 1, 2,
                                          4).reshape(-1, num_heads, head_size)
            value_cache = value_cache.permute(0, 3, 1, 2).reshape(
                -1, num_heads, head_size)
        else:
            key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
            value_cache = kv_cache[1].reshape(-1, num_heads, head_size)
        return key_cache, value_cache

    def put_kv_to_cache(self, model_executable: torch.nn.Module, keys, values,
                        layer, kv_cache, slot_mapping, start_pos, end_pos):

        model_config = model_executable.model.config
        num_heads, head_size = self.get_model_args(model_executable)

        if self.is_deepseek_mla and self.use_mla_opt:
            layer.self_attn.attn = layer.self_attn.mla_attn
            k_c_normed_k_pe = keys.squeeze(1)
            k_c_normed = k_c_normed_k_pe[:, :model_config.kv_lora_rank]
            k_pe = k_c_normed_k_pe[:, model_config.kv_lora_rank:]
            ops.concat_and_cache_mla(
                k_c_normed.to(kv_cache.device),
                k_pe.to(kv_cache.device),
                kv_cache,
                slot_mapping[start_pos:end_pos],
                layer.self_attn.attn.kv_cache_dtype,
                layer.self_attn.attn._k_scale,
            )
        elif self.is_3D_kvcache():
            # 3-D paged attention cache:
            # [2, num_blocks, block_size * num_heads * head_size]
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, num_heads, head_size)
            PagedAttention.write_to_paged_cache(
                keys.to(kv_cache.device),
                values.to(kv_cache.device),
                key_cache,
                value_cache,
                slot_mapping[start_pos:end_pos],
                layer.self_attn.attn.kv_cache_dtype,
                layer.self_attn.attn._k_scale,
                layer.self_attn.attn._v_scale,
            )
        else:
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            ops.reshape_and_cache_flash(
                keys.to(key_cache.device),
                values.to(value_cache.device),
                key_cache,
                value_cache,
                slot_mapping[start_pos:end_pos],
                layer.self_attn.attn.kv_cache_dtype,
                layer.self_attn.attn._k_scale,
                layer.self_attn.attn._v_scale,
            )


def get_kv_connector_cache_layout():
    # NOTE (NickLucche) When running disaggregated PD with NIXL, HND layout is
    # used for faster transfer.
    vllm_config = get_current_vllm_config()
    kv_config = vllm_config.kv_transfer_config
    if kv_config is not None and vllm_config.model_config is None:
        logger.warning_once("Unable to detect current VLLM config. " \
        "Defaulting to NHD kv cache layout.")
    elif kv_config is not None:
        use_mla = vllm_config.model_config.use_mla
        if not use_mla and kv_config.kv_connector == "NixlConnector":
            logger.info_once("NixlConnector detected. Setting KV cache " \
            "layout to HND for better xfer performance.")
            return "HND"
    return "NHD"


class KVOutputAggregator:
    """Utility class to aggregate the output of all workers into a single 
    output corresponding to Rank 0 for scheduler."""

    def __init__(self, world_size: int):
        # Complete transfer tracker. Used to track finished requests
        # [req_id -> n_remaining_workers]
        self._recv_remaining_count = defaultdict[str, int](lambda: world_size)
        self._send_remaining_count = defaultdict[str, int](lambda: world_size)

    def aggregate(self,
                  outputs: list[ModelRunnerOutput],
                  output_rank: int = 0) -> ModelRunnerOutput:
        # aggregate finished_sending, finished_recving from all workers

        def update_finished_set(req_ids: Optional[set[str]],
                                remaining_count_dict: dict[str, int],
                                finished_set: set[str]) -> None:
            for req_id in req_ids or ():
                remaining_count_dict[req_id] -= 1
                if remaining_count_dict[req_id] == 0:
                    finished_set.add(req_id)
                    del remaining_count_dict[req_id]

        finished_sending = set[str]()
        finished_recving = set[str]()
        for output in outputs:
            update_finished_set(output.finished_sending,
                                self._send_remaining_count, finished_sending)
            update_finished_set(output.finished_recving,
                                self._recv_remaining_count, finished_recving)

        # select output of the worker specified by output_rank
        output = outputs[output_rank]

        # set the aggregated finished_sending / finished_recving
        # if output.finished_sending/recving is not empty, but the other ranks
        # still have unfinished send/recv, we want to set the aggregated
        # finished_sending/recving to None until all ranks have finished
        # send/recv
        output.finished_sending = finished_sending if finished_sending else None
        output.finished_recving = finished_recving if finished_recving else None
        return output

    def async_aggregate(self,
                        output_futures: Sequence[Future[ModelRunnerOutput]],
                        output_rank: int = 0) -> Future[ModelRunnerOutput]:
        """Takes a list of futures and returns a single future which resolves
        to the respective list of outputs."""
        result_future: Future[ModelRunnerOutput] = Future()

        outputs: list[Optional[ModelRunnerOutput]] = [None
                                                      ] * len(output_futures)

        def make_callback(idx):

            def callback(fut):
                if result_future.done():
                    return

                try:
                    outputs[idx] = fut.result()
                except CancelledError:
                    result_future.cancel()
                except Exception as e:
                    result_future.set_exception(e)

                # this check assumes io_thread_pool uses a single thread
                if all(outputs):
                    result_future.set_result(
                        self.aggregate(cast(list[ModelRunnerOutput], outputs),
                                       output_rank))

            return callback

        for i, output_future in enumerate(output_futures):
            output_future.add_done_callback(make_callback(i))

        return result_future
