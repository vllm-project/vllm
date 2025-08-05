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
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1)
from vllm.logger import init_logger
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

logger = init_logger(__name__)


class model_aware_kv_ops_helper:

    def __init__(self, config: VllmConfig):
        self.is_deepseek_mla = config.model_config.is_deepseek_mla
        self.use_mla_opt = not envs.VLLM_MLA_DISABLE
        self.tp_size = config.parallel_config.tensor_parallel_size

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

    def get_kv_from_cache(self, kv_cache, num_heads, head_size):
        if self.is_deepseek_mla and self.use_mla_opt:
            key_cache = kv_cache.reshape(-1, num_heads, head_size)
            value_cache = kv_cache.reshape(-1, num_heads, head_size)
        else:
            key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
            value_cache = kv_cache[1].reshape(-1, num_heads, head_size)
        return key_cache, value_cache

    def put_kv_to_cache(self, model_executable: torch.nn.Module, keys, values,
                        layer, kv_cache, slot_mapping, start_pos, end_pos):

        model_config = model_executable.model.config

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
    if kv_config is not None:
        required_kvcache_layout = (
            KVConnectorBase_V1.get_required_kvcache_layout(vllm_config))
        if required_kvcache_layout is not None:
            return required_kvcache_layout
        logger.info_once("Connectors do not specify a " \
                         "kv cache layout, defaulting to NHD.")
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
        # aggregate kv_connector_output from all workers

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
            output = output.kv_connector_output
            update_finished_set(output.finished_sending,
                                self._send_remaining_count, finished_sending)
            update_finished_set(output.finished_recving,
                                self._recv_remaining_count, finished_recving)

        # select output of the worker specified by output_rank
        output = outputs[output_rank]

        output.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending or None,
            finished_recving=finished_recving or None,
        )

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
