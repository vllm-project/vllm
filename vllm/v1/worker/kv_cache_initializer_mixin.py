# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from collections.abc import Iterator
from copy import deepcopy
from typing import Any, Protocol, cast

import torch

from vllm.attention import Attention, AttentionType
from vllm.config import get_layers_from_vllm_config
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils import get_dtype_size
from vllm.v1.kv_cache_interface import (AttentionSpec,
                                        EncoderOnlyAttentionSpec,
                                        KVCacheConfig, KVCacheGroupSpec,
                                        KVCacheSpec, MambaSpec)
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.gpu_input_batch import InputBatch

from .utils import (AttentionGroup, add_kv_sharing_layers_to_kv_cache_groups,
                    bind_kv_cache)


class _KVCacheInitializerSelf(Protocol):
    cache_config: Any
    max_num_reqs: int
    max_model_len: int
    max_encoder_len: int
    max_num_tokens: int
    device: Any
    pin_memory: bool
    model_config: Any
    vllm_config: Any
    input_batch: InputBatch
    is_pooling_model: bool
    shared_kv_cache_layers: dict[str, str]
    kv_sharing_fast_prefill_eligible_layers: set[str]
    runner_only_attn_layers: set[str]
    kv_cache_dtype: torch.dtype
    kv_cache_config: KVCacheConfig
    compilation_config: Any
    kv_caches: Any
    speculative_config: Any
    drafter: Any
    dcp_world_size: int
    attn_groups: list[list[AttentionGroup]]

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        ...


logger = init_logger(__name__)


# Defined as a mixin for GPUModelRunner
class KVCacheInitializerMixin:

    def _runner(self) -> _KVCacheInitializerSelf:
        return cast(_KVCacheInitializerSelf, self)

    def may_reinitialize_input_batch(self,
                                     kv_cache_config: KVCacheConfig) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
        """
        runner = self._runner()
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
        ]
        if block_sizes != [runner.cache_config.block_size]:
            assert runner.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details.")
            runner.input_batch = InputBatch(
                max_num_reqs=runner.max_num_reqs,
                max_model_len=max(runner.max_model_len,
                                  runner.max_encoder_len),
                max_num_batched_tokens=runner.max_num_tokens,
                device=runner.device,
                pin_memory=runner.pin_memory,
                vocab_size=runner.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                is_spec_decode=bool(runner.vllm_config.speculative_config),
                logitsprocs=runner.input_batch.logitsprocs,
                is_pooling_model=runner.is_pooling_model,
                num_speculative_tokens=(runner.vllm_config.speculative_config.
                                        num_speculative_tokens if
                                        runner.vllm_config.speculative_config
                                        else 0),
            )

    def _allocate_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        runner = self._runner()
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            tensor = torch.zeros(kv_cache_tensor.size,
                                 dtype=torch.int8,
                                 device=runner.device)
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in runner.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys(
        )), "Some layers are not correctly initialized"
        return kv_cache_raw_tensors

    def _kv_cache_spec_attn_group_iterator(
            self) -> Iterator[tuple[KVCacheSpec, AttentionGroup]]:
        runner = self._runner()
        if not runner.kv_cache_config.kv_cache_groups:
            return
        for kv_cache_spec_id, attn_groups in enumerate(runner.attn_groups):
            for attn_group in attn_groups:
                yield runner.kv_cache_config.kv_cache_groups[
                    kv_cache_spec_id].kv_cache_spec, attn_group

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        runner = self._runner()
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False
        for kv_cache_spec, group in self._kv_cache_spec_attn_group_iterator():
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in runner.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = (raw_tensor.numel() //
                              kv_cache_spec.page_size_bytes)
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    try:
                        kv_cache_stride_order = \
                            attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(
                            kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(
                            range(len(kv_cache_shape)))
                    kv_cache_shape = tuple(kv_cache_shape[i]
                                           for i in kv_cache_stride_order)
                    inv_order = [
                        kv_cache_stride_order.index(i)
                        for i in range(len(kv_cache_stride_order))
                    ]
                    kv_caches[layer_name] = kv_cache_raw_tensors[
                        layer_name].view(dtype).view(kv_cache_shape).permute(
                            *inv_order)
                elif isinstance(kv_cache_spec, MambaSpec):
                    has_mamba = True
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    state_tensors = []
                    storage_offset_bytes = 0
                    for (shape, dtype) in zip(kv_cache_spec.shapes,
                                              kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        num_element_per_page = (
                            kv_cache_spec.page_size_bytes // dtype_size)
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        target_stride = (num_element_per_page, *stride[1:])
                        assert storage_offset_bytes % dtype_size == 0
                        tensor = torch.as_strided(
                            raw_tensor.view(dtype),
                            size=target_shape,
                            stride=target_stride,
                            storage_offset=storage_offset_bytes // dtype_size,
                        )
                        state_tensors.append(tensor)
                        storage_offset_bytes += stride[0] * dtype_size

                    kv_caches[layer_name] = state_tensors
                else:
                    raise NotImplementedError

        if has_attn and has_mamba:
            self._update_hybrid_attention_mamba_layout(kv_caches)

        return kv_caches

    def _update_hybrid_attention_mamba_layout(
            self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Update the layout of attention layers from (2, num_blocks, ...) to
        (num_blocks, 2, ...).

        Args:
            kv_caches: The KV cache buffer of each layer.
        """

        for kv_cache_spec, group in self._kv_cache_spec_attn_group_iterator():
            for layer_name in group.layer_names:
                kv_cache = kv_caches[layer_name]
                if (isinstance(kv_cache_spec, AttentionSpec)
                        and kv_cache.shape[0] == 2):
                    assert kv_cache.shape[1] != 2, \
                        "Fail to determine whether the layout is " \
                        "(2, num_blocks, ...) or (num_blocks, 2, ...) for " \
                        f"a tensor of shape {kv_cache.shape}"
                    hidden_size = kv_cache.shape[2:].numel()
                    kv_cache.as_strided_(size=kv_cache.shape,
                                         stride=(hidden_size, 2 * hidden_size,
                                                 *kv_cache.stride()[2:]))

    def initialize_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        runner = self._runner()
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config,
                                                   kv_cache_raw_tensors)

        for layer_name, target_layer_name in (
                runner.shared_kv_cache_layers.items()):
            logger.debug("%s reuses KV cache of %s", layer_name,
                         target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        bind_kv_cache(kv_caches,
                      runner.compilation_config.static_forward_context,
                      runner.kv_caches)
        return kv_caches

    def maybe_add_kv_sharing_layers_to_kv_cache_groups(
            self, kv_cache_config: KVCacheConfig) -> None:
        """
        Add layers that re-use KV cache to KV cache group of its target layer.
        Mapping of KV cache tensors happens in `initialize_kv_cache_tensors()`
        """
        runner = self._runner()
        if not runner.shared_kv_cache_layers:
            return

        add_kv_sharing_layers_to_kv_cache_groups(
            runner.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            runner.runner_only_attn_layers,
        )

        if runner.cache_config.kv_sharing_fast_prefill:
            attn_layers = get_layers_from_vllm_config(runner.vllm_config,
                                                      Attention)
            for layer_name in reversed(attn_layers):
                if layer_name in runner.shared_kv_cache_layers:
                    runner.kv_sharing_fast_prefill_eligible_layers.add(
                        layer_name)
                else:
                    break

    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:
        """
        Add encoder-only layers to the KV cache config.
        """
        runner = self._runner()
        block_size = runner.vllm_config.cache_config.block_size
        use_mla = runner.vllm_config.model_config.use_mla
        encoder_only_attn_specs: dict[AttentionSpec,
                                      list[str]] = defaultdict(list)
        attn_layers = get_layers_from_vllm_config(runner.vllm_config,
                                                  Attention)
        for layer_name, attn_module in attn_layers.items():
            if attn_module.attn_type == AttentionType.ENCODER_ONLY:
                attn_spec: AttentionSpec = EncoderOnlyAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=runner.kv_cache_dtype,
                    use_mla=use_mla)
                encoder_only_attn_specs[attn_spec].append(layer_name)
                runner.runner_only_attn_layers.add(layer_name)
        if len(encoder_only_attn_specs) > 0:
            assert len(
                encoder_only_attn_specs
            ) == 1, "Only support one encoder-only attention spec now"
            spec, layer_names = encoder_only_attn_specs.popitem()
            runner.kv_cache_config.kv_cache_groups.append(
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec))

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.

        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        runner = self._runner()
        kv_cache_config = deepcopy(kv_cache_config)
        runner.kv_cache_config = kv_cache_config
        self.may_reinitialize_input_batch(kv_cache_config)
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        runner.initialize_attn_backend(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)

        if runner.speculative_config and runner.speculative_config.use_eagle():
            assert isinstance(runner.drafter, EagleProposer)
            runner.drafter.validate_same_kv_cache_group(kv_cache_config)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)
            if runner.device.type == 'xpu':
                get_kv_transfer_group().set_host_xfer_buffer_ops(
                    copy_kv_blocks)

        if runner.dcp_world_size > 1:
            layer_names = runner.attn_groups[0][0].layer_names
            layers = get_layers_from_vllm_config(
                runner.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
                layer_names,
            )
            for layer in layers.values():
                layer_impl = cast(Any, layer).impl
                assert layer_impl.need_to_return_lse_for_decode, (
                    "DCP requires attention impls to return"
                    " the softmax lse for decode, but the impl "
                    f"{layer_impl.__class__.__name__} "
                    "does not return the softmax lse for decode.")
