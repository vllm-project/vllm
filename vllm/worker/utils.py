# SPDX-License-Identifier: Apache-2.0
'''
Worker-related helper functions.
'''
import enum
import os
from functools import cache

from vllm.platforms import current_platform
from vllm.utils import STR_NOT_IMPL_ENC_DEC_ERR_STRS
from vllm.worker.model_runner import GPUModelRunnerBase


def assert_enc_dec_mr_supported_scenario(
        enc_dec_mr: GPUModelRunnerBase) -> None:
    '''
    Asserted that the provided encoder/decoder model runner instance reflects
    a supported scenario.
    '''

    # Reminder: Please update docs/source/features/compatibility_matrix.md
    # If the feature combo become valid

    if enc_dec_mr.cache_config.enable_prefix_caching:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE'])

    if enc_dec_mr.sliding_window is not None:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_SWA'])

    if enc_dec_mr.scheduler_config.chunked_prefill_enabled:
        raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_ERR_STRS[
            'STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL'])

    if getattr(enc_dec_mr.model_config.hf_config, 'attn_logit_softcapping',
               None) is not None:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_LOGIT_SOFTCAP']
        )

    if enc_dec_mr.lora_config is not None:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_LORA'])

    if enc_dec_mr.parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_PP'])

    if enc_dec_mr.scheduler_config.num_lookahead_slots > 0:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_SPEC_DEC'])

    if enc_dec_mr.prompt_adapter_config is not None:
        raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_ERR_STRS[
            'STR_NOT_IMPL_ENC_DEC_PROMPT_ADAPTER'])


@cache
def get_neuron_framework_to_use():
    """Return the specified framework if corresponding installations are
    available.

    If no framework is specified, use neuronx-distributed-inference by default.
    If that's unavailable, check and switch to transformers-neuronx.
    """
    if not current_platform.is_neuron():
        raise AssertionError(
            f"Neuron Framework unavailable for platform: {current_platform}")

    tnx_installed = current_platform.is_transformers_neuronx()
    nxd_installed = current_platform.is_neuronx_distributed_inference()

    specified_framework = os.environ.get("VLLM_NEURON_FRAMEWORK")
    tnx_framework = NeuronFramework.TRANSFORMERS_NEURONX.value
    nxd_framework = NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE.value
    if specified_framework == tnx_framework and tnx_installed:
        return NeuronFramework.TRANSFORMERS_NEURONX

    if ((specified_framework == nxd_framework and nxd_installed)
            or (specified_framework is None and nxd_installed)):
        return NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE

    if specified_framework is None and tnx_installed:
        return NeuronFramework.TRANSFORMERS_NEURONX

    return None


@cache
def use_neuronx_distributed():
    """
    Return True if the framework determined in get_neuron_framework_to_use() is
    NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE, False otherwise. This is used
    to select the Neuron model framework and framework-specific configuration to
    apply during model compilation.
    """
    nxd_framework = NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE
    return get_neuron_framework_to_use() == nxd_framework


@cache
def use_transformers_neuronx():
    """
    Return True if the framework determined in get_neuron_framework_to_use() is
    NeuronFramework.TRANSFORMERS_NEURONX, False otherwise. This is used to
    select the Neuron model framework and framework-specific configuration to
    apply during model compilation.
    """
    return get_neuron_framework_to_use(
    ) == NeuronFramework.TRANSFORMERS_NEURONX


class NeuronFramework(enum.Enum):
    TRANSFORMERS_NEURONX = "transformers-neuronx"
    NEURONX_DISTRIBUTED_INFERENCE = "neuronx-distributed-inference"
