# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import process_weights_after_loading

logger = init_logger(__name__)

# Notes for Online Quantization
# In terms of state of checkpoints, quantization config and their
# correspondance to online quantization:
# | Use Case      | Checkpoints          |  model_config.quantization |
# | no quant      | high precision       |  None   |
# | offline quant | quantized |  fp8, torchao etc. |
# | online quant  | high precision | torchao etc. |
#
# The process for loading non-quantized checkpoint
# 1. load non-quantized weights (load_weights)
# 2. do any additional post processing (process_weights_after_loading)
#
# The process for loading offline quantized checkpoint
# 1. load offline-quantized weights (load_weights)
# 2. do any additional post processing (process_weights_after_loading)

# The process for unquantized model reloading
# (repeated run in RL training loop)
# first run
#   UI1. load_weights: load bfloat16 weights
#   UI2. process_weights_after_loading: any additional post processing
# subsequent run
#   UC1: load_weights: load bfloat16 weights
#      (shouldn't be any issues since we didn't change any attributes
#       of the weights)
#   UC2: process_weights_after_loading: any additional post processing

# The process for weight reloading with online quantization
# (repeated run in RL training loop)
# first run
#  I1. load_weights: load bfloat16 weights
#  I2. process_weights_after_loading:
#        record weight metadata and attributes for R1 and R2
#        quantize weights to fp8
# subsequent run
#  (beginning model weight is in fp8)
#  load_weights:
#    R1. restore bfloat16 model weight metadata
#    R2. restore the model weight attributes
#    R3. reload bfloat16 weights
#    R4. quantize weights (by calling process_weights_after_loading),
#    also set `process_weights_after_loading_already_called` to
#    True to stop it from running again
#    R5. (workaround for cudagraph), we restore the weight params to original quantized
#    weights params, and use original_weight_param.copy_(updated_weight_param) so that
#    the weight update work well with cudagraph
#  process_weights_after_loading (if called):
#    this will be skipped since it's already ran in
#    load_weights


def maybe_save_metadata_and_attributes_for_weight_reloading(
    model: nn.Module, model_config: ModelConfig
):
    # following is to support on the fly quantization, currently only supported
    # for torchao
    if model_config.quantization != "torchao":
        return

    from vllm.model_executor.model_loader.weight_utils import get_quant_config

    quant_config = get_quant_config(model_config, None)

    # If checkpoint is already torchao serialized, this means it's
    # pre-quantized quantization case, we'll skip saving the metadata
    # Otherwise, this is Step I2 of initialization steps of
    # online quantization
    # This step record the weights metadata and weight attributes so we can
    # restore the bfloat16 model weights during the relad step (R1 and R2)
    # see Notes in online_quantization.py for more details
    if not (
        hasattr(quant_config, "is_checkpoint_torchao_serialized")
        and not quant_config.is_checkpoint_torchao_serialized
    ):
        return

    # This is the I2 step of online quantiztion that saves
    # metadata and attributes of weights so they can be used in R1 and
    # R2 step, note that we only save these during initialization

    # Includes two things
    # 1. save floating point metadata (shape, dtype, device) for init
    # 2. save weight attributes, e.g. `output_dim`, `weight_loader` for init

    if getattr(model, "weight_metadata_and_attr_saved", False):
        return

    # save the dtype, shape and device for model parameter, used for
    # restoring the model high precision parameters before
    # reloading the weights
    assert not hasattr(model, "original_weights_rebuild_keys")
    model.original_weights_rebuild_keys = {}
    for name, p in model.named_parameters():
        model.original_weights_rebuild_keys[name] = {
            "shape": p.shape,
            "dtype": p.dtype,
            "device": p.device,
        }

    # record the weight attributes (loader functions etc.)
    # so these can be recovered later when we reload the weights
    # structure: {"weight_name": {"weight_attr_key": attr}}
    assert not hasattr(model, "recorded_weight_attr")
    model.recorded_weight_attr = {}
    for name, param in model.named_parameters():
        model.recorded_weight_attr[name] = {}
        for key in param.__dict__:
            if hasattr(param, key):
                attr = getattr(param, key)
                if not callable(attr):
                    model.recorded_weight_attr[name][key] = attr
                elif hasattr(attr, "__self__") and param is attr.__self__:
                    # if attr is a bonded method for an instance, and
                    # attr.__self__ points to the instance (param)
                    # we'll record the underlying function object
                    model.recorded_weight_attr[name][key] = attr.__func__
                else:
                    model.recorded_weight_attr[name][key] = attr
    # mark the metadata and attributes saved so we don't run it again
    model._model_config = model_config
    model.weight_metadata_and_attr_saved = True


def _bond_method_to_cls(func, obj):
    if hasattr(func, "__self__") or not callable(func):
        # If the function is already bound to an instance, return it as is
        return func
    else:
        return types.MethodType(func, obj)


def support_quantized_model_reload_from_hp_weights(original_load_weights):
    """Decorator for `load_weights` method for AutoWeightsLoader.load_weights to support
    reloading high precision (bfloat16/float16/float32) weight for an already quantized
    model, this involves restoring the weights to a high precision weights and
    then online quantize the weights
    """
    # online quantization, right now only enabled for
    # torchao
    # R1, R2, R3, R4, R5 in the Notes

    def patched_model_load_weights(
        auto_weight_loader, weights: Iterable[tuple[str, torch.Tensor]], *, mapper=None
    ) -> set[str]:
        model = auto_weight_loader.module
        offline_quantization_or_first_run_of_online_quantization = not getattr(
            model, "weight_metadata_and_attr_saved", False
        )

        # if we don't have `model.weight_metadata_and_attr_saved` defined and
        # set to True, it means that this is either offline quantization case
        # or the first run of online quantization
        # see Notes in this file for more details
        if offline_quantization_or_first_run_of_online_quantization:
            # case 1: offline quantized checkpoint
            # case 2: Step I1 first run of weight loading with
            # online quantization
            return original_load_weights(auto_weight_loader, weights, mapper=mapper)

        model_config = model._model_config

        # TODO: Add fp8 support
        assert model_config.quantization == "torchao", (
            "online quantization is only enabled for torchao currently"
        )
        # TODO: use create_weights to restore the weights to original state

        # Step R1: First restore the quantized weights to original bfloat16
        # weights, with original metadata (shape, dtype, device)
        # and attributes, so that bfloat16 weights can be loaded properly
        # TODO: maybe set remove_duplicate to True?
        original_quantized_weight_dict = dict(
            model.named_parameters(remove_duplicate=False)
        )
        named_modules = dict(model.named_modules(remove_duplicate=False))
        model_device = None

        for name, d in model.original_weights_rebuild_keys.items():
            _shape = d["shape"]
            _dtype = d["dtype"]
            _device = d["device"]
            if model_device is not None:
                assert model_device == _device, (
                    "Expecting all weights "
                    "to be in the same device for now, got both: "
                    f"{model_device} and {_device}"
                )
            else:
                model_device = _device

            if name in original_quantized_weight_dict:
                module_name, weight_name = name.rsplit(".", 1)
                module = named_modules[module_name]
                setattr(
                    module,
                    weight_name,
                    torch.nn.Parameter(
                        torch.empty(_shape, dtype=_dtype, device=_device),
                        requires_grad=False,
                    ),
                )

        # Step R2: recover the weight attributes to the state before first loading
        # recorded_weight_attr is
        # {"weight_name": {"weight_attr_key": attr}}
        # e.g.
        # {
        #   {
        #     "layer.0.weight": {
        #       "weight_loader": weight_loader_function_object,
        #       "input_dim": 0, ...
        #     },
        #     "layer.1.weight": ...,
        #    }
        # }
        for full_weight_name, weight_attr_dict in model.recorded_weight_attr.items():
            for attr_name, attr in weight_attr_dict.items():
                module_name, weight_name = full_weight_name.rsplit(".", 1)
                module = named_modules[module_name]
                weight = getattr(module, weight_name)
                if not hasattr(weight, attr_name):
                    setattr(weight, attr_name, _bond_method_to_cls(attr, weight))

        # Step R3: reload bfloat16 / high precision weights
        updated_params = original_load_weights(
            auto_weight_loader, weights, mapper=mapper
        )

        # Step R4: online quantize the weights
        # manually process weights after loading
        model.process_weights_after_loading_already_called = False
        if model_device is not None:
            process_weights_after_loading(model, model_config, model_device)
        else:
            logger.warning_once(
                "model_device is None, skip calling process_weights_after_loading"
            )

        # Step R5 (workaround for cudagraph): restore the original quantized weights
        # and do a copy_ of the currents weights to the original weights
        updated_quantized_weights = dict(model.named_parameters(remove_duplicate=False))
        for name in model.original_weights_rebuild_keys:
            if name in original_quantized_weight_dict:
                original_quantized_weight = original_quantized_weight_dict[name]
                updated_quantized_weight = updated_quantized_weights[name]

                module_name, weight_name = name.rsplit(".", 1)
                module = named_modules[module_name]
                setattr(module, weight_name, original_quantized_weight)
                with torch.no_grad():
                    original_quantized_weight.copy_(updated_quantized_weight)

        del original_quantized_weight_dict
        del named_modules
        del updated_quantized_weight

        model.process_weights_after_loading_already_called = True
        return updated_params

    return patched_model_load_weights
