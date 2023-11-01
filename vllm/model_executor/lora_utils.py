from vllm.model_executor.parallel_utils.layers import BLoraColumnParallelLinear, BLoraRowParallelLinear, ColumnParallelLinear, RowParallelLinear
from peft.tuners.lora import LoraLayer
from peft import LoraConfig
import re
import torch


WEIGHTS_NAME = "adapter_model.bin"
PREFIX = "base_model.model."
PARAMETER_PREFIX = "lora_"

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def _create_new_module(lora_config, adapter_name, target):
    lora_alpha = lora_config.lora_alpha
    r = lora_config.r
    lora_dropout = lora_config.lora_dropout
    if isinstance(target, ColumnParallelLinear):
        new_module = BLoraColumnParallelLinear(input_size=target.input_size,
                                               output_size=target.output_size_per_partition,
                                               adapter_name=adapter_name,bias=target.bias,
                                                gather_output=target.gather_output,
                                                skip_bias_add=target.skip_bias_add,
                                                quant_config=target.quant_config,
                                                lora_alpha=lora_alpha,
                                                r=r,lora_dropout=lora_dropout)
        return new_module
    if isinstance(target, RowParallelLinear):
        new_module = BLoraRowParallelLinear(input_size=target.input_size_per_partition,
                                            output_size=target.output_size,
                                            adapter_name=adapter_name,
                                            bias=target.bias,
                                            input_is_parallel=target.input_is_parallel,
                                            reduce_results=target.reduce_results,
                                            skip_bias_add=target.skip_bias_add,
                                            quant_config=target.quant_config,
                                            lora_alpha=lora_alpha,
                                            r=r,
                                            lora_dropout=lora_dropout)
        return new_module


def _replace_module(parent, child_name, new_module, child):
    setattr(parent, child_name, new_module)
    new_module.weight = child.weight
    if getattr(child, "state", None) is not None:
        new_module.state = child.state
        new_module.to(child.weight.device)
    # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(child.weight.device)

def _create_and_replace(lora_config, adapter_name, target, target_name, parent):
    if (isinstance(target, (ColumnParallelLinear, RowParallelLinear))
        and not isinstance(target, LoraLayer)):
        new_module = _create_new_module(lora_config, adapter_name, target)
        _replace_module(parent, target_name, new_module, target)
    elif isinstance(target, LoraLayer):
        target.update_layer(adapter_name,
                            lora_config.r,
                            lora_config.lora_alpha,
                            lora_config.lora_dropout,
                            lora_config.init_lora_weights)


def add_lora_adapter(model: torch.nn.Module, lora_path: str, adapter_name: str):
    lora_config = LoraConfig.from_pretrained(lora_path,
                                             revision=None,
                                             use_auth_token=None)
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        # find target module
        target_module_found = any(
                    re.match(f".*\\.{target_key}$", key) for target_key in lora_config.target_modules
                ) or any(target_key == key for target_key in lora_config.target_modules)
        if not target_module_found:
            continue
        parent, target, target_name = _get_submodules(model, key)

        # create and replace
        _create_and_replace(lora_config,
                            adapter_name,
                            target,
                            target_name,
                            parent)

    adapters_weights = torch.load(f"{lora_path}/{WEIGHTS_NAME}")

    processed_adapter_state_dict = {}
    for key, value in adapters_weights.items():
        if key.startswith(PREFIX):
            new_key = key[len(PREFIX) :]
        else:
            new_key = key
        processed_adapter_state_dict[new_key] = value

    state_dict = {}
    for k, v in processed_adapter_state_dict.items():
        if PARAMETER_PREFIX in k:
            suffix = k.split(PARAMETER_PREFIX)[1]
            if "." in suffix:
                to_replace = ".".join(suffix.split(".")[1:])
                k = k.replace(to_replace,
                              f"{adapter_name}.{to_replace}")
            else:
                k = f"{k}.{adapter_name}"
        state_dict[k] = v

    model.load_lora_weights_parallel(state_dict)
    model.cuda()
