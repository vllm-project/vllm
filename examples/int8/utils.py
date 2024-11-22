# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn


def split_decoder_layer_inputs(
    *args: Union[torch.Tensor, Any], **kwargs: Union[torch.Tensor, Any]
) -> Tuple[List[List[Any]], List[Dict[str, Any]]]:
    """This function splits batched decoder layer inputs into individual
    elements.

    Args:
        *args (Union[torch.Tensor, Any]): Positional arguments which could
            be a mix of tensors and other types.
        **kwargs (Union[torch.Tensor, Any]): Keyword arguments which could
            be a mix of tensors and other types.

    Returns:
        Tuple[List[List[Any]], List[Dict[str, Any]]]: A tuple containing two
            lists, one for positional arguments, one for keyword arguments.
            Each list contains individual elements from the batch.
    """

    if not isinstance(args[0], torch.Tensor):
        raise ValueError('The first argument must be a Tensor')

    bs = args[0].size(0)

    batch_args = []
    batch_kwargs = []
    for i in range(bs):
        new_args = []
        # Iterate over each argument. If it's a torch.Tensor and its first
        # dimension equals the batch size, then get the value corresponding
        # to the current index, else directly add the whole value.
        for val in args:
            if isinstance(val, torch.Tensor) and val.size(0) == bs:
                new_args.append(val[i:i + 1])
            else:
                new_args.append(val)

        new_kwargs = {}
        # Execute the same operation for the keyword arguments.
        for name, val in kwargs.items():
            if isinstance(val, torch.Tensor) and val.size(0) == bs:
                new_kwargs[name] = val[i:i + 1]
            else:
                new_kwargs[name] = val

        batch_args.append(new_args)
        batch_kwargs.append(new_kwargs)

    return batch_args, batch_kwargs


def concat_decoder_layer_outputs(
        batch_outputs: List[Tuple[Any]]) -> Tuple[Any]:
    """This function concatenates individual decoder layer outputs into a
    batched output.

    Args:
        batch_outputs (List[Tuple[Any]]): A list of tuples, where each tuple
            represents the output from an individual element in the batch.

    Returns:
        Tuple[Any]: A tuple representing the batched output.
    """

    num_returns = len(batch_outputs[0])

    def is_past_key_value(data: Any) -> bool:
        """Check whether data is a past key-value pair.

        Args:
            data (Any): The data to check.

        Returns:
            bool: True if data is a past key-value pair, False otherwise.
        """
        flag = isinstance(data, tuple)
        flag = flag and len(data) == 2
        flag = flag and isinstance(data[0], torch.Tensor)
        flag = flag and isinstance(data[1], torch.Tensor)
        return flag

    new_outputs = []

    # Iterate over all types of return values.
    for i in range(num_returns):
        # Check if the current element is a past key-value pair.
        flag = is_past_key_value(batch_outputs[0][i])
        if flag:
            # Concatenate the keys and values separately.
            key = torch.cat([out[i][0] for out in batch_outputs])
            value = torch.cat([out[i][1] for out in batch_outputs])
            out_i = (key, value)
        else:
            # If it's not a past key-value pair, concatenate directly.
            out_i = torch.cat([out[i] for out in batch_outputs])
        new_outputs.append(out_i)

    return tuple(new_outputs)


def collect_target_modules(
        model: nn.Module,
        #    target: Union[str, type],
        target: str,
        skip_names: List[str] = None,
        prefix: str = '') -> Dict[str, nn.Module]:
    """Collects the specific target modules from the model.

    Args:
        model : The PyTorch module from which to collect the target modules.
        target : The specific target to be collected. It can be a class of a
            module or the name of a module.
        skip_names : List of names of modules to be skipped during collection.
        prefix : A string to be added as a prefix to the module names.

    Returns:
        A dictionary mapping from module names to module instances.
    """

    # if isinstance(target, LazyAttr):
    #     target = target.build()
    if skip_names is None:
        skip_names = []
    if not isinstance(target, (type, str)):
        raise TypeError('Target must be a string (name of the module) '
                        'or a type (class of the module)')

    def _is_target(n, m):
        if isinstance(target, str):
            return target == type(m).__name__ and n not in skip_names
        return isinstance(m, target) and n not in skip_names

    name2mod = {}
    for name, mod in model.named_modules():
        m_name = f'{prefix}.{name}' if prefix else name
        if _is_target(name, mod):
            name2mod[m_name] = mod
    return name2mod


def bimap_name_mod(
    name2mod_mappings: List[Dict[str, nn.Module]]
) -> Tuple[Dict[str, nn.Module], Dict[nn.Module, str]]:
    """Generates bidirectional maps from module names to module instances and
    vice versa.

    Args:
        name2mod_mappings : List of dictionaries each mapping from module
            names to module instances.

    Returns:
        Two dictionaries providing bidirectional mappings between module
            names and module instances.
    """

    name2mod = {}
    mod2name = {}
    for mapping in name2mod_mappings:
        mod2name.update({v: k for k, v in mapping.items()})
        name2mod.update(mapping)
    return name2mod, mod2name
