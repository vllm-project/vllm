# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Union

import torch
from torch import nn


class GlobalAvailMixin:
    """Mixin class to make instances globally available."""

    _instances: Dict[str, Dict[Union[str, nn.Module], 'GlobalAvailMixin']] = {
        'default': {}
    }

    def global_available(self,
                         key: Union[str, nn.Module] = 'default',
                         group: str = 'default') -> None:
        """Make the instance globally available.

        Args:
            key (Union[str, nn.Module], optional): Key to save the instance.
                Defaults to 'default'.
            group (str, optional): Group to save the instance.
                Defaults to 'default'.
        """
        self._save_instance(self, key, group)

    @classmethod
    def _save_instance(cls,
                       instance: 'GlobalAvailMixin',
                       key: Union[str, nn.Module] = 'default',
                       group: str = 'default') -> None:
        """Save the instance.

        Args:
            instance (GlobalAvailMixin): Instance to save.
            key (Union[str, nn.Module], optional): Key to save the instance.
                Defaults to 'default'.
            group (str, optional): Group to save the instance.
                Defaults to 'default'.
        """
        if group not in cls._instances:
            assert isinstance(group, str)
            cls._instances[group] = {}

        cls._instances[group][key] = instance

    @classmethod
    def find(cls,
             key: Union[str, nn.Module] = 'default',
             group: str = 'default') -> Union[None, 'GlobalAvailMixin']:
        """Find an instance by its key and group.

        Args:
            key (Union[str, nn.Module], optional): Key of the instance.
                Defaults to 'default'.
            group (str, optional): Group of the instance.
                Defaults to 'default'.

        Returns:
            Union[None, GlobalAvailMixin]: The found instance, or None if
                it does not exist.
        """
        return cls._instances.get(group, {}).get(key)

    @classmethod
    def find_group(
            cls,
            group: str) -> Dict[Union[str, nn.Module], 'GlobalAvailMixin']:
        """Find all instances in a group.

        Args:
            group (str): Group of the instances.

        Returns:
            Dict[Union[str, nn.Module], GlobalAvailMixin]: All instances in
                the group.
        """
        return cls._instances.get(group, {})

    @classmethod
    def instances(
            cls) -> Dict[str, Dict[Union[str, nn.Module], 'GlobalAvailMixin']]:
        """Get all instances."""
        return cls._instances


class KVCacheObserver(GlobalAvailMixin):
    """A class to observe and record the max, min, and absolute max value of
    given tensor."""

    def __init__(self, num_head: int, head_dim: int) -> None:
        """Constructor for KVCacheObserver.

        Args:
            num_head : Number of heads
            head_dim : Dimension of each head
        """
        self.num_head = num_head
        self.head_dim = head_dim
        self.max_val = torch.full((num_head, head_dim),
                                  -torch.inf,
                                  dtype=torch.float16)
        self.min_val = torch.full((num_head, head_dim),
                                  torch.inf,
                                  dtype=torch.float16)
        self.absmax_val = torch.full((num_head, head_dim),
                                     0,
                                     dtype=torch.float16)

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        """Function to observe the input tensor and update the max, min, and
        absolute max values.

        Args:
            x : Input tensor
        """
        assert len(x.shape) == 4

        if x.size(1) == self.num_head and x.size(3) == self.head_dim:
            # layout: (bs, heads, seqlen, dims)
            x = x.transpose(1, 2)
        elif x.size(2) != self.num_head or x.size(3) != self.head_dim:
            raise RuntimeError('Unexpected dimensions for x, '
                               'expected (bs, num_head, seqlen, head_dim) '
                               'or (bs, seqlen, num_head, head_dim)')

        # print("x.shape ", x.shape)
        # print("x.flatten(0, 1).shape ", x.flatten(0, 1).shape)
        # print("x.flatten(0, 1).max(0)[0].shape ", x.flatten(0, 1).max(0)[0].shape)
        cur_max = x.flatten(0, 1).max(0)[0].cpu()
        cur_min = x.flatten(0, 1).min(0)[0].cpu()
        cur_absmax = x.flatten(0, 1).abs().max(0)[0].cpu()

        self.max_val = torch.maximum(self.max_val, cur_max)
        self.min_val = torch.minimum(self.min_val, cur_min)
        self.absmax_val = torch.maximum(self.absmax_val, cur_absmax)


class ActivationObserver(GlobalAvailMixin):
    """A class to observe and record the max, min, mean, absolute max, and
    absolute mean value of a given tensor.

    Also keeps track of the number of batches observed.
    """

    def __init__(self, dim: int) -> None:
        """Constructor for ActivationObserver.

        Args:
            dim : Dimension of the tensor
        """
        self.dim = dim
        self.max_val = torch.full((dim, ), -torch.inf, dtype=torch.float16)
        self.min_val = torch.full((dim, ), torch.inf, dtype=torch.float16)
        self.absmax_val = torch.full((dim, ), 0, dtype=torch.float16)
        self.absmean_val = torch.full((dim, ), 0, dtype=torch.float16)
        self.mean_val = torch.full((dim, ), 0, dtype=torch.float16)
        self.num_batches_tracked = 0

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        """Function to observe the input tensor and update the max, min, mean,
        absolute max, absolute mean values and number of batches tracked.

        Args:
            x : Input tensor
        """
        assert len(x.shape) == 3
        assert x.size(2) == self.dim
        cur_val = x.flatten(0, 1)
        cur_max = cur_val.max(0)[0].cpu()
        cur_min = cur_val.min(0)[0].cpu()
        cur_mean = cur_val.mean(0).cpu()

        cur_abs = cur_val.abs()
        cur_absmax = cur_abs.max(0)[0].cpu()
        cur_absmean = cur_abs.mean(0).cpu()

        self.max_val = torch.maximum(self.max_val, cur_max)
        self.min_val = torch.minimum(self.min_val, cur_min)
        self.absmax_val = torch.maximum(self.absmax_val, cur_absmax)

        # Update mean and absmean value with accumulated sum divided
        # by total number of batches
        self.mean_val = (
            (self.mean_val * self.num_batches_tracked + cur_mean) /
            (self.num_batches_tracked + 1))
        self.absmean_val = (
            (self.absmean_val * self.num_batches_tracked + cur_absmean) /
            (self.num_batches_tracked + 1))

        # Increment the count of batches tracked
        self.num_batches_tracked += 1
