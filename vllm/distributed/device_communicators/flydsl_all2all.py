# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import sys

import torch

from vllm.logger import init_logger
from vllm.utils.import_utils import has_flydsl_ep

from .base_device_communicator import All2AllManagerBase, Cache

logger = init_logger(__name__)


class FlydslAll2AllManager(All2AllManagerBase):
    """FlyDSL intranode EP dispatch/combine all2all manager."""

    def __init__(self, cpu_group):
        assert has_flydsl_ep(), (
            "FlyDSL EP requires mori >= 1.2 with mori.ir.flydsl and FlyDSL kernels. "
            "Set FLYDSL_REPO to the FlyDSL repo root or pip install flydsl."
        )
        import mori

        super().__init__(cpu_group)
        self.handle_cache = Cache()
        torch._C._distributed_c10d._register_process_group("mori", cpu_group)
        mori.shmem.shmem_torch_process_group_init("mori")

    def _make_flydsl_kwargs(
        self,
        rank: int,
        num_ep_ranks: int,
        input_dtype: torch.dtype,
        quant_dtype: torch.dtype,
        token_hidden_size: int,
        scale_dim: int,
        scale_type_size: int,
        max_num_tokens_per_dp_rank: int,
        num_local_experts: int,
        num_experts_per_token: int,
    ):
        return dict(
            rank=rank,
            world_size=num_ep_ranks,
            hidden_dim=token_hidden_size,
            max_num_inp_token_per_rank=max_num_tokens_per_dp_rank,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=num_experts_per_token,
            data_type=quant_dtype,
            use_external_inp_buf=False,
            scale_dim=scale_dim,
            scale_type_size=scale_type_size,
            max_token_type_size=max(input_dtype.itemsize, torch.bfloat16.itemsize),
        )

    def _make_handle(self, **kwargs):
        repo = os.environ.get("FLYDSL_REPO", "")
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        from kernels.dispatch_combine_intranode_op import (
            FlyDSLDispatchCombineConfig,
            FlyDSLDispatchCombineIntraNodeOp,
        )

        cfg = FlyDSLDispatchCombineConfig(**kwargs)
        return FlyDSLDispatchCombineIntraNodeOp(cfg)

    def get_handle(self, kwargs):
        fly_kwargs = self._make_flydsl_kwargs(**kwargs)
        logger.debug("FlyDSL all2all args %s", fly_kwargs)
        return self.handle_cache.get_or_create(fly_kwargs, self._make_handle)
