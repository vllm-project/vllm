# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define MFU analysis functionality mixin for model runners.
"""

import time

import torch.nn as nn

from vllm.config import ObservabilityConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.models.deepseek import DeepseekMoE
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
from vllm.v1.core.mfu_utils import analyze_model_mfu, analyze_model_mfu_fast
from vllm.v1.outputs import MFUInfo
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


# We special case for MoE modules
def derive_active_param_count(model):
    def get_module_name(param_name):
        return ".".join(param_name.split(".")[:-1])

    def get_module(param_name):
        for name, module in model.named_modules():
            if name == get_module_name(param_name):
                return module
        return None

    total_params = 0
    for name, param in model.named_parameters():
        module = get_module(name)
        if isinstance(module, FusedMoE):
            ept = module.moe_config.experts_per_token
            ne = module.moe_config.num_experts
            sparsity_factor = ept / ne
        if isinstance(module, Qwen3MoeSparseMoeBlock):
            ept = module.config.num_experts
            ne = module.config.num_experts_per_tok
            sparsity_factor = ept / ne
        if isinstance(module, DeepseekMoE):
            ept = module.config.n_routed_experts
            se = module.config.n_shared_experts
            ne = module.config.num_experts_per_tok
            sparsity_factor = (ept + se) / (ne + se)
        else:
            sparsity_factor = 1
        total_params += param.numel() * sparsity_factor
    return total_params


class MFUAnalyzer:
    def __init__(
        self,
        model: nn.Module,
        observability_config: ObservabilityConfig,
        mfu_info,
        auto_param_count,
    ):
        self.model = model
        self.detailed = observability_config.mfu_analysis_mode == "detailed"
        self.parameter_count = observability_config.mfu_analysis_active_parameters
        if self.parameter_count == 0:
            self.parameter_count = auto_param_count
        self.mfu_info = mfu_info

    def __call__(self, *args, **kwargs):
        with record_function_or_nullcontext("Analyze MFU"):
            if not self.detailed:
                mfu_info = analyze_model_mfu_fast(
                    self.model, self.parameter_count, args, kwargs
                )
            else:
                mfu_info = analyze_model_mfu(self.model, args, kwargs)
            pre_exec = time.perf_counter()
            output = self.model(*args, **kwargs)
            post_exec = time.perf_counter()
            mfu_info.latency_s = (post_exec - pre_exec) + 1e-9  # epsilon

            # copy over to output
            self.mfu_info.flops = mfu_info.flops
            self.mfu_info.read_bytes = mfu_info.read_bytes
            self.mfu_info.write_bytes = mfu_info.write_bytes
            self.mfu_info.latency_s = mfu_info.latency_s
        return output


class MFUAnalyzerMixin:
    def init_mfu_analysis(self, observability_config: ObservabilityConfig):
        self.mfu_info: MFUInfo | None = None
        self.mfu_analysis_interval = observability_config.mfu_analysis_interval
        self.mfu_analysis_detailed = (
            observability_config.mfu_analysis_mode == "detailed"
        )
        # internal trackers, unexposed to the runner
        self._mfu_last_analyzed = 0
        self._mfu_param_count_cache = 0

    # bumps a counter
    def should_analyze_mfu(self):
        if self.mfu_analysis_interval < 0:
            return False
        if self.mfu_analysis_interval == 0:
            return True
        self._mfu_last_analyzed += 1
        if self._mfu_last_analyzed >= self.mfu_analysis_interval:
            self._mfu_last_analyzed = 0
            return True
        return False

    def maybe_wrap_with_mfu_analyzer(
        self, model: nn.Module, observability_config: ObservabilityConfig
    ):
        if (
            not self.should_analyze_mfu()
            or observability_config.mfu_analysis_interval < 0
        ):
            return model
        # lazily compute the approximate active parameters
        if self._mfu_param_count_cache == 0:
            self._mfu_param_count_cache = derive_active_param_count(model)
        self.mfu_info = MFUInfo()
        analyzer = MFUAnalyzer(
            model, observability_config, self.mfu_info, self._mfu_param_count_cache
        )
        return analyzer
