# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Compatibility alias: vllm-ascend references DefaultMoERunner which was
# renamed to MoERunner in upstream vllm.  This shim re-exports it so
# both names resolve to the same class.
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

DefaultMoERunner = MoERunner
