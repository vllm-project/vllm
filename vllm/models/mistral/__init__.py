# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared building blocks for Mistral models.

Model packages (e.g. ``vllm.models.mistral_large4``) compose these blocks;
per-layer architecture choices are resolved once from the config into
:class:`~vllm.models.mistral.common.layer_spec.MistralLayerSpec`.
"""
