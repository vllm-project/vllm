# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 KV-balanced MLA decode: TokenSpeed's FP8 MLA decode kernel with each
request's KV split planned on the device so that every cluster gets about the same
KV, selected with ``additional_config={"mla_decode_backend": "kv_balanced"}``. See
``layer.py``."""
