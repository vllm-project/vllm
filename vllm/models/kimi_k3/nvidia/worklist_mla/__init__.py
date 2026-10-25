# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 MLA decode on a work-list FP8 kernel (TokenSpeed's, with a device-planned
split-KV work list), selected with ``additional_config={"mla_decode_backend":
"worklist"}``. See ``layer.py``."""
