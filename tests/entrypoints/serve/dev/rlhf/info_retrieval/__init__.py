# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP response schemas and weight-version metadata.

test_response_schema checks sleep/wake fields against observed engine state.
test_weight_info checks version reads, updates, and validation. Real transfer
and request scheduling are owned by state_transitions; see the parent package
docstring for component-level coverage outside this directory.
"""
