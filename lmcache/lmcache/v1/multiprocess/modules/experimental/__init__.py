# SPDX-License-Identifier: Apache-2.0
"""
Define the supported experimental feature and the corresponding identifier.
"""

# Query intermediate tensor transfer.
TRANSFER_QUERY: str = "transfer_query"

# The set of all currently supported experimental features.
EXPERIMENTAL_TRANSFER: frozenset[str] = frozenset({TRANSFER_QUERY})
