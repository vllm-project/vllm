# SPDX-License-Identifier: Apache-2.0
# TEST_METADATA: {
#   "expected_status": 500,
#   "expected_contains": ["Error executing script", "division by zero"],
#   "allowed_imports": []
# }
result = 1 / 0
