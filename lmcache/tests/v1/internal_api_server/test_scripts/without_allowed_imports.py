# SPDX-License-Identifier: Apache-2.0
# TEST_METADATA: {
#   "expected_status": 500,
#   "expected_contains": "Error executing script",
#   "allowed_imports": []
# }
# Standard
import json

result = json.dumps({"test": "value"})
