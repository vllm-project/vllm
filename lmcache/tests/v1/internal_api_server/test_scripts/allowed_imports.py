# SPDX-License-Identifier: Apache-2.0
# TEST_METADATA: {
#   "expected_status": 200,
#   "expected_result": "{\"key\": \"value\"}",
#   "allowed_imports": ["json", "sys"]
# }
# Standard
import json

data = {"key": "value"}
result = json.dumps(data)
