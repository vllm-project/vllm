# SPDX-License-Identifier: Apache-2.0
# TEST_METADATA: {
#   "expected_status": 200,
#   "expected_min_length": 1,
#   "allowed_imports": ["os"]
# }
# Standard
import os

result = os.environ.get("PATH", "default_value")
print(result)
