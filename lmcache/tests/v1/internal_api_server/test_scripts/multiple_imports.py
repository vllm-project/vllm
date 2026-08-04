# SPDX-License-Identifier: Apache-2.0
# TEST_METADATA: {
#   "expected_status": 200,
#   "expected_result": "{\"sqrt\": 5.0}",
#   "allowed_imports": ["json", "math", "datetime"]
# }
# Standard
import json
import math

result = json.dumps({"sqrt": math.sqrt(25)})
