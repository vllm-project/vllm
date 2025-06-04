# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from lmdeploy.serve.openai.api_client import APIClient

api_client = APIClient("http://localhost:8000")
model_name = api_client.available_models[0]

print(model_name)
