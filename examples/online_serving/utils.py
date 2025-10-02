# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from openai import APIConnectionError, OpenAI
from openai.pagination import SyncPage
from openai.types.model import Model


def get_first_model(client: OpenAI) -> str:
    """
    Get the first model from the vLLM server.
    """
    try:
        models: SyncPage[Model] = client.models.list()
    except APIConnectionError as e:
        raise RuntimeError(
            "Failed to get the list of models from the vLLM server at "
            f"{client.base_url} with API key {client.api_key}. Check\n"
            "1. the server is running\n"
            "2. the server URL is correct\n"
            "3. the API key is correct"
        ) from e

    if len(models.data) == 0:
        raise RuntimeError(f"No models found on the vLLM server at {client.base_url}")

    return models.data[0].id
