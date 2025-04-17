# SPDX-License-Identifier: Apache-2.0

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


def main():
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    responses = client.embeddings.create(
        input=["Follow the white rabbit."],
        model=model,
        dimensions=1,
    )

    for data in responses.data:
        print(data.embedding)  # List of float of len 1


if __name__ == "__main__":
    main()
