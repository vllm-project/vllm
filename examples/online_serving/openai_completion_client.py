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

    # Completion API
    stream = False
    completion = client.completions.create(
        model=model,
        prompt="A robot may not injure a human being",
        echo=False,
        n=2,
        stream=stream,
        logprobs=3)

    print("-" * 50)
    print("Completion results:")
    if stream:
        for c in completion:
            print(c)
    else:
        print(completion)
    print("-" * 50)


if __name__ == "__main__":
    main()
