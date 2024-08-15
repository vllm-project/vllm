from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


def get_prompts(n=1):
    ps = ['A robot may not injure a human being']
    for i in range(1, n):
        ps.append(' '.join(["hi!"] * i))

    return ps


def main():
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    prompts = get_prompts(50)
    #print (f"{prompts}")
    print(f"# PROMPTS : {len(prompts)}")

    # Completion API
    stream = False
    completion = client.completions.create(model=model,
                                           prompt=prompts,
                                           echo=False,
                                           n=1,
                                           stream=stream)

    print("Completion results:")
    if stream:
        for c in completion:
            print(c)
    else:
        print(completion)


if __name__ == '__main__':
    main()
