import random, time
from threading import Thread
from openai import OpenAI
from datasets import load_dataset

N_CLIENTS = 32
N_CYCLES = 10
MAX_SLEEP_TIME = 10.

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

dataset = load_dataset("fka/awesome-chatgpt-prompts")
prompts = dataset["train"]["prompt"]

def submit_requests(idx, sleep_time):
    for id_cycle in range(N_CYCLES):

        time.sleep(sleep_time)
        print(f"thread {idx} starting in cycle {id_cycle}.")

        completion = client.completions.create(
            model=model,
            prompt=random.choice(prompts),
            echo=False,
            n=1,
            stream=False,
            logprobs=0,
            max_tokens=256,
        )

        print(f"thread {idx} done.")

def simulate():
    threads = [
        Thread(
            target=submit_requests, 
            args=[idx, random.uniform(0.0, MAX_SLEEP_TIME)]
        ) for idx in range(N_CLIENTS)
    ]

    for idx, t in enumerate(threads):
        print(f"launching thread {idx}.")
        t.start()

    for idx, t in enumerate(threads):
        t.join()

if __name__ == "__main__":
    simulate()