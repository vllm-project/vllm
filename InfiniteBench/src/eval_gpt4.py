from openai import OpenAI
import tiktoken
from pathlib import Path
from eval_utils import (
    create_msgs,
    load_data,
    dump_jsonl,
    iter_jsonl,
    get_answer,
)
import time
from args import parse_args

api_key = ""
org_id = ""


client = OpenAI(
    api_key=api_key,
    organization=org_id,
)


def chat(messages: list):
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
    )
    return completion.choices[0].message


if __name__ == "__main__":
    args = parse_args()
    verbose = args.verbose
    task = args.task

    examples = load_data(task)

    result_dir = Path(args.output_dir)
    result_dir.mkdir(exist_ok=True, parents=True)

    output_path = result_dir / f"preds_{task}.jsonl"
    if output_path.exists():
        preds = list(iter_jsonl(output_path))
        start_idx = len(preds)
        stop_idx = len(examples)
    else:
        start_idx = 0
        stop_idx = len(examples)
        preds = []
    tokenizer = tiktoken.encoding_for_model("gpt-4")

    start_time = time.time()
    i = start_idx
    while i < stop_idx:
        eg = examples[i]
        msgs, prompt = create_msgs(
            tokenizer, eg, task, model_name="gpt4", data_dir=args.data_dir
        )
        if verbose:
            print(f"======== Example {i} =========")
            print("Input text:")
            print(prompt[:300])
            print("...")
            print(prompt[-300:])
            print("==============================")
        # Make prediction
        try:
            response = chat(msgs)
            preds.append(
                {
                    "id": i,
                    "prediction": response.content,
                    "ground_truth": get_answer(eg, task),
                }
            )
            # Save result
            dump_jsonl(preds, output_path)
            print("Time spent:", round(time.time() - start_time))
            # exit()
            print(response)
            time.sleep(20)
            i += 1
        except Exception as e:
            print("ERROR:", e)
            print("Retrying...")
            time.sleep(60)
