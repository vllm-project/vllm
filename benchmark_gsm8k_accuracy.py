total = len(problems)
correct = 0
num_evaluated = 0

print(f"Evaluating {total} examples...\n")

for idx, problem in enumerate(problems):
    question = problem["question"]
    
    gold_match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', problem["answer"])
    if not gold_match:
        print(f"Warning: cannot parse gold answer for example {idx}, skipping")
        continue
    gold_answer = gold_match.group(1)

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    try:
        response = requests.post(args.api_url, json=payload, timeout=60)
        response.raise_for_status()
        model_output = response.json()["choices"][0]["message"]["content"]
        pred_answer = extract_answer(model_output)

        if pred_answer == gold_answer:
            correct += 1
        else:
            print(f"Example {idx}:")
            print(f"  Q: {question[:60]}...")
            print(f"  Gold: {gold_answer}, Pred: {pred_answer}")
            print(f"  Model output snippet: {model_output[:120]}...\n")

        num_evaluated += 1   

    except Exception as e:
        print(f"Error on example {idx}: {e}")

    if (idx + 1) % 100 == 0 and num_evaluated > 0:
        current_accuracy = correct / num_evaluated * 100
        print(f"Processed {idx+1}/{total}, Evaluated: {num_evaluated}, Current accuracy: {current_accuracy:.2f}%")


accuracy = correct / num_evaluated * 100 if num_evaluated > 0 else 0
print(f"\n{'='*40}")
print(f"Final accuracy: {correct}/{num_evaluated} = {accuracy:.2f}%")
