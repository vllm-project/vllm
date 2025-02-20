def get_prompts():
    filename = "pile.txt"
    with open(filename, "r") as f:
        prompts = f.readlines()
        print(f"Number of prompts: {len(prompts)}")
    return prompts

def get_prompt_token_ids(model_path, prompts, max_length=1024):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt_token_ids = []
    for prompt in prompts:
        tokens = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        if len(tokens.input_ids[0]) < max_length:
            continue
        prompt_token_ids.append([x.item() for x in tokens.input_ids[0]])
    return prompt_token_ids