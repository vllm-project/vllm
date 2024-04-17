import csv
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


class PromptsGenerator:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.texts = []
        with open('./documents.csv', newline='') as file:
            for line in file:
                self.texts.append(line.strip().strip('"'))
        self.prompt_index = 0

        prompt_template = '''Please summarize the text that is given. Return just the summary and no additional conversational dialog such as ""Sure, here is the summary of the text:"".'''
        self.prompt_template_length = len(self.tokenizer.encode(prompt_template))
        np.random.seed(37)

    def generate(self, average_token: int, variance: float, max_token: int, n: int, show_progress=False) -> List[Tuple[str, int]]:
        if n <= 0:
            return []
        prompts = []
        for i in tqdm(range(n), disable=not show_progress, desc="Generating prompts"):
            prompt_length = min(int(np.random.normal(average_token, variance)), max_token)
            prompt_length = max(prompt_length-self.prompt_template_length, 16)  # avoid prompt too short.
            prompt = self.texts[self.prompt_index]
            self.prompt_index += 1
            if self.prompt_index >= len(self.texts):
                self.prompt_index = 0
            prompt_tokens = self.tokenizer.encode(prompt)[:prompt_length]
            prompt = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            prompt = f'''Please summarize the text that is given. Return just the summary and no additional conversational dialog such as ""Sure, here is the summary of the text:"". {prompt}'''
            prompt = self.tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True)
            prompts.append((prompt, len(self.tokenizer.encode(prompt))))
        return prompts

    def reset(self):
        self.prompt_index = 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate prompt for loadtest")
    parser.add_argument("-k",
                        "--average_token",
                        type=int,
                        default=1024)
    parser.add_argument("--variance",
                        type=float,
                        default=0.3)
    parser.add_argument("-n",
                        type=int,
                        default=32)
    parser.add_argument("--max_token",
                        type=int,
                        default=3072)
    parser.add_argument("--tokenizer",
                        type=str,
                        required=True)
    parser.add_argument("--output",
                        type=str,
                        required=True)
    args, _ = parser.parse_known_args()
    print('\n=============== Argument ===============')
    for key in vars(args):
        print('{}: {}'.format(key, vars(args)[key]))
    print('========================================')
    pg = PromptsGenerator(args.tokenizer)
    prompts = pg.generate(args.average_token, args.average_token*args.variance, args.max_token, args.n, show_progress=True)
    data = []
    for i in prompts:
        data.append([i[0]])
    # Specify the file path
    file_path = args.output
    # Writing to CSV file
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Prompt"])
        # Write the data to the CSV file
        csv_writer.writerows(data)
