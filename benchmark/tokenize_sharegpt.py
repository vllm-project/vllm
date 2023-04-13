"""
Usage:
python3 tokenize_sharegpt.py --model facebook/opt-125m --dataset sharegpt_clean_lang_10k.json
"""
import argparse
import json
import pickle

import ray
from transformers import AutoTokenizer


@ray.remote
class DataParser:
    def __int__(self) -> None:
        pass

    def set_tokenizer(self, model: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def set_dataset(self, dataset):
        self.dataset = dataset

    def parse(self):
        tokenized = []
        for data in self.dataset:
            conv = []
            conversation = data['conversations']
            for seq in conversation:
                speaker = seq['from']
                text = seq['value']

                tokens = self.tokenizer.encode(f'{speaker}: {text}')
                conv.append((speaker, tokens))
            tokenized.append(conv)
        return tokenized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--output', type=str, default="sharegpt_tokenized.pkl")
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    num_cpus = 24

    # Divide dataset into `num_cpus` parts.
    dataset_size = len(dataset)
    seg = (dataset_size + num_cpus - 1) // num_cpus

    dataset_parts = []
    for i in range(num_cpus):
        dataset_parts.append(dataset[i * seg: (i + 1) * seg])

    parsers = [DataParser.remote() for _ in range(num_cpus)]
    for i, parser in enumerate(parsers):
        parser.set_tokenizer.remote(args.model)
        parser.set_dataset.remote(dataset_parts[i])

    tokenized_data = ray.get([parser.parse.remote() for parser in parsers])
    tokenized_data = sum(tokenized_data, [])
    with open(args.output, 'wb') as f:
        pickle.dump(tokenized_data, f)
