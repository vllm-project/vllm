import random
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from typing import List, Tuple, Optional


@dataclass
class TextDatasetArgs:
    num_samples: int
    max_len: int = 4096
    seed: int = 42
    fixed_output_len: Optional[int] = None


def prepare_text_requests(prompts: List[str], completions: List[str],
                         tokenizer: PreTrainedTokenizerBase,
                         dataset_args: TextDatasetArgs,
) -> List[Tuple[str, int, int]]:
    assert len(prompts) == len(completions)
    dataset = []
    for prompt, completion in zip(prompts, completions):
        # Get length.
        prompt_len = len(tokenizer(prompt).input_ids)
        output_len = len(tokenizer(completion).input_ids)
        if dataset_args.fixed_output_len is not None:
            output_len = dataset_args.fixed_output_len

        # Prune too short or long sequences
        if (prompt_len < 4 or output_len < 4
                or prompt_len + output_len > dataset_args.max_len):
            continue

        # Make into dataset tripe.
        dataset.append((prompt, prompt_len, output_len))
        if (len(dataset) >= dataset_args.num_samples * 2):
            break

    # Sample num_requests from the list.
    assert dataset_args.num_samples <= len(dataset)
    random.seed(dataset_args.seed)
    return random.sample(dataset, dataset_args.num_samples)
