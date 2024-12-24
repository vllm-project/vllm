# Copyright (c) OpenMMLab. All rights reserved.
import json
import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(tokenizer, nsamples, seed, seqlen, path=None):
    """Load Wikitext-2 train and test datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: Full tokenized Wikitext-2 test set.
    """
    from datasets import load_dataset
    traindata = load_dataset(path if path else 'wikitext',
                             'wikitext-2-raw-v1',
                             split='train')
    testdata = load_dataset(path if path else 'wikitext',
                            'wikitext-2-raw-v1',
                            split='test')

    trainenc = tokenizer('\n\n'.join(traindata['text']), return_tensors='pt')
    testenc = tokenizer('\n\n'.join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(tokenizer, nsamples, seed, seqlen):
    """Load PTB train and validation datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: Full tokenized PTB validation set.
    """
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only',
                           'penn_treebank',
                           split='validation')

    trainenc = tokenizer('\n\n'.join(traindata['sentence']),
                         return_tensors='pt')
    testenc = tokenizer('\n\n'.join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        print("traindata ", trainenc.input_ids.shape)
        print("seqlen ", seqlen)
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(tokenizer, nsamples, seed, seqlen, path=None):
    """Load C4 train and validation datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: Full tokenized PTB validation set.
    """
    from datasets import load_dataset
    traindata = load_dataset(
        path if path else 'allenai/c4',
        'allenai--c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train',
        use_auth_token=False)
    valdata = load_dataset(
        path if path else 'allenai/c4',
        'allenai--c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation',
        use_auth_token=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_ptb_new(tokenizer, nsamples, seed, seqlen):
    """Load PTB New train and validation datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: Full tokenized PTB validation set.
    """
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(' '.join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(' '.join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(tokenizer, nsamples, seed, seqlen):
    """Load C4 New train and validation datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: Full tokenized PTB validation set.
    """
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4',
        'allenai--c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train')
    valdata = load_dataset(
        'allenai/c4',
        'allenai--c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_pileval(tokenizer, nsamples, seed, path, seqlen=512):
    """Load pileval train dataset and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: Full tokenized PTB validation set.
    """
    from datasets import load_dataset
    from datasets.builder import DatasetGenerationError
    try:
        dataset = load_dataset('json', data_files=path, split='train')
    except DatasetGenerationError as err:
        raise InterruptedError('There have been some issues when generating '
                               'the dataset, you could try to download it '
                               'locally first, and replace the `data_files`'
                               'with local addresses or use other datasets '
                               '(c4, wiki, ptb).') from err
    dataset = dataset.shuffle(seed=seed)
    samples = []
    n_run = 0
    for data in dataset:
        line = data['text']
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == nsamples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // seqlen
    print(f' * Split into {n_split} blocks')
    return [
        cat_samples[:, i * seqlen:(i + 1) * seqlen] for i in range(n_split)
    ], None

#   llamafactory datasets
def get_lf_datasets(tokenizer, nsamples, seed, seqlen, path_to_eval, split_name):
    from datasets import load_dataset
    from typing import Dict
    from tqdm import tqdm, trange
    from template import get_eval_template
    from transformers.utils import cached_file
    CHOICES = ["A", "B", "C", "D"]
    SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]

    mapping = cached_file(
        path_or_repo_id=path_to_eval,
        filename="mapping.json",
    )
    with open(mapping, "r", encoding="utf-8") as f:
        categorys: Dict[str, Dict[str, str]] = json.load(f)
    category_corrects = {subj: np.array([], dtype="bool") for subj in SUBJECTS}
    pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
    trainloader = []
    inputs, labels = [], []
    for subject in pbar:
        dataset = load_dataset(
            path=path_to_eval,
            name=subject,
            # split='train',
            trust_remote_code=True,
        )
        traindata = dataset[split_name]
        pbar.set_postfix_str(categorys[subject]["name"])
        for i in trange(len(traindata), desc="Formatting batches", position=1, leave=False):
            # print("loop i ", i )
            support_set = (
                dataset["train"].shuffle().select(range(min(nsamples, len(dataset["train"]))))
            )
            messages = get_eval_template('zh').format_example(
                target_data=traindata[i],
                support_set=support_set,
                subject_name=categorys[subject]["name"],
            )
            messages[-2]["content"] = '"'+ messages[-2]["content"]+'"'
            # print("**** messages[-2][content] ",messages[-2])
            # print("**** messages[-2][content] ",messages[-2]["content"])
            inputs.append(messages[-2]["content"])
            labels.append(messages[-1]["content"])
            # print(labels)
    trainenc = tokenizer('\n\n'.join(inputs),
                        return_tensors='pt')
    # testenc = tokenizer('\n\n'.join(valdata['sentence']), 
    #                     return_tensors='pt')
    import random
    random.seed(seed)
    # for _ in range(min(nsamples, len(inputs))):
    #     # print("seqlen ", seqlen)
    #     # print("traindata ", trainenc.input_ids.shape)
    #     i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     trainloader.append((inp, tar))
    max_length = trainenc.input_ids.shape[1]
    print("n_requests ", len(inputs))
    print("max_length ", max_length)
    for n in range(max_length):
        # print("seqlen ", seqlen)
        # print("traindata ", trainenc.input_ids.shape)
        i = n*seqlen
        j = i + seqlen
        if j<max_length:
            inp = trainenc.input_ids[:, i:j]
        else:
            s = max_length-1-seqlen
            inp = trainenc.input_ids[:, s:max_length-1]
        tar = inp.clone()
        tar[:, :-1] = -100
        # print("n ", n, " i ",i, " j ",j, " inp ", inp)
        if j>=max_length or len(trainloader)>nsamples:
            break
        trainloader.append((inp, tar))
    return trainloader, None

# ceval_val_cmcc.jsonl
def get_ceval_val_cmcc(tokenizer, nsamples, seed, seqlen, path_to_eval):
    path_to_eval = path_to_eval+'ceval_val_cmcc.jsonl'
    trainloader = []
    inputs=[]
    with open(path_to_eval, 'r') as jsonl_file:
        for line in jsonl_file:
            json_object = json.loads(line)
            inputs.append(json_object["origin_prompt"])

    # inputs=["Please introduce particle physics."]
    trainenc = tokenizer('\n\n'.join(inputs),
                        return_tensors='pt')

    import random
    random.seed(seed)
    # print(trainenc)
    # for _ in range(min(nsamples, len(inputs))):
    #     # print("seqlen ", seqlen)
    #     print("traindata ", trainenc.input_ids.shape)
    #     i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     print("i ",i, " j ",j, " inp ", inp)
    #     trainloader.append((inp, tar))

    max_length = trainenc.input_ids.shape[1]
    print("n_requests ", len(inputs))
    print("max_length ", max_length)
    for n in range(max_length):
        # print("seqlen ", seqlen)
        # print("traindata ", trainenc.input_ids.shape)
        i = n*seqlen
        j = i + seqlen
        if j<max_length:
            inp = trainenc.input_ids[:, i:j]
        else:
            s = max_length-1-seqlen
            inp = trainenc.input_ids[:, s:max_length-1]

        tar = inp.clone()
        tar[:, :-1] = -100
        # print("n ", n, " i ",i, " j ",j, " inp ", inp)
        trainloader.append((inp, tar))
        if j>=max_length:
            break
    return trainloader, None

def get_calib_loaders(name,
                      tokenizer,
                      nsamples=128,
                      seed=0,
                      seqlen=2048,
                      path=None):
    """Get calibration data loaders for a dataset.

    Args:
      name: Dataset name ('wikitext2', 'ptb', 'c4', etc).
      tokenizer: Tokenizer to encode text.
      nsamples: Number of samples to take from train set.
      seed: Random seed for sampling.
      seqlen: Maximum sequence length.

    Returns:
      train_loader: List of sampled and tokenized training examples.
      test_data: Full tokenized validation set.
    """
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer, nsamples, seed, seqlen, path)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(tokenizer, nsamples, seed, seqlen)
        return get_ptb(tokenizer, nsamples, seed, seqlen)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(tokenizer, nsamples, seed, seqlen)
        return get_c4(tokenizer, nsamples, seed, seqlen, path)

    if 'pileval' in name:
        if path is None:
            path = 'https://the-eye.eu/public/AI/pile/val.jsonl.zst'
        return get_pileval(tokenizer, nsamples, seed, path, seqlen)

    if 'pileval' in name:
        if path is None:
            path = 'https://the-eye.eu/public/AI/pile/val.jsonl.zst'
        return get_pileval(tokenizer, nsamples, seed, path, seqlen)

    if 'ceval_val_cmcc' in name:
        return get_ceval_val_cmcc(tokenizer, nsamples, seed, seqlen, path)
    if 'ceval' or 'cmb' or 'cmmlu' or 'medmcqa' or 'medqa' or 'mmlu' in name:
        if name == 'ceval_val_cmcc':
            pass
        split_name = 'test'
        if name == 'ceval':
            split_name = 'test'
        elif name == 'cmb':
            split_name = 'test'
        elif name == 'medmcqa':
            split_name = 'test'
        elif name == 'medqa':
            split_name = 'test'
        elif name == 'mmlu':
            split_name = 'test'

        return get_lf_datasets(tokenizer, nsamples, seed, seqlen, path, split_name)

