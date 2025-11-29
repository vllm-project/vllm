# Benchmark KV Cache Offloading with Multi-Turn Conversations

The requirements (pip) for `benchmark_serving_multi_turn.py` can be found in `requirements.txt`

First start serving your model

```bash
export MODEL_PATH=/models/meta-llama/Meta-Llama-3.1-8B-Instruct/

vllm serve $MODEL_PATH --served-model-name Llama --disable-log-requests
```

The variable `MODEL_PATH` should be a path to the model files (e.g. downloaded from huggingface).

## Synthetic Multi-Turn Conversations

Download the following text file (used for generation of synthetic conversations)

```bash
wget https://www.gutenberg.org/ebooks/1184.txt.utf-8
mv 1184.txt.utf-8 pg1184.txt
```

The filename `pg1184.txt` is used in `generate_multi_turn.json` (see `"text_files"`).

But you may use other text files if you prefer (using this specific file is not required).

Then run the benchmarking script

```bash
export MODEL_PATH=/models/meta-llama/Meta-Llama-3.1-8B-Instruct/

python benchmark_serving_multi_turn.py --model $MODEL_PATH --served-model-name Llama \
--input-file generate_multi_turn.json --num-clients 2 --max-active-conversations 6
```

You can edit the file `generate_multi_turn.json` to change the conversation parameters (number of turns, etc.).

If successful, you will see the following output

```bash
----------------------------------------------------------------------------------------------------
Statistics summary:
runtime_sec = 215.810
requests_per_sec = 0.769
----------------------------------------------------------------------------------------------------
                   count     mean     std      min      25%      50%      75%      90%      99%      max
ttft_ms            166.0    78.22   67.63    45.91    59.94    62.26    64.43    69.66   353.18   567.54
tpot_ms            166.0    25.37    0.57    24.40    25.07    25.31    25.50    25.84    27.50    28.05
latency_ms         166.0  2591.07  326.90  1998.53  2341.62  2573.01  2860.10  3003.50  3268.46  3862.94
input_num_turns    166.0     7.43    4.57     1.00     3.00     7.00    11.00    13.00    17.00    17.00
input_num_tokens   166.0  2006.20  893.56   522.00  1247.75  2019.00  2718.00  3233.00  3736.45  3899.00
output_num_tokens  166.0   100.01   11.80    80.00    91.00    99.00   109.75   116.00   120.00   120.00
output_num_chunks  166.0    99.01   11.80    79.00    90.00    98.00   108.75   115.00   119.00   119.00
----------------------------------------------------------------------------------------------------
```

If you run with `--warmup-step`, the summary will also include `warmup_runtime_sec`
and `total_runtime_incl_warmup_sec` (while `runtime_sec` continues to reflect the
benchmark-only runtime so the reported throughput stays comparable).

### JSON configuration file for synthetic conversations generation

The input flag `--input-file` is used to determine the input conversations for the benchmark.<br/>
When the input is a JSON file with the field `"filetype": "generate_conversations"` the tool will generate synthetic multi-turn (questions and answers) conversations.

The file `generate_multi_turn.json` is an example file.

The file must contain the sections `prompt_input` and `prompt_output`.

The `prompt_input` section must contain `num_turns`, `prefix_num_tokens` and `num_tokens`:

* `num_turns` - Number of total turns in the conversation (both user & assistant).<br/>
The final value will always be rounded to an even number so each user turn has a reply.
* `prefix_num_tokens` - Tokens added at the start of only the **first user turn** in a conversation (unique per conversation).
* `num_tokens` - Total token length of each **user** message (one turn).

The `prompt_output` section must contain `num_tokens`:

* `num_tokens` - Total token length of each **assistant** message (one turn).

### Random distributions for synthetic conversations generation

When creating an input JSON file (such as `generate_multi_turn.json`),<br/>
every numeric field (such as `num_turns` or `num_tokens`) requires a distribution.<br/>
The distribution determines how to randomly sample values for the field.

The available distributions are listed below.

**Note:** The optional `max` field (for lognormal, zipf, and poisson) can be used to cap sampled values at an upper bound.</br>
Can be used to make sure that the total number of tokens in every request does not exceed `--max-model-len`.

#### constant

```json
{
    "distribution": "constant",
    "value": 500
}
```

* `value` - the fixed integer value (always returns the same number).

#### uniform

```json
{
    "distribution": "uniform",
    "min": 12,
    "max": 18
}
```

* `min` - minimum value (inclusive).
* `max` - maximum value (inclusive), should be equal or larger than min.

#### lognormal

```json
{
    "distribution": "lognormal",
    "average": 1000,
    "max": 5000
}
```

You can parameterize the lognormal distribution in one of two ways:

Using the average and optional median ratio:

* `average` - target average value of the distribution.
* `median_ratio` - the ratio of the median to the average; controls the skewness. Must be in the range (0, 1).

Using the parameters of the underlying normal distribution:

* `mean` - mean of the underlying normal distribution.
* `sigma` - standard deviation of the underlying normal distribution.

#### zipf

```json
{
    "distribution": "zipf",
    "alpha": 1.2,
    "max": 100
}
```

* `alpha` - skew parameter (> 1). Larger values produce stronger skew toward smaller integers.

#### poisson

```json
{
    "distribution": "poisson",
    "alpha": 10,
    "max": 50
}
```

* `alpha` - expected value (λ). Also the variance of the distribution.

## ShareGPT Conversations

To run with the ShareGPT data, download the following ShareGPT dataset:
`https://huggingface.co/datasets/philschmid/sharegpt-raw/blob/main/sharegpt_20230401_clean_lang_split.json`

Use the `convert_sharegpt_to_openai.py` script to convert the dataset to a format supported by `benchmark_serving_multi_turn.py`

```bash
python convert_sharegpt_to_openai.py sharegpt_20230401_clean_lang_split.json sharegpt_conv_128.json --seed=99 --max-items=128
```

The script will convert the ShareGPT dataset to a dataset with the standard user/assistant roles.

The flag `--max-items=128` is used to sample 128 conversations from the original dataset (change as needed).

Use the output JSON file `sharegpt_conv_128.json` as the `--input-file` for `benchmark_serving_multi_turn.py`.
