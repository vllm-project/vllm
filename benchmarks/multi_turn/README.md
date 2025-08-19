# Benchmark KV Cache Offloading with Multi-Turn Conversations

The requirements (pip) for `benchmark_serving_multi_turn.py` can be found in `requirements.txt`

First start serving your model

```bash
export MODEL_NAME=/models/meta-llama/Meta-Llama-3.1-8B-Instruct/

vllm serve $MODEL_NAME --disable-log-requests
```

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
export MODEL_NAME=/models/meta-llama/Meta-Llama-3.1-8B-Instruct/

python benchmark_serving_multi_turn.py --model $MODEL_NAME --input-file generate_multi_turn.json \
--num-clients 2 --max-active-conversations 6
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
