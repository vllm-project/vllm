# KV Cache Size Calculator

## Introduction

The KV Cache Size Calculator provides a web interface for calculating the size of the key-value cache required by large language models (LLMs). Users can select a model, specify the data type, and enter the number of tokens to calculate the KV cache size in gigabytes. The web interface includes a form where users can input these parameters and view the results immediately, making it simple and efficient to estimate cache requirements.

This document also provides an overview of the JSON format for model configurations and explains how to use the `generate_config.py` script to generate model configurations using the `transformers` library's `AutoConfig` feature.

## JSON Configuration Format

The JSON configuration file produced by `generate_config.py` or manually maintained should adhere to the following format:

```json
{
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8
}
```

### Fields Description
- **hidden_size**: The size of the hidden layers within the model.
- **num_attention_heads**: The number of attention heads used in each transformer block.
- **num_hidden_layers**: The total number of hidden layers in the model.
- **num_key_value_heads**: (Optional) The number of key-value heads used in certain transformer architectures.

> Note: If an attribute is not applicable to a particular model, it may be set to `null` or omitted altogether.

## How to Use `generate_config.py`

The `generate_config.py` script is used to generate a configuration JSON for a specific model using the Hugging Face `transformers` library. It fetches the model configuration and outputs it in a JSON format to the console.

### Requirements

- **Python 3.6+**
- **transformers library** from Hugging Face
- Install dependencies using:
  ```sh
  pip install transformers
  ```

### Usage

To use the script, run the following command in your terminal:

```sh
python generate_config.py --model <model-name>
```

Replace `<model-name>` with the name or path of the model whose configuration you wish to generate. The `<model-name>` can be any model available on the Hugging Face Hub or a local path containing the model files.

#### Example

```sh
python generate_config.py --model meta-llama/Llama-3.1-8B-Instruct
```

### Output

The script will output the model configuration in JSON format. For example:

```json
{
    "hidden_size": 8192,
    "num_attention_heads": 64,
    "num_hidden_layers": 80,
    "num_key_value_heads": 8
}
```

### Handling Errors

In case the model name is incorrect or the configuration cannot be fetched, the script will print an error message in JSON format:

```json
{
    "error": "Can't load config for '<model-name>'. Make sure that '<model-name>' is the correct path to a directory containing a config.json file"
}
```

### Modifying the Script

You can easily modify the script to save the JSON output to a file instead of printing it to the console. To do so, redirect the output using:

```sh
python generate_config.py --model <model-name> > model_config.json
```

This will save the configuration to a file named `model_config.json` in the current directory.

## Notes
- The script relies on the internet to fetch model configurations unless the model is available locally.
- If certain fields are not available in a model's configuration, they will be set to `null` or excluded from the JSON.

Feel free to modify `generate_config.py` as needed to add more fields or adjust the output format to better suit your requirements.

## How to Contribute

We welcome contributions to improve the KV Cache Size Calculator and related scripts.

### Contribution Guidelines
- Fork the repository and create a new branch for your feature or bug fix.
- Make your changes, ensuring the code is well-documented and adheres to the existing style.
- Submit a pull request describing your changes and the motivation behind them.

If you have any suggestions or find any issues, feel free to open an issue on GitHub. Your contributions are greatly appreciated!


