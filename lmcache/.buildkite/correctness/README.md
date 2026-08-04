# Correctness Test

This is a E2E testing suite for whether having a KV pass through LMCache from vllm (storing and loading) will degrade the accuracy of a transformer. 

Unit tests do not suffice in going from request to response and passing through LMCache. We use offline serving
for this testing suite because the API Server of vllm is assumed to be unproblematic. 

## Setup 

We use the [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) multiple choice dataset downloaded as a compressed tarball in `data.tar.xz`

The following 3 options are viable:
A. Disable prefix caching on vllm to only use lmcache
B. Use lmcache centralized remote server
C. Use lmcache p2p sharing

We choose option A to keep the CI server lightweight (only 2x L4s needed). 

We test one small model for each Attention Architecture. Currently dense/standard attention will use `meta-llama/Llama-3.1-8B-Instruct` and 
MLA (multi-head latent attention) will use `deepseek-ai/DeepSeek-V2-Lite` with tensor parallel 2. We do not care about the objective accuracy on the MMLU benchmark but only on any differential that appears from using LMCache. 

## CI Agent Pre Set-up

To ensure the speed of the MMLU Correctness tests, please conduct the following set up on your runner beforehand (only need to do once and `setup.sh` will renew your environment afterwards). 

1. Create a virtual environment at `~/correctness_venv`

```bash
pip install pandas
```

2. Please run the following commands (these will be run by `setup.sh` every time):
```bash
cd ~
mkdir correctness_repositories
cd correctness_repositories
git clone https://github.com/LMCache/LMCache.git
git clone https://github.com/vllm-project/vllm.git
cd LMCache
pip install -e .
cd ../vllm
pip install -e . 
```

Explanation: "pull" upstream code in this CI suite by pulling the latest upstream default branch
and let the editable virtual environment update the latest code. This allows a little bit of pre-setup to greatly optimize every run afterwards. 

## Directory Structure
`single-mmlu-test.py`

The MMLU question dataset will be sent to this single endpoint and accuracy will be evaluated (correct answers are included
in the dataset). When LMCache is used, queries are sent twice. 