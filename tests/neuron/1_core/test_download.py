# SPDX-License-Identifier: Apache-2.0

# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_download():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
