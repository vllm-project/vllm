# Lab 01: Basic vLLM Offline Inference

## Overview
Learn the fundamentals of vLLM offline inference by implementing a simple text generation application. This lab introduces you to the core concepts of loading models, configuring sampling parameters, and generating text completions.

## Learning Objectives
1. Understand how to initialize the vLLM LLM class for offline inference
2. Configure basic sampling parameters for text generation
3. Handle single and multiple prompt requests
4. Process and display generated outputs
5. Understand the structure of vLLM's output objects

## Estimated Time
1-2 hours

## Prerequisites
- Python 3.8+
- Basic understanding of language models
- Familiarity with Python async/await patterns

## Lab Structure
```
lab01_basic_inference/
├── README.md           # This file
├── starter.py          # Starter code with TODOs
├── solution.py         # Complete solution
├── test_lab.py         # pytest tests
└── requirements.txt    # Dependencies
```

## Instructions

### Step 1: Setup Environment
Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Review the Starter Code
Open `starter.py` and review the structure. You'll find several TODO markers indicating where you need to add code.

### Step 3: Implement the TODOs

#### TODO 1: Initialize the LLM
Create an LLM instance with the specified model and parameters:
- Model: "facebook/opt-125m" (small model for testing)
- Set appropriate tensor parallel size
- Configure GPU memory utilization

#### TODO 2: Configure Sampling Parameters
Create a SamplingParams object with:
- Temperature: 0.8
- Top-p: 0.95
- Max tokens: 100

#### TODO 3: Generate Completions
Use the LLM's generate method to produce completions for the prompts.

#### TODO 4: Process Outputs
Extract and display the generated text from the output objects.

### Step 4: Run Your Implementation
```bash
python starter.py
```

### Step 5: Verify with Tests
```bash
pytest test_lab.py -v
```

## Expected Output

When running the solution, you should see output similar to:
```
=== vLLM Basic Inference Lab ===

Generating completions for 2 prompts...

--- Prompt 1 ---
Input: Hello, my name is
Output: John and I am a software engineer...

--- Prompt 2 ---
Input: The future of AI is
Output: bright and full of possibilities...

Inference completed successfully!
```

## Key Concepts

### LLM Class
The `LLM` class is the primary interface for offline inference in vLLM:
- Loads the model into GPU memory
- Handles batching automatically
- Provides simple API for generation

### SamplingParams
Controls the generation behavior:
- **temperature**: Randomness (0.0 = deterministic, higher = more random)
- **top_p**: Nucleus sampling threshold
- **max_tokens**: Maximum number of tokens to generate
- **top_k**: Top-k sampling parameter

### RequestOutput
The output object contains:
- `prompt`: Original input prompt
- `outputs`: List of completion outputs
- Each output has `text`, `token_ids`, `cumulative_logprob`, etc.

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce `gpu_memory_utilization` or use a smaller model.

### Issue: Model Download Fails
**Solution**: Check internet connection or manually download the model from HuggingFace.

### Issue: Import Errors
**Solution**: Ensure vLLM is properly installed with GPU support.

## Going Further

After completing this lab, try:
1. Experiment with different sampling parameters
2. Try different models from HuggingFace
3. Implement temperature scaling
4. Add logging for timing and performance metrics

## References
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Offline Inference Guide](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [HuggingFace Models](https://huggingface.co/models)
