# Lab 02: Online Serving with AsyncEngine

## Overview
Learn how to implement online inference serving using vLLM's AsyncEngine. This lab teaches you to handle asynchronous request processing, manage concurrent requests, and build a simple async API for real-time inference.

## Learning Objectives
1. Understand the difference between offline (LLM) and online (AsyncEngine) inference
2. Initialize and configure AsyncLLMEngine for serving
3. Implement async request handling with proper coroutine management
4. Process streaming and non-streaming responses
5. Handle concurrent requests efficiently

## Estimated Time
1.5-2 hours

## Prerequisites
- Completion of Lab 01 (Basic Inference)
- Understanding of Python async/await syntax
- Basic knowledge of coroutines and event loops

## Lab Structure
```
lab02_online_serving/
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
Open `starter.py` and review the async structure. You'll implement an async inference server.

### Step 3: Implement the TODOs

#### TODO 1: Initialize AsyncLLMEngine
Create an AsyncLLMEngine with:
- Model configuration
- Engine arguments
- Proper async initialization

#### TODO 2: Generate Request ID
Implement a function to generate unique request IDs for tracking.

#### TODO 3: Submit Async Request
Use the engine's `generate()` method to submit requests asynchronously.

#### TODO 4: Process Async Results
Iterate through the async generator to collect results.

#### TODO 5: Handle Multiple Concurrent Requests
Implement `asyncio.gather()` to process multiple requests concurrently.

### Step 4: Run Your Implementation
```bash
python starter.py
```

### Step 5: Verify with Tests
```bash
pytest test_lab.py -v
```

## Expected Output

```
=== vLLM Async Online Serving Lab ===

Initializing AsyncLLMEngine...
Engine initialized successfully!

Processing single request...
Request ID: req-1234
Prompt: Explain async programming in Python
Generated: Async programming in Python allows...

Processing concurrent requests...
Request 1 (req-5678): Complete
Request 2 (req-5679): Complete
Request 3 (req-5680): Complete

All 3 requests completed in X.XX seconds
Throughput: X.XX requests/second
```

## Key Concepts

### AsyncLLMEngine vs LLM
- **LLM**: Synchronous, offline batch inference
- **AsyncLLMEngine**: Asynchronous, online serving with request queueing

### Async Request Flow
1. Client submits request with unique ID
2. Engine queues request for processing
3. Request is scheduled based on available resources
4. Results streamed back via async generator
5. Client receives final output

### Request Management
- Each request needs a unique `request_id`
- Use `SamplingParams` to control generation
- Results arrive via async iteration

### Concurrency
- Multiple requests processed simultaneously
- Engine handles scheduling and batching
- Use `asyncio.gather()` for parallel requests

## Troubleshooting

### Issue: RuntimeError - No running event loop
**Solution**: Ensure functions are called with `await` inside async functions.

### Issue: Requests hanging indefinitely
**Solution**: Check that you're properly iterating through the async generator.

### Issue: Import error for AsyncLLMEngine
**Solution**: Ensure you're using vLLM >= 0.2.0.

## Going Further

After completing this lab, try:
1. Add request cancellation support
2. Implement request priority queues
3. Add timeout handling
4. Create a simple FastAPI wrapper
5. Monitor queue depths and processing times

## References
- [vLLM AsyncEngine Documentation](https://docs.vllm.ai/)
- [Python asyncio Guide](https://docs.python.org/3/library/asyncio.html)
- [Async/Await Best Practices](https://realpython.com/async-io-python/)
