# Lab 05: Streaming Output Handling

## Overview
Learn to implement streaming responses for real-time text generation. This lab covers async generators, streaming protocols, and building responsive applications with vLLM.

## Learning Objectives
1. Understand streaming vs. non-streaming generation
2. Implement async streaming with AsyncLLMEngine
3. Handle partial outputs and token-by-token generation
4. Build real-time streaming applications
5. Manage streaming errors and cancellation

## Estimated Time
1.5-2 hours

## Prerequisites
- Lab 02 (Online Serving)
- Understanding of async generators
- Knowledge of streaming protocols

## Instructions

### Implement TODOs
1. Create async streaming generator
2. Handle partial outputs incrementally
3. Implement streaming with proper buffering
4. Add cancellation support
5. Build real-time display of streaming text

### Expected Output
```
=== Streaming Output ===
Prompt: Write a story about AI

[Streaming]
The> artificial> intelligence> began> to> learn>...
(each token appears in real-time)

Complete output: The artificial intelligence began to learn...
```

## Key Concepts

### Streaming Benefits
- Lower perceived latency
- Better user experience
- Real-time feedback
- Cancellable generation

### Async Generators
- Use `async for` to iterate
- Yield partial results
- Support cancellation

## Going Further
1. Add token timing metrics
2. Implement streaming to WebSocket
3. Add streaming rate limiting
4. Build SSE (Server-Sent Events) endpoint

## References
- [vLLM Streaming Guide](https://docs.vllm.ai/)
- [Python Async Generators](https://peps.python.org/pep-0525/)
