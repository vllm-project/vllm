# Sleep Mode

Sleep mode is a feature in vLLM that allows you to temporarily offload model weights from GPU memory to CPU memory, freeing up valuable GPU resources when the model is not in use. This is particularly useful in production environments where you want to optimize resource utilization.

## Overview

When a model is put to sleep:
- Model weights are offloaded from GPU memory to CPU memory
- GPU resources are freed up for other tasks
- The model remains loaded but in an inactive state
- API endpoints that don't require model weights continue to function normally

## API Endpoints

vLLM provides several endpoints to control the sleep mode:

- `POST /sleep` - Put the model to sleep, offloading weights from GPU memory
- `POST /wake_up` - Wake up the model, restoring weights to GPU memory
- `GET /is_sleeping` - Check if the model is currently in sleep mode

### Protected Endpoints

When the model is in sleep mode, endpoints that require access to model weights will return a 503 Service Unavailable error with a message indicating that the model is sleeping. The following endpoints are protected:

- `/v1/chat/completions`
- `/v1/completions`
- `/v1/embeddings`
- `/pooling`
- `/score` and `/v1/score`
- `/rerank`, `/v1/rerank`, and `/v2/rerank`
- `/v1/audio/transcriptions`
- `/invocations`

Other endpoints like `/tokenize`, `/detokenize`, and `/v1/models` continue to work normally as they don't require access to model weights.

## Usage

### Putting a Model to Sleep

```bash
curl -X POST http://localhost:8000/sleep
```

You can specify a sleep level (default is 1):

```bash
curl -X POST http://localhost:8000/sleep?level=1
```

### Waking Up a Model

```bash
curl -X POST http://localhost:8000/wake_up
```

### Checking Sleep Status

```bash
curl -X GET http://localhost:8000/is_sleeping
```

Response:
```json
{"is_sleeping": true}
```

### Handling Sleep Mode in Client Applications

When using vLLM with sleep mode enabled, client applications should handle 503 error responses with the `ModelSleepingError` type. Here's an example of the error response:

```json
{
  "error": {
    "message": "Model is currently in sleep mode. Please wake it up first with a POST request to /wake_up",
    "type": "ModelSleepingError",
    "code": 503
  }
}
```

Clients can catch this error and automatically wake up the model before retrying the request.

## Limitations

- Sleep mode is only supported on CUDA devices
- The first request after waking up a model may have higher latency as weights are restored to GPU memory
- Sleep mode state is global for the server - all models managed by the server are either asleep or awake 