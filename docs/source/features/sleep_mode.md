# Sleep Mode

Sleep mode is a feature in vLLM that allows you to temporarily free up GPU resources when the model is not in use. This is particularly useful in production environments where you want to optimize resource utilization.

## Overview

When a model is put to sleep:
- GPU resources are freed up for other tasks
- The model remains in an inactive state but can be reactivated
- API endpoints that don't require model weights continue to function normally

## API Endpoints

vLLM provides several endpoints to control the sleep mode:

- `POST /sleep` - Put the model to sleep
- `POST /wake_up` - Wake up the model
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
- The first request after waking up a model may have higher latency as the model is reloaded
- Sleep mode state is global for the server - all models managed by the server are either asleep or awake 