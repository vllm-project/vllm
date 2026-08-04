# LMCache Health Check
This example demonstrates how to check the health status of the LMCache controller.

## Prerequisites
- The LMCache controller must be running.

## Steps
1. Start the LMCache controller (if not already running):
```bash
PYTHONHASHSEED=123 lmcache_controller --host localhost --port 9000 --monitor-port 9001
```

2. Send a health check request to the controller's monitor port (9001 in this example):
```bash
curl -X POST http://localhost:9000/health -H "Content-Type: application/json" -d '{"instance_id":"lmcache_default_instance"}'
```
`lmcache_default_instance` indicates the `instance_id`. 

3. The expected response is a JSON object indicating the error_codes:
```json
{"event_id":"health47ce328d-f27e-48ae-ab0c-c2218aabce95","error_codes":{"0":0,"1":0}}
```

`event_id` is an identifier of the controller operation, which can be ignored in this functionality.
error_codes formatted to worker_id to error_code pair, and error_code
`0` stand for health, `non zero` means error occurred.