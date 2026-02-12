# Kimi-Audio Prefix Cache Reset and Validation

## Reset prefix cache after deploy/restart
Run once after the server comes up to clear stale entries:

```bash
curl -X POST http://localhost:8093/reset_prefix_cache
```

## Verification checklist
1. **Same audio twice → cache hit + identical output**
   - Send the same audio file twice to `/v1/audio/transcriptions`.
   - Confirm the transcript text matches exactly.
   - Observe server logs for a non-zero **Prefix cache hit rate** on the second request.

2. **Different audio → cache miss + correct output**
   - Send a different audio file immediately after the above test.
   - Confirm the transcript is correct and not truncated or mixed with the prior output.

3. **Log check (required)**
   - Verify log lines like:
     `Prefix cache hit rate: <value>%` in the API server logs.
   - The hit rate should increase only when identical audio is repeated.
