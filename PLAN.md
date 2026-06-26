# Plan: Fix Voxtral Realtime Transcription Crash (Issue #39202)

## 1. Summary of the Issue

When serving `mistralai/Voxtral-Mini-4B-Realtime-2602` for audio transcription via `/v1/audio/transcriptions`, the engine crashes after multiple invocations with:

```
RuntimeError: The size of tensor a (40) must match the size of tensor b (39) at non-singleton dimension 0
```

at `vllm/v1/worker/gpu_model_runner.py:3230`:
```python
self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)
```

**Root cause:** `VoxtralRealtimeGeneration.embed_input_ids()` returns only the flattened multimodal embeddings (shape `[num_mm_tokens, embed_dim]`) instead of a full tensor (shape `[num_all_tokens, embed_dim]`). When a batch contains mixed requests — e.g., a new prefill request (39 tokens, all multimodal) AND a cached decode request (1 token, no multimodal data) — the returned tensor has 39 rows but the caller expects 40 (= `num_scheduled_tokens`).

**Why it only happens after multiple invocations:** Initially, all requests in a batch have audio data. Over time, with concurrent requests, a batch can mix a new prefill request (with audio) and a running decode request (generating text tokens with no new audio). The decode token has no multimodal embedding, so the flattened output is shorter than `num_scheduled_tokens`.

## 2. Files to Modify and Why

### Primary fix

**`vllm/model_executor/models/voxtral_realtime.py` (lines 293-325)**

The `embed_input_ids` override is incorrect. It must return a tensor of shape `[input_ids.shape[0], embed_dim]` — one embedding per scheduled token — but currently returns only the flattened multimodal embeddings (`mm_embeds_flat`), which has fewer rows when some tokens in the batch are non-multimodal.

The fix: allocate a full-sized output tensor, then place multimodal embeddings at the positions indicated by `is_multimodal`. Non-multimodal positions get zeros (which is correct for this model — the `forward` method adds `audio_text_embeds + text_embeds`, so zero audio embedding means text-only).

### Test file

**`tests/models/multimodal/generation/test_voxtral_realtime.py`**

Add a test that exercises the mixed-batch scenario: multiple concurrent requests where one is prefilling (has audio) and another is decoding (no new audio).

## 3. Implementation Approach

### Fix `embed_input_ids` in `VoxtralRealtimeGeneration`

Replace the current implementation (lines 293-325):

```python
def embed_input_ids(
    self,
    input_ids: torch.Tensor,
    multimodal_embeddings: MultiModalEmbeddings | None = None,
    *,
    is_multimodal: torch.Tensor | None = None,
) -> torch.Tensor:
    pool_size = self.config.audio_config.block_pool_size
    embed_dim = self.config.audio_config.d_model * pool_size

    # Always allocate full-sized output — one row per scheduled token.
    # Non-multimodal positions stay zero, which is correct because
    # forward() adds audio_text_embeds + text_embeds.
    output = torch.zeros(
        input_ids.shape[0],
        embed_dim,
        dtype=self.whisper_encoder.dtype,
        device=input_ids.device,
    )

    if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
        logger.warning(
            "Realtime model received empty multimodal embeddings "
            "for %d input tokens. Returning zero embeddings to "
            "avoid engine crash.",
            input_ids.shape[0],
        )
        return output

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)

    if is_multimodal is not None and is_multimodal.any():
        # Place multimodal embeddings at the correct positions
        output[is_multimodal] = mm_embeds_flat.to(dtype=output.dtype)
    else:
        # Fallback: if no mask provided, assume all tokens are multimodal
        # (original behavior for backwards compatibility in single-request case)
        if mm_embeds_flat.shape[0] == input_ids.shape[0]:
            output = mm_embeds_flat
        else:
            # Size mismatch with no mask — place what we have at the start
            output[:mm_embeds_flat.shape[0]] = mm_embeds_flat.to(dtype=output.dtype)

    return output
```

**Key design decisions:**

1. **Always return `[input_ids.shape[0], embed_dim]`** — matches the contract expected by `gpu_model_runner._preprocess()` at line 3230.
2. **Use `is_multimodal` mask** — the mask is already computed by `_gather_mm_embeddings()` and passed through `embed_input_ids()`. This is how the base `SupportsMultiModal.embed_input_ids` works (via `_merge_multimodal_embeddings`).
3. **Zero for non-multimodal positions** — correct because `forward()` computes `audio_text_embeds + text_embeds`, so zero audio embedding means the token gets only its text embedding.

### Verify the call chain

Confirm in `gpu_model_runner.py:3223-3227` that `is_multimodal` (the `is_mm_embed` tensor) is always passed. Current code:

```python
inputs_embeds_scheduled = self.model.embed_input_ids(
    self.input_ids.gpu[:num_scheduled_tokens],
    multimodal_embeddings=mm_embeds,
    is_multimodal=is_mm_embed,
)
```

Yes — `is_mm_embed` is always passed. No changes needed in the model runner.

## 4. Edge Cases and Error Handling

### Edge case 1: All tokens are multimodal
When every token in the batch has audio (common for single-request batches), `is_multimodal.sum() == input_ids.shape[0]` and `mm_embeds_flat` covers all positions. The indexed assignment `output[is_multimodal] = mm_embeds_flat` works correctly.

### Edge case 2: No tokens are multimodal
When `multimodal_embeddings` is empty or None (all decode tokens in a batch), the method returns all-zeros. The warning is logged. This is the existing behavior and is correct.

### Edge case 3: `is_multimodal` is None
Should not happen in the V1 code path (always provided by `_gather_mm_embeddings`), but the fallback handles it. If `mm_embeds_flat` exactly matches `input_ids.shape[0]`, use it directly. Otherwise, place at the start.

### Edge case 4: Mismatch between `is_multimodal.sum()` and `mm_embeds_flat.shape[0]`
This would indicate a bug in `_gather_mm_embeddings`. We should let the RuntimeError from the indexed assignment propagate — it gives a clear error message. This matches the behavior of the base `_merge_multimodal_embeddings` (which also catches this and raises `ValueError`).

### Edge case 5: dtype mismatch
The `mm_embeds_flat.to(dtype=output.dtype)` cast handles potential dtype differences between encoder output and the expected whisper dtype.

### Edge case 6: Empty audio / encoder cache eviction
Already handled by the `multimodal_embeddings is None or len(...) == 0` check, which returns zeros. The warning log is preserved.

## 5. Test Strategy

### Unit-level: Test `embed_input_ids` directly

Add a test that constructs a `VoxtralRealtimeGeneration` model (or mocks the minimal required attributes) and calls `embed_input_ids` with:

1. **All multimodal**: `is_multimodal = [True, True, True]`, 3 embeddings -> output shape `[3, dim]`
2. **Mixed batch**: `is_multimodal = [True, True, True, False]`, 3 embeddings -> output shape `[4, dim]`, last row all zeros
3. **No multimodal**: `multimodal_embeddings = []` -> output shape `[N, dim]`, all zeros
4. **is_multimodal is None**: Falls back to direct assignment

### Integration-level: Reproduce the crash scenario

In `tests/models/multimodal/generation/test_voxtral_realtime.py`, add a test that:
1. Submits multiple transcription requests concurrently (using `engine.generate` with a list of inputs)
2. Ensures the batch scheduler will create mixed batches (some prefilling, some decoding)
3. Verifies no crash occurs and outputs are correct

The existing `test_voxtral_realtime_forward` already sends 2 requests together, which exercises some batching. However, the crash requires one request to be in decode phase while another is in prefill — this is timing-dependent and hard to reproduce deterministically in a unit test. The unit test of `embed_input_ids` with a mixed `is_multimodal` mask is the most reliable way to verify the fix.

### Existing tests

Run the existing test to ensure no regression:
```bash
.venv/bin/python -m pytest tests/models/multimodal/generation/test_voxtral_realtime.py -v
```

### Pre-commit checks
```bash
pre-commit run --all-files
```
