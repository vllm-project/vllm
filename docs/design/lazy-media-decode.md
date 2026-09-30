# Lazy Media Decode

Media that vLLM fetches for a request crosses from the request parser into the multi-modal processor as a *reference*: an encoded payload plus the identity derived from it, with the decode deferred. The processor decodes only the items the cache does not already hold, and decodes them in parallel.

## Motivation

Decoding media is expensive — a video decode dwarfs the rest of handling a request — and [Processor Output Caching](mm_processing.md#processor-output-caching) can skip that work only for an item whose identity is known *before* it is decoded. Hashing decoded pixels or samples would pay the decode cost on every request, including the ones the cache would have served.

So the unit that travels from fetch to processor is not a decoded object. It is an encoded payload, a cache key derived from that payload, and a closure that turns one into the other on demand.

## The media reference

`MediaRef[_T]` in `vllm/multimodal/media/base.py` is the single value type for fetched media. It carries read-only `key` (the cache identity, as `bytes`) and `spec` (a `DecodeSpec`, the resolved decode configuration already folded into the key), alongside `data` (the encoded bytes, `b""` once released), and exposes `is_decoded`, `decode()`, `map()` and `release()`. It appears in the `ImageItem`, `VideoItem` and `AudioItem` aliases in `vllm/multimodal/inputs.py`, so a reference is a legal item everywhere a decoded object is.

### Cache key

`derive_media_key(data, spec)` digests the encoded bytes together with `DecodeSpec.canonical_bytes()`, each chunk preceded by its length so that a chunk boundary cannot be shifted by the content before it. Two properties follow.

The key depends on the *encoded* payload and the decode configuration, never on decoded content. Deriving it has to be cheaper than decoding, or the cache cannot save anything: header parsing at most, never rasterization. Different encodings of the same picture are therefore different items, unless an EXIF `ImageID` identifies them as one (see [Hashing](#hashing)).

The digest is sha256 rather than the configurable `MMHasherAlgorithm`. The key is part of what identifies an item, so it must not vary with which optional hash packages happen to be installed.

Derivation is linear in the payload, so building refs on an event loop is offloaded to the media thread pool, in `MediaConnector._load_refs_async`.

`DecodeSpec` holds every setting that parameterizes the decode or that mutates the media relative to its bytes: target image mode and RGBA background, audio backend, video backend and frame count, and so on. Each `MediaIO` subclass reports its own through `get_decode_spec()`. The spec snapshots settings as canonical JSON bytes; reading `settings` returns a fresh copy. Unsupported values and non-finite numbers are rejected. Because the whole resolved configuration sits inside the key, one place decides what media identity means and it cannot drift from the key.

### Decoding, mapping and release

`decode()` is memoized and lock-guarded, so concurrent callers decode exactly once. Exceptions from the decoder propagate unchanged; wrapping them is the caller's job.

`map(transform, **settings)` takes a transform from `_T` to `_U` and returns a `MediaRef[_U]` decoding to `transform(self.decode())`. It shares the parent's bytes rather than copying them. Supplied `settings` extend the spec and derive a new key from the parent's *key*, without re-digesting the payload; with no settings, the spec and key are preserved. The parse layer uses this to defer resampling, channel normalization and video metadata validation to the parallel decode phase.

`release()` clears this ref's `data` and drops its decode closure, after any in-flight decode has finished. The closure can pin the bytes and, for images, the header-opened PIL image whose `fp` holds a second copy. Decoded media remains readable after release; an undecoded ref raises `RuntimeError` from `decode()` after release.

`fork()` creates an independently releasable wrapper that borrows the parent's decode operation, bytes, spec and key. Processing owns these wrappers and leaves caller-supplied refs reusable.

Release affects this ref's ownership of the payload. A mapped ref does not release a separately retained parent, because another consumer may still need that parent. Decoded results can also retain encoded bytes, such as a video's `original_video_bytes` metadata or a raw-bytes decoder's result. Those owners control the remaining lifetime of the payload.

### Pickling

Pickling materializes the media. `__reduce__` hands the decoded object to `_media_ref_from_pickle`, so the restored ref is already decoded and keeps its key and spec but carries no encoded bytes. The decoder is a closure over the payload and the `MediaIO` that produced it, and neither that nor the lock can cross a process boundary.

### No implicit decoding

`MediaRef` forwards nothing. It defines `__slots__`, `__repr__` and `__reduce__` and no `__getattr__`, `__iter__`, `__getitem__` or `__array__`, so a reference never decodes by accident — not through logging, not through attribute access, not by being handed to numpy. Callers that want the media say so with `decode()`.

## Decoder resolution

`BaseProcessingInfo.get_media_io(modality, media_io_kwargs)` in `vllm/multimodal/processing/context.py` resolves the decoder for a modality. Resolution lives there because that layer has the model config: the video backend bound to a model's HF video processor class is chosen there, with an explicit `video_backend` from `--media-io-kwargs` or the request winning. The request parser asks the model's processing info for a decoder (`BaseMultiModalItemTracker.get_media_io`) and hands it to the connector.

`MediaConnector` in `vllm/multimodal/media/connector.py` is transport only. It owns URL scheme dispatch (`data:`, `http(s):`, `file:`), the allowed-domain check, the encoded-size cap, the on-disk download cache, the redirect policy and error normalization — and nothing about how bytes decode. Its `fetch_image`, `fetch_video` and `fetch_audio`, with their `_async` variants, take a caller-supplied `MediaIO` and return a `MediaRef`.

Custom `MEDIA_CONNECTOR_REGISTRY` plugins must update their constructor and fetch overrides for this API change. Remove `media_io_kwargs` from the constructor, accept the caller-supplied `media_io` argument in every `fetch_*` override (including async variants), and return a `MediaRef`. Video fetch overrides receive the resolved decoder instead of a `video_processor` argument. Delegate to `load_from_url` or `load_from_url_async` with that decoder to preserve the processor's decode configuration and cache identity. Overrides of the shared video/audio fetch methods must also preserve both supplied decoders.

### One download, two decoders

`fetch_video_and_audio` and `fetch_video_and_audio_async` serve one download to two decode specs. `use_audio_in_video` reads the audio track out of the video payload, so building both refs from a single fetch is what keeps the URL from being downloaded twice. The shared download takes the strictest of the decoders' size caps (`_strictest_max_bytes`) and the video fetch timeout.

### Offline helpers

`fetch_image`, `fetch_video` and `fetch_audio` in `vllm/multimodal/utils.py` are the offline entry points, and they `decode()` before returning, so offline callers get plain objects. The serving path calls `MediaConnector.fetch_*` directly and stays lazy.

### Model-declared decoding

A model that wants encoded bytes rather than decoded frames says so through the same hook. `Dots3NoteProcessingInfo.get_media_io` in `vllm/models/dots3_note/common/processor.py` returns `Dots3NoteVideoMediaIO` for video, whose `load_bytes` is the identity and whose spec declares `video_decode: "raw_bytes"`; the HF processor re-opens the container itself. Declaring this as the decoder instead of reaching for `MediaRef.data` while processing keeps the choice inside the spec and therefore inside the key, so a cache hit and a cache miss hand HF the same thing.

## Hashing

`MultiModalHasher.serialize_item` in `vllm/multimodal/hasher.py` has one `MediaRef` branch, ahead of every other, which yields the ref's precomputed `key`. Hashing fetched media never decodes it. The remaining branches cover media that offline user code supplies directly — PIL images, tensors, arrays, bytes and strings, with a pickle fallback — where there is no encoded payload to key off.

`ProcessorInputs.get_mm_hashes` in `vllm/multimodal/processing/inputs.py` drops `media_io_kwargs` from the hash factors for a `MediaRef`, because the decode spec is already inside the key and hashing it again would count it twice. That holds only when the client supplied no UUID for the item: with a UUID the key is never consulted, so every factor still has to be hashed in.

### Images

`ImageMediaIO.load_bytes_ref` parses the image header eagerly and defers only the pixels. The header alone supplies the `VLLM_MAX_IMAGE_PIXELS` check, so an oversized image still fails at fetch time, and the EXIF `ImageID`, which becomes the ref's key when present. Key derivation must never rasterize, and `getexif()` does: on a PNG with no `exif` entry in `info` it reads to EOF looking for a trailing eXIf chunk, which decodes the image. `_exif_key` therefore declines to probe such PNGs, and they key off their bytes. A header that will not parse is left for decode time, where its failure surfaces with the other decode failures.

## The parse layer

`vllm/multimodal/parse.py` gives two views of a batch item.

- `get(index)` returns the decoded object. HF-side helpers such as `is_valid_image` do `isinstance` checks and cannot be handed a reference, so everything the HF processor sees comes through here. By the time items are consumed the batch decode has run, making this a cache read.
- `get_raw(index)` returns the item as stored, which for fetched media is the `MediaRef` itself. Hashing, cache-miss selection and byte release all read through `get_raw`, so none of them pays for a decode. `ModalityDataItems.get_raw` defaults to `get`, so the two views coincide for embeddings and for media the user already decoded.

The parsers prepare fetched refs without decoding them. `_parse_audio_data` maps resampling and channel normalization onto the ref; `_parse_image_data` leaves refs alone, since mode conversion happens inside the image decoder; `_get_video_with_metadata` preserves `(frames, metadata)` in the raw video ref and defers the `video_needs_metadata` check to decode time.

### Selecting parsed items

`ModalityDataItems.select(indices)` selects existing parsed items without repeating validation or normalization. It preserves their concrete class and raw refs. Video selections also select the corresponding metadata entries; dictionary embeddings select their field items together. Cache lookup uses this operation instead of feeding parsed items through `parse_mm_data()` again, so each miss keeps its existing decode transforms and cache key.

Selections retain original request positions through `get_original_index(index)`, including when a selection is selected again. Decode submission uses those positions for errors, so compacting a miss list does not renumber the corrupt item.

### Video views

`VideoProcessorItems.get(index)` returns the view expected by HF: frames alone, or `(frames, metadata)` when `video_needs_metadata` is enabled. `get_frames(index)` always returns the frames, and frame size/count helpers use it regardless of the HF view. `get_metadata(index)` resolves one video's metadata without decoding its siblings. The existing `metadata` property resolves the metadata list for consumers that need the whole batch.

Keeping the raw tuple and exposing these views separately lets cache-miss selection preserve metadata even when HF consumes frames alone. It also lets audio extraction read metadata before requesting frames through `get()`.

## Decode orchestration

Two module-level functions and a processor method in `vllm/multimodal/processing/processor.py` handle decode submission and waiting.

- `_submit_ref_decodes` walks the raw items of every `ProcessorBatchItems` modality, submits each undecoded ref's `decode()` to the shared media thread pool (`global_thread_pool`, sized by `VLLM_MEDIA_LOADING_THREAD_COUNT`), and records `(modality, original_index, future)` triples. It does not wait.
- `_collect_ref_decodes` joins every future — in-flight work is never abandoned — and raises the first failure.
- `_decode_ref_items` composes the two, and is the blocking fallback for items that turn out to need decoding late.

### Two-phase apply

`MultiModalApplyState` carries the inputs, timing context, hashes and decode futures between the phases, and owns cleanup of the parsed refs supplied to processing. Phase 1 hashes, looks up the cache and submits decodes without joining them, so the worker that runs it stays free; the futures are drained off that worker through `wait_decodes()` or `wait_decodes_async()`; phase 2 runs the HF processor and merges the cache.

`ProcessorInputs.can_use_cache` centralizes eligibility for the synchronous and split paths. A cache must be present, and no item may supply passthrough data. Otherwise phase 1 decodes all refs and phase 2 uses the no-cache path, computing hashes after HF processing.

`apply()` is the synchronous composition of the same three steps, via `_cached_apply_hf_processor` and `_build_mm_input`; `apply_phase1` and `apply_phase2` expose the split. `BaseRenderer._process_multimodal_async` uses the split, running both phases on `_mm_executor` and awaiting `wait_decodes_async()` on the event loop in between, so the worker interleaves other requests' phases while the media pool decodes. `supports_two_phase_apply` gates that: it holds only while the processor leaves `apply`, `_cached_apply_hf_processor` and `_apply_hf_processor` unoverridden, and otherwise the renderer uses `_process_multimodal_blocking_async`.

The blocking renderer path and phase 1 both call `_prepare_multimodal_inputs`. This helper resolves the processor, creates a request ID, parses media and UUIDs, applies UUID validation and generation, selects the renderer cache (including `skip_mm_cache`), and retrieves the timing context. Each path then calls its processor entry point under `set_default_torch_num_threads()`.

The unified `vision_chunk` modality is resolved earlier, in the request parser, because its chunk items must be concrete before they reach the renderer. `_predecode_vision_chunk_items` in `vllm/entrypoints/chat_utils.py` decodes those refs concurrently on the same media pool, off the event loop, and raises the first failure instead of letting it be swallowed downstream.

### The single multimodal worker

`BaseRenderer._mm_executor` has one worker. Two independent constraints require it.

1. The frontend `mm_processor_cache`'s `get_and_update` must stay ordered with the engine-core cache's, so both caches evict in the same order and the frontend can predict what the core holds without asking. That is the protocol documented on `BaseMultiModalCache` in `vllm/multimodal/cache/base.py`, and it holds only if one thread at a time runs the cached apply.
2. `_process_multimodal` can enter numba's workqueue parallel region, through Kimi K2.5 vision preprocessing in `vllm/transformers_utils/processors/kimi_k25_vision_fused.py`. The workqueue layer is not thread-safe and aborts the whole process on concurrent access, so the background multimodal warmup is submitted to this same executor to make overlap physically impossible.

The comment above `_mm_warmup_future` in `vllm/renderers/base.py` is the authority on both.

### Byte lifecycle

Refs remain intact between phases. Phase 2 re-derives the cache state rather than reusing phase 1's, because other requests' phases run in between: a miss may have become a hit, and a hit may have been evicted. An evicted hit needs its retained payload for decoding, so releasing a hit in phase 1 would turn that eviction race into a request failure.

`MultiModalApplyState` is a context manager. Synchronous processing and the async renderer each own one context covering the decode wait and processing. On exit, `close()` drains submitted work before releasing the owned wrappers. Refs keep their payloads through HF processing, field construction, prompt-update construction and cache merging. Selected misses share those same wrappers and need no separate release pass. Phase 1 closes the state itself if it fails before returning ownership to its caller.

If decoding fails before phase 2, the wait methods drain submitted decodes and the owning context releases the state's refs before propagating the failure. `wait_decodes_async()` shields the futures from cancellation. If its caller cancels, it waits for all submitted decodes, including through repeated cancellation, then propagates cancellation through the owning context, which releases the refs. This keeps payloads alive while worker threads may still be using them.

State cleanup leaves decoded results readable and does not release separately retained parent refs. Releasing the state's refs therefore does not guarantee that every copy or owner of the encoded payload has disappeared.

## Error surface

A decode failure becomes `VLLMUnprocessableEntityError`, which the API server maps to HTTP 422, carrying the modality and the item's index within that modality. It reports no `parameter`: the processor cannot see which request field the media arrived in, and `audio_url` and `input_audio` both reach it as modality `"audio"`, so any field name would be a guess. The index within the modality is what locates the item for the client.

The wrapping is what makes the status 422. A corrupt payload raises `ValueError` from its decoder, and an unwrapped `ValueError` falls through to the generic 400 `BadRequest` fallback in `vllm/entrypoints/serve/exception_handling/error_response.py`.

## Limitations

- `supports_two_phase_apply` is a runtime reflection probe comparing three methods (`apply`, `_cached_apply_hf_processor`, `_apply_hf_processor`) against the base implementations. A processor that overrides one of them falls back to the blocking path and logs the overriding methods once at debug level. The probe covers only the methods it names.
- The single-worker `_mm_executor` serializes the HF processor call across all concurrent requests. Only the decode wait is lifted off it.
- The connector's on-disk download cache and the download-time size cap it negotiates apply to `http(s)` URLs alone. A `file:` URL shared by two decoders is read once per decoder and a `data:` URL is base64-decoded once per decoder; per-decoder guards such as audio's encoded-size check stay eager on every scheme.
