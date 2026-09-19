# Lazy Media Decode Design

> Goal: move media **decoding** from the chat parsing layer into the mm
> processor, while **downloading** stays in the chat parsing layer.
> Cache-hit items are never decoded; cache-miss items are decoded in parallel
> inside the processor.
> Repo: vllm fork, branch `mm/lazy-decode-in-processor`. Line numbers refer to
> the current code.

## 0. Verified facts and corrections to earlier findings

1. **Where concurrent fetch happens**: `asyncio.gather` lives in
   `AsyncMultiModalItemTracker.resolve_items`
   (`vllm/entrypoints/chat_utils.py:906-945`), not inside
   `AsyncMultiModalContentParser`. The parser only pushes `partial(coro)`
   objects into the tracker (chat_utils.py:1276-1280 etc.).
2. **Decode errors currently surface as 400, not 422**:
   `_wrap_media_fetch_error` (connector.py:58-125) only wraps **download**
   errors (4xx / invalid URL / oversize). Decode errors
   (`ImageMediaIO.load_bytes` raises `ValueError`, image.py:93-94) are not
   wrapped and fall through to the `ValueError -> 400 BadRequest` fallback in
   `vllm/entrypoints/serve/exception_handling/error_response.py:69-72`. This
   design explicitly maps decode errors to `VLLMUnprocessableEntityError`
   (422) — an **intentional behavior change** (400 -> 422) that must be called
   out in the PR description.
3. **`replace_vision_chunk_video_placeholder` (renderers/hf.py:962-985) does
   not touch decoded frames**; it only reads the `prompt` string in chunk
   dicts. The actual render-time decode points are in the chat parsing layer's
   `_resolve_vision_chunk_items` (chat_utils.py:730-789): the image branch
   reads `data.media` (:753-754), and the video branch calls
   `mm_processor.split_video_chunks(video_data)` (:772).
4. **`split_video_chunks` has no in-tree implementation** (repo-wide grep
   finds only the two duck-typed call sites chat_utils.py:764/772); it is a
   hook reserved for external (Kimi) processors. See open questions in §7.
5. **The hasher's video branch triggers decode**:
   `MultiModalHasher.serialize_item` (hasher.py:100-107) compares
   `frames.nbytes < len(original_bytes)` and hashes decoded frames when they
   are smaller — this accesses `.media`. The image branch (hasher.py:88-98)
   also accesses `.media` for EXIF ImageID. Lazy objects therefore need a
   **dedicated hasher branch** that never triggers decode.
6. **`_get_cache_missing_items` (processor.py:1275-1314) fetches items at
   :1301 via `mm_data_items[modality][idx]`**, and
   `ProcessorBatchItems.__getitem__ -> get -> _unwrap` (parse.py:118-126)
   unwraps `.media` — once lazy, this would trigger serial decode ahead of
   time. This is a critical integration point that must change.
7. **EC/EPD never transport raw media**: `ec_connector/example_connector.py`
   only ships `MMMeta` (mm_hash + token counts); only processed tensors cross
   processes. Lazy objects never leave the API server process.
8. A pickle regression test for `MediaWithBytes` exists
   (tests/multimodal/media/test_base.py, upstream issue #30818), but no clear
   production pickle path was found in-tree; see open questions.

Verified key facts (consistent with earlier findings):

- `MediaWithBytes` (media/base.py:16-55): eager dataclass, `__getattr__`
  delegation, `__array__/__iter__/__getitem__` passthrough.
- `global_thread_pool` (connector.py:44-47, 8 workers) offloads download-side
  work; `MediaConnector.fetch_*` (connector.py:468-597).
- The mm processor runs in the API server process on
  `BaseRenderer._mm_executor` (renderers/base.py:115-118, **single worker,
  intentionally ordered**, see the #38418 comment).
- `ProcessorInputs.get_mm_hashes` (processing/inputs.py:31-106);
  `media_io_kwargs` is already a hash factor (:46-71).
- Data normalization in `parse.py`: audio resampling (:696-700) + channel
  normalization (:702-705), image RGB conversion (:730-735, PIL only), video
  metadata unpacking (:648-665, :782-787).
- With `use_audio_in_video`, the same URL is fetched once more for audio
  (chat_utils.py:1178-1186).
- pixtral/voxtral `fetch_images`/`fetch_audio`
  (transformers_utils/processors/pixtral.py:50-62, voxtral.py:60-106) are
  HF-compatible duck-types that receive already-decoded objects; they do not
  go through the connector.
- The GPU video decode memory pool `mm_ipc_gpu_memory_gb`
  (multimodal/gpu_ipc_memory.py) is initialized in the API server process
  (renderers/base.py:141-150) — decoding stays in the same process after the
  move, so this is unaffected.

## 1. Core type design

### 1.1 Recommendation: new sibling class `LazyMedia`; do not modify `MediaWithBytes`

Two options were considered:

- **Option A (make `MediaWithBytes` lazy, or subclass it)**: `MediaWithBytes`
  is a dataclass with `media` as a field. Shadowing the field with a property
  hits a string of problems: dataclass-generated `__repr__`/`__eq__` call
  `getattr(self, "media")` -> **logging/comparison accidentally triggers
  decode**; `__getstate__` copying `__dict__` sweeps the decode callable into
  pickles. The only benefit is saving a few `isinstance` branches.
- **Option B (new standalone class, recommended)**: clean semantics; zero
  risk to eager paths (offline PIL, embeds, the pickle regression test).
  `isinstance(x, MediaWithBytes)` sites number only ~10 repo-wide and are
  highly concentrated (hasher.py:88/100, parse.py:118-120/229-232/414/652/721,
  plus type aliases in inputs.py) — manageable.

### 1.2 `LazyMedia` API shape (vllm/multimodal/media/base.py)

```python
class LazyMedia(Generic[_T]):
    """Holds encoded bytes + a decode callable; decodes on first access.

    Duck-type compatible with MediaWithBytes: `.media`, `.original_bytes`,
    `.io_config`, attribute delegation, __array__/__iter__/__getitem__.
    """

    __slots__ = ("_decoder", "_media", "_lock", "_original_bytes", "io_config")

    def __init__(
        self,
        decoder: Callable[[], _T],       # usually partial(media_io.load_bytes, data)
        original_bytes: bytes,
        io_config: dict[str, Any] | None = None,  # eagerly computable for images, see §1.3
    ) -> None: ...

    @property
    def media(self) -> _T: ...           # first access decodes (thread-safe, locked) and caches
    @property
    def original_bytes(self) -> bytes: ...  # returns b"" after release; hashing must precede release
    @property
    def is_decoded(self) -> bool: ...

    def decode(self) -> _T:
        """Trigger decode explicitly; thread-safe; exceptions propagate
        unchanged (wrapping is the caller's job)."""

    def map(self, transform: Callable[[_T], Any]) -> "LazyMedia":
        """Return a new LazyMedia whose decoder is
        `lambda: transform(self.media)`. Shares the same original_bytes
        reference (no copy). Lets the parse layer stack resample/normalize
        transforms (see §2 parse.py)."""

    def release_bytes(self) -> None:
        """Release original bytes once hashing is done and the item is either
        cache-hit or decoded."""

    def __getattr__(self, name): ...      # delegates to self.media (triggers decode)
    def __array__ / __iter__ / __getitem__  # same semantics as MediaWithBytes
    def __repr__(self): ...               # never decodes: <LazyMedia undecoded, N bytes>
    def __reduce__(self): ...
        # Materialize before pickling; rebuild as
        # (MediaWithBytes, (media, bytes, io_config)) so unpickling yields a
        # plain eager MediaWithBytes and no closure is pickled.
```

Key points:

- **Thread safety**: `decode()` uses a `threading.Lock`; concurrent access
  decodes exactly once and caches the result.
- **Error propagation**: exceptions from the decoder propagate unchanged; the
  processor's decode orchestration wraps them uniformly as
  `VLLMUnprocessableEntityError` (see §2 processor.py).
- **`io_config` timing**: today `MediaWithBytes.io_config` is computed at
  decode time based on whether a mode conversion actually happened
  (image.py:96-101) and participates in hashing. With laziness, hashing
  precedes decoding — see the image special case in §1.3.

### 1.3 Image special case: eager header-open + lazy pixel-decode

PIL `Image.open()` is itself lazy (it parses only the header, not pixels).
Images therefore use two phases:

- **At fetch time (inside the existing global_thread_pool, same cost as
  today's single `Image.open`)**: open the header, read `mode`/`size`/EXIF;
  keep the `VLLM_MAX_IMAGE_PIXELS` check here (image.py:83-89, error timing
  unchanged); **eagerly compute `io_config`** from it (`image.mode !=
  target_mode` implies a conversion will happen).
- **At decode time**: `.load()` + `normalize_image` (EXIF transpose,
  image.py:21-25) + `_convert_image_mode` (image.py:64-77).

This preserves today's hashing semantics exactly for the hasher's EXIF
ImageID branch (hasher.py:88-92) and io_config branch (:93-98) — zero cache
key changes for images. The image variant of `LazyMedia` additionally holds
the header-opened PIL object (which references original_bytes anyway, so no
extra memory).

### 1.4 Audio / video: purely lazy; hashing switches to original_bytes

- Audio today hashes the **decoded array** (the hasher has no audio
  `MediaWithBytes` branch); video hashes decoded frames when they are smaller
  (hasher.py:105-106). After the change, both hash `original_bytes` uniformly
  (same bytes + same decode params -> same hash; determinism holds;
  `media_io_kwargs` is already in the hash factors, inputs.py:46-71). **Hash
  values differ from today**, but the processor cache is in-process memory,
  so there is no cross-version compatibility concern. Must be called out in
  the PR description.
- Video metadata (fps/duration/total_num_frames) is only known after
  decoding. Today it participates as a hash factor (parse.py:412-418); after
  the change it is dropped from the hash factors (metadata is a function of
  bytes+params, hence redundant). The `video_needs_metadata` check
  (parse.py:782-787) is deferred to post-decode for lazy items — see §2.

## 2. File-by-file change list (in dependency order)

### 2.1 `vllm/multimodal/media/base.py`

Add `LazyMedia` (§1.2). Leave `MediaWithBytes` untouched. Export `LazyMedia`
from `media/__init__.py`.

### 2.2 `vllm/multimodal/media/image.py` / `audio.py` / `video.py`

- `ImageMediaIO`: split out `open_header(data) -> (PIL.Image, io_config)`
  (eager) and `_decode_pixels(header_image) -> PIL.Image` (lazy);
  `load_bytes` keeps its signature (= composition of the two, for
  offline/tests); add `load_bytes_lazy(data) -> LazyMedia`.
- `AudioMediaIO.load_bytes_lazy(data) -> LazyMedia`: keep
  `_validate_encoded_size` (audio.py:543-550) **eager**, called before
  construction; decoder is `partial(self.load_bytes, data)`.
- `VideoMediaIO.load_bytes_lazy(data) -> LazyMedia`: decoder is
  `partial(self.load_bytes, data)`; for the jpeg_sequence base64 branch
  (video.py:127-191) argument validation (num_frames/frames_indices etc.) can
  stay eager, while the actual per-frame decode + `np.stack` moves into the
  decoder.

### 2.3 `vllm/multimodal/media/connector.py`

- `load_from_url`/`load_from_url_async` (:367-466): network download logic is
  **completely unchanged** (true-async aiohttp, media disk cache, domain/size
  checks). Replace the trailing `media_io.load_bytes(data)`
  (:385/:402/:435-438/:457-458) with `media_io.load_bytes_lazy(data)`
  (constructing a LazyMedia is O(1), so no further offload needed).
- `fetch_image*` (:494-544): remove the `UnidentifiedImageError` try/except
  (:516-518/:542-544) — decode errors no longer happen here; error
  normalization moves to the processor's decode orchestration.
- data URL / file URL branches likewise switch to lazy construction (base64
  decoding and file reads stay as-is, in global_thread_pool).
- `fetch_*_embedding*` untouched (embeds have no decode concept).

### 2.4 `vllm/multimodal/hasher.py`

Add a `LazyMedia` branch in `serialize_item` **before** the `MediaWithBytes`
branch:

- Image variant: reuse the existing EXIF ImageID -> io_config+bytes -> bytes
  logic (the header is already eagerly opened, so no pixel decode is
  triggered).
- Audio/video: `iter_item_to_bytes(modality, obj.original_bytes)` (if bytes
  were already released, treat as a programming error and raise — hashing
  must precede release, which the processor flow guarantees).

### 2.5 `vllm/multimodal/inputs.py`

Add `LazyMedia[...]` to the `ImageItem` (:58), `VideoItem` (:68-73), and
`AudioItem` (:84) type aliases.

### 2.6 `vllm/multimodal/parse.py`

- `ProcessorBatchItems._unwrap` (:118-120) and `EmbeddingItems._unwrap`
  (:229-232): `isinstance(item, (MediaWithBytes, LazyMedia))` (taking
  `.media` on a lazy item triggers decode — by then batch decode has already
  run, so this is just a cache read).
- Add `ProcessorBatchItems.get_raw(index) -> object` (returns
  `self.data[index]` without unwrapping) for cache-miss selection (§2.7).
- `_parse_audio_data` (:667-709): add a `LazyMedia` branch — do not resample
  here; instead `item.map(transform)` where `transform = decode -> resample
  -> channel normalization` (reusing the logic at :696-705). Resampling thus
  also moves into the parallel decode phase.
- `_parse_image_data` (:711-737): add `LazyMedia` to the isinstance gate
  (:721); the conversion list comprehension (:730-735) only applies to PIL,
  so lazy items are naturally skipped (mode conversion already happens inside
  the decoder).
- `_get_video_with_metadata` (:648-665): add a `LazyMedia` branch returning
  `(item.map(unpack frames), None)` **without decoding**; skip the
  `video_needs_metadata` error (:782-787) for lazy items and defer the check
  to post-decode (the decode orchestration raises if `video_needs_metadata`
  and metadata is missing).
- `VideoProcessorItems.get_item_for_hash` (:412-418): lazy items do not
  concatenate metadata (§1.4); the `metadata` list stores `None` for lazy
  items, and processor-side access goes through `_unwrap` (already decoded by
  then).

### 2.7 `vllm/multimodal/processing/processor.py`

- `_get_cache_missing_items` (:1275-1314): change :1301 to
  `mm_data_items[modality].get_raw(idx)` so unwrapping does not trigger
  decode.
- New `_decode_lazy_items(items: MultiModalDataItems) -> None`:
    - Walk raw items of each modality and collect `LazyMedia` instances;
    - Submit `decode()` calls to the decode thread pool (see §7 open question
    1: reuse connector's `global_thread_pool` vs a dedicated pool) **in
    parallel**;
    - Wait for all to finish (same semantics as the gather at
    chat_utils.py:920-929: do not abandon in-flight tasks), collect
    exceptions; wrap the first decode exception as
    `VLLMUnprocessableEntityError` (parameter = the corresponding modality's
    url parameter) and raise. The exception travels: decode thread ->
    `_mm_executor` worker -> `make_async` future -> event loop ->
    `create_error_response` (error_response.py:45-48) -> **422**.
- `_cached_apply_hf_processor` (:1415-1481): insert
  `self._decode_lazy_items(mm_missing_data_items)` after
  `_get_cache_missing_items` (:1439) and before `_apply_hf_processor_main`
  (:1449); immediately `release_bytes()` on **hit** lazy items (hashing is
  done, bytes no longer needed).
- `_apply_hf_processor` (:1377-1413, the no-cache/passthrough path):
  likewise insert a full `_decode_lazy_items` before
  `_apply_hf_processor_main`.
- `apply` (:1725) / `EncDecMultiModalProcessor.apply` (:1809) need no
  signature changes.
- Timing: add a `timing_ctx.record("decode_mm_items")` span around
  `_decode_lazy_items`.

### 2.8 `vllm/entrypoints/chat_utils.py`

- The sync/async parsers (:1034-1445) themselves **do not change**: what the
  tracker stores changes from "decoded objects" to "LazyMedia", with an
  unchanged interface; `_validate_add` count checks (:686-721) are unaffected
  (they only count).
- `_resolve_vision_chunk_items` (:730-789): `data.media` (:754) and
  `split_video_chunks` (:772) trigger decode. Plan:
    - **Async path**: in `AsyncMultiModalItemTracker.resolve_items`
    (:906-945), after gather completes and before `_resolve_items`, if
    `use_unified_vision_chunk_modality` and any vision_chunk group contains
    lazy items, pre-materialize them concurrently via
    `loop.run_in_executor(global_thread_pool, item.decode)` to avoid
    synchronous decode on the event-loop thread.
    - **Sync path**: inline decode inside `_resolve_vision_chunk_items` is fine
    (the sync parser is fully blocking today — no regression).
    - Note that :785 already has an `except Exception` fallback; after the
    change, decode errors would be swallowed there into a warning + append.
    `LazyMedia` decode errors must **not** be swallowed by that fallback
    (re-raise `VLLMUnprocessableEntityError`), otherwise corrupt video
    becomes a silent error.

### 2.9 `vllm/renderers/base.py`

`_process_multimodal` (:846-884) needs no flow change: parse (:863) -> hash
-> decode orchestration all happen inside the processor. Optional: align
`_mm_timing_registry` span names.

### 2.10 `vllm/multimodal/utils.py` / `vllm/benchmarks/datasets/datasets.py`

- `fetch_image`/`fetch_video`/`fetch_audio` (utils.py:330-390): update return
  type annotations to the lazy type; duck-type compatible (`.media` /
  attribute delegation). `encode_image_url` (utils.py:79-92) triggers decode
  when accessed via `ImageMediaIO._convert_image_mode` — benchmarks
  (datasets.py:427-428) behave correctly, decode just moves to the use site;
  no code change needed.
- Docstrings should note: in offline user code, decode errors now surface at
  the use site (e.g. `LLM.generate`) instead of the fetch site.

## 3. Compatibility checklist

| Call site | Needs adaptation? | Notes |
| --- | --- | --- |
| `chat_utils.py` sync/async parsers | vision_chunk only (§2.8) | Other paths only move objects through the tracker without touching `.media` |
| `renderers/hf.py` / `mistral.py` etc. | No | Parse results go into `_process_multimodal`; decode happens inside the processor |
| Kimi vision_chunk (`_resolve_vision_chunk_items`) | Yes (§2.8) | Async path needs run_in_executor pre-materialization; errors must not be swallowed by the :785 fallback |
| Offline `LLM.generate` (user passes PIL/ndarray directly) | No | parse.py's PIL/ndarray/tuple branches stay; lazy objects only come from the connector |
| Offline `LLM.chat` (sync parser) | No (behavior change to announce) | Decode error timing moves from fetch to generate |
| pixtral/voxtral duck-type `fetch_images`/`fetch_audio` | No | They receive HF processor inputs, already decoded by then (get_processor_data -> `_unwrap`) |
| `benchmarks/datasets.py:427` | No | See §2.10 |
| EPD / EC connector | No | Only hashes/embeddings cross processes; raw media never leaves the API server process (§0.7) |
| `MediaWithBytes` pickle (test_base.py) | No | The eager class is untouched; `LazyMedia.__reduce__` converts to eager |
| In-model `multi_modal_data={...}` (voxtral_realtime / whisper etc.) | No | They pass decoded arrays through parse's ndarray/tuple branches |
| profiling / dummy inputs | No | Dummy data does not go through the connector |
| `use_audio_in_video` (chat_utils.py:1178-1186) | No (optimizable) | The double-download + double-bytes status quo is unchanged; sharing bytes is a follow-up optimization |

## 4. Test plan

### Existing tests to update

- `tests/multimodal/media/test_connector.py`: change fetch assertions from
  "returns decoded object" to "returns LazyMedia; accessing `.media` yields
  the decoded object".
- `tests/multimodal/media/test_image.py` / `test_audio.py` / `test_video.py`:
  `load_bytes` behavior unchanged; add assertions for `load_bytes_lazy`
  (laziness, decode equivalence).
- `tests/multimodal/test_hasher.py`: add lazy cases — hashing **does not
  trigger decode** (counting decoder asserts 0 calls); image EXIF/io_config
  hashing matches the eager path.
- `tests/multimodal/media/test_unprocessable_entity_error.py`: add a
  decode-error -> 422 `create_error_response` case (noting the 400 -> 422
  change).
- `MediaWithBytes` unwrapping at `tests/conftest.py:1718-1719`,
  `tests/entrypoints/multimodal/openai/chat_completion/test_vision.py:201-202`,
  `tests/entrypoints/pooling/embed/test_online_vision.py:171-172`: make
  compatible with `LazyMedia`.

### New test cases

1. **Cache hit skips decode** (core): extend
   `tests/multimodal/test_cache.py`/`test_processing.py` — a LazyMedia with a
   counting decoder passes through `_cached_apply_hf_processor` twice with
   the same bytes; assert the decode count does not increase and
   `release_bytes` was called.
2. **Parallel decode of cache-miss items**: barrier/event counting asserts
   that multiple miss items' decode calls overlap in time; total latency ~
   max, not sum.
3. **Decode error -> 422**: corrupt bytes -> `processor.apply` raises
   `VLLMUnprocessableEntityError`; plus a renderer-level case going through
   `_mm_executor` to verify cross-thread propagation.
4. **Offline compatibility**: add mixed lazy+PIL+ndarray parse cases to
   `tests/multimodal/test_parse.py`; existing offline model tests must not
   regress.
5. **Concurrent downloads not regressed**: multi-image requests still gather
   concurrently (the existing chat_utils test suites not regressing is
   sufficient; add a loose timing-bounded case if necessary).
6. **vision_chunk lazy path**: construct a tracker with
   use_unified_vision_chunk; assert async resolve does not decode on the
   event-loop thread (e.g. check the thread name inside the decoder).
7. **Bytes lifecycle**: on the hit path, `original_bytes == b""` after
   hashing; on the miss path, bytes are released after decode completes.

## 5. Phased rollout (audio -> image -> video)

- **Phase 0 (foundation)**: `LazyMedia` + hasher branch + parse.py branches +
  `get_raw` + processor `_decode_lazy_items` orchestration + unit tests. The
  connector does not switch in this phase; hand-constructed LazyMedia
  exercises the full pipeline. **Independently mergeable, no behavior
  change.**
- **Phase 1 (audio pilot)**: audio has no `MediaWithBytes` and no
  EXIF/metadata entanglement, so the surface is minimal (one connector change
    - one parse `map`). Validate whisper/voxtral-style models e2e and
  cache-hit-skips-decode. **Feasibility: high**; recommended as the first
  validation.
- **Phase 2 (image)**: introduce the eager header-open mechanism
  (io_config/EXIF/max-pixels semantics fully preserved). Largest coverage and
  largest benefit (images are the most frequent modality).
- **Phase 3 (video)**: most complex — deferred metadata, migrating the
  `video_needs_metadata` check, GPU/NVDEC decode concurrency and interaction
  with the `mm_ipc_gpu_memory_gb` pool, vision_chunk/Kimi paths. Recommend a
  separate PR after Phases 1/2 stabilize.
- Each phase switches the connector per modality independently
  (fetch_audio_lazy -> fetch_image_lazy -> fetch_video_lazy); an env-var
  kill switch for fallback is optional.

## 6. Performance and memory expectations

- Hit items: zero decode, zero pixel memory; bytes released right after
  hashing.
- Miss items: decode moves from "serialized in the download thread pool" to
  "targeted parallelism inside the processor"; per-request parallelism is
  preserved; the `_mm_executor` single worker only orchestrates
  (submit+join) and is not occupied by decoding.
- Peak memory: bytes and decoded media coexist briefly (during decode); bytes
  are released as soon as decode succeeds.

## 7. Open questions

1. **Decode thread pool choice**: reuse the connector's
   `global_thread_pool` (8 workers, contends with downloads) or a dedicated
   pool (e.g. `VLLM_MEDIA_DECODE_THREAD_COUNT`)? GPU video decode
   (NVDEC/torchcodec) parallelism also needs alignment with the
   `mm_ipc_gpu_memory_gb` reservation mechanism (gpu_ipc_memory.py:155-202);
   needs measurement.
2. **The production pickle path for `MediaWithBytes` was not located** (the
   regression test for issue #30818 exists). If a cross-process mm_data
   transfer path does exist, `LazyMedia.__reduce__`'s "convert to eager"
   strategy already covers it; the path should be confirmed.
3. **`split_video_chunks` has no in-tree implementation**: the actual
   signature and threading requirements of this duck-typed hook need
   confirmation with the Kimi model owners; if the external implementation
   assumes it is called on the event-loop thread, the pre-materialization
   plan (§2.8) needs them to adapt.
4. **Audio/video hash value change** (decoded -> original_bytes, §1.4): no
   compatibility issue for the in-process cache, but cross-request dedup
   granularity changes (different encodings of the same audio no longer share
   cache entries — in fact they do not today either, since hash inputs
   already differ); confirm no downstream relies on "different bytes, same
   hash" semantics.
5. **Image io_config edge case**: eager header-open assumes "mode differs =>
   conversion definitely happens" is exactly equivalent to the decode-time
   check (image.py:96-101's `converted is not image`); whether today's
   io_config semantics are precise when `normalize_image`'s EXIF transpose
   changes pixels without changing mode needs a second look.
6. **HTTP status for corrupt media changes 400 -> 422** — intentional
   (§0.2); confirm upstream/clients have no retry logic depending on 400.
7. **`use_audio_in_video` double-downloads the same URL / double bytes**:
   whether to share bytes in this PR or leave it as follow-up.
8. **Sync parser (offline `LLM.chat`) decode-error timing moves later**:
   whether offline users need an escape hatch like `fetch_image(...,
   eager=True)`.
