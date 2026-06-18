<!-- markdownlint-disable MD024 -->
# Cohere ASR Tests

> **Registry**: [`observability_matrix.md`](../observability_matrix.md) entries 7.1.1-7.1.7 |
> **Compatibility**: [`feature_matrix.md`](../feature_matrix.md) section ASR

Validates Cohere speech-to-text serving for preprocess-worker configuration,
short- and long-form transcription correctness, and long-audio streaming with
WER gating.

<details>
<summary>Test case 1: ASR preprocess worker configuration</summary>

## How it runs

1. `run_asr()` invokes the dedicated ASR config pytest node as part of the `asr`
   test group.
   - [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh) -- `run_asr`
   - [`tests/cohere/test_asr_config.py`](../../../../tests/cohere/test_asr_config.py)
2. The first test clears any `VLLM_MAX_AUDIO_PREPROCESS_WORKERS` override,
   patches `os.cpu_count()` above the cap, clears env caching if needed, and
   verifies that the exported default remains fixed at `2`.
   - [`tests/cohere/test_asr_config.py`](../../../../tests/cohere/test_asr_config.py) -- `DEFAULT_NUM_WORKERS`, `test_audio_preprocess_workers_default`
   - [`vllm/envs.py`](../../../../vllm/envs.py) -- `VLLM_MAX_AUDIO_PREPROCESS_WORKERS`
3. The second test patches `ThreadPoolExecutor` and
   `make_async_with_semaphore(...)`, constructs `OpenAISpeechToText`, and
   verifies that frontend speech preprocessing is wired to a dedicated executor
   with `max_workers=2`.
   - [`tests/cohere/test_asr_config.py`](../../../../tests/cohere/test_asr_config.py) -- `test_speech_to_text_preprocess_executor_num_workers`
   - [`vllm/entrypoints/openai/speech_to_text/speech_to_text.py`](../../../../vllm/entrypoints/openai/speech_to_text/speech_to_text.py) -- `OpenAISpeechToText`
4. CI shape: this runs inside the dedicated `asr` test group and is currently
   routed only to the 1xH100 runner.
   - [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json) -- `h100.asr`
   - [`.github/workflows/test-pipeline.yaml`](../../../../.github/workflows/test-pipeline.yaml) -- Docker test execution + JUnit reporting

## Checks

1. **Capped default**: `envs.VLLM_MAX_AUDIO_PREPROCESS_WORKERS` resolves to the
   Cohere-specific default of `2` when unset, even when `os.cpu_count()` is
   higher.
   - `test_audio_preprocess_workers_default`
2. **Executor wiring**: `OpenAISpeechToText` constructs its preprocessing
   executor with `max_workers=2` and passes that executor into the async
   preprocessing wrapper.
   - `test_speech_to_text_preprocess_executor_num_workers`

## Measurements

1. Pytest emits **JUnit XML** for the `asr` group, and
   `dorny/test-reporter@v2` surfaces it in GitHub Actions as an `asr Test
   Report`.
   - `test_asr_config` suite -- `PRESENT`
   - [`.github/workflows/test-pipeline.yaml`](../../../../.github/workflows/test-pipeline.yaml) -- `Test Report` step

No `upload-results` summary JSON is emitted for the `asr` group.

## Compatibility

1. **Input**:
   - Audio (compatible)
2. **Cohere Feature**:
3. **Model Architecture**:
4. **Quantization**:
5. **Hardware**:
   - H100 (compatible)
   - A100 (not tested)
   - B200 (not tested)
   - GB200 (not tested)
   - MI300x (not tested)
6. **vLLM Feature**:

## Implementation

Primary test:
[`tests/cohere/test_asr_config.py`](../../../../tests/cohere/test_asr_config.py)
Runtime paths:
[`vllm/envs.py`](../../../../vllm/envs.py),
[`vllm/entrypoints/openai/speech_to_text/speech_to_text.py`](../../../../vllm/entrypoints/openai/speech_to_text/speech_to_text.py)
CI entry:
[`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh) -- `run_asr`

### Setup

1. **Default cap**: `DEFAULT_NUM_WORKERS = 2` captures the intended Cohere
   default for ASR preprocessing.
2. **Env path**: the config test clears `VLLM_MAX_AUDIO_PREPROCESS_WORKERS`,
   patches CPU count high, and validates the exported env getter path.
3. **Serving path**: the constructor test stubs the transcription model and
   patches the executor factory so it can assert the exact
   `ThreadPoolExecutor(max_workers=2, thread_name_prefix="stt-preprocess")`
   call without loading a checkpoint or starting a server.

</details>

<details>
<summary>Test case 2: Short- and long-form WER correctness</summary>

## How it runs

1. The ASR WER regression file contains both the short-form filtered-dataset
   check and a long-form correctness check against the same served Cohere model.
   - [`tests/cohere/test_asr_wer.py`](../../../../tests/cohere/test_asr_wer.py) -- `test_cohere_transcribe_wer_correctness`, `test_cohere_transcribe_long_audio_wer_correctness`
2. The pytest entry resolves the model to the CI-downloaded local checkpoint at
   `${ENGINES_DIR}/cohere-transcribe-03-2026` when present, otherwise falls
   back to the canonical HF model id, then launches a `RemoteOpenAIServer`
   exposing the served model name `CohereLabs/cohere-transcribe-03-2026`.
   - [`tests/cohere/test_asr_wer.py`](../../../../tests/cohere/test_asr_wer.py) -- `_get_server_model`, `test_cohere_transcribe_wer_correctness`, `test_cohere_transcribe_long_audio_wer_correctness`
   - [`tests/utils.py`](../../../../tests/utils.py) -- `RemoteOpenAIServer`
3. The short-form test reuses the shared ASRDataset-aware loader for
   `D4nt3/esb-datasets-earnings22-validation-tiny-filtered`, while the
   long-form test reuses the shared Earnings22 cleaned dataset loader and
   long-form evaluation helpers.
   - [`tests/entrypoints/openai/correctness/test_transcription_api_correctness.py`](../../../../tests/entrypoints/openai/correctness/test_transcription_api_correctness.py) -- `load_shortform_eval_dataset`, `load_longform_dataset`, `run_evaluation`, `run_longform_evaluation`
   - [`tests/cohere/test_asr_wer.py`](../../../../tests/cohere/test_asr_wer.py)
4. CI shape: this suite is documented under the dedicated `asr` group and is
   currently routed only to the 1xH100 runner.
   - [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json) -- `h100.asr`
   - [`.github/workflows/test-pipeline.yaml`](../../../../.github/workflows/test-pipeline.yaml) -- Docker test execution + JUnit reporting

## Checks

1. **Short-form WER regression**: the computed WER for
   `CohereLabs/cohere-transcribe-03-2026` on the tiny filtered dataset matches
   the expected baseline `11.92` within `atol=1e-1, rtol=1e-2`.
   - `test_cohere_transcribe_wer_correctness`
2. **Long-form WER regression**: the same model on the shared long-form
   Earnings22 correctness slice matches the expected baseline `7.5` within
   `atol=1e-1, rtol=1e-2`.
   - `test_cohere_transcribe_long_audio_wer_correctness`

## Measurements

1. Pytest emits **JUnit XML** for the `asr` group, and
   `dorny/test-reporter@v2` surfaces it in GitHub Actions as an `asr Test
   Report`.
   - `test_asr_wer` suite -- `PRESENT`
   - [`.github/workflows/test-pipeline.yaml`](../../../../.github/workflows/test-pipeline.yaml) -- `Test Report` step

No `upload-results` summary JSON is emitted for the `asr` group.

## Compatibility

1. **Input**:
   - Audio (compatible)
   - Long Context (compatible)
2. **Cohere Feature**:
3. **Model Architecture**:
4. **Quantization**:
5. **Hardware**:
   - H100 (compatible)
   - A100 (not tested)
   - B200 (not tested)
   - GB200 (not tested)
   - MI300x (not tested)
6. **vLLM Feature**:

## Implementation

Primary test:
[`tests/cohere/test_asr_wer.py`](../../../../tests/cohere/test_asr_wer.py)
Shared helpers:
[`tests/entrypoints/openai/correctness/test_transcription_api_correctness.py`](../../../../tests/entrypoints/openai/correctness/test_transcription_api_correctness.py)
CI entry:
[`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh) -- `run_asr`
Runtime helpers:
[`tests/utils.py`](../../../../tests/utils.py)

### Setup

1. **Model resolution**: read `ENGINES_DIR` (default `/root/engines`) and use
   `${ENGINES_DIR}/cohere-transcribe-03-2026` when present, otherwise use
   `CohereLabs/cohere-transcribe-03-2026`.
2. **Server args**: pass `--served-model-name=CohereLabs/cohere-transcribe-03-2026`
   and `--trust-remote-code` when required by the model registry entry.
3. **Short-form dataset**: use the filtered dataset
   `D4nt3/esb-datasets-earnings22-validation-tiny-filtered` through
   `load_shortform_eval_dataset(...)`.
4. **Long-form dataset**: use the shared long-form Earnings22 cleaned dataset
   via `load_longform_dataset(...)`, then evaluate with
   `run_longform_evaluation(...)`.

</details>

<details>
<summary>Test case 3: Long-audio streaming WER</summary>

## How it runs

1. `run_asr()` invokes the long-audio streaming pytest node as part of the same
   `asr` test group.
   - [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh) -- `run_asr`
   - [`tests/cohere/test_asr_long_audio_with_output_streaming.py`](../../../../tests/cohere/test_asr_long_audio_with_output_streaming.py) -- `test_asr_long_audio_with_output_streaming`
2. `download_checkpoints.sh asr` prefetches both the Cohere ASR checkpoint and
   the `longform-audio-transcription` dataset; `test-pipeline.yaml` mounts the
   host data cache into the container at `/root/data`, matching the test's
   default `DATA_DIR`.
   - [`tests/cohere/scripts/download_checkpoints.sh`](../../../../tests/cohere/scripts/download_checkpoints.sh) -- `download_asr`, `download_data_if_missing`
   - [`.github/workflows/test-pipeline.yaml`](../../../../.github/workflows/test-pipeline.yaml) -- `/home/runner/_work/data:/root/data` bind mount
   - [`tests/cohere/test_asr_long_audio_with_output_streaming.py`](../../../../tests/cohere/test_asr_long_audio_with_output_streaming.py) -- `_get_long_audio_repo_dir`
3. The pytest entry launches `RemoteOpenAIServer` with
   `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=50`, then sends three concurrent streaming
   transcription requests against the long-form audio dataset and normalizes the
   outputs with a shared Whisper English normalizer before computing WER.
   - [`tests/cohere/test_asr_long_audio_with_output_streaming.py`](../../../../tests/cohere/test_asr_long_audio_with_output_streaming.py) -- `test_asr_long_audio_with_output_streaming`, `stream_long_audio`
   - [`tests/utils.py`](../../../../tests/utils.py) -- `RemoteOpenAIServer`
4. CI shape: this runs only on the 1xH100 `asr` runner, same as the WER
   case.
   - [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json) -- `h100.asr`
   - [`.github/workflows/test-pipeline.yaml`](../../../../.github/workflows/test-pipeline.yaml) -- Docker test execution + JUnit reporting

## Checks

1. **HTTP success**: every transcription request must return HTTP 200; any
   non-200 response raises an `AssertionError` and fails the test immediately.
   - `test_asr_long_audio_with_output_streaming`
2. **Per-sample WER gate**: each long-audio sample must achieve `WER < 0.5`
   after normalization.
   - `test_asr_long_audio_with_output_streaming`
3. **No silent partial pass**: the gathered async results must contain one WER
   value per concurrent request, and none may be `None`.
   - `test_asr_long_audio_with_output_streaming`

## Measurements

1. Pytest emits **JUnit XML** for the `asr` group, and
   `dorny/test-reporter@v2` surfaces it in GitHub Actions as an `asr Test
   Report`.
   - `test_asr_long_audio_with_output_streaming` -- `PRESENT`
   - [`.github/workflows/test-pipeline.yaml`](../../../../.github/workflows/test-pipeline.yaml) -- `Test Report` step

Local transcription text files (`*_transcription_<n>.txt`) are debug outputs
only and are not uploaded CI measurements.

## Compatibility

1. **Input**:
   - Audio (compatible)
   - Long Context (compatible)
2. **Cohere Feature**:
3. **Model Architecture**:
4. **Quantization**:
5. **Hardware**:
   - H100 (compatible)
   - A100 (not tested)
   - B200 (not tested)
   - GB200 (not tested)
   - MI300x (not tested)
6. **vLLM Feature**:

## Implementation

Primary test:
[`tests/cohere/test_asr_long_audio_with_output_streaming.py`](../../../../tests/cohere/test_asr_long_audio_with_output_streaming.py)
CI entry:
[`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh) -- `run_asr`
Dataset prefetch:
[`tests/cohere/scripts/download_checkpoints.sh`](../../../../tests/cohere/scripts/download_checkpoints.sh) -- `download_asr`
Runtime helpers:
[`tests/utils.py`](../../../../tests/utils.py)

### Setup

1. **Model resolution**: same local-checkpoint-first resolution as the WER
   test via `ENGINES_DIR`.
2. **Dataset resolution**: use `ASR_LONG_AUDIO_DATASET_DIR` when set; otherwise
   read `${DATA_DIR}/longform-audio-transcription` with `DATA_DIR` defaulting to
   `/root/data`.
3. **Server env**: set `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=50` when starting the
   in-test `RemoteOpenAIServer` so the long-form inputs are accepted.
4. **Runtime shape**: run three concurrent streaming requests over the
   long-form dataset and gate each normalized transcript with `WER < 0.5`.

</details>

<details>
<summary>Test case 4: Inter-chunk spacing and language-specific joins</summary>

## How it runs

1. `run_asr()` executes the full inter-chunk spacing pytest file as part of the
   the `asr` group.
   - [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh) -- `run_asr`
   - [`tests/entrypoints/openai/speech_to_text/test_transcription_inter_chunk_spacing.py`](../../../../tests/entrypoints/openai/speech_to_text/test_transcription_inter_chunk_spacing.py)
2. The file mixes pure unit coverage of `asr_inter_chunk_separator()` with
   integration-style serving coverage for `OpenAIServingTranscription`, but uses
   stubbed models and mocked preprocessing so it does not load real ASR model
   weights.
   - [`tests/entrypoints/openai/speech_to_text/test_transcription_inter_chunk_spacing.py`](../../../../tests/entrypoints/openai/speech_to_text/test_transcription_inter_chunk_spacing.py) -- `_StubTranscriptionModel`
   - [`vllm/entrypoints/openai/speech_to_text/speech_to_text.py`](../../../../vllm/entrypoints/openai/speech_to_text/speech_to_text.py) -- `asr_inter_chunk_separator`
   - [`vllm/entrypoints/openai/speech_to_text/serving.py`](../../../../vllm/entrypoints/openai/speech_to_text/serving.py) -- `OpenAIServingTranscription`
3. The serving-side checks cover both streaming SSE assembly and non-streaming
   `create_transcription()`, validating English spacing and Chinese no-space
   joins from the same separator policy.
   - [`tests/entrypoints/openai/speech_to_text/test_transcription_inter_chunk_spacing.py`](../../../../tests/entrypoints/openai/speech_to_text/test_transcription_inter_chunk_spacing.py)
4. CI shape: this file is documented under ASR because it is dispatched from the
   `asr` group, which is currently routed only to the 1xH100 runner.
   - [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json) -- `h100.asr`

## Checks

1. **Separator policy**: `SupportsTranscription.no_space_languages` continues to
   include Japanese and Chinese, and `asr_inter_chunk_separator()` returns the
   expected separator for English, Chinese, Japanese, and the default path.
   - `test_default_no_space_languages_includes_zh_and_ja`
   - `test_asr_inter_chunk_separator_matches_protocol` (parametrized)
   - `test_joined_chunks_english_has_space_between`
   - `test_joined_chunks_chinese_has_no_space_between`
2. **Streaming joins**: `OpenAIServingTranscription.transcription_stream_generator()`
   inserts a space between English chunks but does not insert one for Chinese
   chunks when building streamed SSE deltas.
   - `test_transcription_stream_generator_english_inserts_space_between_chunks`
   - `test_transcription_stream_generator_chinese_no_space_between_chunks`
3. **Non-streaming joins**: `create_transcription()` applies the same
   language-specific join behavior for English and Chinese in the JSON response
   path.
   - `test_create_transcription_non_streaming_joins_chunks_by_language`

## Measurements

1. Pytest emits **JUnit XML** for the `asr` group, and
   `dorny/test-reporter@v2` surfaces it in GitHub Actions as an `asr Test
   Report`.
   - `test_transcription_inter_chunk_spacing` suite -- `PRESENT`
   - [`.github/workflows/test-pipeline.yaml`](../../../../.github/workflows/test-pipeline.yaml) -- `Test Report` step

No benchmark summary JSON is uploaded for this coverage file.

## Compatibility

1. **Input**:
   - Audio (compatible)
2. **Cohere Feature**:
3. **Model Architecture**:
4. **Quantization**:
5. **Hardware**:
6. **vLLM Feature**:

## Implementation

Primary test:
[`tests/entrypoints/openai/speech_to_text/test_transcription_inter_chunk_spacing.py`](../../../../tests/entrypoints/openai/speech_to_text/test_transcription_inter_chunk_spacing.py)
Runtime paths:
[`vllm/entrypoints/openai/speech_to_text/speech_to_text.py`](../../../../vllm/entrypoints/openai/speech_to_text/speech_to_text.py),
[`vllm/entrypoints/openai/speech_to_text/serving.py`](../../../../vllm/entrypoints/openai/speech_to_text/serving.py)
CI entry:
[`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh) -- `run_asr`

### Setup

1. **Stubbed transcription model**: `_StubTranscriptionModel` provides the
   speech-to-text config and no-space language set without importing torch or
   loading a checkpoint.
2. **Mocked serving inputs**: the integration-style tests patch
   `get_model_cls(...)` and `_preprocess_speech_to_text(...)`, then drive
   `OpenAIServingTranscription` with synthetic `RequestOutput` chunks.
3. **Language coverage**: the suite explicitly exercises English (`" "`)
   versus Chinese/Japanese (`""`) separator behavior to guard language-specific
   regressions in chunk joining.

</details>

<details>
<summary>Test case 5: Speech-to-text cancellation propagation</summary>

## How it runs

1. `run_asr()` executes the speech-to-text cancellation pytest file as part of
   the `asr` group.
   - [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh) -- `run_asr`
   - [`tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py`](../../../../tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py)
2. The suite constructs `OpenAISpeechToText` via `__new__`, injects mocked
   engine methods (`generate`, `abort`) and preprocessing outputs, then cancels
   in-flight tasks to verify cleanup behavior without loading a real model.
   - [`tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py`](../../../../tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py)
   - [`vllm/entrypoints/openai/speech_to_text/speech_to_text.py`](../../../../vllm/entrypoints/openai/speech_to_text/speech_to_text.py) -- `OpenAISpeechToText`
3. Coverage spans both outer transcription requests and the language-detection
   request path, including the multi-chunk case where preprocessing fans out to
   per-chunk engine requests.
   - [`tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py`](../../../../tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py)
4. CI shape: this file is documented under ASR because it is dispatched from the
   H100-only `asr` group.
   - [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json) -- `h100.asr`

## Checks

1. **Outer-request aborts**: cancelling non-streaming transcription propagates
   the expected engine request ids and calls `engine_client.abort(...)` for both
   single-chunk and multi-chunk preprocess outputs.
   - `test_non_streaming_cancel_aborts_engine_requests`
2. **Chunk fan-out coverage**: in the multi-chunk case, every chunk generator is
   advanced far enough to acquire its own request id before cancellation.
   - `test_non_streaming_cancel_advances_all_chunk_generators`
3. **Language-detection cleanup**: cancelling `_detect_language(...)` aborts the
   dedicated language-detection request id.
   - `test_language_detection_cancel_aborts_engine_request`

## Measurements

1. Pytest emits **JUnit XML** for the `asr` group, and
   `dorny/test-reporter@v2` surfaces it in GitHub Actions as an `asr Test
   Report`.
   - `test_speech_to_text_cancellation` suite -- `PRESENT`
   - [`.github/workflows/test-pipeline.yaml`](../../../../.github/workflows/test-pipeline.yaml) -- `Test Report` step

No benchmark summary JSON is uploaded for this coverage file.

## Compatibility

1. **Input**:
   - Audio (compatible)
2. **Cohere Feature**:
3. **Model Architecture**:
4. **Quantization**:
5. **Hardware**:
6. **vLLM Feature**:

## Implementation

Primary test:
[`tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py`](../../../../tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py)
Runtime path:
[`vllm/entrypoints/openai/speech_to_text/speech_to_text.py`](../../../../vllm/entrypoints/openai/speech_to_text/speech_to_text.py)
CI entry:
[`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh) -- `run_asr`

### Setup

1. **Mocked engine client**: the suite uses `SimpleNamespace` plus mocked
   `generate`, `abort`, and tracing hooks so cancellation behavior can be
   asserted directly.
2. **Synthetic preprocessing outputs**: `_preprocess_speech_to_text(...)`
   returns either one engine input or several chunk inputs, which lets the tests
   validate request-id generation and abort fan-out.
3. **Task cancellation path**: the tests create `asyncio` tasks around
   `_create_speech_to_text(...)` and `_detect_language(...)`, yield once, then
   cancel and assert `CancelledError` plus abort-side effects.

</details>
