# Offline word timestamps

The native BF16 Nemotron-3.5-Transcribe implementation can align its finished
transcript to the input audio without the NeMo plugin. This opt-in offline
feature returns a start and end time in seconds for each transcription unit.

Install the audio and alignment dependencies:

```bash
uv pip install 'vllm[audio,audio-alignment]'
```

## Generate and align

```python
import soundfile as sf

from vllm import LLM, SamplingParams
from vllm.entrypoints.speech_to_text.offline import transcribe_with_word_timestamps

llm = LLM(
    model="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
    dtype="bfloat16",
    hf_overrides={"enable_ctc_timestamps": True},
    enforce_eager=True,
    enable_prefix_caching=False,
    max_model_len=4096,
    max_num_seqs=1,
    limit_mm_per_prompt={"audio": {"count": 1, "length": 480000}},
    speculative_config={"method": "mtp", "num_speculative_tokens": 2},
)
audio, sample_rate = sf.read("clip.wav", dtype="float32")
prompt = llm.get_tokenizer().apply_chat_template(
    [{"role": "user", "content": (
        "Audio transcription. Language: as spoken. "
        "Wording: verbatim, including fillers and repetitions. "
        "Formatting: punctuation and capitalization. "
        "Output: transcript only. <|audio|>"
    )}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
result = transcribe_with_word_timestamps(
    llm,
    {"prompt": prompt, "multi_modal_data": {"audio": (audio, sample_rate)}},
    SamplingParams(temperature=0, max_tokens=1024),
)[0]
print(result.output.outputs[0].text)
if result.error is not None:
    raise RuntimeError(result.error)
for word in result.words:
    print(word.word, word.start, word.end)
```

A runnable version is available in
[the offline example](../../examples/speech_to_text/offline_word_timestamps.py).

Omit `speculative_config` to use ordinary decoding. Each prompt must contain
one raw audio item and request one completion. Pass a list of prompts to align
a batch; results follow input order. Do not call the helper concurrently or
while the same engine has queued requests.

The helper captures the target audio encoder's state during its original
forward. After generation finishes, it aligns the final accepted transcript,
including when MTP is enabled. The drafter does not load another CTC head.

## Results and language units

`WordTimestampResult.output` is the original `RequestOutput`. Its `words` list
contains `word`, `start` and `end`; times are relative to the input audio. An
empty transcript succeeds with an empty list and `error=None`. Alignment
failures and truncated transcripts retain the generated output and set
`error`; callers must check it before using timestamps. A length-truncated
transcript returns `error="truncated_transcript"` and no timestamps; its CTC
scoring is skipped.

The segmentation and alignment follow the public NeMo algorithm at commit
[`0d1ec2f8`](https://github.com/NVIDIA-NeMo/Speech/commit/0d1ec2f89764422312d6cd0a201fbecf55b78e18).
Chinese and Japanese units are characters rather than semantic words. Thai
uses the reference `newmm` segmentation. Automatic script selection applies to
the transcript as a whole, including mixed-script text. The returned `word`
field therefore does not promise linguistic word segmentation for every language.
The CTC tokenizer has a separate vocabulary from the language model. A valid
transcript can contain a character that the CTC tokenizer cannot represent;
that request returns an explicit alignment error and keeps its transcript.

The alignment search follows the fixed NeMo algorithm. vLLM adapts its inputs
for transcripts without `<spk:N>` tags: it combines all speaker activity into
one speech timeline, so an unlabelled transcript can span speaker changes.
Explicit speaker tags keep their separate identity priors. Raw encoder captures
are preserved. This input policy differs from the original NeMo caller's
single-speaker interpretation of unlabelled text.

Speaker information is used internally; this interface does not return speaker
attribution or diarization segments.

## Resource use and supported execution

The first implementation supports synchronous offline execution on one CUDA
worker with BF16, Model Runner V2 and eager execution. Prefix caching must be
disabled. CPU weight offloading is not supported when CTC timestamps are enabled.
HTTP timestamps, streaming, CUDA Graphs, quantized checkpoints, distributed
execution and speculative methods other than MTP are not covered.
The internal `capture_ctc_timestamps` processor option and alignment handle are
managed by the helper; setting them on a raw `LLM.generate()` call does not
request a supported timestamp result.

The head is loaded only when `enable_ctc_timestamps` is enabled. Captures are
kept in bounded host memory, shared with requests for the same captured audio.
The default 16 GiB host budget includes captures and alignment workspaces;
`ctc_timestamp_max_cpu_bytes` in `hf_overrides` adjusts it. Budget failures
produce an explicit error. Capture owners are released even after a failed
call. An unowned capture can remain while its encoder embedding is cached,
so another timestamp request can reuse it; encoder eviction or reset releases it.

The CTC head runs in windows and copies only the transcript's required score
columns to the host. It does not retain a complete hour-long vocabulary matrix
on the GPU. Ordinary generation does not capture alignment state.

The model limit is one hour per audio. For long inputs, raise both
`max_model_len` (for example, 65536) and the audio sample limit. Size the KV cache
for that context and measure memory for the actual workload. The small example's
30-second limit and context are intentionally independent of long-audio settings.
