# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The attention metadata and n-gram inputs of uniform decode steps, replayed.

The problem
-----------
Model Runner V2 replays a whole decode step's forward as one FULL CUDA graph.
The graph reads its attention metadata from persistent device buffers: the
metadata builders' own (GDN, the PLE short convolution, the QSA indexer
caches), the runner's (query starts, sequence lengths, block tables, slot
mappings) and the mamba state context's aligned state indices. Before every
replay, the host rebuilds that metadata, which rewrites those buffers: for
Qwen3.8-Flash-Next that is eight attention groups of Python, two dozen small
kernels and two host-to-device copies, about 1.5 ms of host time per step, and
at one request per TP4 rank the host, not the GPU, sets the step time. The PLE
n-gram context adds about 0.2 ms more.

When the fast path applies
--------------------------
To a step that is a uniform decode on a FULL graph: every request schedules
`num_speculative_tokens + 1` tokens, all of them its drafts plus one, no
request prefills, adaptive verification is off (`max_query_len` unset) and
the step runs the target model's attention groups (the ones this state
recorded with). Such a step is identified by its key, `(requests, padded
requests, tokens, padded tokens)`. Every other step, and a key that was not
recorded, builds its metadata the stock way.

What is recorded per key, and why it is constant
------------------------------------------------
For a uniform decode step every host value the stock build reads is a function
of the key: the query starts (`tokens_per_request * i`, then the token count
for the padding requests), the draft counts (`num_speculative_tokens` for a
request, -1 for padding), the prefill flags (none), the decode/prefill splits
the builders derive from them, and the slices they take of their buffers. The
only exception is `max_seq_len`, the largest sequence length, which only the
QSA indexer metadata carries, as a host field. So, during CUDA graph capture,
`DecodeMetadataGraphs.record` runs the stock build
(`MambaHybridModelState.prepare_attn`) of every key once eagerly, to compile
its kernels, and once inside a CUDA graph, on a batch of that key whose request
indices are a persistent buffer of this module. It keeps the graph and the
metadata objects the recorded build returned. Their device tensors are views of
the persistent buffers above, at the addresses the forward graphs read; their
host fields are the key's constants.

The stock build stages two host masks (the speculative-request masks of GDN
and of the PLE short convolution) through pinned memory, and a host-to-device
copy waits several microseconds for the host memory. Those masks are constants
of the key, so the recorded build copies them from device tensors made by the
build before the capture instead (`_device_staging`), which the recording
keeps alive; the captured build raises unless it stages the same masks.

What a replay rewrites each step
--------------------------------
`DecodeMetadataGraphs.replay` copies the step's request indices into the
persistent buffer and replays the key's graph, which runs the stock build's
kernels on the step's device inputs: the accepted-token counts of the step's
requests, the aligned mamba state indices of all mamba groups, the GDN and PLE
state indices, speculative masks, query starts and accepted counts (padding
requests get the null block, False, the last query start and one), and the QSA
indexer caches' token rows. On the host it then sets `max_seq_len` on the
recorded objects that carry it and points the GDN builders'
`mamba_aligned_state_indices` at the aligned indices, as the stock build
does, and returns the recorded objects.

The n-gram inputs of such a step (`ngram_context` and the PLE query starts)
are filled by one Triton kernel, `_ngram_inputs_kernel`, instead of the
stock state's ten tensor operations; the result is the same. The kernel does
not specialize on the request counts, and the model state compiles it after
recording (`warm_up_ngram_inputs`), so no live step compiles it.

Overlapping the replay with the forward graph
---------------------------------------------
The replay is about thirty small dependent kernels and copies, 40 to 50 us of
GPU time in which the forward graph, which must follow it, cannot start. But
the forward graph's first kernels (the embedding, its all-reduce, the PLE
prefetch and the first hyper-connection read) read none of the metadata. So,
with `additional_config["decode_metadata_overlap"]` on (the default), the
replay of a key runs on a side stream of this module when the FULL forward
graph of its padded token count waits for it (`_Recording.overlap`), and
the forward graph starts at once on the runner's stream:

- the executor, while capturing a FULL forward graph, makes the capture wait
  (`join`, an external event wait node) for `gdn_ready` before the first
  GDN layer and for `ready` before the first PLE or QSA layer, and at the
  end of the layers at the latest (`CaptureWaits`), then notes the padded
  token count (`joined`); a FULL capture that runs the stock layers notes
  its padded token count as one without the waits (`captured_stock`), and
  recordings of such a count never overlap; `join` raises outside a
  capture or on the side stream;
- every recording records `gdn_ready` (an external event record node) right
  after the build has built the metadata of the last GDN group
  (`_after_gdn_builds`); the build before the capture checks that the rest
  of the build leaves the GDN metadata unchanged, so that the GDN layers may
  run while the rest of the build does;
- an overlapped replay makes the side stream wait for the runner's stream,
  copies the request indices and replays there, and records `ready` after
  the graph.

Several keys share one forward graph, so the build cannot be captured into
it: the graph is captured for its padded size, the build for the key.

Which records the waits see. The host runs ahead of the GPU: it may launch
the replays of steps k + 1 and k + 2, which record both events again, before
the GPU reaches the waits of step k's forward graph. CUDA's rule for
`cudaEventRecord` (CUDA Runtime API, Event Management) is that a wait uses
the event's most recently recorded state at the time of the call that
enqueues it and is not affected by later records; an event record node
records the event at each launch of its graph, and an event wait node waits
for the work captured in the event (Graph Management,
`cudaGraphAddEventRecordNode` and `cudaGraphAddEventWaitNode`). The wait
nodes of a forward graph are enqueued by its launch, so they bind to the
records of the replay launched just before it, step k's, never to a later
step's. The test `test_a_graph_wait_binds_to_the_records_before_its_launch`
checks this on the GPU with the host far ahead. A wait bound when it runs
would instead see step k + 2's records, which follow step k's forward graph
on the side stream: a deadlock.

Why the recordings may share the graph pool
-------------------------------------------
The recordings capture into the global CUDA graph pool, which the forward
graphs also use. `torch.cuda.graph` captures them on torch's side capture
stream; the forward graphs are captured on the runner's capture stream, a
different one. The caching allocator reuses a freed block only for an
allocation on the stream that freed it. So a temporary that a recording
frees can only back an allocation of a later metadata recording, never one
of a forward graph, and no forward graph's temporary backs a recording.
Metadata recordings replay one at a time (a replay on the runner's stream
first waits for the last one on the side stream, `settle`), and a replay's
temporaries are dead once it ends, so two recordings that share a temporary
never run at the same time. The metadata objects a recording returns stay
alive with it, so no later capture reuses their memory.

How the step's work is ordered
------------------------------
The step runs in this order on the runner's current stream (main), except
steps 3 and 4 of an overlapped replay, which run on the side stream (side):

1. the runner's `prepare_inputs` and `prepare_attn`: the host-to-device
   copies of the step's inputs (its request indices, query starts and
   sequence lengths among them), and the gathers of the block tables and
   slot mappings;
2. `preprocess_state`: moves mamba state across block boundaries, advances
   the state indices and resets the accepted-token counts that the replay
   reads;
3. the device-to-device copy of the step's request indices into
   `DecodeMetadataGraphs.idx_mapping`;
4. the replay of the key's recording, which reads all of the above and
   writes the builders' buffers;
5. `_ngram_inputs_kernel`, which writes the n-gram buffers;
6. the replay of the forward graph, which reads the buffers of steps 1 to 5.

Without overlap, stream order makes each step see the writes of the steps
before it, and the next decode step's writes wait for this step's forward
graph. With overlap these invariants keep the same order:

- I1 (fork): side waits for main after step 2 is enqueued, so steps 3-4 see
  steps 1-2 and run after every earlier user of the buffers they write, the
  previous step's forward graph and drafts among them.
- I2 (join): every forward graph of an overlapped key's padded size waits for
  `gdn_ready` before its first GDN layer and for `ready` before its first
  PLE or QSA layer and before its end; the GDN metadata are final at
  `gdn_ready`; the main-stream work between the fork and those waits (the
  n-gram kernel, the embedding, the PLE prefetch and the first
  hyper-connection read) neither reads nor writes what steps 3-4 write or
  read. A graph launch binds its waits to the events' last records before it
  (previous section), so the waits of step k's forward graph see step k's
  replay, never a later one, however far the host runs ahead.
- I3 (completion): main work after the forward graph, the next step and the
  drafts among it, runs after the graph's end and so after `ready`; a stock
  build or a replay on main first waits for `ready` (`settle`) should no
  forward graph have followed an overlapped replay.
- I4 (lifetime): the step's request indices, which main frees, are marked as
  used by side (`record_stream`); the recordings' constants and graphs are
  dropped only after side is synchronized (`release`).
- I5 (memory): a recording's temporaries live in the global graph pool on
  torch's capture stream; the forward graphs capture on the runner's capture
  stream, so no forward graph's memory backs a recording's temporary and the
  two may run at the same time (section on the graph pool).

`tests/models/qwen4_exp/specialized_decode/test_metadata_protocol.py` checks
I1-I4 over every interleaving of the two streams, with the host three steps
ahead, on the operations this module and `CaptureWaits` enqueue, and catches
a missing fork, a missing record, a wait after a reader, a build that changes
the GDN metadata after `gdn_ready`, a missing `record_stream` and waits
bound when they run.

Persistent buffers the fast path relies on
------------------------------------------
- The runner's `InputBuffers.query_start_loc` and `seq_lens`, its
  `BlockTables.input_block_tables` and `slot_mappings`: written by the
  runner before the model state runs; read by the graph.
- `MambaHybridModelState.num_accepted_tokens_gpu` (read) and the mamba
  context's `aligned_state_indices` (written).
- The builders' staging buffers, written by the graph and read by the forward
  graphs: `GDNAttentionMetadataBuilder.spec_state_indices_tensor`,
  `spec_sequence_masks`, `spec_token_indx`, `spec_query_start_loc` and
  `num_accepted_tokens`; the same names but `spec_token_indx` on
  `PleShortConvAttentionMetadataBuilder`; `token_to_req_buffer`,
  `logical_positions_buffer`, `visible_blocks_buffer`,
  `slot_mapping_buffer` and `k_work_metadata_buffer` on
  `QSAIndexerMetadataBuilder`.
- The model state's `ngram_context` and `ple_query_start_loc`.

`tests/models/qwen4_exp/specialized_decode/test_metadata.py` pins these
buffers and shows a replay equal to the stock build at every captured size.
Each recording binds the builders, buffers and attention groups it was made
with: initializing a KV cache (new builders and buffers) drops every
recording, and the next capture records them again.

The runtime check
-----------------
With `additional_config={"decode_metadata_check_steps": N}`, the first N
replays of every key are checked: the replay's outputs (every tensor the
returned metadata references and every device buffer of the target builders
and of the mamba context, the n-gram buffers for the inputs) are saved, those
buffers are restored to their state before the replay, the stock build runs,
and the two results must be equal: the same metadata fields, tensor contents
and addresses. A difference raises.
"""

import dataclasses
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import ModuleType
from typing import Any

import numpy as np
import torch

from vllm.compilation.cudagraph_pool import capture_pool
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.compilation import (
    decode_cudagraph_capture_max_tokens,
    decode_cudagraph_capture_shapes,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.attention.backends import gdn_attn, short_conv_attn
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup

from ....common.qsa import Qwen4ExpQSAMetadataBuilder
from ....common.qsa_indexer import QSAIndexerMetadataBuilder

logger = init_logger(__name__)

# (requests, padded requests, tokens, padded tokens) of a uniform decode step.
Key = tuple[int, int, int, int]
# The stock attention metadata build of a batch.
StockBuild = Callable[[InputBatch], dict[str, Any]]

# The builders whose stock build of a uniform decode batch records: its device
# work writes only persistent buffers and staged copies, and its host
# values follow from the key.
RECORDABLE_BUILDERS = frozenset(
    {
        GDNAttentionMetadataBuilder,
        PleShortConvAttentionMetadataBuilder,
        Qwen4ExpQSAMetadataBuilder,
        QSAIndexerMetadataBuilder,
    }
)

# The modules whose stock build stages host masks through pinned memory.
_PINNED_STAGING_MODULES: tuple[ModuleType, ...] = (gdn_attn, short_conv_attn)


def capture_keys(vllm_config: VllmConfig, decode_query_len: int) -> list[Key]:
    """The keys of the uniform decode steps of `decode_query_len` tokens a
    request (the drafts plus one) that the FULL decode graphs of
    `vllm_config` serve: each request count from one to the largest graph's,
    padded to the smallest captured graph that holds it."""
    config = vllm_config.compilation_config
    max_reqs = vllm_config.get_decode_cudagraph_query_lens(decode_query_len).get(
        decode_query_len, 0
    )
    padded_sizes = [
        num_reqs
        for num_reqs, _ in decode_cudagraph_capture_shapes(
            config.decode_cudagraph_capture_request_sizes or (),
            {decode_query_len: max_reqs},
            decode_cudagraph_capture_max_tokens(config),
        )
    ]
    keys: list[Key] = []
    first = 1
    for padded in padded_sizes:
        keys += [
            (num_reqs, padded, num_reqs * decode_query_len, padded * decode_query_len)
            for num_reqs in range(first, padded + 1)
        ]
        first = padded + 1
    return keys


def uniform_decode_key(
    input_batch: InputBatch, cudagraph_mode: CUDAGraphMode, decode_query_len: int
) -> Key | None:
    """The key of `input_batch` if it is a uniform decode step on a FULL
    graph: every request schedules `decode_query_len` tokens, its drafts plus
    one, none prefills and adaptive verification is off. None otherwise."""
    if (
        cudagraph_mode != CUDAGraphMode.FULL
        or input_batch.has_prefill
        or input_batch.max_query_len is not None
    ):
        return None
    num_reqs = input_batch.num_reqs
    num_drafts = input_batch.num_draft_tokens_per_req
    if (
        num_drafts is None
        or not (input_batch.num_scheduled_tokens[:num_reqs] == decode_query_len).all()
        or not (num_drafts[:num_reqs] == decode_query_len - 1).all()
    ):
        return None
    return (
        num_reqs,
        input_batch.num_reqs_after_padding,
        input_batch.num_tokens,
        input_batch.num_tokens_after_padding,
    )


def unrecordable_builders(attn_groups: Sequence[Sequence[AttentionGroup]]) -> list[str]:
    """The names of the builder classes of `attn_groups` (by KV cache
    group) whose builds do not record."""
    return sorted(
        {
            type(group.get_metadata_builder()).__name__
            for groups in attn_groups
            for group in groups
            if type(group.get_metadata_builder()) not in RECORDABLE_BUILDERS
        }
    )


@contextmanager
def _device_staging(staged: list["_Staged"], capturing: bool) -> Iterator[None]:
    """Make the stock builds' `async_tensor_h2d` copy from device constants.

    Inside, `async_tensor_h2d` of the modules in `_PINNED_STAGING_MODULES`
    copies its host data from a device tensor holding the same values instead
    of a pinned buffer, so a CUDA graph that records the copy holds a
    device-to-device copy (no host-to-device latency) from memory that lives
    as long as `staged`. The data of a recorded key's build are constants of
    the key. The build before the capture (`capturing` False) makes the
    constants, one a call, in call order; the captured build (`capturing`
    True) takes them in the same order and raises unless each call's data
    equal the constant's (the same build of the same batch).
    """
    calls = iter(staged)

    def async_tensor_h2d(
        data: list | np.ndarray | torch.Tensor,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`vllm.utils.torch_utils.async_tensor_h2d` from a device constant."""
        if dtype is None and out is not None:
            dtype = out.dtype
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if isinstance(data, torch.Tensor):
            host = data.to(dtype=dtype or data.dtype, copy=True)
        else:
            host = torch.tensor(data, dtype=dtype)
        target = out.device if out is not None else device
        assert target is not None, "must provide destination tensor or device"
        if capturing:
            constant = next(calls, None)
            if constant is None or not (
                constant.host.dtype == host.dtype
                and constant.host.shape == host.shape
                and torch.equal(constant.host, host)
            ):
                raise RuntimeError(
                    "Qwen4Exp decode metadata: a recorded build staged other "
                    "host data than the build before it."
                )
        else:
            constant = _Staged(host, host.to(device=target))
            staged.append(constant)
        if out is not None:
            return out.copy_(constant.device)
        return constant.device.clone()

    saved = [(module, module.async_tensor_h2d) for module in _PINNED_STAGING_MODULES]
    for module, _ in saved:
        module.__dict__["async_tensor_h2d"] = async_tensor_h2d
    try:
        yield
    finally:
        for module, function in saved:
            module.__dict__["async_tensor_h2d"] = function


@contextmanager
def _after_gdn_builds(
    builders: Sequence[GDNAttentionMetadataBuilder],
    num_builds: int,
    done: Callable[[list[Any]], None],
) -> Iterator[None]:
    """Call `done` right after the stock build has built the metadata of
    its `num_builds` GDN groups, whose builders are `builders`.

    Inside, every call of a builder's `build` or `update_block_table` (the
    stock build calls one of them for each GDN group) counts; the call that
    makes `num_builds` calls `done` with the metadata the calls returned,
    before the build moves on to the next group (with no GDN groups, `done`
    runs before the build). Raises after the build if it did not make exactly
    `num_builds` such calls.
    """
    built: list[Any] = []
    if num_builds == 0:
        done([])

    def counted(method: Callable[..., Any]) -> Callable[..., Any]:
        """`method` that counts its call and calls `done` at the last."""

        def call(*args: Any, **kwargs: Any) -> Any:
            """The builder method, then the count."""
            metadata = method(*args, **kwargs)
            built.append(metadata)
            if len(built) == num_builds:
                done(list(built))
            return metadata

        return call

    for builder in builders:
        builder.__dict__["build"] = counted(builder.build)
        builder.__dict__["update_block_table"] = counted(builder.update_block_table)
    try:
        yield
    finally:
        for builder in builders:
            del builder.__dict__["build"], builder.__dict__["update_block_table"]
    if len(built) != num_builds:
        raise RuntimeError(
            f"Qwen4Exp decode metadata: the stock build built {len(built)} GDN "
            f"group metadata, not {num_builds}."
        )


@dataclass(frozen=True)
class _Staged:
    """Host data a stock build stages, and a device tensor of the same
    values that a recorded build copies from (`_device_staging`)."""

    host: torch.Tensor
    device: torch.Tensor


@dataclass
class _Recording:
    """One key's recorded build: the graph, what it returned, and the host
    state the stock build also sets."""

    graph: torch.cuda.CUDAGraph
    attn_metadata: dict[str, Any]
    # The returned metadata objects with a ``max_seq_len`` host field.
    with_max_seq_len: list[Any]
    # (builder, value) of every builder's ``mamba_aligned_state_indices``.
    aligned_indices: list[tuple[Any, torch.Tensor | None]]
    # Device constants the graph's staged copies read.
    staged: list[_Staged]
    # Whether the replay runs on the side stream: the FULL forward graph of
    # the key's padded token count waits for it.
    overlap: bool
    # Replays checked against the stock build so far.
    checked: int = 0
    inputs_checked: int = 0


@dataclass(frozen=True)
class _TensorRecord:
    """A tensor's layout, address and contents at one point of a check."""

    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    device: torch.device
    # Device tensors with elements: the address the forward graphs read.
    data_ptr: int | None
    contents: torch.Tensor | None


def _record_tensor(tensor: torch.Tensor) -> _TensorRecord:
    """The layout, address and a copy of `tensor`."""
    empty = tensor.numel() == 0
    return _TensorRecord(
        dtype=tensor.dtype,
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        device=tensor.device,
        data_ptr=None if empty or tensor.device.type == "cpu" else tensor.data_ptr(),
        contents=None if empty else tensor.clone(),
    )


def _flatten(value: Any, path: str, out: dict[str, Any], seen: dict[int, str]) -> None:
    """Record `value` under `path`: tensors by `_record_tensor`,
    dataclasses, mappings and sequences by element (an object already recorded
    by the path it was first recorded under), others as they are."""
    if isinstance(value, torch.Tensor):
        out[path] = _record_tensor(value)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        if id(value) in seen:
            out[path] = ("same object as", seen[id(value)])
            return
        seen[id(value)] = path
        out[path] = type(value)
        for field in dataclasses.fields(value):
            _flatten(getattr(value, field.name), f"{path}.{field.name}", out, seen)
    elif isinstance(value, dict):
        out[path] = (dict, tuple(value))
        for name, item in value.items():
            _flatten(item, f"{path}[{name!r}]", out, seen)
    elif isinstance(value, (list, tuple)):
        out[path] = (type(value), len(value))
        for index, item in enumerate(value):
            _flatten(item, f"{path}[{index}]", out, seen)
    elif isinstance(value, np.ndarray):
        out[path] = (np.ndarray, value.dtype, value.shape, value.tobytes())
    else:
        out[path] = value


def snapshot(attn_metadata: dict[str, Any], buffers: Sequence[torch.Tensor]) -> dict:
    """Every field of `attn_metadata` (metadata by layer name) and the
    contents of the device `buffers`, for `differences`."""
    out: dict[str, Any] = {}
    _flatten(attn_metadata, "metadata", out, {})
    for index, buffer in enumerate(buffers):
        out[f"buffer[{index}]"] = _record_tensor(buffer)
    return out


def differences(fast: dict, stock: dict, limit: int = 8) -> list[str]:
    """Where the `snapshot` of a replay (`fast`) and of the stock build
    (`stock`) differ, at most `limit` places."""
    found: list[str] = []
    for path in sorted(fast.keys() | stock.keys()):
        if len(found) >= limit:
            break
        if path not in fast or path not in stock:
            found.append(f"{path}: only in {'fast' if path in fast else 'stock'}")
            continue
        a, b = fast[path], stock[path]
        if isinstance(a, _TensorRecord) and isinstance(b, _TensorRecord):
            layout_a = (a.dtype, a.shape, a.stride, a.device, a.data_ptr)
            layout_b = (b.dtype, b.shape, b.stride, b.device, b.data_ptr)
            if layout_a != layout_b:
                found.append(f"{path}: layout {layout_a} != {layout_b}")
            elif a.contents is not None and not torch.equal(a.contents, b.contents):
                rows = (a.contents != b.contents).nonzero()[:4].tolist()
                found.append(f"{path}: contents differ at {rows}")
        elif type(a) is not type(b) or a != b:
            found.append(f"{path}: {a!r} != {b!r}")
    return found


def _device_buffers(objects: Sequence[Any]) -> list[torch.Tensor]:
    """The CUDA tensors among the attributes of `objects`, each once."""
    buffers: dict[int, torch.Tensor] = {}
    for obj in objects:
        for value in vars(obj).values():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                buffers.setdefault(id(value), value)
    return list(buffers.values())


class CaptureWaits:
    """The waits of one FULL forward graph capture for the overlapped replay
    (invariant I2 of the module docstring).

    The executor calls `before_gdn` before each GDN layer, `before_all`
    before each PLE or QSA layer, and `finish` after the layers: the first
    call of each kind places its wait (`DecodeMetadataGraphs.join`), a wait
    for the whole build also covers the GDN metadata, and `finish` places the
    wait for the whole build if no layer has, then notes the capture
    (`DecodeMetadataGraphs.joined`). Without `metadata` (the setting off,
    or no replayed metadata) nothing is placed.
    """

    def __init__(self, metadata: "DecodeMetadataGraphs | None") -> None:
        """The waits of one capture for the replays of `metadata`."""
        self._metadata = metadata
        self._gdn = self._all = metadata is None

    def before_gdn(self) -> None:
        """Before a GDN layer: wait for the GDN groups' metadata."""
        if not self._gdn:
            assert self._metadata is not None
            self._metadata.join(gdn_only=True)
            self._gdn = True

    def before_all(self) -> None:
        """Before a PLE or QSA layer: wait for the whole build."""
        if not self._all:
            assert self._metadata is not None
            self._metadata.join(gdn_only=False)
            self._gdn = self._all = True

    def finish(self, rows: int) -> None:
        """After the layers of a capture of `rows` padded tokens: wait for
        the whole build if no layer has, and note the capture."""
        self.before_all()
        if self._metadata is not None:
            self._metadata.joined(rows)


class DecodeMetadataGraphs:
    """Records the stock attention metadata build of every uniform decode key
    during CUDA graph capture, and replays it.

    `decode_query_len` is the tokens a request schedules in a decode step
    (the drafts plus one), `max_num_reqs` the most requests of a step, the
    first `check_steps` replays of every key are checked against the stock
    build, and `overlap` lets replays run on the side stream (module
    docstring); without it every replay runs on the current stream.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        decode_query_len: int,
        max_num_reqs: int,
        device: torch.device,
        check_steps: int = 0,
        overlap: bool = True,
    ) -> None:
        self._vllm_config = vllm_config
        self.decode_query_len = decode_query_len
        self.check_steps = check_steps
        self.overlap = overlap
        # The request indices the recorded builds read; each replay copies the
        # step's into it.
        self.idx_mapping = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        self._recordings: dict[Key, _Recording] = {}
        self._attn_groups: list[list[AttentionGroup]] | None = None
        # Device buffers a stock build may write: restored before the stock
        # build of a check.
        self._buffers: list[torch.Tensor] = []
        # The stream that overlapped replays run on, and the events that the
        # FULL forward graphs wait for: ``gdn_ready`` once the build has built
        # the GDN groups' metadata (an event record node of every recording),
        # ``ready`` once the whole build has run (recorded after the replay).
        self._stream = torch.cuda.Stream(device=device)
        self.gdn_ready = torch.cuda.Event(external=True)
        self.ready = torch.cuda.Event(external=True)
        # Torch creates an event at its first record, and a wait for an event
        # that does not exist yet does nothing, also while capturing.
        self.gdn_ready.record()
        self.ready.record()
        # The padded token counts whose FULL forward graph captures wait for
        # both events (``join``, ``joined``), and those with a FULL capture
        # that ran the stock layers, so without the waits (``captured_stock``).
        self._joined: set[int] = set()
        self._stock_captures: set[int] = set()
        # Whether the last replay ran on the side stream and the current
        # stream has not waited for it since.
        self._pending = False

    @property
    def keys(self) -> list[Key]:
        """The recorded keys."""
        return list(self._recordings)

    @property
    def overlapped_keys(self) -> list[Key]:
        """The recorded keys whose replays run on the side stream."""
        return [key for key, recording in self._recordings.items() if recording.overlap]

    def join(self, gdn_only: bool) -> None:
        """While a FULL forward graph captures: make the capture wait for the
        last overlapped replay, for its GDN groups' metadata (`gdn_only`) or
        for all of it (an external event wait node). The executor calls it,
        through `CaptureWaits`, before the first layer that reads that
        metadata. Raises outside a capture or on the side stream: the wait
        must be a node of the forward graph."""
        stream = torch.cuda.current_stream()
        if not torch.cuda.is_current_stream_capturing() or stream == self._stream:
            raise RuntimeError(
                "Qwen4Exp decode metadata: the wait for the replay must be "
                "captured into the forward graph on the runner's stream."
            )
        stream.wait_event(self.gdn_ready if gdn_only else self.ready)

    def joined(self, rows: int) -> None:
        """Note that the FULL forward graph of `rows` padded tokens, just
        captured, waits for both events before its layers read the metadata:
        recordings of that padded token count replay on the side stream,
        unless the setting is off or another FULL capture of `rows` ran the
        stock layers."""
        self._joined.add(rows)

    def captured_stock(self, rows: int) -> None:
        """Note that a FULL forward graph of `rows` padded tokens was just
        captured with the stock layers, so without the waits: recordings of
        that padded token count replay on the current stream."""
        self._stock_captures.add(rows)

    def forget_joins(self) -> None:
        """Forget the captured forward graphs (new ones will be captured)."""
        self._joined.clear()
        self._stock_captures.clear()

    def overlaps(self, rows: int) -> bool:
        """Whether recordings of `rows` padded tokens replay on the side
        stream: the setting is on, a FULL capture of `rows` placed the
        waits and none ran the stock layers."""
        return (
            self.overlap and rows in self._joined and rows not in self._stock_captures
        )

    def settle(self) -> None:
        """Make the current stream wait for the last overlapped replay, if it
        has not yet: before a stock build or a replay on the current stream
        writes the builders' buffers. Outside a capture only (a wait would
        become part of the captured graph)."""
        if self._pending and not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream().wait_event(self.ready)
            self._pending = False

    def key(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        attn_groups: list[list[AttentionGroup]],
    ) -> Key | None:
        """The key of `input_batch` if its build replays a recording: a
        uniform decode step of a recorded key that builds the recorded
        `attn_groups`. None otherwise."""
        if attn_groups is not self._attn_groups:
            return None
        key = uniform_decode_key(input_batch, cudagraph_mode, self.decode_query_len)
        if key is not None and key not in self._recordings:
            logger.warning_once(
                "Qwen4Exp specialized decode: a uniform decode step has no "
                "recorded attention-metadata build; building it eagerly."
            )
            return None
        return key

    def record(
        self,
        keys: Sequence[Key],
        input_buffers: InputBuffers,
        attn_groups: list[list[AttentionGroup]],
        buffers: Sequence[torch.Tensor],
        stock_build: StockBuild,
        max_model_len: int,
    ) -> None:
        """Record `stock_build` (the stock build of `attn_groups`) of a
        batch of every key in `keys`, which reads the runner's
        `input_buffers`. `buffers` are the device buffers outside the
        builders that the build writes (the mamba context's aligned state
        indices), restored before the stock build of a check. The batches'
        sequence length bound is `max_model_len`, the worst case, should a
        build size work by it."""
        self.release()
        builders = [
            group.get_metadata_builder() for groups in attn_groups for group in groups
        ]
        gdn_builders = list(
            {
                id(builder): builder
                for builder in builders
                if isinstance(builder, GDNAttentionMetadataBuilder)
            }.values()
        )
        num_gdn = sum(
            isinstance(builder, GDNAttentionMetadataBuilder) for builder in builders
        )
        for key in keys:
            batch = self._decode_batch(key, input_buffers, max_model_len)
            staged: list[_Staged] = []
            gdn_built: list[Any] = []
            at_gdn_ready: list[dict] = []

            def gdn_done(
                built: list[Any],
                kept: list[Any] = gdn_built,
                snapshots: list[dict] = at_gdn_ready,
            ) -> None:
                """Keep the GDN metadata and a snapshot of them."""
                kept[:] = built
                snapshots.append(snapshot({"gdn": built}, []))

            # Compile, allocate lazily, warm up; make the staged constants.
            with (
                _device_staging(staged, capturing=False),
                _after_gdn_builds(gdn_builders, num_gdn, gdn_done),
            ):
                stock_build(batch)
            # The GDN layers of an overlapped replay wait only for
            # ``gdn_ready``: the rest of the build must not change what their
            # metadata hold.
            final = snapshot({"gdn": gdn_built}, [])
            if self.overlap and (found := differences(at_gdn_ready[0], final)):
                raise RuntimeError(
                    f"Qwen4Exp decode metadata: the build of {key} changes the "
                    "GDN metadata after building them: " + "; ".join(found)
                )
            graph = torch.cuda.CUDAGraph()
            with (
                _device_staging(staged, capturing=True),
                _after_gdn_builds(
                    gdn_builders, num_gdn, lambda built: self.gdn_ready.record()
                ),
                capture_pool(
                    current_platform.get_global_graph_pool(), self._vllm_config
                ) as pool,
                torch.cuda.graph(graph, pool),
            ):
                attn_metadata = stock_build(batch)
            self._recordings[key] = _Recording(
                graph=graph,
                attn_metadata=attn_metadata,
                with_max_seq_len=list(
                    {
                        id(m): m
                        for m in attn_metadata.values()
                        if hasattr(m, "max_seq_len")
                    }.values()
                ),
                aligned_indices=[
                    (builder, builder.mamba_aligned_state_indices)
                    for builder in builders
                    if hasattr(builder, "mamba_aligned_state_indices")
                ],
                staged=staged,
                overlap=self.overlaps(key[3]),
            )
        self._attn_groups = attn_groups
        self._buffers = _device_buffers(builders) + [
            buffer
            for groups in attn_groups
            for group in groups
            if (buffer := group.kernel_block_table) is not None
        ]
        self._buffers += list(buffers)

    def replay(
        self, key: Key, input_batch: InputBatch, stock_build: StockBuild
    ) -> dict[str, Any]:
        """Replay the recorded build of `key` for `input_batch` and return
        its metadata. `stock_build` builds the step's metadata the stock way;
        it runs only on the checked replays."""
        recording = self._recordings[key]
        if recording.checked < self.check_steps:
            recording.checked += 1
            return self._checked_replay(key, recording, input_batch, stock_build)
        return self._replay(recording, input_batch, recording.overlap)

    def _replay(
        self, recording: _Recording, input_batch: InputBatch, overlap: bool
    ) -> dict:
        """Replay `recording` and set the host state of `input_batch`; on
        the side stream if `overlap`, else on the current stream."""
        num_reqs = input_batch.num_reqs
        if overlap:
            current = torch.cuda.current_stream()
            self._stream.wait_stream(current)
            # The step's request indices are freed on the current stream.
            input_batch.idx_mapping.record_stream(self._stream)
            with torch.cuda.stream(self._stream):
                self.idx_mapping[:num_reqs].copy_(input_batch.idx_mapping)
                recording.graph.replay()
                self.ready.record()
            self._pending = True
        else:
            self.settle()
            self.idx_mapping[:num_reqs].copy_(input_batch.idx_mapping)
            recording.graph.replay()
        num_reqs_padded = input_batch.num_reqs_after_padding
        max_seq_len = int(input_batch.seq_lens_cpu_upper_bound[:num_reqs_padded].max())
        for metadata in recording.with_max_seq_len:
            metadata.max_seq_len = max_seq_len
        for builder, indices in recording.aligned_indices:
            builder.mamba_aligned_state_indices = indices
        return recording.attn_metadata

    def _checked_replay(
        self,
        key: Key,
        recording: _Recording,
        input_batch: InputBatch,
        stock_build: StockBuild,
    ) -> dict[str, Any]:
        """Replay, then run the stock build from the same buffer state and
        raise if the two differ."""
        self.settle()
        before = [buffer.clone() for buffer in self._buffers]
        attn_metadata = self._replay(recording, input_batch, overlap=False)
        fast = snapshot(attn_metadata, self._buffers)
        fast["aligned_indices"] = self._aligned_indices(recording)
        for buffer, saved in zip(self._buffers, before):
            buffer.copy_(saved)
        expected = stock_build(input_batch)
        stock = snapshot(expected, self._buffers)
        stock["aligned_indices"] = self._aligned_indices(recording)
        found = differences(fast, stock)
        if found:
            raise RuntimeError(
                f"Qwen4Exp decode metadata check: the replay of {key} differs "
                "from the stock build: " + "; ".join(found)
            )
        if recording.checked in (1, self.check_steps):
            logger.info(
                "Qwen4Exp decode metadata check: %d of %d replays of %s equal the "
                "stock build.",
                recording.checked,
                self.check_steps,
                key,
            )
        return attn_metadata

    @staticmethod
    def _aligned_indices(recording: _Recording) -> list[tuple | None]:
        """Where the builders' `mamba_aligned_state_indices` point."""
        return [
            None
            if (indices := builder.mamba_aligned_state_indices) is None
            else (indices.data_ptr(), tuple(indices.shape), tuple(indices.stride()))
            for builder, _ in recording.aligned_indices
        ]

    def check_inputs(self, key: Key) -> bool:
        """Whether the n-gram inputs of this replayed step of `key` are
        checked; counts the step if so."""
        recording = self._recordings[key]
        if recording.inputs_checked >= self.check_steps:
            return False
        recording.inputs_checked += 1
        return True

    def release(self) -> None:
        """Drop the recordings and the buffers they bind, once the last
        overlapped replay is done with them."""
        self._stream.synchronize()
        self._pending = False
        self._recordings.clear()
        self._attn_groups = None
        self._buffers = []

    def _decode_batch(
        self, key: Key, input_buffers: InputBuffers, max_model_len: int
    ) -> InputBatch:
        """A batch of `key`: the host values of every uniform decode step of
        the key, the runner's persistent device buffers, this module's request
        indices, and `max_model_len` as its sequence length bound."""
        num_reqs, num_reqs_padded, num_tokens, num_tokens_padded = key
        query_len = self.decode_query_len
        batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
        query_start_loc_np = np.full(num_reqs_padded + 1, num_tokens, dtype=np.int32)
        query_start_loc_np[: num_reqs + 1] = np.arange(num_reqs + 1) * query_len
        seq_lens_bound = torch.zeros(num_reqs_padded, dtype=torch.int32)
        seq_lens_bound[:num_reqs] = max_model_len
        return replace(
            batch,
            num_reqs_after_padding=num_reqs_padded,
            num_tokens_after_padding=num_tokens_padded,
            idx_mapping=self.idx_mapping[:num_reqs],
            num_scheduled_tokens=np.full(num_reqs, query_len, dtype=np.int32),
            num_draft_tokens=num_reqs * (query_len - 1),
            num_draft_tokens_per_req=np.full(num_reqs, query_len - 1, dtype=np.int32),
            query_start_loc=input_buffers.query_start_loc[: num_reqs_padded + 1],
            query_start_loc_np=query_start_loc_np,
            seq_lens=input_buffers.seq_lens[:num_reqs_padded],
            seq_lens_cpu_upper_bound=seq_lens_bound,
            is_prefilling_np=np.zeros(num_reqs, dtype=bool),
            has_prefill=False,
            prefill_runs_as_decode_np=None,
            input_ids=input_buffers.input_ids[:num_tokens_padded],
            positions=input_buffers.positions[:num_tokens_padded],
            is_padding=input_buffers.is_padding[:num_tokens_padded],
            max_query_len=None,
        )


# The request counts change from step to step and the history row length is
# the runner's, so none of them is specialized on: one compile, at capture
# (``warm_up_ngram_inputs``), serves every step.
@triton.jit(do_not_specialize=["num_reqs", "num_reqs_padded", "token_ids_stride"])
def _ngram_inputs_kernel(
    idx_mapping_ptr,  # [num_reqs] batch row -> request state row
    num_computed_ptr,  # [max_num_reqs] tokens computed per request state row
    token_ids_ptr,  # [max_num_reqs, max_model_len] token history
    token_ids_stride,
    context_ptr,  # [max_num_reqs, CONTEXT_LEN] n-gram context, written
    context_stride,
    query_start_loc_ptr,  # [num_reqs_padded + 1] the runner's query starts
    ple_query_start_loc_ptr,  # [max_num_reqs + 1] PLE query starts, written
    num_reqs,
    num_reqs_padded,
    max_num_reqs,
    eos_token_id,
    CONTEXT_LEN: tl.constexpr,
    BLOCK_CONTEXT: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    # The last CONTEXT_LEN tokens before each live request's first query token;
    # EOS before the sequence start and in every other row.
    live = rows < num_reqs
    req = tl.load(idx_mapping_ptr + rows, mask=live, other=0).to(tl.int64)
    end = tl.load(num_computed_ptr + req, mask=live, other=0).to(tl.int64)
    cols = tl.arange(0, BLOCK_CONTEXT)
    in_context = cols < CONTEXT_LEN
    token_index = end[:, None] - CONTEXT_LEN + cols[None, :]
    valid = live[:, None] & in_context[None, :] & (token_index >= 0)
    tokens = tl.load(
        token_ids_ptr + req[:, None] * token_ids_stride + token_index,
        mask=valid,
        other=eos_token_id,
    )
    tl.store(
        context_ptr + rows[:, None].to(tl.int64) * context_stride + cols[None, :],
        tokens,
        mask=(rows < max_num_reqs)[:, None] & in_context[None, :],
    )
    # The runner's query starts, then its last one for every unused request.
    in_starts = rows <= max_num_reqs
    starts = tl.load(
        query_start_loc_ptr + tl.minimum(rows, num_reqs_padded), mask=in_starts
    )
    tl.store(ple_query_start_loc_ptr + rows, starts, mask=in_starts)


def fill_ngram_inputs(
    input_batch: InputBatch,
    num_computed_tokens: torch.Tensor,
    token_ids: torch.Tensor,
    ngram_context: torch.Tensor,
    ple_query_start_loc: torch.Tensor,
    eos_token_id: int,
) -> None:
    """Write the n-gram inputs of `input_batch` in one kernel, as
    `Qwen4ExpModelState.prepare_inputs` does: row i of `ngram_context`
    (`[max_num_reqs, ngram_size - 1]`) gets the tokens of `token_ids` (the
    history by request state row) before request i's first query token
    (`num_computed_tokens`), `eos_token_id` where there are none and in the
    rows of no request; `ple_query_start_loc` (`[max_num_reqs + 1]`) gets
    the batch's query starts, then its last one."""
    _launch_ngram_inputs(
        input_batch.idx_mapping,
        input_batch.query_start_loc,
        input_batch.num_reqs,
        input_batch.num_reqs_after_padding,
        num_computed_tokens,
        token_ids,
        ngram_context,
        ple_query_start_loc,
        eos_token_id,
    )


def warm_up_ngram_inputs(
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    ple_query_start_loc: torch.Tensor,
    eos_token_id: int,
) -> None:
    """Compile `_ngram_inputs_kernel` before serving, so that no live step
    compiles it: launch it once for one request with no history, on stand-ins
    of the request state with its dtypes (int64 request indices, int32
    computed-token counts and token history), the runner's
    `query_start_loc` buffer and the n-gram buffers `fill_ngram_inputs`
    writes. The launch leaves EOS in every `ngram_context` row; every
    replayed step rewrites both n-gram buffers before its forward reads
    them."""
    device = ngram_context.device
    _launch_ngram_inputs(
        torch.zeros(1, dtype=torch.int64, device=device),
        query_start_loc,
        1,
        1,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(1, 1, dtype=torch.int32, device=device),
        ngram_context,
        ple_query_start_loc,
        eos_token_id,
    )


def _launch_ngram_inputs(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_reqs: int,
    num_reqs_padded: int,
    num_computed_tokens: torch.Tensor,
    token_ids: torch.Tensor,
    ngram_context: torch.Tensor,
    ple_query_start_loc: torch.Tensor,
    eos_token_id: int,
) -> None:
    """Launch `_ngram_inputs_kernel` for a batch of `num_reqs` requests
    (`num_reqs_padded` with padding) at request state rows `idx_mapping`
    and query starts `query_start_loc`."""
    max_num_reqs, context_len = ngram_context.shape
    assert token_ids.stride(1) == 1 and ngram_context.stride(1) == 1
    block_rows = 16
    _ngram_inputs_kernel[(triton.cdiv(max_num_reqs + 1, block_rows),)](
        idx_mapping,
        num_computed_tokens,
        token_ids,
        token_ids.stride(0),
        ngram_context,
        ngram_context.stride(0),
        query_start_loc,
        ple_query_start_loc,
        num_reqs,
        num_reqs_padded,
        max_num_reqs,
        eos_token_id,
        CONTEXT_LEN=context_len,
        BLOCK_CONTEXT=triton.next_power_of_2(context_len),
        BLOCK_ROWS=block_rows,
    )


def supported(has_triton: bool = HAS_TRITON) -> bool:
    """Whether the fast path's kernels can run here (CUDA and Triton)."""
    return has_triton and current_platform.is_cuda()
