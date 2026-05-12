# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepStream decode pool — GStreamer pipelines + CUDA frame capture.

N pipelines run on N daemon threads inside a single process, all
sharing one CUDA context. Each pipeline's buffer probes copy decoded
NVMM frames into a CUDA tensor allocated from PyTorch's caching
allocator and the caller uses that tensor directly — no D2H/H2D
round-trip, no IPC handle, no per-decode tensor reconstruction.

:class:`DecodePool` is the file-decode pool. Workers pre-build a
``filesrc → parsebin → nvv4l2decoder → nvvideoconvert →
capsfilter[NVMM RGB] → fakesink`` pipeline and swap source URIs
between requests. ``parsebin`` auto-routes H.264, H.265, and the
containers wrapping them (MP4, MKV, MPEG-TS).
"""

from __future__ import annotations

import bisect
import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


# ----------------------------------------------------------------------
# DeepStream runtime preload (matches deepstream_runtime/decode_demo.py)
# ----------------------------------------------------------------------
# Loaded at module import — both the parent and any spawned workers re-run
# this when they import the module, so GStreamer's nv* plugins can resolve
# their nvbufsurface / nvds* deps without LD_LIBRARY_PATH manipulation.
def _preload_deepstream_libs() -> None:
    try:
        import deepstream_runtime as dsr
    except ImportError:
        return
    lib_dir = Path(dsr.lib_dir())
    for soname in (
        "libnvbuf_fdmap.so",
        "libnvbufsurface.so",
        "libnvbufsurftransform.so",
        "libnvdsbufferpool.so",
        "libnvdsgst_helper.so",
        "libnvdsgst_meta.so",
        "libnvds_meta.so",
        "libgstnvdsseimeta.so",
        "libgstnvcustomhelper.so",
        "libcuvidv4l2.so",
        "libnvv4l2.so",
    ):
        path = lib_dir / soname
        if path.exists():
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                logger.warning("[ds_decode_pool] preload failed: %s (%s)",
                               path, e)
    os.environ["GST_PLUGIN_PATH"] = (
        dsr.gst_plugin_dir() + os.pathsep
        + os.environ.get("GST_PLUGIN_PATH", "")
    )


_preload_deepstream_libs()


# ----------------------------------------------------------------------
# NvBufSurface ctypes layout
# ----------------------------------------------------------------------
# Mirrors the public ``nvbufsurface.h`` definitions for DeepStream 9.x.
# We only need to read pitch + dataPtr from surfaceList[0]; the trailing
# fields are present so sizeof() lines up with the C struct, but their
# values are ignored.
_NVBUF_MAX_PLANES = 4
_STRUCTURE_PADDING = 4


class _NvBufSurfacePlaneParams(ctypes.Structure):
    _fields_ = [
        ("num_planes",  ctypes.c_uint32),
        ("width",       ctypes.c_uint32 * _NVBUF_MAX_PLANES),
        ("height",      ctypes.c_uint32 * _NVBUF_MAX_PLANES),
        ("pitch",       ctypes.c_uint32 * _NVBUF_MAX_PLANES),
        ("offset",      ctypes.c_uint32 * _NVBUF_MAX_PLANES),
        ("psize",       ctypes.c_uint32 * _NVBUF_MAX_PLANES),
        ("bytesPerPix", ctypes.c_uint32 * _NVBUF_MAX_PLANES),
        ("_reserved",   ctypes.c_void_p * (_STRUCTURE_PADDING
                                           * _NVBUF_MAX_PLANES)),
    ]


class _NvBufSurfaceMappedAddr(ctypes.Structure):
    _fields_ = [
        ("addr",      ctypes.c_void_p * _NVBUF_MAX_PLANES),
        ("eglImage",  ctypes.c_void_p),
        ("nvmmPtr",   ctypes.c_void_p),
        ("cudaPtr",   ctypes.c_void_p),
        ("_reserved", ctypes.c_void_p * _STRUCTURE_PADDING),
    ]


class _NvBufSurfaceParams(ctypes.Structure):
    _fields_ = [
        ("width",       ctypes.c_uint32),
        ("height",      ctypes.c_uint32),
        ("pitch",       ctypes.c_uint32),
        ("colorFormat", ctypes.c_int),
        ("layout",      ctypes.c_int),
        ("bufferDesc",  ctypes.c_uint64),
        ("dataSize",    ctypes.c_uint32),
        ("dataPtr",     ctypes.c_void_p),
        ("planeParams", _NvBufSurfacePlaneParams),
        ("mappedAddr",  _NvBufSurfaceMappedAddr),
        ("paramex",     ctypes.c_void_p),
        ("cudaBuffer",  ctypes.c_void_p),
        ("_reserved",   ctypes.c_void_p * _STRUCTURE_PADDING),
    ]


class _NvBufSurface(ctypes.Structure):
    _fields_ = [
        ("gpuId",         ctypes.c_uint32),
        ("batchSize",     ctypes.c_uint32),
        ("numFilled",     ctypes.c_uint32),
        ("isContiguous",  ctypes.c_bool),
        ("memType",       ctypes.c_int),
        ("surfaceList",   ctypes.POINTER(_NvBufSurfaceParams)),
        ("isImportedBuf", ctypes.c_bool),
        ("_reserved",     ctypes.c_void_p * _STRUCTURE_PADDING),
    ]


# ----------------------------------------------------------------------
# Request / Result dataclasses (picklable)
# ----------------------------------------------------------------------
@dataclass
class _DecodeRequest:
    job_id: int
    uri: str
    target_pts_ns: tuple = ()       # sorted nanosecond PTS targets (file mode)
    target_indices: tuple = ()      # sorted segment-relative indices (RTSP)
    use_pts_mode: bool = True       # True = file/PTS, False = RTSP/index
    max_frames: int = 8
    timeout_sec: float = 30.0


@dataclass
class _DecodeResult:
    job_id: int
    worker_id: int
    n_kept: int = 0
    n_total: int = 0
    fps: float = 0.0
    error: str = ""
    # CUDA tensor (N, H, W, 3) uint8 written by the worker's CUDA stream.
    # Caller uses it directly — same address space.
    frames: Any = None


# ----------------------------------------------------------------------
# Common worker state — pipeline + probe context + CUDA stream
# ----------------------------------------------------------------------
class _BaseWorkerState:
    """Shared probe / capture logic for both file and RTSP workers."""

    def __init__(self, worker_id: int, drop_interval: int):
        self.worker_id = worker_id
        self.drop_interval = max(0, drop_interval)
        self.pipeline = None
        self.elements: dict[str, Any] = {}
        self._Gst = None
        self._gst_imported = False

        # CUDA / cudart bindings — created lazily after torch.cuda.init().
        self._cudart = None
        self._cuda_stream = ctypes.c_void_p(0)

        # Per-decode mutable probe state — reset by _reset_for_decode.
        self.target_pts: tuple[int, ...] = ()
        self.target_indices: tuple[int, ...] = ()
        self.use_pts_mode = True
        self.max_frames = 0
        self.kept = 0
        self.total = 0
        self.target_cursor = 0
        self.early_eos = False
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.has_error = False
        self.err_msg = ""

        # Pre-allocated destination tensor for this decode (N, H, W, 3).
        # Allocated on the first kept frame — H/W are unknown until then.
        self.dst_tensor: torch.Tensor | None = None
        self.dst_ptr = 0
        self.frame_bytes = 0
        self.dst_pitch = 0

        # GOP-aware parser drop state (file mode only).
        self.gop_initialized = False
        self.gop_i_pts = 0
        self.gop_duration = 0
        # Cached "any unmatched target falls in this GOP" decision, set
        # on each I-frame and re-evaluated on deltas as the select probe
        # consumes targets. Defaults to True until the first I-frame so
        # we don't drop an opening fragment of P-frames before any I-frame
        # has been parsed.
        self.gop_has_target = True
        # Inferred frame interval (next-delta-pts − last-i-frame-pts).
        # Reset on every I-frame, set on the first delta after it.
        self.frame_duration = 0

        # True once the pipeline has actually streamed (PLAYING run to
        # EOS or a prior decode loop). Used by the file worker to skip
        # the no-op FLUSH-seek to byte 0 on a freshly-built pipeline,
        # avoiding the seek/PAUSED-transition race.
        self.pipeline_has_streamed = False

        # RTSP-only "decode complete" signalling. Live pipelines never
        # accept EOS; the count probe sets this once max_frames is hit.
        self._done_event = None

    # ------------------------------------------------------------------
    # Lazy GStreamer / cudart setup
    # ------------------------------------------------------------------
    def _ensure_gst(self):
        if self._gst_imported:
            return self._Gst
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # type: ignore
        Gst.init(None)
        self._Gst = Gst
        self._gst_imported = True
        return Gst

    def _ensure_cudart(self):
        if self._cudart is not None:
            return self._cudart
        lib = ctypes.CDLL("libcudart.so")
        # cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch,
        #     const void* src, size_t spitch, size_t width, size_t height,
        #     enum cudaMemcpyKind kind, cudaStream_t stream);
        lib.cudaMemcpy2DAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_int,    ctypes.c_void_p,
        ]
        lib.cudaMemcpy2DAsync.restype = ctypes.c_int
        lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.cudaStreamCreate.restype = ctypes.c_int
        lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        lib.cudaStreamSynchronize.restype = ctypes.c_int
        if lib.cudaStreamCreate(ctypes.byref(self._cuda_stream)) != 0:
            raise RuntimeError("cudaStreamCreate failed")
        self._cudart = lib
        return lib

    # ------------------------------------------------------------------
    # Per-decode reset
    # ------------------------------------------------------------------
    def _reset_for_decode(self, req: _DecodeRequest) -> None:
        self.target_pts = tuple(sorted(req.target_pts_ns))
        self.target_indices = tuple(sorted(req.target_indices))
        self.use_pts_mode = req.use_pts_mode
        self.max_frames = max(1, req.max_frames)
        self.kept = 0
        self.total = 0
        self.target_cursor = 0
        self.early_eos = False
        self.fps = 0.0
        # Width/height are sticky across decodes for the same pipeline —
        # only clear them when the source URI changes (caller's job).
        self.has_error = False
        self.err_msg = ""
        self.dst_tensor = None
        self.dst_ptr = 0
        self.frame_bytes = 0
        self.dst_pitch = 0
        self.gop_initialized = False
        self.gop_i_pts = 0
        self.gop_duration = 0
        self.gop_has_target = True
        self.frame_duration = 0
        if self._done_event is not None:
            self._done_event.clear()

    # ------------------------------------------------------------------
    # Probe callbacks — all accept the GIL implicitly
    # ------------------------------------------------------------------
    def _parser_probe(self, pad, info, _ud):
        """Parser src pad — drop GOPs containing no target PTS.

        On each I-frame we cache whether any unmatched target falls in
        the upcoming GOP and reuse that decision on the deltas — avoids
        a bisect per buffer. Deltas re-evaluate too: as the select
        probe consumes targets via ``target_cursor``, a GOP that
        started with a target may run out of them mid-GOP, in which
        case we drop the trailing deltas.
        """
        Gst = self._Gst
        if self.early_eos:
            return Gst.PadProbeReturn.DROP
        if not self.use_pts_mode or not self.target_pts:
            return Gst.PadProbeReturn.OK

        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK

        pts = buf.pts
        is_keyframe = not (buf.get_flags() & Gst.BufferFlags.DELTA_UNIT)

        if is_keyframe:
            if self.gop_initialized and pts > self.gop_i_pts:
                gap = pts - self.gop_i_pts
                if gap > self.gop_duration:
                    self.gop_duration = gap
            self.gop_i_pts = pts
            self.gop_initialized = True
            # Frame interval is reinferred from the first delta of the
            # new GOP — codecs occasionally vary it across GOPs.
            self.frame_duration = 0
            self.gop_has_target = self._gop_contains_target(pts)
            return Gst.PadProbeReturn.OK

        # First delta after the I-frame: estimate frame interval. Used
        # below to ignore targets that land in the very last frame slot
        # of this GOP — they're matched by the next GOP's I-frame anyway.
        if self.frame_duration == 0 and self.gop_initialized:
            self.frame_duration = pts - self.gop_i_pts

        if not self.gop_initialized or self.gop_duration == 0:
            return Gst.PadProbeReturn.OK

        if not self.gop_has_target:
            return Gst.PadProbeReturn.DROP

        # Re-check from the current delta pts — once the select probe
        # has matched the last target in this GOP, drop the rest.
        self.gop_has_target = self._gop_contains_target(pts)
        if not self.gop_has_target:
            return Gst.PadProbeReturn.DROP

        return Gst.PadProbeReturn.OK

    def _gop_contains_target(self, lower_bound_pts: int) -> bool:
        """Whether an unmatched target falls within the current GOP.

        ``lower_bound_pts`` is the I-frame pts on a fresh GOP, or the
        current delta pts when re-checking mid-GOP. ``target_cursor``
        skips targets the select probe has already matched. The upper
        bound shrinks by ``frame_duration`` when known: a target sitting
        in the last frame slot is matched by the next GOP's I-frame and
        doesn't need this GOP's deltas at all.
        """
        gop_end = self.gop_i_pts + self.gop_duration
        upper = (gop_end - self.frame_duration
                 if self.frame_duration > 0 else gop_end)
        lo = max(self.target_cursor,
                 bisect.bisect_left(self.target_pts, lower_bound_pts))
        return lo < len(self.target_pts) and self.target_pts[lo] < upper

    def _select_probe(self, pad, info, _ud):
        """nvvideoconvert sink pad — pick which decoded frames to keep."""
        Gst = self._Gst
        if self.early_eos:
            return Gst.PadProbeReturn.DROP

        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK

        # First frame: capture fps from caps so the caller can report it.
        if self.total == 0:
            caps = pad.get_current_caps()
            if caps and caps.get_size():
                s = caps.get_structure(0)
                ok, num, den = s.get_fraction("framerate")
                if ok and den > 0:
                    self.fps = num / den

        self.total += 1

        if self.use_pts_mode:
            if not self.target_pts:
                return Gst.PadProbeReturn.OK
            pts = buf.pts
            if pts == Gst.CLOCK_TIME_NONE:
                return Gst.PadProbeReturn.OK
            if self.target_cursor >= len(self.target_pts):
                return Gst.PadProbeReturn.DROP
            if pts < self.target_pts[self.target_cursor]:
                return Gst.PadProbeReturn.DROP
            while (self.target_cursor < len(self.target_pts)
                   and self.target_pts[self.target_cursor] <= pts):
                self.target_cursor += 1
        else:
            if self.target_indices:
                idx = self.total - 1
                lo, hi = 0, len(self.target_indices) - 1
                hit = False
                while lo <= hi:
                    m = (lo + hi) >> 1
                    v = self.target_indices[m]
                    if v == idx:
                        hit = True
                        break
                    if v < idx:
                        lo = m + 1
                    else:
                        hi = m - 1
                if not hit:
                    return Gst.PadProbeReturn.DROP

        if self.kept >= self.max_frames:
            return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    def _copy_probe(self, pad, info, _ud):
        """capsfilter src pad — copy NVMM RGB buffer into dst tensor."""
        Gst = self._Gst
        if self.early_eos or self.has_error:
            return Gst.PadProbeReturn.OK
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        if self.kept >= self.max_frames or self.max_frames == 0:
            return Gst.PadProbeReturn.OK

        src_ptr, src_pitch, p_width, p_height = _read_nvbuf_surface_first(
            buf, Gst)
        if not src_ptr:
            return Gst.PadProbeReturn.OK

        # Lazy-allocate destination tensor on the first kept frame.
        if self.dst_tensor is None:
            self.width = p_width
            self.height = p_height
            self.dst_pitch = self.width * 3
            self.frame_bytes = self.height * self.dst_pitch
            self.dst_tensor = torch.empty(
                (self.max_frames, self.height, self.width, 3),
                dtype=torch.uint8, device="cuda")
            self.dst_ptr = self.dst_tensor.data_ptr()

        dst = self.dst_ptr + self.kept * self.frame_bytes
        if not src_pitch:
            src_pitch = self.dst_pitch
        rc = self._cudart.cudaMemcpy2DAsync(
            ctypes.c_void_p(dst), self.dst_pitch,
            ctypes.c_void_p(src_ptr), src_pitch,
            self.dst_pitch, self.height,
            3,  # cudaMemcpyDeviceToDevice
            self._cuda_stream,
        )
        if rc != 0:
            self.has_error = True
            self.err_msg = f"cudaMemcpy2DAsync rc={rc}"
            return Gst.PadProbeReturn.OK

        self.kept += 1
        if self.kept >= self.max_frames and not self.early_eos:
            self.early_eos = True
            self._on_max_frames_reached(pad)
        return Gst.PadProbeReturn.OK

    def _on_max_frames_reached(self, pad) -> None:
        """File mode pushes EOS; RTSP signals the worker thread instead."""
        Gst = self._Gst
        if self._done_event is not None:
            self._done_event.set()
        else:
            pad.push_event(Gst.Event.new_eos())

    # ------------------------------------------------------------------
    # Build a populated _DecodeResult after the pipeline returns
    # ------------------------------------------------------------------
    def _finalize_result(self, req: _DecodeRequest,
                         error: str) -> _DecodeResult:
        if self._cudart is not None:
            self._cudart.cudaStreamSynchronize(self._cuda_stream)

        frames = None
        if self.dst_tensor is not None and self.kept > 0 and not self.has_error:
            # Slice down to the actual kept count, ``contiguous()`` so the
            # CUDA-IPC handoff carries only the populated rows.
            frames = self.dst_tensor[: self.kept].contiguous()

        err = error or (self.err_msg if self.has_error else "")
        return _DecodeResult(
            job_id=req.job_id,
            worker_id=self.worker_id,
            n_kept=self.kept,
            n_total=self.total,
            fps=self.fps,
            error=err,
            frames=frames,
        )


# ----------------------------------------------------------------------
# File-mode worker — explicit pipeline, swap source on URI change
# ----------------------------------------------------------------------
class _FileWorkerState(_BaseWorkerState):
    def __init__(self, worker_id: int, drop_interval: int):
        super().__init__(worker_id, drop_interval)
        self.current_uri: str | None = None
        self._probe_ids: dict[str, int] = {}

    # ------------------------------------------------------------------
    def ensure_pipeline(self, uri: str) -> None:
        if self.pipeline is None:
            self._build(uri)
        elif uri != self.current_uri:
            self._swap_uri(uri)

    def _build(self, uri: str) -> None:
        Gst = self._ensure_gst()
        self._ensure_cudart()
        path = uri[7:] if uri.startswith("file://") else uri

        # ``parsebin`` auto-detects the container (MP4/MKV/TS/…) and the
        # codec (H.264/H.265/…), then exposes a dynamic src pad carrying
        # the parsed elementary stream. ``nvv4l2decoder`` accepts H.264
        # and H.265 input via caps negotiation on its sink pad.
        elems = {
            "filesrc":  Gst.ElementFactory.make("filesrc",        None),
            "parsebin": Gst.ElementFactory.make("parsebin",       None),
            "nvdec":    Gst.ElementFactory.make("nvv4l2decoder",  None),
            "nvvconv":  Gst.ElementFactory.make("nvvideoconvert", None),
            "capsf":    Gst.ElementFactory.make("capsfilter",     None),
            "sink":     Gst.ElementFactory.make("fakesink",       None),
        }
        missing = [k for k, v in elems.items() if v is None]
        if missing:
            raise RuntimeError(f"GStreamer element creation failed: {missing}")

        elems["filesrc"].set_property("location", path)
        elems["nvdec"].set_property("drop-frame-interval", self.drop_interval)
        elems["nvdec"].set_property("num-extra-surfaces", 4)
        elems["nvvconv"].set_property("nvbuf-memory-type", 2)
        elems["sink"].set_property("sync", False)
        caps = Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=RGB")
        elems["capsf"].set_property("caps", caps)

        pipeline = Gst.Pipeline.new(None)
        for e in elems.values():
            pipeline.add(e)

        # filesrc -> parsebin : static.
        if not elems["filesrc"].link(elems["parsebin"]):
            raise RuntimeError("filesrc->parsebin link failed")

        # parsebin -> nvv4l2decoder : dynamic. parsebin only exposes its
        # src pad after it has type-detected the container/codec, so we
        # link in the pad-added callback. Filter to video/* in case the
        # source has audio tracks. The GOP-drop probe must also live on
        # this dynamic pad — there's no static parser pad to attach it
        # to, since parsebin encapsulates the parser.
        BUF = Gst.PadProbeType.BUFFER

        def _on_parsebin_pad(_pb, pad):
            cap = pad.get_current_caps() or pad.query_caps(None)
            if not cap or cap.get_size() == 0:
                return
            if not cap.get_structure(0).get_name().startswith("video/"):
                return
            sink_pad = elems["nvdec"].get_static_pad("sink")
            if sink_pad is None or sink_pad.is_linked():
                return
            if pad.link(sink_pad) != Gst.PadLinkReturn.OK:
                return
            self._probe_ids["parser"] = pad.add_probe(
                BUF, self._parser_probe, None)
        elems["parsebin"].connect("pad-added", _on_parsebin_pad)

        if not (elems["nvdec"].link(elems["nvvconv"])
                and elems["nvvconv"].link(elems["capsf"])
                and elems["capsf"].link(elems["sink"])):
            raise RuntimeError("downstream link failed")

        self.pipeline = pipeline
        self.elements = elems
        self.current_uri = uri

        self._probe_ids["select"] = elems["nvvconv"].get_static_pad(
            "sink").add_probe(BUF, self._select_probe, None)
        self._probe_ids["copy"] = elems["capsf"].get_static_pad(
            "src").add_probe(BUF, self._copy_probe, None)

        # Warm up: play to EOS so NVDEC session + NVMM surfaces are ready,
        # then park in PAUSED so the next decode starts from a primed
        # pipeline.
        pipeline.set_state(Gst.State.PLAYING)
        bus = pipeline.get_bus()
        msg = bus.timed_pop_filtered(
            30 * Gst.SECOND,
            Gst.MessageType.EOS | Gst.MessageType.ERROR)
        if msg is not None and msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            raise RuntimeError(f"warmup error: {err.message} ({dbg})")
        pipeline.set_state(Gst.State.PAUSED)
        pipeline.get_state(5 * Gst.SECOND)
        # Warmup ran the pipeline to EOS — the next decode must seek to
        # rewind even when the chunk starts at byte 0.
        self.pipeline_has_streamed = True

    def _swap_uri(self, new_uri: str) -> None:
        """NULL-cycle the pipeline and point ``filesrc`` at the new path.

        An in-place element swap (replace only filesrc + parsebin while
        keeping NVDEC alive) would skip the NVDEC re-init cost, but
        PyGObject's ref counting interacts poorly with mid-pipeline
        element removal. The NULL/PAUSED reset is reliable and the cost
        is amortised across many decodes against the same URI.
        """
        Gst = self._Gst
        path = new_uri[7:] if new_uri.startswith("file://") else new_uri
        self.pipeline.set_state(Gst.State.NULL)
        self.pipeline.get_state(5 * Gst.SECOND)
        self.elements["filesrc"].set_property("location", path)
        self.current_uri = new_uri
        self.width = 0
        self.height = 0
        self.pipeline.set_state(Gst.State.PAUSED)
        self.pipeline.get_state(5 * Gst.SECOND)
        # PAUSED preroll advances the pipeline past byte 0 (sometimes all
        # the way to EOS for short files). Mark the pipeline as having
        # streamed so the next decode issues a real FLUSH-seek to rewind,
        # even for a chunk-0 request. This matches what ``_build`` does
        # at line 623; a previous "fresh source at byte 0 — skip seek"
        # comment here was incorrect and produced zero-frame results
        # the first time a worker decoded a freshly-swapped file.
        self.pipeline_has_streamed = True

    # ------------------------------------------------------------------
    def decode(self, req: _DecodeRequest) -> _DecodeResult:
        Gst = self._ensure_gst()
        try:
            self.ensure_pipeline(req.uri)
        except Exception as e:
            return _DecodeResult(
                job_id=req.job_id, worker_id=self.worker_id,
                error=f"{type(e).__name__}: {e}")

        self._reset_for_decode(req)

        seek_pts = self.target_pts[0] if self.target_pts else 0
        # Skip the no-op FLUSH-seek to byte 0 on a freshly-built or
        # freshly-swapped pipeline. Any pipeline that has already
        # streamed (post-warmup or post-decode) is parked at EOS or a
        # later segment and must be rewound, even for chunk 0.
        if seek_pts > 0 or self.pipeline_has_streamed:
            self.pipeline.seek_simple(
                Gst.Format.TIME,
                (Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT
                 | Gst.SeekFlags.SNAP_BEFORE),
                seek_pts,
            )
        bus = self.pipeline.get_bus()
        while bus.pop_filtered(
                Gst.MessageType.EOS | Gst.MessageType.ERROR) is not None:
            pass

        self.pipeline.set_state(Gst.State.PLAYING)

        timeout_ns = int(req.timeout_sec * 1e9)
        msg = bus.timed_pop_filtered(
            timeout_ns,
            Gst.MessageType.EOS | Gst.MessageType.ERROR)
        error = ""
        if msg is None:
            error = f"timeout after {req.timeout_sec}s"
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            error = f"{err.message} ({dbg})"

        # Pipeline has now advanced past byte 0 — any subsequent reuse
        # needs a real seek to rewind, even for chunk 0.
        self.pipeline_has_streamed = True
        # Leave the pipeline in PLAYING — next decode's flush-seek resets it.
        return self._finalize_result(req, error)

    def shutdown(self) -> None:
        if self.pipeline is None:
            return
        self.pipeline.set_state(self._Gst.State.NULL)
        self.pipeline = None
        self.elements = {}


# ----------------------------------------------------------------------
# RTSP / persistent-stream worker — pipeline stays in PLAYING forever
# ----------------------------------------------------------------------
def _read_nvbuf_surface_first(buf, Gst):
    """Read pitch + dataPtr + width + height from ``surfaceList[0]``.

    ``buf.map(GST_MAP_READ)`` exposes the GstBuffer's memory as a Python
    bytes view; for NVMM buffers that view is the raw NvBufSurface
    struct (numFilled, surfaceList, …). The struct contains a
    ``surfaceList`` pointer pointing to NvBufSurfaceParams in the same
    allocation, which we then dereference via ctypes to read the GPU
    pointer (``dataPtr``) and pitch.

    Returns ``(data_ptr, pitch, width, height)``. All zero on failure.
    """
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return 0, 0, 0, 0
    try:
        if mapinfo.size < _NVBUF_SURFACE_HEAD_SZ:
            return 0, 0, 0, 0
        # mapinfo.data is a bytes-like view; copy out the head of the
        # NvBufSurface struct so we can parse it. The pointer values
        # inside the copy still reference the original NVMM allocation.
        head = bytes(mapinfo.data[:_NVBUF_SURFACE_HEAD_SZ])
        surf = _NvBufSurface.from_buffer_copy(head)
        if surf.numFilled == 0 or not surf.surfaceList:
            return 0, 0, 0, 0
        p = surf.surfaceList[0]
        data_ptr = int(p.dataPtr or 0)
        if not data_ptr and p.cudaBuffer:
            # NvBufSurfaceCudaBuffer layout: void* basePtr; void* dataPtr; …
            # The second pointer is the page-aligned image data.
            data_ptr = int(ctypes.c_void_p.from_address(
                int(p.cudaBuffer) + 8).value or 0)
        return data_ptr, int(p.pitch), int(p.width), int(p.height)
    finally:
        buf.unmap(mapinfo)


# ======================================================================
# Public Pool / Stream API
# ======================================================================
@dataclass
class DecodeFrames:
    """Result of a single decode call. ``frames`` is a CUDA tensor
    ``(n_kept, H, W, 3)`` uint8 in the same process; access directly."""
    frames: torch.Tensor | None
    n_kept: int
    n_total: int
    fps: float
    error: str = ""


def _file_worker_loop(worker_id: int,
                      drop_interval: int,
                      warmup_uri: str | None,
                      req_q,
                      res_q,
                      closed_flag: "list[bool]") -> None:
    state = _FileWorkerState(worker_id, drop_interval)
    if warmup_uri:
        try:
            state.ensure_pipeline(warmup_uri)
        except Exception as e:
            logger.warning(
                "[ds_decode_pool worker %d] warmup failed: %s",
                worker_id, e)
    try:
        while not closed_flag[0]:
            req: _DecodeRequest | None = req_q.get()
            if req is None:
                break
            res = state.decode(req)
            res_q.put(res)
    finally:
        state.shutdown()


class DecodePool:
    """Pool of N file-decode pipelines on N daemon threads, sharing one
    CUDA context. Probe GIL transitions are negligible at our probe call
    rate, and a single CUDA context lets the driver pipeline NVDEC
    sessions efficiently across pool slots.

    Frames returned by :meth:`decode` are CUDA tensors in this same
    process — no IPC handle, no D2H/H2D round-trip; the caller uses the
    tensor directly.
    """

    def __init__(self,
                 num_workers: int = 8,
                 drop_interval: int = 0,
                 warmup_uri: str | None = None) -> None:
        import queue as _queue
        import threading

        # Force CUDA context init in the main thread before workers build
        # pipelines, so nvv4l2decoder/nvvideoconvert pick up the same
        # primary context PyTorch uses.
        if torch.cuda.is_available():
            torch.cuda.init()

        self._req_q: "_queue.Queue" = _queue.Queue()
        self._res_q: "_queue.Queue" = _queue.Queue()
        self._closed_flag = [False]
        self._workers: list[threading.Thread] = []
        for i in range(num_workers):
            t = threading.Thread(
                target=_file_worker_loop,
                args=(i, drop_interval, warmup_uri,
                      self._req_q, self._res_q, self._closed_flag),
                daemon=True,
                name=f"ds-decode-thread-{i}",
            )
            t.start()
            self._workers.append(t)

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._results: dict[int, _DecodeResult] = {}
        self._next_id = 0
        self._closed = False
        self._collector = threading.Thread(
            target=self._collect_loop, daemon=True,
            name="ds-decode-pool-collector")
        self._collector.start()

    def decode(self,
               uri: str,
               *,
               target_pts_ns: list[int] | None = None,
               max_frames: int = 8,
               timeout_sec: float = 30.0) -> DecodeFrames:
        if self._closed:
            raise RuntimeError("DecodePool is closed")
        with self._lock:
            job_id = self._next_id
            self._next_id += 1
        self._req_q.put(_DecodeRequest(
            job_id=job_id,
            uri=uri,
            target_pts_ns=tuple(target_pts_ns) if target_pts_ns else (),
            use_pts_mode=True,
            max_frames=max_frames,
            timeout_sec=timeout_sec,
        ))
        with self._cv:
            while job_id not in self._results and not self._closed:
                self._cv.wait()
            if self._closed and job_id not in self._results:
                raise RuntimeError(
                    "DecodePool was closed before result")
            res = self._results.pop(job_id)
        return _to_decode_frames(res)

    def _collect_loop(self) -> None:
        import queue as _queue
        while not self._closed:
            try:
                res = self._res_q.get(timeout=0.5)
            except _queue.Empty:
                continue
            with self._cv:
                self._results[res.job_id] = res
                self._cv.notify_all()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._closed_flag[0] = True
        for _ in self._workers:
            self._req_q.put(None)
        for t in self._workers:
            t.join(timeout=5)
        with self._cv:
            self._cv.notify_all()


def _to_decode_frames(res: _DecodeResult) -> DecodeFrames:
    return DecodeFrames(
        frames=res.frames,
        n_kept=res.n_kept,
        n_total=res.n_total,
        fps=res.fps,
        error=res.error,
    )
