# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ws_load.py: N concurrent real-time-paced /v1/realtime websocket streams.

Protocol mirrored from examples/speech_to_text/realtime/openai_realtime_client.py:
    recv session.created -> send session.update {model} -> send commit (ready)
    -> loop input_audio_buffer.append {audio: base64 PCM16 @16kHz}
    -> input_audio_buffer.commit {final: true}
    -> recv transcription.delta* then transcription.done

Paces audio to wall-clock in ~frame-ms PCM frames (real-time ingest), records
per-frame->partial latency + final transcript per stream + an aggregate summary.

Usage:
    .venv/bin/python ws_load.py -n 8 --audio ref.wav --out results.json
    .venv/bin/python ws_load.py -n 16 --frame-ms 30 --duration-cap 60
"""

import argparse
import asyncio
import json
import statistics
import sys
import time

import numpy as np
import pybase64 as base64
import websockets

from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio

SAMPLE_RATE = 16000


def load_pcm16(audio_path: str, loops: int = 1) -> bytes:
    audio, _ = load_audio(audio_path, sr=SAMPLE_RATE, mono=True)
    pcm = (audio * 32767).astype(np.int16).tobytes()
    # Repeat the clip so a single session accumulates enough decoder positions
    # to cross the re-anchor threshold (the unbounded-duration test).
    return pcm * max(1, loops)


async def run_one(
    idx, uri, model, pcm, frame_bytes, frame_dt, duration_cap, collect_text,
    pace=True, start_byte=0
):
    res = {
        "stream": idx,
        "ok": False,
        "error": None,
        "session_id": None,
        "frames_sent": 0,
        "audio_seconds_sent": 0.0,
        "first_partial_latency_s": None,
        "final_text": None,
        "wall_seconds": None,
        "audio_sent_wall_s": None,   # wall time when the last audio frame was sent
        "flush_latency_s": None,     # transcription.done arrival - last audio frame
        # (elapsed_s, cumulative_words) per partial: raw transcription-progress
        # timeline, for offline inspection of how output tracks the audio clock.
        "lat_timeline": [],
    }
    t_start = time.perf_counter()
    first_partial_seen = False
    cum_words = 0
    try:
        async with websockets.connect(uri, max_size=None, ping_interval=20) as ws:
            hello = json.loads(await ws.recv())
            if hello.get("type") != "session.created":
                res["error"] = f"unexpected first message: {hello.get('type')}"
                return res
            res["session_id"] = hello.get("id")
            await ws.send(json.dumps({"type": "session.update", "model": model}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            done = asyncio.Event()

            async def receiver():
                nonlocal first_partial_seen, cum_words
                while not done.is_set():
                    try:
                        msg = json.loads(await ws.recv())
                    except websockets.ConnectionClosed:
                        return
                    mtype = msg.get("type")
                    if mtype == "transcription.delta":
                        now = time.perf_counter()
                        if not first_partial_seen:
                            first_partial_seen = True
                            res["first_partial_latency_s"] = now - t_start
                        delta = msg.get("delta", "")
                        if collect_text:
                            res["final_text"] = (res["final_text"] or "") + delta
                        cum_words += len(delta.split())
                        res["lat_timeline"].append((round(now - t_start, 3), cum_words))
                    elif mtype == "transcription.done":
                        if collect_text and msg.get("text"):
                            res["final_text"] = msg["text"]
                        res["flush_latency_s"] = (
                            time.perf_counter()
                            - t_start
                            - (res["audio_sent_wall_s"] or 0.0)
                        )
                        res["ok"] = True
                        done.set()
                        return
                    elif mtype == "error":
                        res["error"] = msg.get("error")
                        done.set()
                        return

            recv_task = asyncio.create_task(receiver())

            total = len(pcm)
            sent = start_byte % max(total, 1)
            next_send = time.perf_counter()
            while sent < total and not done.is_set():
                if duration_cap > 0 and res["audio_seconds_sent"] >= duration_cap:
                    break
                chunk = pcm[sent : sent + frame_bytes]
                sent += len(chunk)
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("utf-8"),
                        }
                    )
                )
                res["frames_sent"] += 1
                res["audio_seconds_sent"] = (
                    res["frames_sent"] * frame_bytes / 2 / SAMPLE_RATE
                )
                if pace:
                    next_send += frame_dt
                    slack = next_send - time.perf_counter()
                    if slack > 0:
                        await asyncio.sleep(slack)
                else:
                    # fast-feed: advance positions as fast as the server accepts
                    # (NOT real-time; wall vs audio is meaningless in this mode).
                    await asyncio.sleep(0)

            res["audio_sent_wall_s"] = time.perf_counter() - t_start
            await ws.send(
                json.dumps({"type": "input_audio_buffer.commit", "final": True})
            )
            try:
                await asyncio.wait_for(recv_task, timeout=120)
            except asyncio.TimeoutError:
                res["error"] = res["error"] or "timeout waiting for transcription.done"
                done.set()
                recv_task.cancel()
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        res["wall_seconds"] = time.perf_counter() - t_start
    return res


def pct(data, p):
    if not data:
        return float("nan")
    s = sorted(data)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def summarize(results):
    ok = [r for r in results if r["ok"]]
    first = [
        r["first_partial_latency_s"]
        for r in results
        if r["first_partial_latency_s"] is not None
    ]
    flush = [r["flush_latency_s"] for r in results if r["flush_latency_s"] is not None]
    return {
        "n_streams": len(results),
        "n_ok": len(ok),
        "n_error": len(results) - len(ok),
        "errors": [r["error"] for r in results if r["error"]][:8],
        "first_partial_latency_s": {
            "mean": statistics.fmean(first) if first else None,
            "p50": pct(first, 50),
            "p95": pct(first, 95),
            "max": max(first) if first else None,
        },
        "flush_latency_s": {
            "mean": statistics.fmean(flush) if flush else None,
            "p50": pct(flush, 50),
            "p95": pct(flush, 95),
            "max": max(flush) if flush else None,
        },
    }


async def main_async(args):
    audio_path = args.audio or str(AudioAsset("mary_had_lamb").get_local_path())
    if not args.audio:
        print(f"[ws_load] no --audio, default asset: {audio_path}", file=sys.stderr)
    pcm = load_pcm16(audio_path, args.loops)
    samples_per_frame = int(args.frame_ms / 1000 * SAMPLE_RATE)
    frame_bytes = samples_per_frame * 2
    frame_dt = args.frame_ms / 1000.0
    uri = f"ws://{args.host}:{args.port}/v1/realtime"
    # De-sync: each stream starts at a different position in the audio, so at any
    # instant the streams carry different content (no perfectly-aligned batches).
    stagger_bytes = (
        int(args.pos_stagger_s * SAMPLE_RATE) * 2 // frame_bytes
    ) * frame_bytes
    print(
        f"[ws_load] {args.n} streams -> {uri} frame={args.frame_ms}ms "
        f"audio={len(pcm) / 2 / SAMPLE_RATE:.1f}s cap={args.duration_cap}s "
        f"pos-stagger={args.pos_stagger_s}s/stream",
        file=sys.stderr,
    )
    tasks = [
        run_one(
            i,
            uri,
            args.model,
            pcm,
            frame_bytes,
            frame_dt,
            args.duration_cap,
            not args.no_text,
            not args.no_pace,
            start_byte=i * stagger_bytes,
        )
        for i in range(args.n)
    ]
    results = await asyncio.gather(*tasks)
    summary = summarize(results)
    with open(args.out, "w") as fh:
        json.dump({"summary": summary, "streams": results}, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[ws_load] per-stream results -> {args.out}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Concurrent realtime websocket load test")
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--model", default="mistralai/Voxtral-Mini-4B-Realtime-2602")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--frame-ms", type=float, default=30.0)
    ap.add_argument(
        "--duration-cap",
        type=float,
        default=0.0,
        help="cap audio sec/stream (0=full file)",
    )
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument(
        "--no-pace",
        action="store_true",
        help="fast-feed (advance positions ASAP, not real-time)",
    )
    ap.add_argument(
        "--loops",
        type=int,
        default=1,
        help="repeat the audio clip N times per stream (unbounded-duration test)",
    )
    ap.add_argument(
        "--pos-stagger-s",
        type=float,
        default=0.0,
        help="start stream i at i*this many seconds into the audio (de-sync so "
        "concurrent streams carry different content, not perfectly-aligned batches)",
    )
    ap.add_argument("--out", default="ws_load_results.json")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
