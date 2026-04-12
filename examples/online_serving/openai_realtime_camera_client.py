# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demo client for the vLLM Realtime Video WebSocket API using live camera.

Reads frames from the camera, samples every Nth frame (--frame-interval),
and sends to the server. When the client-side frame queue is full, old
frames are discarded and newer frames are kept (drop-oldest policy).

Before running, start vLLM with a vision model that supports video, e.g.:

    vllm serve Qwen2.5-VL-7B-Instruct --enforce-eager

Requirements:
- websockets
- Pillow
- opencv-python
- gradio (for --gradio UI)

Usage:
  # Default camera (device 0), every frame
  python openai_realtime_camera_client.py

  # Set output resolution (exact size, may stretch image)
  python openai_realtime_camera_client.py --output-resolution 640x480

  # Set max size (keep aspect ratio)
  python openai_realtime_camera_client.py --max-size 640

  # Keep aspect ratio when resizing to exact resolution
  python openai_realtime_camera_client.py --output-resolution 640x480 --keep-aspect-ratio

  # Gradio UI: set all parameters in the interface, send new prompts anytime
  python openai_realtime_camera_client.py --gradio

  # Gradio with CLI defaults (override in UI)
  python openai_realtime_camera_client.py --gradio --port 8000 --camera-id 0

Press Ctrl+C to stop.
"""

import argparse
import asyncio
import base64
import collections
import io
import json
import os
import queue
import sys
import threading

import websockets

try:
    import gradio as gr
except ImportError:
    gr = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2

    # cv2.setLogLevel exists in OpenCV 4.8+; Gradio may call it
    if not hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel = lambda _: None  # no-op for older OpenCV
except ImportError:
    cv2 = None

# Shared state
frame_queue: collections.deque | None = None
is_capturing = False
is_stopping = False
dropped_count = 0
latest_display_frame = None  # RGB numpy array for Gradio
response_text = ""  # Model output for Gradio

# For Gradio: prompts to send (session.update) when user sends new prompt
_prompt_queue: queue.Queue | None = None
# For Gradio: trigger events to send (generation.trigger) for asking questions
_trigger_queue: queue.Queue | None = None

# Shared resize configuration (for runtime modification)
_resize_config = {
    "output_resolution": None,
    "max_size": None,
    "keep_aspect_ratio": False,
}


def _append_response(s: str) -> None:
    """Append to response_text for Gradio display."""
    global response_text
    response_text += s


def _parse_resolution(resolution_str: str) -> tuple[int, int] | None:
    """Parse resolution string 'WxH' into (width, height). Returns None if invalid."""
    if not resolution_str:
        return None
    try:
        parts = resolution_str.lower().replace(" ", "").split("x")
        if len(parts) == 2:
            width = int(parts[0])
            height = int(parts[1])
            if width > 0 and height > 0:
                return width, height
    except (ValueError, AttributeError):
        pass
    return None


def frame_to_base64_jpeg(
    bgr,
    quality: int = 85,
    output_resolution: str | None = None,
    max_size: int | None = None,
    keep_aspect_ratio: bool = False,
) -> str:
    """Convert BGR frame to base64-encoded JPEG with optional resize."""
    if Image is None:
        raise RuntimeError("PIL is required. Install with: pip install Pillow")

    original_height, original_width = bgr.shape[:2]

    # Determine output dimensions
    output_width, output_height = None, None

    if max_size:
        # Use max_size with aspect ratio preservation
        if original_width > max_size or original_height > max_size:
            ratio = min(max_size / original_width, max_size / original_height)
            output_width = int(original_width * ratio)
            output_height = int(original_height * ratio)
    elif output_resolution:
        parsed_res = _parse_resolution(output_resolution)
        if parsed_res:
            if keep_aspect_ratio:
                # Fit image within the target resolution while maintaining aspect ratio
                target_width, target_height = parsed_res
                target_ratio = target_width / target_height
                original_ratio = original_width / original_height

                if original_ratio > target_ratio:
                    # Image is wider, fit width
                    output_width = target_width
                    output_height = int(target_width / original_ratio)
                else:
                    # Image is taller, fit height
                    output_height = target_height
                    output_width = int(target_height * original_ratio)
            else:
                # Exact resize (may stretch)
                output_width, output_height = parsed_res

    # Resize if needed
    if output_width and output_height:
        bgr = cv2.resize(bgr, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def camera_capture_loop(
    camera_id: int,
    frame_interval: int,
    queue: "collections.deque[str]",
    quality: int,
    update_display_frame: bool = False,
    output_resolution: str | None = None,
    max_size: int | None = None,
    keep_aspect_ratio: bool = False,
) -> None:
    """Capture frames from camera, sample, and append to queue. Drops oldest when full."""
    global dropped_count, latest_display_frame
    if cv2 is None:
        raise RuntimeError(
            "opencv-python is required. Install with: pip install opencv-python"
        )
    # On Windows, CAP_DSHOW often works better for webcams
    if sys.platform == "win32":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    # Get camera original resolution
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine effective resolution
    effective_res = f"{orig_width}x{orig_height}"
    if max_size:
        effective_res = f"max={max_size} (from {orig_width}x{orig_height})"
    elif output_resolution:
        if keep_aspect_ratio:
            effective_res = f"fit in {output_resolution} (from {orig_width}x{orig_height})"
        else:
            effective_res = f"{output_resolution} (from {orig_width}x{orig_height})"

    if update_display_frame:
        _append_response(f"[Camera] Original: {orig_width}x{orig_height}, Output: {effective_res}\n")
    else:
        print(f"Camera: {orig_width}x{orig_height} -> Output: {effective_res}", flush=True)

    frame_idx = 0
    try:
        while is_capturing:
            ret, bgr = cap.read()
            if not ret:
                break
            if update_display_frame:
                latest_display_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if frame_idx % frame_interval == 0:
                # Use shared config for runtime updates
                out_res = output_resolution or _resize_config.get("output_resolution")
                max_sz = max_size or _resize_config.get("max_size")
                keep_ar = _resize_config.get("keep_aspect_ratio", False) or keep_aspect_ratio

                b64 = frame_to_base64_jpeg(
                    bgr,
                    quality=quality,
                    output_resolution=out_res,
                    max_size=max_sz,
                    keep_aspect_ratio=keep_ar,
                )
                # deque(maxlen=N): when full, append() automatically drops leftmost (oldest)
                if len(queue) >= queue.maxlen:
                    dropped_count += 1
                queue.append(b64)
            frame_idx += 1
    finally:
        cap.release()


async def run_realtime_camera(
    host: str,
    port: int,
    model: str,
    prompt: str | None,
    camera_id: int,
    frame_interval: int,
    batch_size: int,
    queue_size: int,
    quality: int,
    prompt_queue: queue.Queue | None = None,
    trigger_queue: queue.Queue | None = None,
    update_display_frame: bool = False,
    output_resolution: str | None = None,
    max_size: int | None = None,
    keep_aspect_ratio: bool = False,
):
    """Stream camera frames to realtime video WebSocket.

    Client runs in silent mode by default (frames sent but no output).
    Use 'generation.trigger' to ask questions and get responses.

    Parameters:
        batch_size: Upper limit on frames per commit. Actual frames sent is
            min(batch_size, available_server_capacity, frames_in_queue).
            Server capacity is controlled by its max_queue_size (typically 3-4).
    """
    global frame_queue, is_capturing

    frame_queue = collections.deque(maxlen=queue_size)
    is_capturing = True

    # Set shared config for runtime updates
    _resize_config["output_resolution"] = output_resolution
    _resize_config["max_size"] = max_size
    _resize_config["keep_aspect_ratio"] = keep_aspect_ratio

    uri = f"ws://{host}:{port}/v1/realtime_video"
    capture_thread = threading.Thread(
        target=camera_capture_loop,
        args=(
            camera_id,
            frame_interval,
            frame_queue,
            quality,
            update_display_frame,
            output_resolution,
            max_size,
            keep_aspect_ratio,
        ),
        daemon=True,
    )
    capture_thread.start()

    print(
        f"Camera {camera_id}: frame_interval={frame_interval}, "
        f"batch_size={batch_size}, queue_size={queue_size} (drop oldest when full)"
    )
    print("Mode: Silent by default. Use 'Trigger' button to ask questions.")
    print("Press Ctrl+C to stop.\n")

    try:
        try:
            async with websockets.connect(uri, open_timeout=10) as ws:
                msg = json.loads(await ws.recv())
                if msg.get("type") == "error":
                    print(f"Error: {msg.get('error', msg)}")
                    return
                if msg.get("type") != "session.created":
                    print(f"Unexpected: {msg}")
                    return
                initial_water = msg.get("input_video_buffer") or {}
                print(f"Session created: {msg.get('id', '')}")

                payload = {"type": "session.update", "model": model}
                if prompt:
                    payload["prompt"] = prompt
                await ws.send(json.dumps(payload))

                # Background task: handle session.update and generation.trigger from UI
                prompt_task: asyncio.Task | None = None
                if prompt_queue is not None:

                    async def prompt_sender() -> None:
                        while is_capturing:
                            try:
                                # Check if stopping signal received
                                global is_stopping
                                if is_stopping:
                                    # Send final commit to signal server we're done
                                    await ws.send(
                                        json.dumps({"type": "input_video_buffer.commit", "final": True})
                                    )
                                    if update_display_frame:
                                        _append_response(f"\n[Stopping] Sent final commit to server\n")
                                    else:
                                        print(f"\n[Stopping] Sent final commit to server", flush=True)
                                    break

                                # Check for session.update (silent mode prompt change)
                                try:
                                    new_prompt = prompt_queue.get_nowait()
                                    await ws.send(
                                        json.dumps({
                                            "type": "session.update",
                                            "model": model,
                                            "prompt": new_prompt or "",
                                        })
                                    )
                                    if update_display_frame:
                                        _append_response(f"\n[Prompt updated] {new_prompt}\n")
                                    else:
                                        print(f"\n[Prompt updated] {new_prompt}", flush=True)
                                except queue.Empty:
                                    pass

                                # Check for generation.trigger (trigger question)
                                if trigger_queue is not None:
                                    try:
                                        trigger_data = trigger_queue.get_nowait()
                                        trigger_payload = {"type": "generation.trigger"}
                                        if trigger_data.get("prompt"):
                                            trigger_payload["prompt"] = trigger_data["prompt"]
                                        if trigger_data.get("max_tokens"):
                                            trigger_payload["max_tokens"] = trigger_data["max_tokens"]
                                        await ws.send(json.dumps(trigger_payload))
                                        if update_display_frame:
                                            _append_response(f"\n[Trigger] {trigger_data.get('prompt', '')}\n")
                                        else:
                                            print(f"\n[Trigger] {trigger_data.get('prompt', '')}", flush=True)
                                    except queue.Empty:
                                        pass

                            except Exception as e:
                                if update_display_frame:
                                    _append_response(f"\n[Sender error] {e}\n")
                                else:
                                    print(f"\n[Sender error] {e}", flush=True)
                            await asyncio.sleep(0.05)

                    prompt_task = asyncio.create_task(prompt_sender())

                queue_depth = 0
                max_queue_size = initial_water.get("max_queue_size", 3)
                received_done_count = 0
                err: str | None = None
                waiting_for_response = False  # Track if we're waiting for a trigger response

                try:
                    while err is None and is_capturing:
                        # Calculate how many frames we can actually send
                        available_server_capacity = max_queue_size - queue_depth
                        frames_in_queue = len(frame_queue)
                        frames_to_send = min(batch_size, available_server_capacity, frames_in_queue)

                        # Collect frames from queue (only what we plan to send)
                        batch: list[str] = []
                        for _ in range(frames_to_send):
                            try:
                                batch.append(frame_queue.popleft())
                            except IndexError:
                                break

                        # Send when we have frames (>=2 for model compatibility) and server has capacity
                        # Note: Qwen2-VL/Qwen3-VL require at least 2 frames per batch
                        should_send = (
                            len(batch) >= 2
                            and queue_depth < max_queue_size
                        )

                        if should_send:
                            for b64 in batch:
                                await ws.send(
                                    json.dumps(
                                        {
                                            "type": "input_video_buffer.append",
                                            "video": b64,
                                            "format": "image/jpeg",
                                        }
                                    )
                                )
                            # For live camera, we never send final=True until we stop
                            await ws.send(
                                json.dumps({"type": "input_video_buffer.commit", "final": False})
                            )
                            queue_depth += 1
                            continue  # Sent successfully, try to send more

                        # If we collected frames but couldn't send, put them back
                        if batch:
                            for b64 in reversed(batch):
                                frame_queue.appendleft(b64)

                        # When queue is empty and not waiting for response, sleep briefly
                        if len(frame_queue) == 0 and not waiting_for_response:
                            await asyncio.sleep(0.01)
                            continue

                        # Receive message
                        if prompt_queue is not None:
                            try:
                                msg_bytes = await asyncio.wait_for(ws.recv(), timeout=1)
                            except asyncio.TimeoutError:
                                continue
                        else:
                            # Always use timeout to allow graceful stopping in non-Gradio mode too
                            try:
                                msg_bytes = await asyncio.wait_for(ws.recv(), timeout=1)
                            except asyncio.TimeoutError:
                                continue
                        response = json.loads(msg_bytes)
                        t = response.get("type")
                        if t == "completion.delta":
                            waiting_for_response = True
                            delta = response.get("delta", "")
                            if update_display_frame:
                                _append_response(delta)
                            else:
                                print(delta, end="", flush=True)
                        elif t == "completion.done":
                            waiting_for_response = False
                            text = response.get("text", "")
                            only_show_non_empty = True  # Only show non-empty responses (triggered)
                            if text or not only_show_non_empty:
                                if update_display_frame:
                                    _append_response(f"\n\n[Response] {text}")
                                    if response.get("usage"):
                                        _append_response(f"\nUsage: {response['usage']}")
                                else:
                                    print(f"\n\n[Response] {text}")
                                    if response.get("usage"):
                                        print(f"Usage: {response['usage']}")
                            received_done_count += 1
                            buf = response.get("input_video_buffer")
                            if buf is not None:
                                queue_depth = buf.get("queue_depth", queue_depth)
                                max_queue_size = buf.get("max_queue_size", max_queue_size)
                        elif t == "input_video_buffer.water_level":
                            queue_depth = response.get("queue_depth", queue_depth)
                            max_queue_size = response.get("max_queue_size", max_queue_size)
                        elif t == "session.stop":
                            is_stopping = True
                            is_capturing = False
                            if update_display_frame:
                                _append_response("\n[Session stopped by server]\n")
                            else:
                                print("\n[Session stopped by server]", flush=True)
                            break
                        elif t == "error":
                            err = response.get("error", response.get("message", str(response)))
                            err_str = f"\nError: {err}"
                            if response.get("code"):
                                err_str += f"\nCode: {response['code']}"
                            if update_display_frame:
                                _append_response(err_str)
                            else:
                                print(err_str, flush=True)
                        else:
                            print(f"[Received type={t!r}] {response}", flush=True)

                    # If we exited loop normally (not due to error), send final commit to server
                    if is_stopping:
                        try:
                            await ws.send(
                                json.dumps({"type": "input_video_buffer.commit", "final": True})
                            )
                            if update_display_frame:
                                _append_response("\n[Stopped] Sent final commit to server\n")
                            else:
                                print("\n[Stopped] Sent final commit to server", flush=True)
                        except Exception as e:
                            # WebSocket may already be closed, that's ok
                            pass
                finally:
                    if prompt_task is not None and not prompt_task.done():
                        prompt_task.cancel()
                        try:
                            await prompt_task
                        except asyncio.CancelledError:
                            pass

                    # Close WebSocket connection gracefully
                    try:
                        await ws.close()
                    except Exception:
                        pass
        except Exception as conn_err:
            err_msg = f"Connection error: {conn_err}\nIs the vLLM server running at {uri}?"
            if update_display_frame:
                _append_response(err_msg)
            else:
                print(err_msg)
            # Keep camera running for preview so user can see it works
            while is_capturing:
                await asyncio.sleep(0.2)
    except asyncio.CancelledError:
        pass
    finally:
        is_capturing = False
        if dropped_count > 0:
            print(f"\nDropped {dropped_count} frame(s) (queue was full).", flush=True)


def websocket_handler(
    host: str,
    port: int,
    model: str,
    prompt: str | None,
    camera_id: int,
    frame_interval: int,
    batch_size: int,
    queue_size: int,
    quality: int,
    trigger_queue: queue.Queue | None = None,
    output_resolution: str | None = None,
    max_size: int | None = None,
    keep_aspect_ratio: bool = False,
) -> None:
    """Run WebSocket + camera in event loop (for Gradio background thread)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            run_realtime_camera(
                host,
                port,
                model,
                prompt,
                camera_id,
                frame_interval,
                batch_size,
                queue_size,
                quality,
                prompt_queue=_prompt_queue,
                trigger_queue=trigger_queue,
                update_display_frame=True,
                output_resolution=output_resolution,
                max_size=max_size,
                keep_aspect_ratio=keep_aspect_ratio,
            )
        )
    except Exception as e:
        print(f"WebSocket error: {e}")


def send_prompt(prompt: str) -> tuple:
    """Gradio callback: enqueue new prompt for session.update."""
    global _prompt_queue
    if _prompt_queue is not None and prompt and prompt.strip():
        _prompt_queue.put(prompt.strip())
    return gr.update(value="")  # Clear the input


def send_trigger(prompt: str) -> tuple:
    """Gradio callback: enqueue trigger for generation."""
    global _trigger_queue
    if _trigger_queue is not None and prompt and prompt.strip():
        _trigger_queue.put({"prompt": prompt.strip(), "max_tokens": 512})
    return gr.update(value="")  # Clear the input


def update_resolution_settings(
    resolution: str, max_size: int, keep_aspect_ratio: bool
) -> tuple:
    """Runtime update resolution settings."""
    global _resize_config
    _resize_config["output_resolution"] = resolution if resolution else None
    _resize_config["max_size"] = max_size if max_size > 0 else None
    _resize_config["keep_aspect_ratio"] = keep_aspect_ratio

    msg = f"[Resolution updated] "
    if max_size > 0:
        msg += f"max_size={max_size}, keep_aspect={keep_aspect_ratio}"
    elif resolution:
        msg += f"{resolution}, keep_aspect={keep_aspect_ratio}"
    else:
        msg += "using camera original resolution"
    _append_response(f"\n{msg}\n")

    return gr.update(), gr.update(), gr.update()


def start_camera_service(
    host: str,
    port: int,
    model: str,
    prompt: str,
    camera_id: int,
    frame_interval: int,
    batch_size: int,
    queue_size: int,
    quality: int,
    output_resolution: str | None = None,
    max_size: int = 0,
    keep_aspect_ratio: bool = False,
) -> tuple:
    """Start the camera + WebSocket service (Gradio Start button)."""
    global response_text, _trigger_queue, is_capturing, is_stopping
    response_text = ""
    _trigger_queue = queue.Queue()
    is_capturing = True
    is_stopping = False
    thread = threading.Thread(
        target=websocket_handler,
        args=(
            host or "localhost",
            int(port) if port else 8000,
            model or "Qwen2.5-VL-7B-Instruct",
            prompt.strip() if prompt else None,
            int(camera_id) if camera_id is not None else 0,
            int(frame_interval) if frame_interval else 1,
            int(batch_size) if batch_size else 16,
            int(queue_size) if queue_size else 64,
            int(quality) if quality else 85,
            _trigger_queue,
            output_resolution.strip() if output_resolution else None,
            max_size if max_size > 0 else None,
            keep_aspect_ratio,
        ),
        daemon=True,
    )
    thread.start()
    return gr.update(interactive=False), gr.update(interactive=True)


def stop_camera_service() -> tuple:
    """Stop the camera + WebSocket service (Gradio Stop button)."""
    global is_capturing, is_stopping
    is_capturing = False
    is_stopping = True  # Signal graceful shutdown to send final commit
    _append_response("\n[Stop] Stopping camera service...\n")
    return gr.update(interactive=True), gr.update(interactive=False)


def get_latest_display() -> tuple:
    """Return latest frame and response text for Gradio periodic update."""
    global latest_display_frame, response_text
    img = latest_display_frame
    return img, response_text


def create_gradio_demo(
    host: str = "localhost",
    port: int = 8000,
    model: str = "Qwen2.5-VL-7B-Instruct",
    prompt: str | None = None,
    camera_id: int = 0,
    frame_interval: int = 1,
    batch_size: int = 16,
    queue_size: int = 64,
    quality: int = 85,
    output_resolution: str | None = None,
    max_size: int = 0,
    keep_aspect_ratio: bool = False,
) -> "gr.Blocks":
    """Create Gradio interface with parameter inputs and prompt send."""
    # Inline JS: stick to bottom unless user scrolls up; when at bottom, keep auto-scrolling
    scroll_js = """
    <script>
    (function(){
      var userScrolledUp = false;
      function isAtBottom(ta){
        return ta.scrollHeight - ta.scrollTop - ta.clientHeight < 10;
      }
      function run(){
        var el = document.getElementById("model-response-output");
        if(!el) return;
        var ta = el.tagName==="TEXTAREA" ? el : el.querySelector("textarea");
        if(!ta) return;
        if(!userScrolledUp || isAtBottom(ta)) ta.scrollTop = ta.scrollHeight;
      }
      function onScroll(){
        var el = document.getElementById("model-response-output");
        if(!el) return;
        var ta = el.tagName==="TEXTAREA" ? el : el.querySelector("textarea");
        if(!ta) return;
        userScrolledUp = !isAtBottom(ta);
      }
      function start(){
        var el = document.getElementById("model-response-output");
        if(!el){ setTimeout(start, 200); return; }
        var ta = el.tagName==="TEXTAREA" ? el : el.querySelector("textarea");
        if(ta) ta.addEventListener("scroll", onScroll, {passive:true});
        setInterval(run, 100);
      }
      if(document.readyState==="loading")
        document.addEventListener("DOMContentLoaded", start);
      else
        setTimeout(start, 500);
    })();
    </script>
    """
    with gr.Blocks(title="Real-time Camera Vision", head=scroll_js) as demo:
        gr.Markdown("# Real-time Camera Vision")
        gr.Markdown(
            "Set parameters below (or use defaults from command line), click **Start**. "
            "Camera runs in **silent mode** by default. Use **Trigger** button to ask questions and get responses."
        )
        with gr.Row():
            with gr.Column(scale=1):
                host_in = gr.Textbox(label="Host", value=host or "localhost")
                port_in = gr.Number(label="Port", value=port or 8000, precision=0)
                model_in = gr.Textbox(
                    label="Model",
                    value=model or "Qwen2.5-VL-7B-Instruct",
                )
                prompt_in = gr.Textbox(
                    label="Initial Prompt (optional)",
                    value=prompt or "",
                    placeholder="Describe what you see.",
                    lines=2,
                )
                camera_id_in = gr.Number(
                    label="Camera ID",
                    value=camera_id,
                    precision=0,
                )
                frame_interval_in = gr.Number(
                    label="Frame Interval",
                    value=frame_interval,
                    precision=0,
                )
                batch_size_in = gr.Number(
                    label="Batch Size",
                    value=batch_size,
                    precision=0,
                )
                queue_size_in = gr.Number(
                    label="Queue Size",
                    value=queue_size,
                    precision=0,
                )
                quality_in = gr.Number(
                    label="Quality (1-100)",
                    value=quality,
                    precision=0,
                )
                # Resolution settings
                gr.Markdown("### Resolution Settings")
                resolution_in = gr.Textbox(
                    label="Output Resolution (e.g., 640x480, 1280x720)",
                    value=output_resolution or "",
                    placeholder="e.g., 640x480, 1280x720. Leave empty for original.",
                )
                max_size_in = gr.Number(
                    label="Max Size (0=disabled, keep aspect ratio)",
                    value=max_size,
                    precision=0,
                    minimum=0,
                )
                keep_aspect_in = gr.Checkbox(
                    label="Keep Aspect Ratio",
                    value=keep_aspect_ratio,
                )
                update_res_btn = gr.Button("Update Resolution", variant="secondary", size="sm")
                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary")
                    stop_btn = gr.Button("Stop", variant="stop", interactive=False)
            with gr.Column(scale=1):
                image_out = gr.Image(label="Camera", height=400)
                text_out = gr.Textbox(
                    label="Model Response",
                    lines=12,
                    max_lines=25,
                    elem_id="model-response-output",
                )
                gr.Markdown("### Trigger Question (Ask & Get Response)")
                with gr.Row():
                    trigger_in = gr.Textbox(
                        label="Question",
                        placeholder="Type and click Trigger to ask the model",
                        scale=4,
                    )
                    trigger_btn = gr.Button("Trigger", variant="primary", scale=1)
                gr.Markdown("### Update Silent Mode Prompt (No output)")
                with gr.Row():
                    new_prompt_in = gr.Textbox(
                        label="New Silent Prompt",
                        placeholder="Type and click Send to update prompt",
                        scale=4,
                    )
                    send_btn = gr.Button("Send", scale=1)

        start_btn.click(
            start_camera_service,
            inputs=[
                host_in,
                port_in,
                model_in,
                prompt_in,
                camera_id_in,
                frame_interval_in,
                batch_size_in,
                queue_size_in,
                quality_in,
                resolution_in,
                max_size_in,
                keep_aspect_in,
            ],
            outputs=[start_btn, stop_btn],
        )
        stop_btn.click(
            stop_camera_service,
            outputs=[start_btn, stop_btn],
        )
        # Runtime resolution update
        update_res_btn.click(
            update_resolution_settings,
            inputs=[resolution_in, max_size_in, keep_aspect_in],
            outputs=[resolution_in, max_size_in, keep_aspect_in],
        )
        trigger_btn.click(
            send_trigger,
            inputs=[trigger_in],
            outputs=[trigger_in],
        )
        send_btn.click(
            send_prompt,
            inputs=[new_prompt_in],
            outputs=[new_prompt_in],
        )
        # Gradio 6+: use Timer instead of demo.load(every=...)
        timer = gr.Timer(value=0.1)  # Refresh every 100ms
        timer.tick(
            get_latest_display,
            outputs=[image_out, text_out],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Realtime Video WebSocket client for vLLM (live camera)"
    )
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device index. Default: 0.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Send every Nth frame (1=every frame, 10=every 10th frame). Default: 1.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Frames per batch; one commit per batch. "
            "Send rhythm is controlled by server water level (backpressure)."
        ),
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=64,
        help=(
            "Max frames in client queue. When full, oldest frames are "
            "dropped to make room for newer ones. Default: 64."
        ),
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG encoding quality (1-100). Default: 85.",
    )
    parser.add_argument(
        "--output-resolution",
        type=str,
        default=None,
        help=(
            "Output image resolution (exact size), format WIDTHxHEIGHT "
            "(e.g., 640x480, 1280x720). May stretch image. Default: use "
            "camera original resolution."
        ),
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=0,
        help=(
            "Maximum dimension for resize (0=disabled). Keeps aspect ratio. "
            "Larger of width/height will be scaled to this value. Default: 0."
        ),
    )
    parser.add_argument(
        "--keep-aspect-ratio",
        action="store_true",
        help=(
            "When using --output-resolution, fit image within target "
            "resolution while maintaining aspect ratio."
        ),
    )
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Launch Gradio UI (image + text side by side).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link (use with --gradio).",
    )
    args = parser.parse_args()

    if args.frame_interval < 1:
        parser.error("--frame-interval must be >= 1")

    if args.gradio:
        if gr is None:
            raise RuntimeError(
                "gradio is required for --gradio. "
                "Install with: pip install gradio"
            )
        global _prompt_queue
        _prompt_queue = queue.Queue()
        demo = create_gradio_demo(
            host=args.host,
            port=args.port,
            model=args.model,
            prompt=args.prompt,
            camera_id=args.camera_id,
            frame_interval=args.frame_interval,
            batch_size=args.batch_size,
            queue_size=args.queue_size,
            quality=args.quality,
            output_resolution=args.output_resolution,
            max_size=args.max_size,
            keep_aspect_ratio=args.keep_aspect_ratio,
        )
        demo.launch(share=args.share)
        return

    try:
        asyncio.run(
            run_realtime_camera(
                args.host,
                args.port,
                args.model,
                args.prompt,
                args.camera_id,
                args.frame_interval,
                args.batch_size,
                args.queue_size,
                args.quality,
                output_resolution=args.output_resolution,
                max_size=args.max_size if args.max_size > 0 else None,
                keep_aspect_ratio=args.keep_aspect_ratio,
            )
        )
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)


if __name__ == "__main__":
    main()