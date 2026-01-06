# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example Python client for OpenAI Realtime API using vLLM API server

This script demonstrates how to connect to the vLLM Realtime API WebSocket
endpoint and exchange messages using the OpenAI Realtime protocol.

NOTE: Start a vLLM server with an audio model first, e.g.:
    export VLLM_TARGET_DEVICE=cpu
    python -m vllm.entrypoints.openai.api_server \
        --model fixie-ai/ultravox-v0_5-llama-3_2-1b \
        --trust-remote-code --dtype float32 --enforce-eager

Then run this client:
    python openai_realtime_client.py --mode text
    python openai_realtime_client.py --mode audio --audio-file /path/to/audio.wav
"""

import argparse
import asyncio
import base64
import json
import struct
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    raise


async def send_event(ws, event: dict) -> None:
    """Send an event to the server."""
    await ws.send(json.dumps(event))
    print(f">>> Sent: {event['type']}")


async def receive_events(ws, timeout: float = 5.0) -> list[dict]:
    """Receive events from the server with timeout."""
    events = []
    try:
        while True:
            try:
                data = await asyncio.wait_for(ws.recv(), timeout=timeout)
                event = json.loads(data)
                events.append(event)
                print(f"<<< Received: {event['type']}")
                
                # Print more details for certain events
                if event["type"] == "error":
                    print(f"    Error: {event['error']['message']}")
                elif event["type"] == "response.text.delta":
                    print(f"    Text delta: {event['delta']}")
                elif event["type"] == "response.audio_transcript.delta":
                    print(f"    Transcript delta: {event['delta']}")
                elif event["type"] == "response.done":
                    # Response is complete, return
                    return events
            except asyncio.TimeoutError:
                # No more events within timeout
                break
    except Exception as e:
        print(f"Error receiving: {e}")
    return events


def generate_mock_audio(duration_ms: int = 500, sample_rate: int = 24000) -> str:
    """Generate mock audio data (silence) as base64-encoded PCM16.
    
    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Base64-encoded PCM16 audio data
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    silence = struct.pack(f"<{num_samples}h", *([0] * num_samples))
    return base64.b64encode(silence).decode("ascii")


def load_audio_file(
    audio_path: str,
    target_sample_rate: int = 24000,
    pcm_sample_rate: int = 24000,
) -> str:
    """Load an audio file and convert to base64-encoded PCM16.
    
    Supports both standard audio formats (wav, mp3, etc.) via librosa,
    and raw PCM16 files (.pcm extension).
    
    Args:
        audio_path: Path to audio file (wav, mp3, pcm, etc.)
        target_sample_rate: Target sample rate for the model
        pcm_sample_rate: Sample rate of input PCM file (only used for .pcm files)
        
    Returns:
        Base64-encoded PCM16 audio data
    """
    import numpy as np
    
    # Check if it's a raw PCM file
    if audio_path.lower().endswith('.pcm'):
        # Read raw PCM16 data directly
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        
        # Convert to numpy array (PCM16 = int16, little-endian)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Resample if needed
        if pcm_sample_rate != target_sample_rate:
            try:
                import librosa
            except ImportError:
                print("Please install librosa for resampling: pip install librosa")
                raise
            
            # Convert to float for resampling
            audio_float = audio_int16.astype(np.float32) / 32767.0
            audio_resampled = librosa.resample(
                audio_float, orig_sr=pcm_sample_rate, target_sr=target_sample_rate
            )
            # Convert back to int16
            audio_int16 = (audio_resampled * 32767).astype(np.int16)
        
        audio_bytes = audio_int16.tobytes()
        duration = len(audio_int16) / target_sample_rate
        print(f"Loaded PCM audio: {duration:.2f}s at {target_sample_rate}Hz")
        
        return base64.b64encode(audio_bytes).decode("ascii")
    
    # Standard audio format - use librosa
    try:
        import librosa
    except ImportError:
        print("Please install librosa: pip install librosa")
        raise
    
    # Load audio and resample to target sample rate
    audio, sr = librosa.load(audio_path, sr=target_sample_rate, mono=True)
    
    # Convert float32 [-1.0, 1.0] to PCM16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Pack as little-endian bytes
    audio_bytes = audio_int16.tobytes()
    
    print(f"Loaded audio: {len(audio) / target_sample_rate:.2f}s at {target_sample_rate}Hz")
    
    return base64.b64encode(audio_bytes).decode("ascii")


async def test_text_response(ws) -> None:
    """Test text-only response generation."""
    print("\n=== Testing Text Response ===")
    
    # Update session to text-only mode
    await send_event(ws, {
        "type": "session.update",
        "session": {
            "modalities": ["text"],
            "instructions": "You are a helpful assistant.",
        }
    })
    
    # Wait for session.updated
    await receive_events(ws, timeout=2.0)
    
    # Create a user message
    await send_event(ws, {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": "Hello, how are you?"
            }]
        }
    })
    
    # Wait for item created
    await receive_events(ws, timeout=2.0)
    
    # Request a response
    await send_event(ws, {
        "type": "response.create",
        "response": {
            "modalities": ["text"]
        }
    })
    
    # Receive all response events (longer timeout for CPU inference)
    await receive_events(ws, timeout=120.0)


async def test_audio_response(
    ws,
    audio_file: str | None = None,
    pcm_sample_rate: int = 24000,
) -> None:
    """Test audio input and text response generation.
    
    Args:
        ws: WebSocket connection
        audio_file: Optional path to audio file. If None, mock audio is used.
        pcm_sample_rate: Sample rate of input PCM file (for .pcm files only)
    """
    print("\n=== Testing Audio Input ===")
    
    # Update session - note: Ultravox is audio-to-text only, so output is text
    await send_event(ws, {
        "type": "session.update",
        "session": {
            "modalities": ["text"],  # Output is text only for Ultravox
            "instructions": "You are a helpful assistant. Respond to the audio input.",
        }
    })
    
    # Wait for session.updated
    await receive_events(ws, timeout=2.0)
    
    # Load or generate audio data
    if audio_file:
        print(f"Loading audio from: {audio_file}")
        audio_data = load_audio_file(audio_file, pcm_sample_rate=pcm_sample_rate)
    else:
        print("Using mock audio (500ms silence)")
        audio_data = generate_mock_audio(500)  # 500ms of silence
    
    # Send audio in chunks (max 15MB per chunk recommended)
    # For simplicity, send all at once if small enough
    chunk_size = 1024 * 1024  # 1MB chunks
    audio_bytes = base64.b64decode(audio_data)
    
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        chunk_b64 = base64.b64encode(chunk).decode("ascii")
        await send_event(ws, {
            "type": "input_audio_buffer.append",
            "audio": chunk_b64
        })
    
    print(f"Sent {len(audio_bytes)} bytes of audio data")
    
    # Commit the audio buffer
    await send_event(ws, {
        "type": "input_audio_buffer.commit"
    })
    
    # Wait for buffer committed and item created
    await receive_events(ws, timeout=2.0)
    
    # Request a response (text only for Ultravox)
    await send_event(ws, {
        "type": "response.create",
        "response": {
            "modalities": ["text"]
        }
    })
    
    # Receive all response events
    print("\nWaiting for model response...")
    events = await receive_events(ws, timeout=60.0)  # Longer timeout for real inference
    
    # Collect full text response
    full_text = ""
    for e in events:
        if e["type"] == "response.text.delta":
            full_text += e.get("delta", "")
    
    if full_text:
        print(f"\n=== Model Response ===\n{full_text}")


async def main(args):
    """Main function to run the Realtime API client."""
    uri = f"ws://{args.host}:{args.port}/v1/realtime"
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as ws:
            print("Connected!")
            
            # Wait for initial session.created and conversation.created events
            print("\n=== Initial Connection ===")
            events = await receive_events(ws, timeout=5.0)
            
            if not events:
                print("No initial events received. Server may not support Realtime API.")
                return
            
            # Check if we got session.created
            session_created = any(e["type"] == "session.created" for e in events)
            if not session_created:
                print("Did not receive session.created event")
                return
            
            print("\nSession established successfully!")
            
            # Run tests based on mode
            if args.mode in ("text", "both"):
                await test_text_response(ws)
            
            if args.mode in ("audio", "both"):
                await test_audio_response(ws, args.audio_file, args.pcm_sample_rate)
            
            print("\n=== Tests Complete ===")
            
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed: {e}")
    except ConnectionRefusedError:
        print(f"Could not connect to {uri}. Is the vLLM server running?")
    except Exception as e:
        print(f"Error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test client for vLLM OpenAI Realtime API"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["text", "audio", "both"],
        default="both",
        help="Test mode: text, audio, or both (default: both)"
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Path to audio file for audio mode (wav, mp3, pcm, etc.). "
             "If not provided, mock audio (silence) is used."
    )
    parser.add_argument(
        "--pcm-sample-rate",
        type=int,
        default=24000,
        help="Sample rate of input PCM file (default: 24000). "
             "Only used when audio-file is a .pcm file."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))

