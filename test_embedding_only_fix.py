#!/usr/bin/env python3
"""
Combined test script: Starts server and runs embedding-only test
This allows agentic access to test the fix in one command
"""

import subprocess
import time
import sys
import signal
import os
import threading
from pathlib import Path

try:
    import requests
except ImportError:
    print("Installing requests...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

# Import the test client
from example_embedding_only_serve import main as run_client_test
from openai import OpenAI

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
PORT = 8000
SERVER_URL = f"http://localhost:{PORT}"

def check_server_ready(max_wait=300, process=None):
    """Check if server is ready by polling /health endpoint"""
    print(f"Waiting up to {max_wait} seconds for server to be ready...")
    for i in range(max_wait):
        # Check if process is still alive
        if process and process.poll() is not None:
            print(f"\n✗ Server process died unexpectedly (exit code: {process.returncode})")
            # Try to read any output
            try:
                output = process.stdout.read()
                if output:
                    print("Server output:")
                    print(output[-2000:])  # Last 2000 chars
            except:
                pass
            return False
        
        try:
            response = requests.get(f"{SERVER_URL}/health", timeout=2)
            if response.status_code == 200:
                print(f"✓ Server is ready (took {i+1}s)")
                return True
        except:
            pass
        
        # Show progress every 10 seconds
        if (i + 1) % 10 == 0:
            print(f"  Still waiting... ({i+1}s/{max_wait}s)")
        
        time.sleep(1)
    
    print(f"\n✗ Server failed to start within {max_wait} seconds")
    if process:
        # Try to read any output
        try:
            output = process.stdout.read()
            if output:
                print("\nLast server output:")
                print(output[-2000:])  # Last 2000 chars
        except:
            pass
    return False

def start_server():
    """Start vLLM server in background"""
    print("Starting vLLM server...")
    print(f"Model: {MODEL}")
    print(f"Port: {PORT}")
    
    # Use a clean HuggingFace cache to avoid corrupted cache issues
    import os
    env = os.environ.copy()
    clean_cache_dir = "/tmp/hf_cache_clean"
    os.makedirs(clean_cache_dir, exist_ok=True)
    env["HF_HOME"] = clean_cache_dir
    print(f"Using HuggingFace cache: {clean_cache_dir}")
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--enable-mm-embeds",
        "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
        "--port", str(PORT),
        "--max-model-len", "2048",  # Smaller for faster startup
    ]
    
    # Start server process - write to both file and capture
    log_file = open("/tmp/vllm_server.log", "w")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )
    
    print(f"Server process started (PID: {process.pid})")
    print("Server logs are being written to /tmp/vllm_server.log")
    print("You can monitor progress with: tail -f /tmp/vllm_server.log")
    
    # Start a thread to copy output to log file
    def log_output():
        try:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                # Also print important lines
                if any(keyword in line.lower() for keyword in ["error", "exception", "ready", "uvicorn", "started"]):
                    print(f"  [SERVER] {line.rstrip()}")
        except:
            pass
    
    log_thread = threading.Thread(target=log_output, daemon=True)
    log_thread.start()
    
    if not check_server_ready(max_wait=300, process=process):
        print("✗ Server failed to start within timeout")
        log_file.close()
        process.terminate()
        return None
    
    log_file.close()
    return process

def run_test():
    """Run the client test"""
    print("\n" + "="*60)
    print("Running client test...")
    print("="*60 + "\n")
    
    try:
        run_client_test()
        print("\n✓ Test completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner"""
    server_process = None
    
    try:
        # Start server
        server_process = start_server()
        if not server_process:
            print("Failed to start server")
            return 1
        
        # Run test
        success = run_test()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    finally:
        # Cleanup: kill server
        if server_process:
            print("\nShutting down server...")
            try:
                server_process.terminate()
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Force killing server...")
                server_process.kill()
                server_process.wait()
            print("Server stopped")

if __name__ == "__main__":
    sys.exit(main())

