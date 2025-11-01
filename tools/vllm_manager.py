"""
vLLM Manager - FastAPI middleware to manage vLLM server
Provides endpoints to start/stop vLLM, download models, check disk space, and switch models
"""

import os
import sys
import signal
import shutil
import subprocess
import psutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import asyncio
import aiohttp

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, HttpUrl
import json
from collections import deque

# GPU monitoring
try:
    import pynvml
    GPU_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: pynvml not available. GPU monitoring will be disabled.")
except Exception as e:
    GPU_AVAILABLE = False
    print(f"Warning: GPU monitoring unavailable: {e}")


# Configuration
MODELS_DIR = os.getenv("VLLM_MODELS_DIR", str(Path.home() / ".cache" / "huggingface" / "hub"))
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_HOST = os.getenv("VLLM_HOST", "0.0.0.0")

# Crash loop detection configuration
CRASH_LOOP_THRESHOLD = 3  # Number of failures before stopping auto-restart
CRASH_LOOP_WINDOW = 300  # Time window in seconds (5 minutes)
HEALTH_CHECK_INTERVAL = 10  # Health check every 10 seconds
RESTART_DELAY = 5  # Wait 5 seconds before restarting

# Crash loop detector class
class CrashLoopDetector:
    def __init__(self, threshold: int = CRASH_LOOP_THRESHOLD, window: int = CRASH_LOOP_WINDOW):
        self.threshold = threshold
        self.window = window
        self.failures = deque()
        self.crash_loop_active = False
        self.crash_loop_message = ""

    def record_failure(self, reason: str = "Service stopped unexpectedly"):
        """Record a service failure"""
        now = datetime.now()
        self.failures.append((now, reason))

        # Clean old failures outside the window
        cutoff = now.timestamp() - self.window
        while self.failures and self.failures[0][0].timestamp() < cutoff:
            self.failures.popleft()

        # Check if we've hit crash loop threshold
        if len(self.failures) >= self.threshold:
            self.crash_loop_active = True
            self.crash_loop_message = f"Crash loop detected: {len(self.failures)} failures in {self.window}s. Last reason: {reason}"
            return True
        return False

    def is_crash_loop(self) -> bool:
        """Check if we're in a crash loop"""
        return self.crash_loop_active

    def reset(self):
        """Reset crash loop state"""
        self.failures.clear()
        self.crash_loop_active = False
        self.crash_loop_message = ""

    def get_status(self) -> dict:
        """Get current crash loop status"""
        return {
            "crash_loop_active": self.crash_loop_active,
            "message": self.crash_loop_message,
            "failure_count": len(self.failures),
            "threshold": self.threshold
        }

# Global state
current_model: Optional[str] = None
context_window: Optional[int] = None
SERVICE_NAME = "vllm.service"
crash_detector = CrashLoopDetector()
auto_restart_enabled = False
last_known_state = "stopped"  # Track service state

app = FastAPI(
    title="vLLM Manager",
    description="Middleware to manage vLLM server lifecycle and models",
    version="1.0.0"
)


# Pydantic models
class StartVLLMRequest(BaseModel):
    model_name: str
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = 30000  # Default to 30k context window
    tensor_parallel_size: int = 1
    additional_args: Optional[Dict[str, Any]] = None


class DownloadModelRequest(BaseModel):
    model_url: str
    model_name: Optional[str] = None


class ModelSwitchRequest(BaseModel):
    model_name: str
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = 30000  # Default to 30k context window
    tensor_parallel_size: int = 1


# Helper functions
def get_disk_space():
    """Get disk space information for the models directory"""
    try:
        usage = shutil.disk_usage(MODELS_DIR)
        return {
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "percent_used": round((usage.used / usage.total) * 100, 2),
            "models_dir": MODELS_DIR
        }
    except Exception as e:
        return {"error": str(e)}


def get_gpu_info():
    """Get GPU usage and memory information"""
    if not GPU_AVAILABLE:
        return None
    
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        gpus = []
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get GPU name
            name_bytes = pynvml.nvmlDeviceGetName(handle)
            name = name_bytes.decode('utf-8') if isinstance(name_bytes, bytes) else str(name_bytes)
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = mem_info.total / (1024**3)  # Convert to GB
            used_memory = mem_info.used / (1024**3)
            free_memory = mem_info.free / (1024**3)
            memory_percent = (mem_info.used / mem_info.total) * 100
            
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = util.gpu
            memory_utilization = util.memory
            
            # Get temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = None
            
            # Get power usage
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            except:
                power = None
            
            gpus.append({
                "id": i,
                "name": name,
                "memory_total_gb": round(total_memory, 2),
                "memory_used_gb": round(used_memory, 2),
                "memory_free_gb": round(free_memory, 2),
                "memory_percent": round(memory_percent, 1),
                "gpu_utilization": gpu_utilization,
                "memory_utilization": memory_utilization,
                "temperature_c": temp,
                "power_watts": power
            })
        
        return {
            "available": True,
            "count": device_count,
            "gpus": gpus
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


def run_systemctl_command(action: str) -> tuple[bool, str]:
    """Run systemctl command and return success status and output"""
    try:
        result = subprocess.run(
            ['sudo', 'systemctl', action, SERVICE_NAME],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def get_service_status() -> tuple[bool, str]:
    """Check if vLLM service is active"""
    try:
        result = subprocess.run(
            ['systemctl', 'is-active', SERVICE_NAME],
            capture_output=True,
            text=True,
            timeout=10
        )
        is_active = result.returncode == 0 and result.stdout.strip() == 'active'
        return is_active, result.stdout.strip()
    except Exception as e:
        return False, str(e)


def is_vllm_running():
    """Check if vLLM service is running"""
    is_active, _ = get_service_status()
    return is_active


async def get_vllm_model_info():
    """Query vLLM API to get current model and context window"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f'http://localhost:{VLLM_PORT}/v1/models', timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data and len(data['data']) > 0:
                        model_data = data['data'][0]
                        return {
                            'model_path': model_data.get('id', ''),
                            'model_name': model_data.get('id', '').split('/')[-1],
                            'max_model_len': model_data.get('max_model_len')
                        }
    except Exception as e:
        print(f"Failed to query vLLM API: {e}")
    return None


def get_vllm_process_info():
    """Get detailed information about the vLLM service"""
    is_active, status = get_service_status()
    if not is_active:
        return None

    try:
        # Get service status details
        result = subprocess.run(
            ['systemctl', 'show', SERVICE_NAME, '--property=MainPID,ActiveState,SubState'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return None

        # Parse systemctl output
        properties = {}
        for line in result.stdout.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                properties[key] = value

        pid = int(properties.get('MainPID', '0'))
        if pid == 0:
            return None

        # Get process info using psutil
        proc = psutil.Process(pid)
        return {
            "pid": pid,
            "status": properties.get('SubState', 'unknown'),
            "cpu_percent": proc.cpu_percent(interval=0.1),
            "memory_mb": round(proc.memory_info().rss / (1024**2), 2),
            "create_time": datetime.fromtimestamp(proc.create_time()).isoformat(),
            "model": current_model
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, subprocess.TimeoutExpired):
        return None


async def download_model_from_url(url: str, destination: str):
    """Download a model file from a URL"""
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download model: HTTP {response.status}")

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(destination, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"Download progress: {progress:.2f}%")


async def check_vllm_health() -> tuple[bool, str]:
    """Check if vLLM API is responsive"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f'http://localhost:{VLLM_PORT}/health',
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    return True, "healthy"
                else:
                    return False, f"API returned status {response.status}"
    except asyncio.TimeoutError:
        return False, "health check timeout"
    except aiohttp.ClientConnectorError:
        return False, "connection refused"
    except Exception as e:
        return False, f"health check failed: {str(e)}"


async def auto_restart_service():
    """Attempt to restart the service"""
    global last_known_state

    if crash_detector.is_crash_loop():
        print(f"⚠️  Not restarting - crash loop detected: {crash_detector.crash_loop_message}")
        return False

    print(f"🔄 Auto-restarting vLLM service...")
    success, output = run_systemctl_command("restart")

    if success:
        print(f"✅ Service restarted successfully")
        await asyncio.sleep(RESTART_DELAY)
        return True
    else:
        print(f"❌ Failed to restart service: {output}")
        crash_detector.record_failure(f"Restart failed: {output}")
        return False


async def monitor_service_health():
    """Background task to monitor service health and auto-restart if needed"""
    global last_known_state, auto_restart_enabled

    while True:
        try:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)

            # Check if service is running
            is_running = is_vllm_running()

            if is_running:
                # Service is running, check API health
                is_healthy, health_msg = await check_vllm_health()

                if is_healthy:
                    # Service is healthy
                    if last_known_state != "running":
                        print(f"✅ vLLM service is now healthy")
                        crash_detector.reset()  # Reset crash detector on successful recovery
                    last_known_state = "running"
                else:
                    # Service running but API unhealthy
                    print(f"⚠️  vLLM API unhealthy: {health_msg}")

                    if auto_restart_enabled and not crash_detector.is_crash_loop():
                        crash_detector.record_failure(f"API unhealthy: {health_msg}")
                        await auto_restart_service()

                    last_known_state = "unhealthy"
            else:
                # Service not running
                if last_known_state == "running":
                    # Service was running but now stopped - unexpected
                    print(f"❌ vLLM service stopped unexpectedly")
                    crash_detector.record_failure("Service stopped unexpectedly")

                    if auto_restart_enabled and not crash_detector.is_crash_loop():
                        await auto_restart_service()
                    elif crash_detector.is_crash_loop():
                        print(f"🛑 Crash loop detected - stopping auto-restart")
                        print(f"   {crash_detector.crash_loop_message}")
                        # Stop the service to prevent further restart attempts
                        run_systemctl_command("stop")

                last_known_state = "stopped"

        except Exception as e:
            print(f"Error in health monitor: {e}")


# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vLLM Manager</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }

        h1 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2em;
        }

        .version {
            color: #888;
            margin-bottom: 30px;
            font-size: 0.9em;
        }

        .status-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .status-item {
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 10px;
        }

        .status-label {
            font-size: 0.85em;
            opacity: 0.9;
            margin-bottom: 5px;
        }

        .status-value {
            font-size: 1.2em;
            font-weight: bold;
        }

        .section {
            margin-bottom: 30px;
        }

        .section-title {
            font-size: 1.3em;
            color: #333;
            margin-bottom: 15px;
            font-weight: 600;
        }

        .button-group {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        button {
            padding: 15px 25px;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            flex: 1;
            min-width: 120px;
        }

        .btn-start {
            background: #10b981;
            color: white;
        }

        .btn-start:hover {
            background: #059669;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(16, 185, 129, 0.4);
        }

        .btn-stop {
            background: #ef4444;
            color: white;
        }

        .btn-stop:hover {
            background: #dc2626;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(239, 68, 68, 0.4);
        }

        .btn-refresh {
            background: #3b82f6;
            color: white;
        }

        .btn-refresh:hover {
            background: #2563eb;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
        }

        .btn-download {
            background: #8b5cf6;
            color: white;
        }

        .btn-download:hover {
            background: #7c3aed;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(139, 92, 246, 0.4);
        }

        .btn-switch {
            background: #f59e0b;
            color: white;
        }

        .btn-switch:hover {
            background: #d97706;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(245, 158, 11, 0.4);
        }

        .btn-delete {
            background: #dc2626;
            color: white;
        }

        .btn-delete:hover {
            background: #991b1b;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(220, 38, 38, 0.4);
        }

        input, select {
            width: 100%;
            padding: 15px;
            border: 2px solid #e5e7eb;
            border-radius: 10px;
            font-size: 1em;
            margin-bottom: 10px;
            transition: border 0.3s;
        }

        input[type="number"] {
            -moz-appearance: textfield;
        }

        input[type="number"]::-webkit-outer-spin-button,
        input[type="number"]::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }

        select {
            min-width: 100%;
            max-width: 100%;
            -webkit-appearance: none;
            -moz-appearance: none;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23667eea' d='M6 9L1 4h10z'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 15px center;
            padding-right: 40px;
        }

        select option {
            padding: 10px;
            min-width: 100%;
            white-space: normal;
            word-wrap: break-word;
        }

        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }

        .input-group {
            margin-bottom: 15px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 600;
        }

        .notification {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: none;
        }

        .notification.success {
            background: #d1fae5;
            color: #065f46;
            border: 2px solid #10b981;
        }

        .notification.error {
            background: #fee2e2;
            color: #991b1b;
            border: 2px solid #ef4444;
        }

        .notification.info {
            background: #dbeafe;
            color: #1e40af;
            border: 2px solid #3b82f6;
        }

        .indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .indicator.running {
            background: #10b981;
            box-shadow: 0 0 10px #10b981;
            animation: pulse 2s infinite;
        }

        .indicator.stopped {
            background: #ef4444;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .spinner {
            border: 4px solid #f3f4f6;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @media (max-width: 600px) {
            .container {
                padding: 20px;
            }

            h1 {
                font-size: 1.5em;
            }

            .button-group {
                flex-direction: column;
            }

            button {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>vLLM Manager</h1>
        <div class="version">v1.0.0</div>

        <div id="notification" class="notification"></div>

        <div class="status-box">
            <h2 style="margin-bottom: 5px;">System Status</h2>
            <div style="display: flex; gap: 20px; margin-bottom: 15px; flex-wrap: wrap;">
                <div>
                    <span class="indicator running"></span>
                    <span style="font-size: 0.9em;">Manager: Running</span>
                </div>
                <div>
                    <span id="statusIndicator" class="indicator stopped"></span>
                    <span style="font-size: 0.9em;">vLLM Inference: <span id="vllmStatusText">Stopped</span></span>
                </div>
            </div>
            <!-- Full width current model display -->
            <div class="status-item" style="margin-bottom: 15px; grid-column: 1 / -1;">
                <div class="status-label">Current Model</div>
                <div class="status-value" id="currentModel" style="font-size: 1em;">None</div>
            </div>

            <!-- Progress bar for loading states -->
            <div id="loadingProgress" style="display: none; margin-bottom: 15px;">
                <div style="background: rgba(255,255,255,0.2); border-radius: 10px; height: 8px; overflow: hidden;">
                    <div id="progressBar" style="background: #10b981; height: 100%; width: 0%; transition: width 0.3s;"></div>
                </div>
                <div style="font-size: 0.85em; margin-top: 5px; opacity: 0.9;" id="loadingMessage">Loading model...</div>
            </div>

            <div class="status-grid">
                <div class="status-item">
                    <div class="status-label">Context Window</div>
                    <div class="status-value" id="contextWindow">-</div>
                </div>
                <div class="status-item">
                    <div class="status-label">vLLM Port</div>
                    <div class="status-value" id="vllmPort">8000</div>
                </div>
                <div class="status-item">
                    <div class="status-label">CPU Usage</div>
                    <div class="status-value" id="cpuUsage">-</div>
                </div>
                <div class="status-item">
                    <div class="status-label">GPU Usage</div>
                    <div class="status-value" id="gpuUsage">-</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Memory</div>
                    <div class="status-value" id="memoryUsage">-</div>
                </div>
                <div class="status-item">
                    <div class="status-label">GPU Memory</div>
                    <div class="status-value" id="gpuMemory">-</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Free Disk</div>
                    <div class="status-value" id="freeDisk">-</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Total Models</div>
                    <div class="status-value" id="totalModels">-</div>
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Control Panel</div>
            <div class="input-group">
                <label for="startContextWindow">Context Window (for new start):</label>
                <input type="number" id="startContextWindow" placeholder="30000" min="1024" max="100000" step="1024" />
            </div>
            <div class="button-group">
                <button class="btn-start" onclick="startVLLM()">▶ Start vLLM</button>
                <button class="btn-stop" onclick="stopVLLM()">■ Stop vLLM</button>
                <button class="btn-refresh" onclick="refreshStatus()">↻ Refresh</button>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Health Monitoring & Auto-Restart</div>
            <div id="crashLoopAlert" class="notification error" style="display: none;">
                <strong>⚠️ Crash Loop Detected!</strong>
                <p id="crashLoopMessage"></p>
            </div>
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 200px;">
                    <div style="font-weight: 600; margin-bottom: 5px;">Auto-Restart:</div>
                    <div id="autoRestartStatus" style="font-size: 0.9em; color: #666;">Disabled</div>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <div style="font-weight: 600; margin-bottom: 5px;">API Health:</div>
                    <div id="apiHealthStatus" style="font-size: 0.9em; color: #666;">-</div>
                </div>
            </div>
            <div class="button-group">
                <button id="toggleAutoRestart" class="btn-start" onclick="toggleAutoRestart()">Enable Auto-Restart</button>
                <button class="btn-refresh" onclick="resetCrashLoop()">Reset Crash Loop</button>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Switch Model</div>
            <div class="input-group">
                <label for="modelSelect">Select Model:</label>
                <select id="modelSelect">
                    <option value="">Loading models...</option>
                </select>
            </div>
            <div class="input-group">
                <label for="contextWindow">Context Window:</label>
                <input type="number" id="contextWindow" placeholder="30000" min="1024" max="100000" step="1024" />
            </div>
            <div class="button-group">
                <button class="btn-switch" onclick="switchModel()">Switch Model</button>
                <button class="btn-delete" onclick="deleteModel()">🗑️ Delete Model</button>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Download Model</div>
            <div class="input-group">
                <label for="modelUrl">Model URL or HuggingFace ID:</label>
                <input type="text" id="modelUrl" placeholder="e.g., meta-llama/Llama-2-7b-hf or https://..." />
            </div>
            <button class="btn-download" onclick="downloadModel()">⬇ Download Model</button>
        </div>

        <div id="loading" class="loading">
            <div class="spinner"></div>
            <p style="margin-top: 10px;">Processing...</p>
        </div>
    </div>

    <script>
        let selectedModelName = '';

        function showNotification(message, type = 'info') {
            const notification = document.getElementById('notification');
            notification.textContent = message;
            notification.className = 'notification ' + type;
            notification.style.display = 'block';
            setTimeout(() => {
                notification.style.display = 'none';
            }, 5000);
        }

        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }

        function showProgress(show, message = 'Loading model...', progress = 0) {
            const progressEl = document.getElementById('loadingProgress');
            const progressBar = document.getElementById('progressBar');
            const messageEl = document.getElementById('loadingMessage');

            progressEl.style.display = show ? 'block' : 'none';
            if (show) {
                progressBar.style.width = progress + '%';
                messageEl.textContent = message;
            }
        }

        async function refreshStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();

                // Update status
                const isRunning = data.vllm_running;
                document.getElementById('vllmStatusText').textContent = isRunning ? 'Running' : 'Stopped';
                document.getElementById('currentModel').textContent = data.current_model || 'None';

                const indicator = document.getElementById('statusIndicator');
                indicator.className = 'indicator ' + (isRunning ? 'running' : 'stopped');

                if (data.process_info) {
                    document.getElementById('cpuUsage').textContent = data.process_info.cpu_percent.toFixed(1) + '%';
                    document.getElementById('memoryUsage').textContent = data.process_info.memory_mb.toFixed(0) + ' MB';
                } else {
                    document.getElementById('cpuUsage').textContent = '-';
                    document.getElementById('memoryUsage').textContent = '-';
                }

                // Update GPU information
                if (data.gpu_info && data.gpu_info.available && data.gpu_info.gpus.length > 0) {
                    const gpu = data.gpu_info.gpus[0]; // Use first GPU for display
                    document.getElementById('gpuUsage').textContent = gpu.gpu_utilization + '%';
                    document.getElementById('gpuMemory').textContent = gpu.memory_used_gb.toFixed(1) + ' / ' + gpu.memory_total_gb.toFixed(1) + ' GB';
                } else {
                    document.getElementById('gpuUsage').textContent = '-';
                    document.getElementById('gpuMemory').textContent = '-';
                }

                if (data.disk_space) {
                    document.getElementById('freeDisk').textContent = data.disk_space.free_gb + ' GB';
                }

                // Update context window
                document.getElementById('contextWindow').textContent = data.context_window || '-';

                // Update context window input fields
                if (data.context_window) {
                    const contextValue = data.context_window.replace(/,/g, '');
                    document.getElementById('startContextWindow').value = contextValue;
                    document.getElementById('contextWindow').value = contextValue;
                } else {
                    document.getElementById('startContextWindow').value = '30000';
                    document.getElementById('contextWindow').value = '30000';
                }

                // Update auto-restart status
                const autoRestartBtn = document.getElementById('toggleAutoRestart');
                if (data.auto_restart_enabled) {
                    document.getElementById('autoRestartStatus').textContent = '✅ Enabled';
                    document.getElementById('autoRestartStatus').style.color = '#10b981';
                    autoRestartBtn.textContent = 'Disable Auto-Restart';
                    autoRestartBtn.className = 'btn-stop';
                } else {
                    document.getElementById('autoRestartStatus').textContent = '❌ Disabled';
                    document.getElementById('autoRestartStatus').style.color = '#ef4444';
                    autoRestartBtn.textContent = 'Enable Auto-Restart';
                    autoRestartBtn.className = 'btn-start';
                }

                // Update API health status
                if (isRunning) {
                    if (data.api_healthy) {
                        document.getElementById('apiHealthStatus').textContent = '✅ Healthy';
                        document.getElementById('apiHealthStatus').style.color = '#10b981';
                    } else {
                        document.getElementById('apiHealthStatus').textContent = '❌ ' + data.health_message;
                        document.getElementById('apiHealthStatus').style.color = '#ef4444';
                    }
                } else {
                    document.getElementById('apiHealthStatus').textContent = 'Service not running';
                    document.getElementById('apiHealthStatus').style.color = '#888';
                }

                // Update crash loop status
                if (data.crash_loop_status && data.crash_loop_status.crash_loop_active) {
                    document.getElementById('crashLoopAlert').style.display = 'block';
                    document.getElementById('crashLoopMessage').textContent = data.crash_loop_status.message;
                } else {
                    document.getElementById('crashLoopAlert').style.display = 'none';
                }

                // Load models
                const modelsResponse = await fetch('/models');
                const modelsData = await modelsResponse.json();
                document.getElementById('totalModels').textContent = modelsData.count;

                const select = document.getElementById('modelSelect');
                const currentSelection = select.value; // Save current selection
                select.innerHTML = '<option value="">-- Select a model --</option>';
                modelsData.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.name;
                    option.textContent = model.name + ' (' + model.size_gb + ' GB)';
                    select.appendChild(option);
                });

                // Restore previous selection
                if (currentSelection) {
                    select.value = currentSelection;
                } else if (data.current_model) {
                    // Or set to currently running model
                    const runningModel = 'models--' + data.current_model.replace('/', '--');
                    select.value = runningModel;
                }

            } catch (error) {
                showNotification('Error refreshing status: ' + error.message, 'error');
            }
        }

        async function startVLLM() {
            const modelName = document.getElementById('modelSelect').value;
            if (!modelName) {
                showNotification('Please select a model first', 'error');
                return;
            }

            const contextWindow = document.getElementById('startContextWindow').value || 30000;

            showLoading(true);
            showProgress(true, 'Starting vLLM...', 10);

            try {
                showProgress(true, 'Loading model weights...', 30);
                const response = await fetch('/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_name: modelName.replace('models--', '').replace('--', '/'),
                        gpu_memory_utilization: 0.85,
                        max_model_len: parseInt(contextWindow)
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    showProgress(true, 'Model loaded successfully!', 100);
                    showNotification('vLLM started successfully with model: ' + data.model, 'success');
                    setTimeout(() => {
                        showProgress(false);
                        refreshStatus();
                    }, 1000);
                } else {
                    showProgress(false);
                    showNotification('Failed to start: ' + data.detail, 'error');
                }
            } catch (error) {
                showProgress(false);
                showNotification('Error starting vLLM: ' + error.message, 'error');
            } finally {
                showLoading(false);
            }
        }

        async function stopVLLM() {
            showLoading(true);
            try {
                const response = await fetch('/stop', { method: 'POST' });
                const data = await response.json();

                if (response.ok) {
                    showNotification('vLLM stopped successfully', 'success');
                    setTimeout(refreshStatus, 1000);
                } else {
                    showNotification('Error: ' + data.detail, 'error');
                }
            } catch (error) {
                showNotification('Error stopping vLLM: ' + error.message, 'error');
            } finally {
                showLoading(false);
            }
        }

        async function switchModel() {
            const modelName = document.getElementById('modelSelect').value;
            if (!modelName) {
                showNotification('Please select a model first', 'error');
                return;
            }

            const contextWindow = document.getElementById('contextWindow').value || 30000;

            showLoading(true);
            showProgress(true, 'Stopping current model...', 10);

            try {
                showProgress(true, 'Unloading previous model...', 30);
                await new Promise(resolve => setTimeout(resolve, 500));

                showProgress(true, 'Loading new model...', 50);
                const response = await fetch('/switch-model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_name: modelName.replace('models--', '').replace('--', '/'),
                        gpu_memory_utilization: 0.85,
                        max_model_len: parseInt(contextWindow)
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    showProgress(true, 'Model switched successfully!', 100);
                    showNotification('Switched to model: ' + data.model, 'success');
                    setTimeout(() => {
                        showProgress(false);
                        refreshStatus();
                    }, 1000);
                } else {
                    showProgress(false);
                    showNotification('Failed to switch: ' + data.detail, 'error');
                }
            } catch (error) {
                showProgress(false);
                showNotification('Error switching model: ' + error.message, 'error');
            } finally {
                showLoading(false);
            }
        }

        async function downloadModel() {
            const modelUrl = document.getElementById('modelUrl').value.trim();
            if (!modelUrl) {
                showNotification('Please enter a model URL or HuggingFace ID', 'error');
                return;
            }

            showLoading(true);
            try {
                const response = await fetch('/download-model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_url: modelUrl })
                });

                const data = await response.json();

                if (response.ok) {
                    showNotification('Model download started. Check server logs for progress.', 'success');
                    document.getElementById('modelUrl').value = '';
                    setTimeout(refreshStatus, 3000);
                } else {
                    showNotification('Error: ' + data.detail, 'error');
                }
            } catch (error) {
                showNotification('Error downloading model: ' + error.message, 'error');
            } finally {
                showLoading(false);
            }
        }

        async function deleteModel() {
            const modelName = document.getElementById('modelSelect').value;
            if (!modelName) {
                showNotification('Please select a model to delete', 'error');
                return;
            }

            if (!confirm(`Are you sure you want to delete ${modelName}?\n\nThis cannot be undone.`)) {
                return;
            }

            showLoading(true);
            try {
                const response = await fetch('/delete-model', {
                    method: 'DELETE',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_name: modelName })
                });

                const data = await response.json();

                if (response.ok) {
                    showNotification('Model deleted successfully', 'success');
                    refreshStatus();
                } else {
                    showNotification('Error: ' + data.detail, 'error');
                }
            } catch (error) {
                showNotification('Error deleting model: ' + error.message, 'error');
            } finally {
                showLoading(false);
            }
        }

        async function toggleAutoRestart() {
            showLoading(true);
            try {
                // Check current status to determine action
                const statusResponse = await fetch('/status');
                const statusData = await statusResponse.json();
                const isEnabled = statusData.auto_restart_enabled;

                const endpoint = isEnabled ? '/auto-restart/disable' : '/auto-restart/enable';
                const response = await fetch(endpoint, { method: 'POST' });
                const data = await response.json();

                if (response.ok) {
                    showNotification(data.message, 'success');
                    refreshStatus();
                } else {
                    showNotification('Error: ' + data.detail, 'error');
                }
            } catch (error) {
                showNotification('Error toggling auto-restart: ' + error.message, 'error');
            } finally {
                showLoading(false);
            }
        }

        async function resetCrashLoop() {
            showLoading(true);
            try {
                const response = await fetch('/crash-loop/reset', { method: 'POST' });
                const data = await response.json();

                if (response.ok) {
                    showNotification(data.message, 'success');
                    refreshStatus();
                } else {
                    showNotification('Error: ' + data.detail, 'error');
                }
            } catch (error) {
                showNotification('Error resetting crash loop: ' + error.message, 'error');
            } finally {
                showLoading(false);
            }
        }

        // Auto-refresh status every 30 seconds
        setInterval(refreshStatus, 30000);

        // Initial load
        refreshStatus();
    </script>
</body>
</html>
    """

@app.get("/api")
async def api_info():
    """API endpoint with information"""
    return {
        "service": "vLLM Manager",
        "version": "1.0.0",
        "status": "running",
        "vllm_running": is_vllm_running(),
        "current_model": current_model
    }


@app.get("/status")
async def get_status():
    """Get current status of vLLM and system"""
    is_running = is_vllm_running()

    # Try to get model info from API if service is running
    model_name = current_model
    ctx_window = context_window

    if is_running:
        api_info = await get_vllm_model_info()
        if api_info:
            model_name = api_info['model_name']
            ctx_window = api_info['max_model_len']

    ctx_window_formatted = f"{ctx_window:,}" if ctx_window else None

    # Check API health
    api_healthy = False
    health_msg = "not checked"
    if is_running:
        api_healthy, health_msg = await check_vllm_health()

    return {
        "vllm_running": is_running,
        "api_healthy": api_healthy,
        "health_message": health_msg,
        "current_model": model_name,
        "context_window": ctx_window_formatted,
        "process_info": get_vllm_process_info(),
        "disk_space": get_disk_space(),
        "gpu_info": get_gpu_info(),
        "vllm_port": VLLM_PORT,
        "models_dir": MODELS_DIR,
        "auto_restart_enabled": auto_restart_enabled,
        "crash_loop_status": crash_detector.get_status()
    }


@app.post("/start")
async def start_vllm(request: StartVLLMRequest):
    """Start vLLM service with specified model"""
    global current_model, context_window

    if is_vllm_running():
        raise HTTPException(
            status_code=400,
            detail=f"vLLM service is already running with model: {current_model}"
        )

    # Update the service file with new parameters if needed
    # For now, we'll just start the existing service
    # In a production setup, you might want to dynamically update the service file
    
    try:
        # Start the vLLM service
        success, output = run_systemctl_command("start")
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to start vLLM service: {output}")
        
        # Update global state
        current_model = request.model_name
        context_window = request.max_model_len

        # Wait a moment to check if it started successfully
        await asyncio.sleep(3)

        if not is_vllm_running():
            raise HTTPException(status_code=500, detail="vLLM service failed to start")

        return {
            "status": "started",
            "model": current_model,
            "service": SERVICE_NAME,
            "port": VLLM_PORT,
            "message": "vLLM service started successfully"
        }
    except Exception as e:
        current_model = None
        context_window = None
        raise HTTPException(status_code=500, detail=f"Failed to start vLLM service: {str(e)}")


@app.post("/stop")
async def stop_vllm():
    """Stop the running vLLM service"""
    global current_model

    if not is_vllm_running():
        raise HTTPException(status_code=400, detail="vLLM service is not running")

    try:
        old_model = current_model

        # Stop the vLLM service
        success, output = run_systemctl_command("stop")
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to stop vLLM service: {output}")

        # Wait for service to stop
        for _ in range(10):
            if not is_vllm_running():
                break
            await asyncio.sleep(1)

        current_model = None
        context_window = None

        return {
            "status": "stopped",
            "model": old_model,
            "service": SERVICE_NAME,
            "message": "vLLM service has been stopped"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop vLLM service: {str(e)}")


@app.post("/restart")
async def restart_vllm(request: StartVLLMRequest):
    """Restart vLLM with new model or settings"""
    if is_vllm_running():
        await stop_vllm()
        # Wait a bit to ensure clean shutdown
        await asyncio.sleep(2)

    return await start_vllm(request)


@app.post("/switch-model")
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model (stops current and starts new)"""
    was_running = is_vllm_running()

    if was_running:
        await stop_vllm()
        await asyncio.sleep(2)

    start_request = StartVLLMRequest(
        model_name=request.model_name,
        gpu_memory_utilization=request.gpu_memory_utilization,
        max_model_len=request.max_model_len,
        tensor_parallel_size=request.tensor_parallel_size
    )

    return await start_vllm(start_request)


@app.get("/disk-space")
async def disk_space():
    """Get disk space information"""
    space_info = get_disk_space()
    if "error" in space_info:
        raise HTTPException(status_code=500, detail=space_info["error"])
    return space_info


@app.post("/download-model")
async def download_model(request: DownloadModelRequest, background_tasks: BackgroundTasks):
    """Download a model from a URL (supports HuggingFace or direct links)"""

    url = request.model_url

    # Check if it's a HuggingFace model
    if "huggingface.co" in url or not url.startswith("http"):
        # Assume it's a HuggingFace model name/repo
        model_name = request.model_name or url.split("/")[-1]

        # Use HuggingFace Hub to download
        try:
            from huggingface_hub import snapshot_download

            # Download in background
            def download_hf_model():
                snapshot_download(
                    repo_id=url if not url.startswith("http") else url.split("huggingface.co/")[-1],
                    cache_dir=MODELS_DIR
                )

            background_tasks.add_task(download_hf_model)

            return {
                "status": "downloading",
                "model": url,
                "method": "huggingface_hub",
                "destination": MODELS_DIR,
                "message": "Model download started in background"
            }
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="huggingface_hub not installed. Install with: pip install huggingface_hub"
            )
    else:
        # Direct URL download
        model_name = request.model_name or url.split("/")[-1]
        destination = os.path.join(MODELS_DIR, model_name)

        # Download in background
        background_tasks.add_task(download_model_from_url, url, destination)

        return {
            "status": "downloading",
            "url": url,
            "method": "direct_download",
            "destination": destination,
            "message": "Model download started in background"
        }


@app.get("/models")
async def list_models():
    """List available models in the models directory"""
    try:
        models = []
        models_path = Path(MODELS_DIR)

        if models_path.exists():
            for item in models_path.iterdir():
                if item.is_dir():
                    # Get directory size
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    models.append({
                        "name": item.name,
                        "path": str(item),
                        "size_gb": round(size / (1024**3), 2),
                        "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    })

        return {
            "models_dir": MODELS_DIR,
            "count": len(models),
            "models": models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.delete("/delete-model")
async def delete_model(request: dict):
    """Delete a model from disk"""
    model_name = request.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name required")

    # Check if model is currently running
    if current_model and model_name in current_model:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete currently running model. Stop vLLM first."
        )

    model_path = Path(MODELS_DIR) / model_name
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        shutil.rmtree(model_path)
        return {
            "status": "deleted",
            "model": model_name,
            "message": f"Model {model_name} has been deleted"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")


@app.get("/gpu")
async def get_gpu_status():
    """Get detailed GPU information"""
    gpu_info = get_gpu_info()
    if gpu_info is None:
        raise HTTPException(status_code=503, detail="GPU monitoring not available")
    return gpu_info


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/auto-restart/enable")
async def enable_auto_restart():
    """Enable auto-restart functionality"""
    global auto_restart_enabled, last_known_state
    auto_restart_enabled = True
    last_known_state = "running" if is_vllm_running() else "stopped"
    print(f"✅ Auto-restart enabled")
    return {
        "status": "enabled",
        "message": "Auto-restart is now enabled"
    }


@app.post("/auto-restart/disable")
async def disable_auto_restart():
    """Disable auto-restart functionality"""
    global auto_restart_enabled
    auto_restart_enabled = False
    print(f"⏸️  Auto-restart disabled")
    return {
        "status": "disabled",
        "message": "Auto-restart is now disabled"
    }


@app.post("/crash-loop/reset")
async def reset_crash_loop():
    """Reset crash loop detector"""
    crash_detector.reset()
    print(f"🔄 Crash loop detector reset")
    return {
        "status": "reset",
        "message": "Crash loop detector has been reset"
    }


@app.get("/crash-loop/status")
async def get_crash_loop_status():
    """Get crash loop status"""
    return crash_detector.get_status()


@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup"""
    global last_known_state

    # Initialize state
    last_known_state = "running" if is_vllm_running() else "stopped"

    # Start health monitoring task
    asyncio.create_task(monitor_service_health())

    print(f"🚀 vLLM Manager started")
    print(f"📊 Health monitoring active (checking every {HEALTH_CHECK_INTERVAL}s)")
    print(f"🔄 Auto-restart: {'enabled' if auto_restart_enabled else 'disabled'}")
    print(f"⚠️  Crash loop threshold: {CRASH_LOOP_THRESHOLD} failures in {CRASH_LOOP_WINDOW}s")


if __name__ == "__main__":
    import uvicorn

    print(f"Starting vLLM Manager on port 7999...")
    print(f"Models directory: {MODELS_DIR}")
    print(f"vLLM will run on port: {VLLM_PORT}")

    uvicorn.run(app, host="0.0.0.0", port=7999)
