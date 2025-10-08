#!/usr/bin/env pwsh

[CmdletBinding()] param(
	[switch]$Build,
	[switch]$Interactive,
	[string]$Command = "",
	[switch]$Setup,
	[switch]$GPUCheck,
	[switch]$Mirror,
	[switch]$Recreate,
	[string]$WorkVolume = "",
	[string]$WorkDirHost = "",
	[switch]$Progress,
	[switch]$NoCache,
	[switch]$Pull,
	[string[]]$Env,
	[switch]$Help
)

if ($Help) {
	Write-Host "Usage: extras/podman/run.ps1 [options]"
	Write-Host "  -Build                Build the dev image (reads extras/configs/build.env)"
	Write-Host "  -Interactive          Start an interactive shell"
	Write-Host "  -Command <cmd>        Run a command inside the dev container"
	Write-Host "  -Setup                Run project setup inside the container"
	Write-Host "  -GPUCheck             Run a CUDA/Torch sanity check"
	Write-Host "  -Mirror               Use local mirror registries if configured"
	Write-Host "  -Recreate             Recreate the container if running"
	Write-Host "  -WorkVolume <name>    Named volume to mount at /opt/work"
	Write-Host "  -WorkDirHost <path>   Host dir to mount at /opt/work"
	Write-Host "  -Progress             Show progress bars in setup"
	Write-Host "  -NoCache              Build image without using cache"
	Write-Host "  -Pull                 Always attempt to pull newer base image"
	Write-Host "  -Env KEY=VALUE        Additional environment variable(s) to inject (can repeat)"
	return
}

if (-not $Interactive -and [string]::IsNullOrEmpty($Command) -and -not $GPUCheck -and -not $Setup) { $Interactive = $true }

if (-not (Get-Command podman -ErrorAction SilentlyContinue)) { Write-Host "❌ Podman not found in PATH" -ForegroundColor Red; exit 1 }

$ContainerName = "vllm-dev"
$ImageTag = "vllm-dev:latest"
$SourceDir = (Get-Location).Path

function Get-DefaultPodmanMachine {
	if ($Env:PODMAN_MACHINE_NAME) { return $Env:PODMAN_MACHINE_NAME }
	try {
		$machine = (& podman machine list --format "{{range .}}{{if .Default}}{{.Name}}{{end}}{{end}}" 2>$null)
		if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($machine)) {
			return $machine.Trim()
		}
	} catch {
	}
	return "podman-machine-default"
}

function Test-PodmanMachinePath {
	param([string]$Path)
	if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
	$escaped = $Path.Replace('"', '\"')
	try {
		& podman machine ssh -- "test -e \"$escaped\"" *> $null
		if ($LASTEXITCODE -eq 0) { return $true }
	} catch {
	}
	try {
		$machine = Get-DefaultPodmanMachine
		$unc = "\\\\wsl.localhost\\$machine" + ($Path -replace '/', '\\')
		return (Test-Path -LiteralPath $unc)
	} catch {
		return $false
	}
}

Write-Host "🐋 vLLM Dev Container (Podman)" -ForegroundColor Green

# Normalize CRLF -> LF for mounted shell scripts to avoid /usr/bin/env 'bash\r' errors inside container
try {
	$normalizeTargets = @()
	$normalizeTargets += (Join-Path $SourceDir 'extras/podman')
	$normalizeTargets += (Join-Path $SourceDir 'extras/patches')
	foreach ($targetDir in $normalizeTargets) {
		if (-not (Test-Path $targetDir)) { continue }
		$shellScripts = Get-ChildItem -Path $targetDir -Recurse -Filter *.sh -File -ErrorAction SilentlyContinue
		foreach ($f in $shellScripts) {
			$raw = Get-Content -Raw -Path $f.FullName
			if ($raw -like "*`r*") {
				$raw -replace "`r", "" | Set-Content -NoNewline -Encoding UTF8 $f.FullName
			}
		}
	}
} catch { Write-Host "⚠️  Script newline normalization skipped: $($_.Exception.Message)" -ForegroundColor Yellow }

if ($Build) {
	Write-Host "🔨 Building image (honoring extras/configs/build.env)..." -ForegroundColor Yellow
	$configPath = Join-Path $SourceDir "extras/configs/build.env"
	$dockerfilePath = Join-Path $SourceDir "extras/Dockerfile"
	$cudaVer = $null
	$baseFlavor = $null
	$archList = $null
	$cudaArchs = $null
	$requireFfmpegArg = '1'
	function Get-DockerArgDefault([string]$name, [string]$fallback) {
		if (Test-Path $dockerfilePath) {
			$df = Get-Content -Raw -Path $dockerfilePath
			$m = [regex]::Match($df, "(?m)^\s*ARG\s+${name}\s*=\s*([^\r\n]+)")
			if ($m.Success) {
				return $m.Groups[1].Value.Trim()
			}
		}
		return $fallback
	}
	if (Test-Path $configPath) {
		$cfg = Get-Content -Raw -Path $configPath
		function Get-EnvDefault([string]$name, [string]$fallback) {
			# Match a line like: export NAME=VALUE
			$line = [regex]::Match($cfg, "(?m)^\s*export\s+${name}\s*=\s*([^\r\n]+)")
			if (-not $line.Success) { return $fallback }
			$val = $line.Groups[1].Value.Trim()
			# Strip wrapping quotes if present
			if (($val.StartsWith('"') -and $val.EndsWith('"')) -or ($val.StartsWith("'") -and $val.EndsWith("'"))) { $val = $val.Substring(1, $val.Length-2) }
			# If value is Bash-style ${NAME:-default}, extract default
			if ($val.StartsWith('${') -and $val.Contains(':-')) {
				$idx = $val.IndexOf(':-'); $end = $val.IndexOf('}', $idx)
				if ($idx -ge 0 -and $end -gt $idx) {
					$def = $val.Substring($idx+2, $end-($idx+2)).Trim()
					if (($def.StartsWith('"') -and $def.EndsWith('"')) -or ($def.StartsWith("'") -and $def.EndsWith("'"))) { $def = $def.Substring(1, $def.Length-2) }
					return $def
				}
			}
			return $val
		}
		$cudaVer = Get-EnvDefault -name 'CUDA_VERSION' -fallback (Get-DockerArgDefault 'CUDA_VERSION' '13.0.0')
		$baseFlavor = Get-EnvDefault -name 'BASE_FLAVOR' -fallback (Get-DockerArgDefault 'BASE_FLAVOR' 'rockylinux9')
		$archList = Get-EnvDefault -name 'TORCH_CUDA_ARCH_LIST' -fallback (Get-DockerArgDefault 'TORCH_CUDA_ARCH_LIST' '8.0 8.6 8.9 9.0 12.0 13.0')
		$cudaArchs = Get-EnvDefault -name 'CUDA_ARCHS' -fallback (Get-DockerArgDefault 'CUDA_ARCHS' '80;86;89;90;120')
	# No longer used: wheels-only installs for torchvision/torchaudio
		$requireFfmpeg = Get-EnvDefault -name 'REQUIRE_FFMPEG' -fallback (Get-DockerArgDefault 'REQUIRE_FFMPEG' '1')
		if ($requireFfmpeg -match '^[01]$') { $requireFfmpegArg = $requireFfmpeg } else { $requireFfmpegArg = '1' }
	}
	# Derive PyTorch nightly index from CUDA version (e.g., 13.0 -> cu130, 12.9 -> cu129)
	$torchCudaIndex = if ($cudaVer -match '^13\.') { 'cu130' } elseif ($cudaVer -match '^12\.9') { 'cu129' } else {
		$parts = $cudaVer.Split('.')
		if ($parts.Length -ge 2) { 'cu' + $parts[0] + $parts[1] + '0' } else { 'cu129' }
	}
	Write-Host ("Config: CUDA={0} BASE_FLAVOR={1} TORCH_CUDA_INDEX={2} ARCH_LIST=({3}) CUDA_ARCHS={4}" -f $cudaVer,$baseFlavor,$torchCudaIndex,$archList,$cudaArchs) -ForegroundColor DarkGray
	$buildCmd = @("build","-f","extras/Dockerfile",
		"--build-arg","CUDA_VERSION=$cudaVer",
		"--build-arg","BASE_FLAVOR=$baseFlavor",
		"--build-arg","TORCH_CUDA_INDEX=$torchCudaIndex",
		"--build-arg","TORCH_CUDA_ARCH_LIST=$archList",
		"--build-arg","CUDA_ARCHS=$cudaArchs",
	"--build-arg","REQUIRE_FFMPEG=$requireFfmpegArg",
		"-t",$ImageTag,".")
	# Use cache by default; add --no-cache only when requested
	if ($NoCache) { $buildCmd = @($buildCmd[0],"--no-cache") + $buildCmd[1..($buildCmd.Length-1)] }
	if ($Pull) { $buildCmd = @($buildCmd[0],"--pull=always") + $buildCmd[1..($buildCmd.Length-1)] }
	& podman @buildCmd
	if ($LASTEXITCODE -ne 0) { Write-Host "❌ Build failed" -ForegroundColor Red; exit 1 }
	Write-Host "✅ Build ok" -ForegroundColor Green
}

# Already running?
$running = podman ps --filter "name=$ContainerName" --format "{{.Names}}" 2>$null

if ($Recreate -and $running -eq $ContainerName) {
	Write-Host "♻️  Removing existing container '$ContainerName'" -ForegroundColor Yellow
	podman rm -f $ContainerName | Out-Null
	$running = $null
}

if ($running -eq $ContainerName) {
	if ($GPUCheck) {
		Write-Host "🔍 GPU check (existing container)" -ForegroundColor Yellow
		$cmd = @'
source /home/vllmuser/venv/bin/activate && python - <<'PY'
import torch, os
print("PyTorch:", getattr(torch,"__version__","n/a"))
print("CUDA:", torch.cuda.is_available())
print("Devices:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
if torch.cuda.is_available():
		try:
				print("GPU 0:", torch.cuda.get_device_name(0))
		except Exception as e:
				print("GPU name error:", e)
PY
nvidia-smi || true
'@
		$cmd = "export NVIDIA_VISIBLE_DEVICES=all; " + $cmd
		podman exec $ContainerName bash -lc $cmd
		exit $LASTEXITCODE
	}
	if ($Setup) {
		Write-Host "🔧 Running dev setup in existing container" -ForegroundColor Yellow
		$envs = @()
		if ($Mirror) { $envs += @('LOCAL_MIRROR=1') }
		if ($Progress) { $envs += @('PROGRESS_WATCH=1') }
		$envs += @('NVIDIA_VISIBLE_DEVICES=all')
			   $envs += @('PYTHON_PATCH_OVERLAY=1')
		$envStr = ($envs | ForEach-Object { "export $_;" }) -join ' '
		$cmd = @'
TMP_RUN=$(mktemp /tmp/run-dev-setup.XXXX.sh)
tr -d '\r' < ./extras/podman/dev-setup.sh > "$TMP_RUN" 2>/dev/null || cp ./extras/podman/dev-setup.sh "$TMP_RUN"
chmod +x "$TMP_RUN" 2>/dev/null || true
if [ -x ./extras/patches/apply_patches_overlay.sh ]; then
	bash ./extras/patches/apply_patches_overlay.sh || true
elif [ -x ./extras/patches/apply_patches.sh ]; then
	bash ./extras/patches/apply_patches.sh || true
fi
"$TMP_RUN"
'@
		$cmd = "$envStr $cmd"
		if ($Progress) { podman exec -it $ContainerName bash -lc $cmd } else { podman exec $ContainerName bash -lc $cmd }
		exit $LASTEXITCODE
	}
	if ($Command) {
		Write-Host "🚀 Running command in existing container" -ForegroundColor Green
		$runCmd = "source /home/vllmuser/venv/bin/activate && $Command"
		podman exec $ContainerName bash -c $runCmd
		exit $LASTEXITCODE
	}
	$resp = Read-Host "Attach to running container? [Y/n]"
	if ($resp -eq "" -or $resp -match '^[Yy]$') { podman exec -it $ContainerName bash; exit $LASTEXITCODE } else { exit 0 }
}

# Ensure image exists
podman image exists $ImageTag
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Image missing. Use -Build." -ForegroundColor Red; exit 1 }

# Base args (no default /tmp tmpfs; can be enabled via VLLM_TMPFS_TMP_SIZE)
$runArgs = @("run","--rm","--security-opt=label=disable","--shm-size","8g","-v","${SourceDir}:/workspace:Z")
if (-not [string]::IsNullOrWhiteSpace($WorkVolume)) { $runArgs += @('-v',"${WorkVolume}:/opt/work:Z") }
elseif ($WorkDirHost -and (Test-Path $WorkDirHost)) { $runArgs += @('-v',"${WorkDirHost}:/opt/work:Z") }
$runArgs += @('-w','/workspace','--name',"$ContainerName",'--user','vllmuser','--env','ENGINE=podman')
# Use a tiny entrypoint to apply patches before executing the requested command
$runArgs += @('--entrypoint','/workspace/extras/podman/entrypoint/apply-patches-then-exec.sh')

$tmpfsSize = [Environment]::GetEnvironmentVariable('VLLM_TMPFS_TMP_SIZE')
if (-not [string]::IsNullOrEmpty($tmpfsSize) -and $tmpfsSize -ne '0') { $runArgs += @('--tmpfs',"/tmp:size=$tmpfsSize") }

if (-not $Env:VLLM_DISABLE_CDI) { # Request GPU via CDI hooks
	$runArgs = @("run","--rm","--security-opt=label=disable","--device=nvidia.com/gpu=all") + $runArgs[3..($runArgs.Length-1)]
} else {
	Write-Host "⚠️  Skipping CDI GPU request (VLLM_DISABLE_CDI set)" -ForegroundColor Yellow
}

$wslRoot = '/usr/lib/wsl'
$wslRootExists = Test-PodmanMachinePath $wslRoot
if ($wslRootExists) {
	$runArgs += @('-v',"$($wslRoot):$($wslRoot):ro")
	$libCandidates = @(
		"$wslRoot/drivers/libcuda.so.1.1",
		"$wslRoot/drivers/libcuda.so.1",
		"$wslRoot/lib/libcuda.so.1",
		"$wslRoot/lib/libcuda.so"
	)
	$libFound = $false
	foreach ($candidate in $libCandidates) {
		if (Test-PodmanMachinePath $candidate) { $libFound = $true; break }
	}
	if (-not $libFound) {
		Write-Host "⚠️  Could not locate libcuda under $wslRoot; container GPU may fail" -ForegroundColor Yellow
	}
} else {
	Write-Host "⚠️  WSL GPU libraries not detected; continuing without /usr/lib/wsl mount" -ForegroundColor Yellow
}

if (Test-PodmanMachinePath '/dev/dxg') {
	$runArgs += @('--device','/dev/dxg')
} else {
	Write-Host "⚠️  /dev/dxg not available in podman machine; GPU passthrough may be disabled" -ForegroundColor Yellow
}
if ($Mirror) { $runArgs += @('--env','LOCAL_MIRROR=1') }
foreach ($ev in 'NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES','NVIDIA_REQUIRE_CUDA') {
	$val = [Environment]::GetEnvironmentVariable($ev)
	if ($val) { $runArgs += @('--env',"$ev=$val") }
}

# Forward any FA3_* host environment variables (e.g., FA3_MEMORY_SAFE_MODE)
$fa3Vars = Get-ChildItem Env: | Where-Object { $_.Name -like 'FA3_*' }
foreach ($v in $fa3Vars) { if ($v.Value) { $runArgs += @('--env',"$($v.Name)=$($v.Value)") } }

# User provided generic env KEY=VALUE pairs via -Env
if ($Env) {
	foreach ($pair in $Env) {
		if ($pair -match '^[A-Za-z_][A-Za-z0-9_]*=') {
			$runArgs += @('--env',$pair)
		} else {
			Write-Host "⚠️  Ignoring invalid -Env entry: $pair" -ForegroundColor Yellow
		}
	}
}
$runArgs += @('--env','ENGINE=podman','--env','NVIDIA_VISIBLE_DEVICES=all','--env','NVIDIA_DRIVER_CAPABILITIES=compute,utility','--env','NVIDIA_REQUIRE_CUDA=')

if ($GPUCheck) {
	$pyDiag = @'
import json, torch, os
out = {
		"torch_version": getattr(torch, "__version__", "n/a"),
		"torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", "n/a"),
		"cuda_available": torch.cuda.is_available(),
		"ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
}
try:
		out["device_count"] = torch.cuda.device_count()
except Exception as e:
		out["device_count_error"] = str(e)
if out["cuda_available"] and out.get("device_count", 0) > 0:
		try:
				cap = torch.cuda.get_device_capability(0)
				out["device_0"] = {"name": torch.cuda.get_device_name(0), "capability": f"sm_{cap[0]}{cap[1]}"}
		except Exception as e:
				out["device_0_error"] = str(e)
else:
		out["diagnostics"] = ["Missing /dev/nvidia* or podman machine without GPU passthrough"]
print(json.dumps(out, indent=2))
'@
	$pyB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($pyDiag))
	$gpuScript = @'
echo '=== GPU Check ==='
which nvidia-smi && nvidia-smi || echo 'nvidia-smi unavailable'
echo '--- /dev/nvidia* ---'
ls -l /dev/nvidia* 2>/dev/null || echo 'no /dev/nvidia* nodes'
echo '--- Environment (NVIDIA_*) ---'
env | grep -E '^NVIDIA_' || echo 'no NVIDIA_* env vars'
if [ "$NVIDIA_VISIBLE_DEVICES" = "void" ]; then echo 'WARN: NVIDIA_VISIBLE_DEVICES=void (no GPU mapped)'; fi
echo '--- LD_LIBRARY_PATH ---'
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
source /home/vllmuser/venv/bin/activate 2>/dev/null || true
echo __PY_B64__ | base64 -d > /tmp/gpucheck.py
python /tmp/gpucheck.py || true
rm -f /tmp/gpucheck.py
'@
	$gpuScript = "export NVIDIA_VISIBLE_DEVICES=all; export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/wsl/drivers:`$LD_LIBRARY_PATH; " + ($gpuScript -replace '__PY_B64__', $pyB64) -replace "`r",""
	$runArgs += @('--user','root', $ImageTag,'bash','-lc',$gpuScript)
} elseif ($Setup) {
	# Use robust setup entrypoint that finds the right script (extras/dev-setup.sh or image helper)
	# Avoid in-place edits on Windows-mounted files; run a CRLF-normalized temp copy instead
	$prefix = 'TMP_RUN=$(mktemp /tmp/run-dev-setup.XXXX.sh); tr -d "\r" < ./extras/podman/dev-setup.sh > "$TMP_RUN" || cp ./extras/podman/dev-setup.sh "$TMP_RUN"; chmod +x "$TMP_RUN" 2>/dev/null || true; export PYTHON_PATCH_OVERLAY=1; if [ -x ./extras/patches/apply_patches_overlay.sh ]; then bash ./extras/patches/apply_patches_overlay.sh || true; elif [ -x ./extras/patches/apply_patches.sh ]; then bash ./extras/patches/apply_patches.sh || true; fi; '
	$envPrefix = ''
	if ($Mirror) { $envPrefix += 'export LOCAL_MIRROR=1; ' }
	if ($Progress) { $envPrefix += 'export PROGRESS_WATCH=1; ' }
	# Pass configured archs from build.env (the Dockerfile already defaults to safe values)
	if ($archList) { $envPrefix += "export TORCH_CUDA_ARCH_LIST='$archList'; " }
	if ($cudaArchs) { $envPrefix += "export CUDAARCHS='$cudaArchs'; " }
	$envPrefix += 'export TMPDIR=/opt/work/tmp; export TMP=/opt/work/tmp; export TEMP=/opt/work/tmp; mkdir -p /opt/work/tmp; '
	$setupCmd = $prefix + $envPrefix + '"$TMP_RUN"'
	# Use bash -lc always; rely on entrypoint normalization only for simple path cases.
	if ($Progress) { $runArgs += @('-it', $ImageTag, 'bash','-lc', $setupCmd) } else { $runArgs += @($ImageTag, 'bash','-lc', $setupCmd) }
	Write-Host "🔧 Running dev setup" -ForegroundColor Green
} elseif ($Interactive -and -not $Command) {
	$runArgs += @('-it',$ImageTag,'bash')
	Write-Host "🚀 Interactive shell" -ForegroundColor Green
} elseif ($Command) {
	$runArgs += @($ImageTag,'bash','-lc',"export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/wsl/drivers:`$LD_LIBRARY_PATH; source /home/vllmuser/venv/bin/activate && $Command")
	Write-Host "🚀 Running command" -ForegroundColor Green
} else {
	$runArgs += @($ImageTag)
}

Write-Host "Command: podman $($runArgs -join ' ')" -ForegroundColor Gray
& podman @runArgs

if ($LASTEXITCODE -eq 0 -and $Interactive) { Write-Host "Exited cleanly" -ForegroundColor Green }
