#!/usr/bin/env python3
"""vLLM for AMD ROCm - Interactive Installer"""
import os, sys, subprocess, shutil, glob
from pathlib import Path

def main():
    print("=" * 45)
    print("  vLLM for AMD ROCm - Interactive Installer")
    print("=" * 45)
    print()

    # Where to install
    default = "E:\\VLLM"
    install_dir = input(f"Install folder [{default}]: ").strip() or default
    install_dir = Path(install_dir)
    install_dir.mkdir(parents=True, exist_ok=True)
    print(f"Installed to: {install_dir}")
    print()

    # Python venv
    if input("Create Python venv? (Y/N) [Y]: ").strip().upper() or "Y":
        venv_dir = install_dir / ".venv"
        if not (venv_dir / "Scripts" / "python.exe").exists():
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        python = str(venv_dir / "Scripts" / "python.exe")
        pip = str(venv_dir / "Scripts" / "pip.exe")
        print(f"Venv: {venv_dir}")
    else:
        python = sys.executable
        pip = os.path.join(os.path.dirname(python), "pip.exe") if os.name == "nt" else "pip"
    print()

    # Detect ROCm
    hip_path = detect_rocm()
    if hip_path:
        rocm_ver = get_rocm_version(hip_path)
        print(f"[OK] ROCm at {hip_path} (version {rocm_ver})")
    else:
        print("[!!] ROCm not found.")
        user_path = input("ROCm path (or press Enter to skip): ").strip()
        if not user_path:
            print("[!!] ROCm is required. Install from https://rocm.docs.amd.com")
            sys.exit(1)
        if not (Path(user_path) / "bin" / "hipcc.exe").exists():
            print(f"[!!] hipcc.exe not found at {user_path}")
            sys.exit(1)
        hip_path = user_path
        rocm_ver = get_rocm_version(hip_path) or "7.13"
        print(f"[OK] ROCm at {hip_path} (version {rocm_ver})")

    # sitecustomize.py
    site_pkg = get_site_packages(python)
    if site_pkg:
        site_file = Path(site_pkg) / "sitecustomize.py"
        if site_file.exists():
            site_file.unlink()
        site_file.write_text(
            "import os\n"
            f"os.environ.setdefault('HIP_PATH', r'{hip_path}')\n"
            "os.environ.setdefault('VLLM_NO_USAGE_STATS', 'true')\n"
        )
        print("[OK] sitecustomize.py created")
    print()

    # Install PyTorch with ROCm
    if input(f"Install PyTorch with ROCm? (Y/N) [Y]: ").strip().upper() or "Y":
        torch_ver, tv_ver = get_torch_versions(rocm_ver)
        print(f"Installing torch {torch_ver} from AMD repo...")
        # Try AMD repo first
        cmd = [pip, "install", f"torch=={torch_ver}", f"torchvision=={tv_ver}",
               "--extra-index-url", "https://repo.amd.com/rocm/whl/gfx120X-all/",
               "--timeout", "120"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            # Fallback: direct wheel URL
            py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
            wheel_url = f"https://repo.amd.com/rocm/whl/gfx120X-all/torch/torch-{torch_ver}-{py_tag}-{py_tag}-win_amd64.whl"
            print("AMD index failed, trying direct wheel...")
            cmd = [pip, "install", wheel_url, "--timeout", "120"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                # Last resort: official PyTorch ROCm repo
                print("Direct wheel failed, trying PyTorch official repo...")
                cmd = [pip, "install", "torch", "torchvision",
                       f"--index-url=https://download.pytorch.org/whl/rocm{rocm_ver}",
                       "--timeout", "120"]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"PyTorch install failed:\n{r.stderr}")
                    sys.exit(1)
        print("PyTorch + ROCm installed.")
    print()

    # Clone vLLM
    src_dir = install_dir / "vllm-windows"
    if not (install_dir / "vllm-windows" / "vllm" / "__init__.py").exists():
        if not src_dir.exists():
            print("Cloning vLLM Windows port...")
            subprocess.run(["git", "clone", "https://github.com/Maxritz/vllm-windows.git", str(src_dir)], check=True)
        subprocess.run(["git", "checkout", "WINDOWS-PORT"], cwd=src_dir, capture_output=True)
    print(f"vLLM source: {src_dir}")
    print()

    # Install vLLM package
    print("Installing vLLM package...")
    subprocess.run([pip, "install", "-e", "."], cwd=src_dir, capture_output=True)
    print("vLLM package installed.")

    # Copy _C.pyd
    pyd_src = Path(__file__).parent / "_C.pyd"
    if pyd_src.exists():
        shutil.copy2(pyd_src, src_dir / "vllm" / "_C.pyd")
        print(f"_C.pyd installed ({pyd_src.stat().st_size // 1048576} MB)")

    # Copy vllm.exe
    exe_src = Path(__file__).parent / "vllm.exe"
    if exe_src.exists():
        shutil.copy2(exe_src, src_dir)
        print("vllm.exe copied")

    print()
    print("=" * 45)
    print("  INSTALLATION COMPLETE")
    print("=" * 45)
    print()
    print(f"vLLM is ready in: {src_dir}")
    print(f"Run: {src_dir / 'vllm.exe'}")
    print()

def detect_rocm():
    # Check env vars
    for var in ["ROCM_HOME", "ROCM_PATH", "HIP_PATH"]:
        val = os.environ.get(var)
        if val and (Path(val) / "bin" / "hipcc.exe").exists():
            return val
    # Check common paths
    for pattern in [
        "C:/Program Files/AMD/ROCm/*",
        "C:/ROCm/*",
        "D:/ROCM-*",
        "E:/ROCM-*",
    ]:
        for p in glob.glob(pattern):
            if (Path(p) / "bin" / "hipcc.exe").exists():
                return str(Path(p))
    return None

def get_rocm_version(hip_path):
    try:
        r = subprocess.run([str(Path(hip_path) / "bin" / "hipcc.exe"), "--version"],
                          capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            if "HIP version" in line:
                parts = line.replace(":", " ").replace(".", " ").split()
                for i, p in enumerate(parts):
                    if p == "version" and i + 2 < len(parts):
                        return f"{parts[i+1]}.{parts[i+2]}"
    except: pass
    return "7.13"

def get_torch_versions(rocm_ver):
    versions = {
        "7.13": ("2.11.0+rocm7.13.0", "0.22.0+rocm7.13.0"),
        "7.12": ("2.10.0+rocm7.12.0", "0.21.0+rocm7.12.0"),
        "7.11": ("2.9.1+rocm7.11.0", "0.20.1+rocm7.11.0"),
        "7.10": ("2.9.1+rocm7.10.0", "0.20.1+rocm7.10.0"),
    }
    return versions.get(rocm_ver, ("2.11.0+rocm7.13.0", "0.22.0+rocm7.13.0"))

def get_site_packages(python):
    try:
        r = subprocess.run([python, "-c", "import sys; [print(p) for p in sys.path if 'site-packages' in p]"],
                          capture_output=True, text=True, timeout=5)
        for line in r.stdout.strip().splitlines():
            if line.strip():
                return line.strip()
    except: pass
    return None

if __name__ == "__main__":
    main()
