#!/usr/bin/env python3
"""vLLM for AMD ROCm - Interactive Installer"""
import os, sys, subprocess, shutil, glob, webbrowser
from pathlib import Path

def main():
    print("=" * 50)
    print("  vLLM for AMD ROCm - Installer")
    print("=" * 50)
    print()

    # 1. Find Python 3.12
    python312 = find_python312()
    if not python312:
        print("[!!] Python 3.12 not found.")
        dl = input("Download Python 3.12? (Y/N) [Y]: ").strip().upper() or "Y"
        if dl == "Y":
            url = "https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe"
            installer = Path(os.environ.get("TEMP", ".")) / "python-3.12.9-amd64.exe"
            print(f"Downloading from {url}...")
            import urllib.request
            urllib.request.urlretrieve(url, str(installer))
            print("Running installer...")
            subprocess.run([str(installer), "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0"], check=True)
            python312 = find_python312()
        if not python312:
            print("[!!] Python 3.12 required. Install it manually from python.org")
            input("Press Enter to exit...")
            sys.exit(1)
    print(f"[OK] Python 3.12: {python312}")

    # 2. Install folder
    default = "E:\\VLLM"
    install_dir = input(f"\nInstall folder [{default}]: ").strip() or default
    install_dir = Path(install_dir)
    install_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] {install_dir}")

    # 3. Create venv with Python 3.12
    venv_dir = install_dir / ".venv"
    if input("\nCreate Python venv? (Y/N) [Y]: ").strip().upper() or "Y":
        subprocess.run([python312, "-m", "venv", str(venv_dir)], check=True)
        python = str(venv_dir / "Scripts" / "python.exe")
        pip = str(venv_dir / "Scripts" / "pip.exe")
        # Verify venv is 3.12
        r = subprocess.run([python, "--version"], capture_output=True, text=True)
        print(f"[OK] Venv: {venv_dir} ({r.stdout.strip()})")
    else:
        python = python312
        pip = str(Path(python312).parent / "pip.exe") if os.name == "nt" else "pip"

    # 4. Detect ROCm
    print("\n[..] Detecting ROCm...")
    hip_path = detect_rocm()
    if hip_path:
        rocm_ver = get_rocm_version(hip_path)
        print(f"[OK] ROCm {rocm_ver} at {hip_path}")
    else:
        print("[!!] ROCm not found.")
        user_path = input("Enter ROCm path: ").strip()
        if not user_path or not (Path(user_path) / "bin" / "hipcc.exe").exists():
            print("[!!] ROCm required. Install from https://rocm.docs.amd.com")
            input("Press Enter to exit...")
            sys.exit(1)
        hip_path = user_path
        rocm_ver = get_rocm_version(hip_path) or "7.13"
        print(f"[OK] ROCm {rocm_ver}")

    # 5. Generate requirements.txt with AMD wheel URLs
    # List wheels the user needs to download
    py_tag = "cp312"
    base_amd = "https://repo.amd.com/rocm/whl/gfx120X-all"
    torch_ver, tv_ver = get_torch_versions(rocm_ver)
    wheel_dir = install_dir / "wheels"
    wheel_dir.mkdir(exist_ok=True)

    wheels = [
        f"{base_amd}/rocm-sdk-core/rocm_sdk_core-7.13.0-py3-none-win_amd64.whl",
        f"{base_amd}/rocm-sdk-devel/rocm_sdk_devel-7.13.0-py3-none-win_amd64.whl",
        f"{base_amd}/torch/torch-{torch_ver}-{py_tag}-{py_tag}-win_amd64.whl",
        f"{base_amd}/torchvision/torchvision-{tv_ver}-{py_tag}-{py_tag}-win_amd64.whl",
    ]
    dl_file = install_dir / "download_urls.txt"
    content = "Open each URL in your browser and save the .whl file.\n"
    content += "Save all files into a folder (e.g. C:\\wheels).\n"
    content += "Then run: pip install --no-index --find-links C:\\wheels torch torchvision rocm-sdk-core rocm-sdk-devel\n\n"
    content += "\n".join(wheels) + "\n"
    dl_file.write_text(content)

    # 6. Check if GPU shows up
    print("\n[..] Checking for AMD GPU...")
    try:
        r = subprocess.run([str(Path(hip_path) / "bin" / "rocm-smi.exe"), "--showproductname"],
                          capture_output=True, text=True, timeout=10)
        if "RX" in r.stdout or "AMD" in r.stdout:
            for line in r.stdout.splitlines():
                if line.strip():
                    print(f"  GPU: {line.strip()}")
    except: pass

    # 7. Print instructions
    print()
    print("=" * 50)
    print("  READY TO INSTALL PYTORCH")
    print("=" * 50)
    print()
    print("Run these commands in your terminal (as administrator):")
    print()
    for w in wheels:
        print(f"  {w}")
    print()
    print("Open each URL in your browser, right-click -> Save As.")
    print("Save all .whl files into a folder (e.g. C:\\wheels).")
    print()
    print(f"Then in your terminal, run:")
    print(f"  {pip} install --no-index --find-links C:\\wheels torch torchvision rocm-sdk-core rocm-sdk-devel")
    print()
    print("See download_urls.txt for the full list.")
    print()
    input("Type DONE after installing the wheels, or ENTER to skip: ")

    # 8. Clean stale sitecustomize.py
    for p in sys.path:
        f = Path(p) / "sitecustomize.py"
        if f.exists():
            try: f.unlink()
            except: pass

    # 9. Write fresh sitecustomize.py
    site_pkg = get_site_packages(python)
    if site_pkg:
        site_file = Path(site_pkg) / "sitecustomize.py"
        site_file.write_text(
            "import os\n"
            f"os.environ.setdefault('HIP_PATH', r'{hip_path}')\n"
            "os.environ.setdefault('VLLM_NO_USAGE_STATS', 'true')\n"
        )
        print("[OK] sitecustomize.py created")

    # 10. Clone vLLM
    src_dir = install_dir / "vllm-windows"
    if not src_dir.exists():
        print("\n[..] Cloning vLLM...")
        subprocess.run(["git", "clone", "https://github.com/Maxritz/vllm-windows.git", str(src_dir)], check=True)
        subprocess.run(["git", "checkout", "WINDOWS-PORT"], cwd=src_dir, capture_output=True)
    print(f"[OK] vLLM source at {src_dir}")

    # 11. Install vLLM package
    print("\n[..] Installing vLLM package...")
    subprocess.run([pip, "install", "-e", "."], cwd=src_dir, capture_output=True)

    # 12. Copy binaries
    pyd_src = Path(__file__).parent / "_C.pyd"
    if pyd_src.exists():
        shutil.copy2(pyd_src, src_dir / "vllm" / "_C.pyd")
        print(f"[OK] _C.pyd ({pyd_src.stat().st_size // 1048576} MB)")

    exe_src = Path(__file__).parent / "vllm.exe"
    if exe_src.exists():
        shutil.copy2(exe_src, src_dir)
        print("[OK] vllm.exe copied")

    print()
    print("=" * 50)
    print("  INSTALLATION COMPLETE")
    print("=" * 50)
    print()
    print(f"  vLLM: {src_dir}")
    print(f"  Run:  {src_dir / 'vllm.exe'}")
    print()
    input("Press Enter to exit...")

def find_python312():
    # 1. Try py launcher
    try:
        r = subprocess.run(["py", "-3.12", "--version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            # Find where py's 3.12 is
            r2 = subprocess.run(["py", "-3.12", "-c", "import sys; print(sys.executable)"], capture_output=True, text=True, timeout=5)
            if r2.returncode == 0:
                return r2.stdout.strip()
    except: pass
    # 2. Check PATH
    try:
        r = subprocess.run(["python3.12", "--version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            r2 = subprocess.run(["python3.12", "-c", "import sys; print(sys.executable)"], capture_output=True, text=True, timeout=5)
            if r2.returncode == 0:
                return r2.stdout.strip()
    except: pass
    # 3. Check common install paths
    for p in [
        r"C:\Python312\python.exe",
        r"C:\Program Files\Python312\python.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Python\Python312\python.exe"),
    ]:
        if Path(p).exists():
            return p
    return None

def detect_rocm():
    for var in ["ROCM_HOME", "ROCM_PATH", "HIP_PATH"]:
        val = os.environ.get(var)
        if val and (Path(val) / "bin" / "hipcc.exe").exists():
            return val
    for pattern in ["C:/Program Files/AMD/ROCm/*", "C:/ROCm/*", "D:/ROCM-*", "E:/ROCM-*"]:
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
    v = {
        "7.13": ("2.11.0+rocm7.13.0", "0.22.0+rocm7.13.0"),
        "7.12": ("2.10.0+rocm7.12.0", "0.21.0+rocm7.12.0"),
        "7.11": ("2.9.1+rocm7.11.0", "0.20.1+rocm7.11.0"),
        "7.10": ("2.9.1+rocm7.10.0", "0.20.1+rocm7.10.0"),
    }
    return v.get(rocm_ver, ("2.11.0+rocm7.13.0", "0.22.0+rocm7.13.0"))

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
