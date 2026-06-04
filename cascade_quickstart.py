#!/usr/bin/env python3
"""
Quick start script to verify Cascade implementation and run tests.
Place in repository root: python cascade_quickstart.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, check=True)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e}")
        return False
    except FileNotFoundError as e:
        print(f"⚠️  {description} - SKIPPED (tool not found)")
        return None


def main():
    """Run Cascade implementation verification."""
    print("""
╔══════════════════════════════════════════════════════════╗
║     Cascade MLSys 2026 - vLLM V1 Implementation         ║
║              Quick Start Verification                    ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    print("\n📁 Checking files...")
    files = {
        "vllm/v1/spec_decode/cascade.py": "Core Cascade module",
        "tests/v1/spec_decode/test_cascade.py": "Test suite",
        "docs/CASCADE_INTEGRATION.md": "Integration guide",
        "IMPLEMENTATION_CHECKLIST.md": "Implementation checklist",
        "CASCADE_README.md": "Branch README",
        "IMPLEMENTATION_SUMMARY.md": "Implementation summary",
    }
    
    all_exist = True
    for filepath, description in files.items():
        path = Path(__file__).parent / filepath
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {filepath}: {description}")
        all_exist = all_exist and exists
    
    if not all_exist:
        print("\n⚠️  Some files are missing. Please ensure all files are committed.")
        return 1
    
    print("\n✅ All files present!")
    
    # Run tests
    print("\n" + "="*60)
    print("🧪 Running Test Suite")
    print("="*60)
    
    test_result = run_command(
        [
            sys.executable, "-m", "pytest",
            "tests/v1/spec_decode/test_cascade.py",
            "-v", "--tb=short"
        ],
        "Cascade Unit Tests"
    )
    
    # Check imports
    print("\n" + "="*60)
    print("📦 Checking Imports")
    print("="*60)
    
    import_result = run_command(
        [
            sys.executable, "-c",
            "from vllm.v1.spec_decode.cascade import CascadeController, PerRequestCascadeState; print('✅ Imports successful')"
        ],
        "Import Verification"
    )
    
    # Summary
    print("\n" + "="*60)
    print("📊 Verification Summary")
    print("="*60)
    
    print(f"""
✅ Files: All present
✅ Tests: {'Passing' if test_result else 'Check output above'}
✅ Imports: {'Working' if import_result else 'Check output above'}

📚 Next Steps:
  1. Read docs/CASCADE_INTEGRATION.md for detailed integration guide
  2. Follow IMPLEMENTATION_CHECKLIST.md step-by-step
  3. Implement Phase 2-5 modifications
  4. Run full test suite after each modification
  5. Benchmark with real MoE models

📖 Documentation:
  - CASCADE_README.md: Branch overview
  - CASCADE_INTEGRATION.md: Complete integration steps
  - IMPLEMENTATION_SUMMARY.md: Current status
  - IMPLEMENTATION_CHECKLIST.md: Task tracking

🔗 Resources:
  - Paper: https://arxiv.org/abs/2506.20675
  - Issue: https://github.com/vllm-project/vllm/issues/44506
  - Branch: https://github.com/JOSH1024/vllm/tree/feat/cascade-moe-spec-decode
    """)
    
    return 0 if test_result else 1


if __name__ == "__main__":
    sys.exit(main())
