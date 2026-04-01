#!/bin/bash
# Quick module availability check for PSC
# Run this interactively on PSC login node before submitting jobs

echo "=== Available GCC modules ==="
module avail gcc 2>&1 | grep -i gcc

echo ""
echo "=== Available CUDA modules ==="
module avail cuda 2>&1 | grep -i cuda

echo ""
echo "=== Currently loaded modules ==="
module list
