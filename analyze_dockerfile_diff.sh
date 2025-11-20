#!/bin/bash

echo "========================================"
echo "Dockerfile Optimization Analysis"
echo "========================================"
echo ""

# Get original Dockerfile from git
git show HEAD:docker/Dockerfile > /tmp/dockerfile_original.txt

# Current optimized Dockerfile
cp docker/Dockerfile /tmp/dockerfile_optimized.txt

echo "📊 Analyzing Dockerfile Structure..."
echo ""

# Count RUN commands
echo "1. RUN Command Analysis (Base Stage)"
echo "----------------------------------------"
ORIG_RUN_BASE=$(sed -n '/^FROM.*AS base/,/^FROM/p' /tmp/dockerfile_original.txt | grep -c "^RUN")
OPT_RUN_BASE=$(sed -n '/^FROM.*AS base/,/^FROM/p' /tmp/dockerfile_optimized.txt | grep -c "^RUN")

echo "Original base stage RUN commands: $ORIG_RUN_BASE"
echo "Optimized base stage RUN commands: $OPT_RUN_BASE"
echo "Reduction: $((ORIG_RUN_BASE - OPT_RUN_BASE)) RUN commands"
echo ""

# Count apt-get update
echo "2. apt-get update Analysis"
echo "----------------------------------------"
ORIG_APT_UPDATE=$(grep -c "apt-get update" /tmp/dockerfile_original.txt)
OPT_APT_UPDATE=$(grep -c "apt-get update" /tmp/dockerfile_optimized.txt)

echo "Original apt-get update calls: $ORIG_APT_UPDATE"
echo "Optimized apt-get update calls: $OPT_APT_UPDATE"
echo ""

# Count --no-install-recommends
echo "3. --no-install-recommends Usage"
echo "----------------------------------------"
ORIG_NO_REC=$(grep "apt-get install" /tmp/dockerfile_original.txt | grep -c "\-\-no-install-recommends")
ORIG_TOTAL=$(grep -c "apt-get install" /tmp/dockerfile_original.txt)
OPT_NO_REC=$(grep "apt-get install" /tmp/dockerfile_optimized.txt | grep -c "\-\-no-install-recommends")
OPT_TOTAL=$(grep -c "apt-get install" /tmp/dockerfile_optimized.txt)

echo "Original: $ORIG_NO_REC / $ORIG_TOTAL apt-get install commands use --no-install-recommends"
echo "Optimized: $OPT_NO_REC / $OPT_TOTAL apt-get install commands use --no-install-recommends"
echo ""

# Count cleanup commands
echo "4. Cleanup Commands (rm -rf /var/lib/apt/lists/*)"
echo "----------------------------------------"
ORIG_CLEANUP=$(grep -c "rm -rf /var/lib/apt/lists/" /tmp/dockerfile_original.txt)
OPT_CLEANUP=$(grep -c "rm -rf /var/lib/apt/lists/" /tmp/dockerfile_optimized.txt)

echo "Original cleanup commands: $ORIG_CLEANUP"
echo "Optimized cleanup commands: $OPT_CLEANUP"
echo "Added: $((OPT_CLEANUP - ORIG_CLEANUP)) cleanup commands"
echo ""

# Analyze GCC-10 installation
echo "5. GCC-10 Installation Analysis"
echo "----------------------------------------"
echo "Original approach:"
grep -A 2 "gcc-10" /tmp/dockerfile_original.txt | head -5
echo ""
echo "Optimized approach:"
grep -B 2 -A 2 "gcc-10" /tmp/dockerfile_optimized.txt | head -10
echo ""

# Calculate estimated savings
echo "6. Estimated Size Savings"
echo "----------------------------------------"
echo "Based on Docker best practices:"
echo ""
echo "  • Package lists cleanup: ~50-100 MB per stage"
echo "    Stages with cleanup: $OPT_CLEANUP"
echo "    Estimated savings: $((OPT_CLEANUP * 75)) MB"
echo ""
echo "  • --no-install-recommends: ~20-50 MB per apt-get install"
echo "    New usage: $((OPT_NO_REC - ORIG_NO_REC)) commands"
echo "    Estimated savings: $((35 * (OPT_NO_REC - ORIG_NO_REC))) MB"
echo ""
TOTAL_SAVINGS=$((OPT_CLEANUP * 75 + 35 * (OPT_NO_REC - ORIG_NO_REC)))
echo "  📦 Total estimated savings: ~${TOTAL_SAVINGS} MB"
echo ""

# Layer reduction
echo "7. Docker Layer Reduction"
echo "----------------------------------------"
LAYER_REDUCTION=$((ORIG_RUN_BASE - OPT_RUN_BASE))
echo "Base stage layer reduction: $LAYER_REDUCTION layers"
echo "This improves:"
echo "  • Build cache efficiency"
echo "  • Image pull speed"
echo "  • Storage efficiency"
echo ""

# Generate summary
echo "========================================"
echo "📋 OPTIMIZATION SUMMARY"
echo "========================================"
echo ""
echo "✅ Achievements:"
echo "  • Reduced RUN commands in base stage: $LAYER_REDUCTION"
echo "  • Added --no-install-recommends: $((OPT_NO_REC - ORIG_NO_REC)) locations"
echo "  • Added cleanup commands: $((OPT_CLEANUP - ORIG_CLEANUP))"
echo "  • Estimated size reduction: ~${TOTAL_SAVINGS} MB"
echo ""
echo "📈 Coverage:"
echo "  • --no-install-recommends: $OPT_NO_REC/$OPT_TOTAL ($(( OPT_NO_REC * 100 / OPT_TOTAL ))%)"
echo "  • Cleanup after apt-get: $OPT_CLEANUP stages"
echo ""

# Cleanup
rm -f /tmp/dockerfile_original.txt /tmp/dockerfile_optimized.txt

echo "✨ Analysis complete!"

