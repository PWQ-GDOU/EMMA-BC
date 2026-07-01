#!/bin/bash
# ==============================================================
# MODMA Cross-Dataset + Multi-Target Experiment — Server Runner
# ==============================================================
# Run after deploy.py uploads the scripts.
# Usage: bash run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "MODMA Cross-Dataset Experiment Pipeline"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# Check GPU
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

# Check disk
echo ""
echo "Disk:"
df -h /data/disk1/ | tail -1

# Step 1: Extract MODMA wav2vec2 features
echo ""
echo "============================================"
echo "STEP 1/4: Extract MODMA wav2vec2 features"
echo "============================================"
python3 run_modma_experiments.py --step extract 2>&1 | tee -a experiment.log

# Step 2: Cross-dataset validation
echo ""
echo "============================================"
echo "STEP 2/4: Cross-Dataset Validation"
echo "============================================"
python3 run_modma_experiments.py --step cross 2>&1 | tee -a experiment.log

# Step 3: MODMA multi-target training
echo ""
echo "============================================"
echo "STEP 3/4: MODMA Multi-Target Regression"
echo "============================================"
python3 run_modma_experiments.py --step train 2>&1 | tee -a experiment.log

# Step 4: Generate summary
echo ""
echo "============================================"
echo "STEP 4/4: Generate Summary Report"
echo "============================================"
python3 run_modma_experiments.py --step summary 2>&1 | tee -a experiment.log

echo ""
echo "============================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results: $SCRIPT_DIR/results/"
echo "============================================"
