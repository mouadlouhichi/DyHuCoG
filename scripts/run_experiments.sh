#!/bin/bash
# Run all experiments for DyHuCoG paper

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p results/figures
mkdir -p logs
mkdir -p checkpoints

# Download data
echo "Downloading MovieLens-100K dataset..."
bash scripts/download_data.sh ml-100k

# Preprocess data
echo "Preprocessing data..."
python scripts/preprocess.py --dataset ml-100k

# Train models
echo "Training DyHuCoG..."
python scripts/train.py --dataset ml-100k --model dyhucog --epochs 100

echo "Training baselines..."
python scripts/train.py --dataset ml-100k --model lightgcn --epochs 100
python scripts/train.py --dataset ml-100k --model ngcf --epochs 100

# Run baseline comparison
echo "Running baseline experiments..."
python experiments/run_baselines.py --dataset ml-100k --n_runs 5

# Run ablation study
echo "Running ablation study..."
python experiments/run_ablation.py --dataset ml-100k

# Run statistical tests
echo "Running statistical significance tests..."
python experiments/run_statistical_tests.py --results_dir results/baselines

# Generate explanations
echo "Generating SHAP explanations..."
BEST_CHECKPOINT=$(ls checkpoints/dyhucog*/best_model.pth | head -1)
python scripts/explain.py --checkpoint $BEST_CHECKPOINT --analysis_type all

# Run enhanced SHAP analysis
echo "Running enhanced SHAP analysis..."
python experiments/enhanced_shap_analysis.py --checkpoint $BEST_CHECKPOINT

# Generate paper figures
echo "Generating paper figures..."
python scripts/generate_paper_figures.py

echo "All experiments completed! Check results/ directory for outputs."