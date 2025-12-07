#!/bin/bash
# eval_lora_models.sh - Evaluate all LoRA models in a directory
#
# Usage:
#   ./eval_lora_models.sh _models/lora/EleutherAI_pythia-70m
#   ./eval_lora_models.sh _models/lora/EleutherAI_pythia-70m --split test[:100]
#   ./eval_lora_models.sh _models/lora/EleutherAI_pythia-70m --device cuda:0

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Missing required argument${NC}"
    echo "Usage: $0 <lora_models_directory> [--split SPLIT] [--device DEVICE] [--batch_size SIZE] [--max_tokens TOKENS]"
    echo ""
    echo "Example:"
    echo "  $0 _models/lora/EleutherAI_pythia-70m"
    echo "  $0 _models/lora/EleutherAI_pythia-70m --split test[:100] --device cuda:0"
    exit 1
fi

LORA_DIR="$1"
shift  # Remove first argument

# Default parameters
SPLIT="test"
DEVICE="auto"
BATCH_SIZE=8
MAX_TOKENS=256
TEMPERATURE=0.7

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if directory exists
if [ ! -d "$LORA_DIR" ]; then
    echo -e "${RED}Error: Directory not found: $LORA_DIR${NC}"
    exit 1
fi

# Extract model name from directory path
MODEL_NAME=$(basename "$LORA_DIR")

echo -e "${BLUE}========================================"
echo "LoRA Model Evaluation Script"
echo "========================================${NC}"
echo -e "Model directory: ${GREEN}$LORA_DIR${NC}"
echo -e "Split: ${GREEN}$SPLIT${NC}"
echo -e "Device: ${GREEN}$DEVICE${NC}"
echo -e "Batch size: ${GREEN}$BATCH_SIZE${NC}"
echo -e "Max tokens: ${GREEN}$MAX_TOKENS${NC}"
echo -e "Temperature: ${GREEN}$TEMPERATURE${NC}"
echo ""

# Map directory names to dataset names for evaluate.py
# Hardcoded mappings for known datasets
map_dir_to_dataset() {
    local dir_name="$1"
    
    case "$dir_name" in
        "allenai_ai2_arc_ARC-Easy")
            echo "allenai/ai2_arc.ARC-Easy"
            ;;
        "allenai_ai2_arc_ARC-Challenge")
            echo "allenai/ai2_arc.ARC-Challenge"
            ;;
        "openai_gsm8k_main")
            echo "openai/gsm8k.main"
            ;;
        "google_boolq")
            echo "google/boolq"
            ;;
        "stanfordnlp_snli")
            echo "stanfordnlp/snli"
            ;;
        "winogrande_winogrande_m")
            echo "winogrande.winogrande_m"
            ;;
        "allenai_openbookqa_main")
            echo "allenai/openbookqa.main"
            ;;
        "Rowan_hellaswag")
            echo "Rowan/hellaswag"
            ;;
        *)
            # Default: try to convert underscores to slashes/dots
            # Three parts: provider_repo_config -> provider/repo.config
            if [[ "$dir_name" =~ ^([^_]+)_([^_]+)_(.+)$ ]]; then
                echo "${BASH_REMATCH[1]}/${BASH_REMATCH[2]}.${BASH_REMATCH[3]}"
            # Two parts: provider_repo -> provider/repo
            elif [[ "$dir_name" =~ ^([^_]+)_(.+)$ ]]; then
                echo "${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
            else
                echo "$dir_name"
            fi
            ;;
    esac
}

# Count total models
total_models=$(find "$LORA_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
current=0
success=0
failed=0

# Array to store results summary
declare -a results_summary

echo -e "${BLUE}Found $total_models LoRA models to evaluate${NC}"
echo ""

# Iterate through each subdirectory (each represents a dataset-specific LoRA model)
for model_path in "$LORA_DIR"/*/ ; do
    # Remove trailing slash
    model_path="${model_path%/}"
    
    # Get dataset directory name
    dataset_dir=$(basename "$model_path")
    
    # Skip if not a directory or doesn't contain adapter_config.json
    if [ ! -f "$model_path/adapter_config.json" ]; then
        echo -e "${YELLOW}⊘ Skipping $dataset_dir (not a valid LoRA model)${NC}"
        continue
    fi
    
    current=$((current + 1))
    
    # Convert directory name to dataset name
    dataset_name=$(map_dir_to_dataset "$dataset_dir")
    
    echo -e "${BLUE}[$current/$total_models] Evaluating: ${GREEN}$dataset_dir${NC}"
    echo -e "  Dataset: ${GREEN}$dataset_name${NC}"
    echo -e "  Model path: $model_path"
    
    # Run evaluation and capture output
    eval_output=$(python evaluate.py \
        --model_path "$model_path" \
        --model_type lora \
        --datasets "$dataset_name" \
        --split "$SPLIT" \
        --device "$DEVICE" \
        --evaluator.batch_size "$BATCH_SIZE" \
        --evaluator.max_new_tokens "$MAX_TOKENS" \
        --evaluator.temperature "$TEMPERATURE" \
        2>&1 | tee /tmp/eval_${dataset_dir}.log)
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully evaluated $dataset_dir${NC}"
        success=$((success + 1))
        
        # Extract accuracy from output (look for pattern like "24.88%")
        accuracy=$(echo "$eval_output" | grep -oP '(?<=\s)\d+\.\d+%' | head -1)
        samples=$(echo "$eval_output" | grep -oP '\d+(?=\s*$)' | tail -1)
        
        # Store result for summary
        if [ -n "$accuracy" ]; then
            results_summary+=("$dataset_name|$accuracy|$samples")
        fi
    else
        echo -e "${RED}✗ Failed to evaluate $dataset_dir${NC}"
        echo -e "${YELLOW}  See log: /tmp/eval_${dataset_dir}.log${NC}"
        failed=$((failed + 1))
    fi
    
    echo ""
done

# Print summary
echo -e "${BLUE}========================================"
echo "Evaluation Complete"
echo "========================================${NC}"
echo -e "Total models: ${BLUE}$total_models${NC}"
echo -e "Successful: ${GREEN}$success${NC}"
echo -e "Failed: ${RED}$failed${NC}"
echo ""

if [ $success -gt 0 ]; then
    echo -e "${GREEN}Results saved to: _results/${MODEL_NAME}_lora/${NC}"
    echo ""
    
    # Print results summary table
    if [ ${#results_summary[@]} -gt 0 ]; then
        echo -e "${BLUE}========================================"
        echo "Results Summary"
        echo "========================================${NC}"
        printf "%-40s %-15s %-10s\n" "Dataset" "Accuracy" "Samples"
        echo "=========================================================================================="
        
        for result in "${results_summary[@]}"; do
            IFS='|' read -r dataset accuracy samples <<< "$result"
            printf "%-40s %-15s %-10s\n" "$dataset" "$accuracy" "$samples"
        done
        echo ""
    fi
fi

if [ $failed -gt 0 ]; then
    echo -e "${YELLOW}Check logs in /tmp/eval_*.log for failed evaluations${NC}"
    exit 1
fi