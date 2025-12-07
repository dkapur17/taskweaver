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

# Start timing
SCRIPT_START_TIME=$(date +%s)

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Missing required argument${NC}"
    echo "Usage: $0 <lora_models_directory> [--split SPLIT] [--device DEVICE] [--batch_size SIZE] [--max_tokens TOKENS] [--num_pass K]"
    echo ""
    echo "Example:"
    echo "  $0 _models/lora/EleutherAI_pythia-70m"
    echo "  $0 _models/lora/EleutherAI_pythia-70m --split test[:100] --device cuda:0"
    echo "  $0 _models/lora/EleutherAI_pythia-70m --num_pass 5"
    exit 1
fi

LORA_DIR="$1"
shift  # Remove first argument

# Default parameters
SPLIT="test"
DEVICE="cuda"
BATCH_SIZE=4
MAX_TOKENS=256
TEMPERATURE=0.7
NUM_PASS=2

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
        --num_pass)
            NUM_PASS="$2"
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
echo -e "Num generations: ${GREEN}$NUM_PASS${NC}"
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

# Display GPU information
if command -v nvidia-smi &> /dev/null; then
    echo -e "${BLUE}GPU Information:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while read line; do
        echo -e "  ${GREEN}$line${NC}"
    done
    echo ""
fi

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
        --evaluator.num_pass "$NUM_PASS" \
        2>&1 | tee /tmp/eval_${dataset_dir}.log)
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully evaluated $dataset_dir${NC}"
        success=$((success + 1))
        
        # Extract the entire summary line for this dataset
        summary_line=$(echo "$eval_output" | grep -E "^$dataset_name\s+" | head -1)
        
        if [ -n "$summary_line" ]; then
            # Store the entire line for the summary table
            results_summary+=("$summary_line")
        fi
    else
        echo -e "${RED}✗ Failed to evaluate $dataset_dir${NC}"
        echo -e "${YELLOW}  See log: /tmp/eval_${dataset_dir}.log${NC}"
        failed=$((failed + 1))
    fi
    
    echo ""
done

# Calculate elapsed time
SCRIPT_END_TIME=$(date +%s)
ELAPSED_TIME=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
ELAPSED_HOURS=$((ELAPSED_TIME / 3600))
ELAPSED_MINUTES=$(((ELAPSED_TIME % 3600) / 60))
ELAPSED_SECONDS=$((ELAPSED_TIME % 60))

# Print summary
echo -e "${BLUE}========================================"
echo "Evaluation Complete"
echo "========================================${NC}"
echo -e "Total models: ${BLUE}$total_models${NC}"
echo -e "Successful: ${GREEN}$success${NC}"
echo -e "Failed: ${RED}$failed${NC}"
echo -e "Total time: ${BLUE}${ELAPSED_HOURS}h ${ELAPSED_MINUTES}m ${ELAPSED_SECONDS}s${NC}"
echo ""

if [ $success -gt 0 ]; then
    echo -e "${GREEN}Results saved to: _results/${MODEL_NAME}_lora/${NC}"
    echo ""
    
    # Print results summary table
    if [ ${#results_summary[@]} -gt 0 ]; then
        echo -e "${BLUE}========================================"
        if [ "$NUM_PASS" -gt 1 ]; then
            echo "Results Summary (Pass@$NUM_PASS)"
        else
            echo "Results Summary"
        fi
        echo "========================================${NC}"
        
        for result in "${results_summary[@]}"; do
            echo "$result"
        done
        echo ""
    fi
fi

if [ $failed -gt 0 ]; then
    echo -e "${YELLOW}Check logs in /tmp/eval_*.log for failed evaluations${NC}"
    exit 1
fi