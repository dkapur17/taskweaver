#!/bin/bash
# evaluate_all_taskweaver.sh - Evaluate TaskWeaver hypernetwork model on datasets
#
# Usage:
#   ./evaluate_all_taskweaver.sh _models/hypernet/EleutherAI_pythia-70m/mix_8_datasets
#   ./evaluate_all_taskweaver.sh _models/hypernet/EleutherAI_pythia-70m/mix_8_datasets --split test[:100]
#   ./evaluate_all_taskweaver.sh _models/hypernet/EleutherAI_pythia-70m/mix_8_datasets --datasets "allenai/ai2_arc.ARC-Easy google/boolq"

set -e  # Exit on error

# Start timing
SCRIPT_START_TIME=$(date +%s)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Missing required argument${NC}"
    echo "Usage: $0 <hypernet_model_path> [--datasets DATASETS] [--split SPLIT] [--device DEVICE] [--batch_size SIZE] [--max_tokens TOKENS] [--num_pass K]"
    echo ""
    echo "Example:"
    echo "  $0 _models/hypernet/EleutherAI_pythia-70m/mix_8_datasets"
    echo "  $0 _models/hypernet/EleutherAI_pythia-70m/mix_8_datasets --datasets all"
    echo "  $0 _models/hypernet/EleutherAI_pythia-70m/mix_8_datasets --datasets \"allenai/ai2_arc.ARC-Easy google/boolq\""
    exit 1
fi

HYPERNET_PATH="$1"
shift  # Remove first argument

# Default parameters
DATASETS_ARG="all"
SPLIT="test"
DEVICE="auto"
BATCH_SIZE=8
MAX_TOKENS=256
TEMPERATURE=0.7
NUM_PASS=3

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)
            DATASETS_ARG="$2"
            shift 2
            ;;
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
if [ ! -d "$HYPERNET_PATH" ]; then
    echo -e "${RED}Error: Directory not found: $HYPERNET_PATH${NC}"
    exit 1
fi

# Check if it's a valid hypernetwork model
if [ ! -f "$HYPERNET_PATH/config.json" ] || [ ! -f "$HYPERNET_PATH/pytorch_model.bin" ]; then
    echo -e "${RED}Error: Not a valid hypernetwork model (missing config.json or pytorch_model.bin)${NC}"
    exit 1
fi

# Extract model name from path
MODEL_NAME=$(basename "$(dirname "$HYPERNET_PATH")")

echo -e "${BLUE}========================================"
echo "TaskWeaver Hypernetwork Evaluation"
echo "========================================${NC}"
echo -e "Model path: ${GREEN}$HYPERNET_PATH${NC}"
echo -e "Datasets: ${GREEN}$DATASETS_ARG${NC}"
echo -e "Split: ${GREEN}$SPLIT${NC}"
echo -e "Device: ${GREEN}$DEVICE${NC}"
echo -e "Batch size: ${GREEN}$BATCH_SIZE${NC}"
echo -e "Max tokens: ${GREEN}$MAX_TOKENS${NC}"
echo -e "Temperature: ${GREEN}$TEMPERATURE${NC}"
echo -e "Num generations: ${GREEN}$NUM_PASS${NC}"
echo ""

# Display GPU information
if command -v nvidia-smi &> /dev/null; then
    echo -e "${BLUE}GPU Information:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while read line; do
        echo -e "  ${GREEN}$line${NC}"
    done
    echo ""
fi

# Run evaluation
echo -e "${BLUE}Running evaluation...${NC}"
echo ""

python evaluate.py \
    --model_path "$HYPERNET_PATH" \
    --model_type hypernet \
    --datasets "$DATASETS_ARG" \
    --split "$SPLIT" \
    --device "$DEVICE" \
    --evaluator.batch_size "$BATCH_SIZE" \
    --evaluator.max_new_tokens "$MAX_TOKENS" \
    --evaluator.temperature "$TEMPERATURE" \
    --evaluator.num_pass "$NUM_PASS"

exit_code=$?

# Calculate elapsed time
SCRIPT_END_TIME=$(date +%s)
ELAPSED_TIME=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
ELAPSED_HOURS=$((ELAPSED_TIME / 3600))
ELAPSED_MINUTES=$(((ELAPSED_TIME % 3600) / 60))
ELAPSED_SECONDS=$((ELAPSED_TIME % 60))

echo ""
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}========================================"
    echo "Evaluation Complete"
    echo "========================================${NC}"
    echo -e "${GREEN}Results saved to: _results/${MODEL_NAME}_hypernet/${NC}"
    echo -e "Total time: ${BLUE}${ELAPSED_HOURS}h ${ELAPSED_MINUTES}m ${ELAPSED_SECONDS}s${NC}"
else
    echo -e "${RED}========================================"
    echo "Evaluation Failed"
    echo "========================================${NC}"
    echo -e "Total time: ${BLUE}${ELAPSED_HOURS}h ${ELAPSED_MINUTES}m ${ELAPSED_SECONDS}s${NC}"
    exit 1
fi