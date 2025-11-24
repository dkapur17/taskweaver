# Evaluation Framework

## Quick Start

1. Choose or create a config file in `configs/`:
   - `quick_test.yaml` - Small subset for testing (10 samples each)
   - `base_model.yaml` - For non-chat models (5% of datasets)
   - `eval_config_template.yaml` - Template for custom configs

2. Run evaluation:
```bash
cd src/eval
python run_eval.py configs/quick_test.yaml
```

## Config Format

```yaml
model_name: "Qwen/Qwen3-0.6B"
is_chat_model: true  # false for base models

datasets:
  gsm8k:
    enabled: true
    split: "test[:5%]"  # or "test[:100]"
    batch_size: 8
    max_new_tokens: 256
```

## Datasets

- **gsm8k**: Math word problems (numeric answers)
- **snli**: Natural language inference (0/1/2 classification)
- **squad_v2**: Reading comprehension (exact match)
- **arc_easy**: Science multiple choice (A/B/C/D)

## Results

Results saved to `results/{model_name}/{timestamp}.json` with:
- Summary statistics (avg accuracy, per-task metrics)
- Individual task results
- Optional: predictions and references (set `include_predictions: true`)
