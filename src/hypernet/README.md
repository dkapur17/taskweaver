# TaskWeaver Hypernetwork

A hypernetwork-based approach to dynamic LoRA (Low-Rank Adaptation) for language models. TaskWeaver generates task-specific LoRA weights on-the-fly based on input prompts, enabling efficient multi-task adaptation.

## Project Structure

```
hypernet/
├── __init__.py                 # Package initialization
├── dynamic_lora.py            # Dynamic LoRA linear layer
├── hypernetwork.py            # TaskWeaver hypernetwork implementation
├── collator.py                # Data collator with prompt lengths
├── dataset.py                 # Multi-dataset creation utilities
├── train.py                   # Training script with TensorBoard logging
├── configs/
│   ├── dataset_config.json    # Single dataset configuration
│   └── multi_dataset_config.json  # Multi-dataset configuration
├── dev.ipynb                  # Development notebook (original POC)
└── README.md                  # This file
```

## Quick Start

### Installation

Ensure you have the required dependencies:

```bash
pip install torch transformers datasets einops tensorboard python-dotenv
```

### Basic Training

Train with default settings (GSM8K dataset):

```bash
cd src/hypernet
python train.py --use_default_dataset
```

### Training with Custom Configuration

1. Create or modify a dataset configuration file in `configs/`
2. Run training:

```bash
python train.py --dataset_config configs/multi_dataset_config.json
```

### Training Arguments

```bash
python train.py \
  --model_name EleutherAI/pythia-70M-deduped \
  --hidden_dim 1024 \
  --lora_rank 2 \
  --lora_alpha 16 \
  --lora_target_layers query_key_value \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --num_epochs 3 \
  --output_dir ./output \
  --tensorboard_dir ./runs
```

### Monitoring Training

View training metrics in TensorBoard:

```bash
tensorboard --logdir ./runs
```

## Usage

### Saving a Trained Model

After training, save your TaskWeaver model:

```python
# Save the hypernetwork (stores config and weights, but not the base LM)
hypernet.save_pretrained('./my_trained_model')
tokenizer.save_pretrained('./my_trained_model')
```

This saves:
- `config.json`: Model configuration including the base LM name
- `pytorch_model.bin`: Hypernetwork weights (excluding the base language model)

### Loading a Trained Model

Load a saved TaskWeaver model:

```python
from hypernetwork import TaskWeaver
from transformers import AutoTokenizer

# Load the TaskWeaver model (automatically loads the base LM using transformers)
hypernet = TaskWeaver.from_pretrained('./my_trained_model')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('./my_trained_model')

# Optionally specify a different device
hypernet = TaskWeaver.from_pretrained('./my_trained_model', device='cuda')

# Or override the base model name
hypernet = TaskWeaver.from_pretrained(
    './my_trained_model',
    model_name='EleutherAI/pythia-160M-deduped'
)
```

### Generating Text

```python
# Prepare input
prompt = "What is 2 + 2?"
inputs = tokenizer(prompt, return_tensors='pt')

# Generate with task-adapted model
outputs = hypernet.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=100,
    temperature=0.7
)

# Decode output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Adding New Datasets

To add support for a new dataset:

1. Create a new processor class in `dataset.py`:

```python
class MyDatasetProcessor(DatasetProcessor):
    def process(self, examples: Dict) -> Dict:
        prompts = []
        completions = []

        # Extract prompts and completions from examples
        for example in examples:
            prompts.append(example['input_field'])
            completions.append(example['output_field'])

        return self._prepare_examples(prompts, completions)
```

2. Register the processor in `create_dataset()` function:

```python
creator.register_processor('my_dataset', MyDatasetProcessor(tokenizer))
```

3. Add the dataset to your configuration file:

```json
[
  {
    "name": "my_dataset",
    "dataset_path": "path/to/dataset",
    "dataset_config": "config_name",
    "num_samples": 1000
  }
]
```

## Architecture

### DynamicLoraLinear

A custom linear layer that accepts batch-specific LoRA matrices (A and B), enabling instance-level adaptation.

### TaskWeaver

The main hypernetwork that:
1. Processes input prompts through the base language model to extract semantic embeddings
2. Generates task-specific LoRA weights using the hypernetwork
3. Injects these weights into DynamicLoraLinear layers
4. Runs the adapted model for training or generation

### Data Flow

```
Input Prompt → LM (frozen) → Semantic Embedding → Hypernetwork → LoRA Weights →
→ Inject into DynamicLoraLinear → Adapted LM → Output
```

## Configuration Files

### Dataset Configuration Format

```json
[
  {
    "name": "processor_name",        // Registered processor name
    "dataset_path": "dataset/path",  // HuggingFace dataset path
    "dataset_config": "config",      // Dataset configuration (optional)
    "split": "train",                // Dataset split (optional)
    "num_samples": 1000              // Number of samples (optional)
  }
]
```

## License

[Add your license here]

## Citation

If you use this code, please cite:

```
[Add citation information]
```
