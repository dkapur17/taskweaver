"""
Simple integration test for TaskWeaver evaluation pipeline.

This test verifies that:
1. TaskWeaver can be loaded
2. Task detection works (is_taskweaver check)
3. prompt_lengths are calculated correctly
4. Generation works with prompt_lengths
5. The full evaluation pipeline runs end-to-end
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from datasets import Dataset

from hypernetwork import TaskWeaver
from eval.task import Task
from eval.eval_configs import NumericConfig


def test_taskweaver_eval():
    """Test TaskWeaver evaluation pipeline end-to-end."""
    print("="*70)
    print("TaskWeaver Evaluation Integration Test")
    print("="*70)
    
    # Step 1: Create a tiny TaskWeaver model
    print("\n[1/6] Creating TaskWeaver model...")
    from transformers import AutoModelForCausalLM
    
    model_name = "EleutherAI/pythia-14m"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      Using device: {device}")
    
    lm = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    hypernet = TaskWeaver(
        lm=lm,
        hidden_dim=64,
        lora_rank=2,
        lora_target_layers=['query_key_value'],
        lora_alpha=8,
        lora_dropout=0.01,
        model_name=model_name
    ).to(device)
    hypernet.eval()
    
    print("      ✓ TaskWeaver created")
    
    # Step 2: Test TaskWeaver detection
    print("\n[2/6] Testing TaskWeaver detection...")
    is_taskweaver = hasattr(hypernet, '_hypernet_forward')
    assert is_taskweaver, "FAIL: TaskWeaver not detected!"
    print(f"      ✓ TaskWeaver detected: {is_taskweaver}")
    
    # Step 3: Test prompt_lengths calculation
    print("\n[3/6] Testing prompt_lengths calculation...")
    test_prompts = [
        "What is 2+2? ",
        "What is 3+3? ",
        "What is 5+5? "
    ]
    
    model_inputs = tokenizer(test_prompts, return_tensors='pt', padding=True).to(device)
    prompt_lengths = model_inputs.attention_mask.sum(dim=1)
    
    print(f"      Input IDs shape: {model_inputs.input_ids.shape}")
    print(f"      Attention mask shape: {model_inputs.attention_mask.shape}")
    print(f"      Prompt lengths: {prompt_lengths.tolist()}")
    
    assert len(prompt_lengths) == len(test_prompts), "FAIL: Wrong number of prompt_lengths!"
    assert all(pl > 0 for pl in prompt_lengths), "FAIL: Invalid prompt_lengths!"
    print("      ✓ prompt_lengths calculated correctly")
    
    # Step 4: Test generation with prompt_lengths
    print("\n[4/6] Testing generation with prompt_lengths...")
    with torch.no_grad():
        outputs = hypernet.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            prompt_lengths=prompt_lengths,
            max_new_tokens=10,
            temperature=1.0,
            do_sample=True
        )
    
    assert outputs.shape[0] == len(test_prompts), "FAIL: Wrong batch size in outputs!"
    print(f"      Output shape: {outputs.shape}")
    print("      ✓ Generation successful")
    
    # Step 5: Create a simple evaluation task
    print("\n[5/6] Creating evaluation task...")
    
    # Create a tiny dataset
    test_data = Dataset.from_dict({
        'question': [
            "What is 2+2?",
            "What is 3+3?",
            "What is 5+5?"
        ],
        'answer': ["4", "6", "10"]
    })
    
    task = Task(
        task_name="test_math",
        dataset=test_data,
        user_template="{question}",
        assistant_template="{answer}",
        system_prompt=None,
        eval_config=NumericConfig(),
        is_chat_task=False,
        skip_formatting=False
    )
    
    print(f"      Task created: {task}")
    print(f"      Dataset size: {len(task)}")
    print("      ✓ Task created")
    
    # Step 6: Run full evaluation
    print("\n[6/6] Running full evaluation pipeline...")
    result = task.evaluate(
        model=hypernet,
        tokenizer=tokenizer,
        batch_size=2,
        max_new_tokens=10,
        temperature=1.0,
        progress=False
    )
    
    print(f"      Evaluation type: {result.eval_type}")
    print(f"      Number of samples: {result.num_samples}")
    print(f"      Predictions: {result.predictions}")
    print(f"      References: {result.references}")
    
    if result.metrics:
        print(f"      Metrics: {result.metrics}")
    
    assert result.num_samples == len(test_data), "FAIL: Wrong number of samples!"
    assert len(result.predictions) == len(test_data), "FAIL: Wrong number of predictions!"
    assert len(result.references) == len(test_data), "FAIL: Wrong number of references!"
    
    print("      ✓ Evaluation completed")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nTaskWeaver evaluation pipeline is working correctly:")
    print("  ✓ Model loading")
    print("  ✓ TaskWeaver detection")
    print("  ✓ prompt_lengths calculation")
    print("  ✓ Generation with dynamic LoRA")
    print("  ✓ Full evaluation pipeline")
    print("\nYou can now evaluate trained TaskWeaver models with:")
    print("  python run_eval.py eval/configs/hypernet_model.yaml")


if __name__ == '__main__':
    test_taskweaver_eval()
