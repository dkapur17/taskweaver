"""Quick test script to verify SFTTrainer integration works correctly."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

from hypernetwork import TaskWeaver
from collator import DataCollatorWithPromptLengths


def test_sft_trainer():
    """Test SFTTrainer with a tiny model and small dataset."""
    print("Testing SFTTrainer integration...")
    
    # Use Pythia-14M for fast testing (smallest model that works with TaskWeaver)
    model_name = "EleutherAI/pythia-14m"  # Very small model for testing (~56MB)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    lm = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create TaskWeaver
    hypernet = TaskWeaver(
        lm=lm,
        hidden_dim=64,
        lora_rank=2,
        lora_target_layers=['query_key_value'],  # Pythia uses query_key_value
        lora_alpha=8,
        lora_dropout=0.01,
        model_name=model_name
    )
    
    # Move to device
    hypernet = hypernet.to(device)
    
    print("TaskWeaver created successfully!")
    hypernet.print_trainable_parameters()
    
    # Create tiny test dataset
    test_data = {
        'prompt': [
            "What is 2+2? ",
            "What is 3+3? ",
            "What is 5+5? ",
        ],
        'completion': [
            "4",
            "6",
            "10",
        ]
    }
    
    # Tokenize manually
    tokenized_data = {'input_ids': [], 'attention_mask': [], 'labels': [], 'prompt_length': []}
    
    for prompt, completion in zip(test_data['prompt'], test_data['completion']):
        prompt_tokens = tokenizer(prompt, add_special_tokens=True)
        completion_tokens = tokenizer(completion, add_special_tokens=False)
        
        input_ids = prompt_tokens['input_ids'] + completion_tokens['input_ids']
        attention_mask = prompt_tokens['attention_mask'] + completion_tokens['attention_mask']
        prompt_length = len(prompt_tokens['input_ids'])
        labels = [-100] * prompt_length + completion_tokens['input_ids']
        
        tokenized_data['input_ids'].append(input_ids)
        tokenized_data['attention_mask'].append(attention_mask)
        tokenized_data['labels'].append(labels)
        tokenized_data['prompt_length'].append(prompt_length)
    
    dataset = Dataset.from_dict(tokenized_data)
    print(f"Test dataset created with {len(dataset)} examples")
    
    # Create data collator
    data_collator = DataCollatorWithPromptLengths(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors='pt'
    )
    
    # Test collator preserves prompt_length
    batch = data_collator([dataset[0], dataset[1]])
    assert 'prompt_lengths' in batch, "FAIL: prompt_lengths not in batch!"
    print(f"✓ Data collator preserves prompt_lengths: {batch['prompt_lengths']}")
    
    # Create minimal training args
    training_args = TrainingArguments(
        output_dir='./test_output',
        per_device_train_batch_size=2,
        max_steps=3,  # Just 3 steps for quick test
        logging_steps=1,
        save_steps=999,  # Don't save during test
        remove_unused_columns=False,  # Critical!
        report_to=[],  # No logging
    )
    
    # Create SFTTrainer
    print("\nCreating SFTTrainer...")
    trainer = SFTTrainer(
        model=hypernet,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer
    )
    
    print("✓ SFTTrainer created successfully!")
    
    # Test forward pass
    print("\nTesting forward pass with prompt_lengths...")
    hypernet.eval()
    
    # Move batch tensors to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = hypernet(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            prompt_lengths=batch['prompt_lengths']
        )
    print(f"✓ Forward pass successful! Loss: {outputs.loss.item():.4f}")
    
    # Test training for a few steps
    print("\nTesting training (3 steps)...")
    trainer.train()
    print("✓ Training successful!")
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED!")
    print("="*50)
    print("\nSFTTrainer integration is working correctly.")
    print("You can now run the full training script with confidence.")


if __name__ == '__main__':
    test_sft_trainer()
