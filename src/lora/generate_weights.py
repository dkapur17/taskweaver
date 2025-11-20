from typing import Optional, Any, Dict

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
import torch


class LORA_Finetune:
    def __init__(self, model_name: str,
                 dataset_name: str,
                 dataset_subset: Optional[str] = None,
                 lora_rank: int = 8,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset

        self.peft_config = LoraConfig(
            r=lora_rank,  # Rank dimension - typically between 4-32
            lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
            lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
            # Bias type for LoRA. the corresponding biases will be updated during training.
            bias="none",
            target_modules="all-linear",  # Which modules to apply LoRA to
            task_type="CAUSAL_LM",  # Task type for model architecture
        )

        self.train_data = None
        self.eval_dataset = None
        self.model = None
        self.tokenizer = None

    def load_model_and_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )
    
    def load_dataset(self, build_text, split: str = "train") -> None:
        if self.dataset_subset:
            dataset = load_dataset(self.dataset_name, self.dataset_subset, split=split)
        else:
            dataset = load_dataset(self.dataset_name, split=split)

        dataset = dataset.map(build_text)
        print("Sample data:", dataset[0])
        
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
        self.train_dataset = train_test_split["train"]
        self.eval_dataset = train_test_split["test"]


    def train(self) -> None:

        assert self.model is not None, "Model not loaded. Call load_model_and_tokenizer() first."
        assert self.train_dataset is not None, "Dataset not loaded. Call load_dataset() first."

        output_dir = self.model_name.replace("/", "-") + "-lora-finetuned-" + self.dataset_name.replace("/", "-")

        training_arguments = TrainingArguments(
            max_steps=1000,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=10,
            output_dir=output_dir,
            save_total_limit=2,
            save_steps=200,
            remove_unused_columns=False,
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            peft_config=self.peft_config,
            processing_class=self.tokenizer,
            args=training_arguments,
        )
        trainer.train()

        metrics = trainer.evaluate()
        print("Evaluation metrics:", metrics)

        print("Saving LoRA adapter...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)




if __name__ == "__main__":
    lora_finetuner = LORA_Finetune(
        model_name="google/gemma-3-270m",
        dataset_name="allenai/ai2_arc",
        dataset_subset="ARC-Easy",
    )
    def build_text(example):
        q = example["question"]
        labels = example["choices"]["label"]
        texts = example["choices"]["text"]

        # Build the multiple-choice block
        mc_block = "\n".join([f"{label}. {txt}" for label, txt in zip(labels, texts)])

        # The answer is a letter like "B"
        answer = example["answerKey"]

        example["text"] = (
            f"Question: {q}\n"
            f"{mc_block}\n\n"
            f"Answer: {answer}"
        )
        return example

    lora_finetuner.load_model_and_tokenizer()
    lora_finetuner.load_dataset(build_text)
    lora_finetuner.train()
