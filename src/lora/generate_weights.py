from typing import Optional, Any, Dict

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch


class LORA_Finetune:
    def __init__(self, model_name: str,
                 dataset_name: str,
                 dataset_subset: Optional[str] = None,
                 lora_rank: int = 2,
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
    
    OUTPUT_DIR = "lora_weights"

    def load_model_and_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.float16,
        # )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            # quantization_config=bnb_config,
            device_map="auto",
        )
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()
    
    def load_dataset(self, split: str = "train", formatting_func=None) -> None:
        assert self.tokenizer is not None, "Tokenizer not loaded. Call load_model_and_tokenizer() first."

        if self.dataset_subset:
            dataset = load_dataset(self.dataset_name, self.dataset_subset, split=split)
        else:
            dataset = load_dataset(self.dataset_name, split=split)
        
        dataset = dataset.map(formatting_func, remove_columns=dataset.column_names)

        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
        self.train_dataset = train_test_split["train"]
        self.eval_dataset = train_test_split["test"]


    def train(self) -> None:

        assert self.model is not None, "Model not loaded. Call load_model_and_tokenizer() first."
        assert self.train_dataset is not None, "Dataset not loaded. Call load_dataset() first."

        output_dir = self.OUTPUT_DIR + "/" + self.model_name.replace("/", "-") + "-lora-finetuned-" + self.dataset_name.replace("/", "-")

        training_arguments = TrainingArguments(
            max_steps=500,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
            bf16=True,
            logging_steps=10,
            output_dir=output_dir,
            save_total_limit=2,
            save_steps=50,
            remove_unused_columns=False,
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
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
        model_name="google/gemma-3-270m-it",
        dataset_name="allenai/ai2_arc",
        dataset_subset="ARC-Easy",
    )
    def process_sample(example):
        q = example["question"]
        labels = example["choices"]["label"]
        texts = example["choices"]["text"]
        mc_block = "\n".join([f"{label}. {txt}" for label, txt in zip(labels, texts)])
        answer = example["answerKey"]
        text = (
            f"Answer the multiple-choice question based on the provided options.\n\n"
            f"Question: {q}\n"
            f"{mc_block}\n\n"
            f"Answer: {answer}"
        )
        return {"text": text}

    lora_finetuner.load_model_and_tokenizer()
    lora_finetuner.load_dataset(formatting_func=process_sample)
    lora_finetuner.train()