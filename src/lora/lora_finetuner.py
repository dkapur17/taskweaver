from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import os
from dsconf.dataset_configs import DatasetConfig

from typing import Optional, Any, Dict, List, Tuple, Type

class LoraFinetuner:

    def __init__(self,
                 model_path: str,
                 dataset_config: Type[DatasetConfig],
                 dataset_train_split: str = "train",
                 dataset_test_split: Optional[str] = "test",
                 lora_rank: int = 2,
                 lora_alpha: int = 8,
                 lora_dropout: float = 0.05,
                 target_modules: List[str] = ['q_proj', 'v_proj'],
                 device_map: str = 'auto',
                 output_dir: str = 'lora_models'):
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias='none',
            target_modules=target_modules,
            task_type='CAUSAL_LM'
        )

        self.model, self.tokenizer = self._get_pretrained(model_path, device_map, lora_config)
        self.is_chat = self.tokenizer.chat_template is not None
        self.train_dataset, self.test_dataset = self._get_datasets(dataset_config, self.is_chat, dataset_train_split, dataset_test_split)
        self.output_dir = os.path.join(output_dir, model_path.replace('/', '_'), dataset_config.id().replace('/', '_').replace('.', '_'))


    def _get_pretrained(
            self, 
            model_path: str, 
            device_map:str, 
            lora_config: LoraConfig
            ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        
        model =  AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        peft_model = get_peft_model(model, lora_config)

        return peft_model, tokenizer
    
    def _get_datasets(
            self,
            dataset_config: Type[DatasetConfig],
            is_chat: bool,
            dataset_train_split: str,
            dataset_test_split: Optional[str],
        ) -> Tuple[Dataset, Optional[Dataset]]:

        train_dataset = dataset_config.load_and_process(is_chat, dataset_train_split)
        test_dataset = dataset_config.load_and_process(is_chat, dataset_test_split) if dataset_test_split is not None else None

        return train_dataset, test_dataset
    
    def train(
            self,
            num_train_epochs: float = 3.0,
            per_device_train_batch_size: int = 2,
            gradient_accumulation_steps: int = 2,
            learning_rate: float = 5e-5,
            bf16: bool = False,
            logging_steps: int = 10,
            save_total_limit: int = 2,
            save_steps: int = 100,
            **trainer_kwargs) -> None:

        training_args = TrainingArguments(
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            bf16=bf16,
            logging_steps=logging_steps,
            output_dir=self.output_dir,
            save_total_limit=save_total_limit,
            save_steps=save_steps,
            remove_unused_columns=False,
            **trainer_kwargs
        )


        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            processing_class=self.tokenizer,
            args=training_args,
        )

        trainer.train()

        metrics = trainer.evaluate()
        print(f"Evaluation metrics: {metrics}")

    def save(self):
        print(f"Saving mode to {self.output_dir}")
        self.model.save_pretrained(self.output_dir)

if __name__ == "__main__":

    lora_finetuner = LoraFinetuner(
        model_path='google/gemma-3-270m-it',
        dataset_config=DatasetConfig.from_dataset_path('openai/gsm8k', 'main'),
        dataset_train_split='train[:100]',
        dataset_test_split='test[:20]'
    )

    lora_finetuner.train()