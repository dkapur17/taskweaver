#!/bin/bash

# All datasets except held out datasets

echo "Training LoRA adapters for Pythia 70M on all datasets"

python train_lora.py \
--model EleutherAI/pythia-70m \
--datasets all \
--ignore_datasets tau/commonsense_qa ChilleD/SVAMP ehovy/race.middle \
--lora.target_modules query_key_value \
--lora.rank 2 \
--lora.alpha 8 \
--train.per_device_train_batch_size 16 \
--train.gradient_accumulation_steps 1 \
--dataset.train_split train[:10000]


echo "Training LoRA adapters for Gemma3 270M on all datasets"
python train_lora.py \
--model google/gemma-3-270m-it \
--datasets all \
--ignore_datasets tau/commonsense_qa ChilleD/SVAMP ehovy/race.middle \
--lora.target_modules q_proj v_proj \
--lora.rank 2 \
--lora.alpha 8 \
--train.per_device_train_batch_size 4 \
--train.gradient_accumulation_steps 4 \
--dataset.train_split train[:10000]


echo "Training LoRA adapters for Qwen3 0.6B on all datasets"
python train_lora.py \
--model Qwen/Qwen3-0.6B \
--datasets all \
--ignore_datasets tau/commonsense_qa ChilleD/SVAMP ehovy/race.middle \
--lora.target_modules q_proj v_proj \
--lora.rank 2 \
--lora.alpha 8 \
--train.per_device_train_batch_size 4 \
--train.gradient_accumulation_steps 4 \
--dataset.train_split train[:10000]

# Mixed dataset except held out sets

echo "Training LoRA adapters for Pythia 70M on mixed dataset"
python train_lora.py \
--model EleutherAI/pythia-70m \
--datasets mix \
--ignore_datasets tau/commonsense_qa ChilleD/SVAMP ehovy/race.middle \
--lora.target_modules query_key_value \
--lora.rank 2 \
--lora.alpha 8 \
--train.per_device_train_batch_size 16 \
--train.gradient_accumulation_steps 1 \
--dataset.train_split train[:10000]


echo "Training LoRA adapters for Gemma3 270M on all datasets"
python train_lora.py \
--model google/gemma-3-270m-it \
--datasets mix \
--ignore_datasets tau/commonsense_qa ChilleD/SVAMP ehovy/race.middle \
--lora.target_modules q_proj v_proj \
--lora.rank 2 \
--lora.alpha 8 \
--train.per_device_train_batch_size 4 \
--train.gradient_accumulation_steps 4 \
--dataset.train_split train[:10000]


echo "Training LoRA adapters for Qwen3 0.6B on all datasets"
python train_lora.py \
--model Qwen/Qwen3-0.6B \
--datasets mix \
--ignore_datasets tau/commonsense_qa ChilleD/SVAMP ehovy/race.middle \
--lora.target_modules q_proj v_proj \
--lora.rank 2 \
--lora.alpha 8 \
--train.per_device_train_batch_size 4 \
--train.gradient_accumulation_steps 4 \
--dataset.train_split train[:10000]