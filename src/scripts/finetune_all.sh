#!/bin/bash

echo "Finetuning Pythia 70M on all datasets"
python lora_finetune.py \
--model EleutherAI/pythia-70m \
--datasets all \
--ignore_datasets tau/commonsense_qa ChilleD/SVAMP ehovy/race.middle \
--lora.target_modules query_key_value \
--lora.rank 2 \
--lora.alpha 8 \
--train.per_device_train_batch_size 16 \
--dataset.train_split train[:10000]


echo "Finetuning Gemma3 270M on all datasets"
python lora_finetune.py \
--model google/gemma-3-270m-it \
--datasets all \
--ignore_datasets tau/commonsense_qa ChilleD/SVAMP ehovy/race.middle \
--lora.target_modules q_proj v_proj \
--lora.rank 2 \
--lora.alpha 8 \
--train.per_device_train_batch_size 8 \
--dataset.train_split train[:10000]


echo "Finetuning Qwen3 0.6B 270M on all datasets"
python lora_finetune.py \
--model Qwen/Qwen3-0.6B \
--datasets all \
--ignore_datasets tau/commonsense_qa ChilleD/SVAMP ehovy/race.middle \
--lora.target_modules q_proj v_proj \
--lora.rank 2 \
--lora.alpha 8 \
--train.per_device_train_batch_size 4 \
--dataset.train_split train[:10000]

# Mixed dataset finetuning

echo "Finetuning Pythia 70M on mixed dataset"
python lora_finetune.py \
--model EleutherAI/pythia-70m \
--datasets mix \
--ignore_datasets tau/commonsense_qa ChilleD/SVAMP ehovy/race.middle \
--lora.target_modules query_key_value \
--lora.rank 2 \
--lora.alpha 8 \
--train.per_device_train_batch_size 16 \
--dataset.train_split train[:10000]

echo "Finetuning Gemma3 270M on all datasets"
python lora_finetune.py \
--model google/gemma-3-270m-it \
--datasets mix \
--ignore_datasets tau/commonsense_qa ChilleD/SVAMP ehovy/race.middle \
--lora.target_modules q_proj v_proj \
--lora.rank 2 \
--lora.alpha 8 \
--train.per_device_train_batch_size 8 \
--dataset.train_split train[:10000]


echo "Finetuning Qwen3 0.6B 270M on all datasets"
python lora_finetune.py \
--model Qwen/Qwen3-0.6B \
--datasets mix \
--ignore_datasets tau/commonsense_qa ChilleD/SVAMP ehovy/race.middle \
--lora.target_modules q_proj v_proj \
--lora.rank 2 \
--lora.alpha 8 \
--train.per_device_train_batch_size 4 \
--dataset.train_split train[:10000]