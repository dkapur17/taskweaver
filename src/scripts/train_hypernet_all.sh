#!/bin/bash

echo "Training TaskWeaver with Pythia 70M"

python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--ignore_datasets tau/commonsense_qa ChilleD/SVAMP ehovy/race.middle \
--hypernet.target_modules query_key_value \
--hypernet.hidden_dim 1024 \
--hypernet.lora_rank 2 \
--hypernet.lora_alpha 8 \
--train.per_device_train_batch_size 8 \
--train.gradient_accumulation_steps 2


echo "Training TaskWeaver with Gemma3 270m"

python train_hypernet.py \
--model google/gemma-3-270m-it \
--datasets all \
--ignore_datasets tau/commonsense_qa ChilleD/SVAMP ehovy/race.middle \
--hypernet.target_modules q_proj v_proj \
--hypernet.hidden_dim 1024 \
--hypernet.lora_rank 2 \
--hypernet.lora_alpha 8 \
--train.per_device_train_batch_size 4 \
--train.gradient_accumulation_steps 4



echo "Training TaskWeaver with Qwen 3 0.6B"

python train_hypernet.py \
--model Qwen/Qwen3-0.6B \
--datasets all \
--ignore_datasets tau/commonsense_qa ChilleD/SVAMP ehovy/race.middle \
--hypernet.target_modules q_proj v_proj \
--hypernet.hidden_dim 1024 \
--hypernet.lora_rank 2 \
--hypernet.lora_alpha 8 \
--train.per_device_train_batch_size 2 \
--train.gradient_accumulation_steps 8
