#!/bin/bash

python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--hypernet.target_modules query_key_value \
--hypernet.hidden_dim 512 \
--train.per_device_train_batch_size 8

python train_hypernet.py \
--model google/gemma-3-270m-it \
--datasets all \
--hypernet.target_modules q_proj v_proj \
--hypernet.hidden_dim 2048 \
--train.per_device_train_batch_size 4


python train_hypernet.py \
--model Qwen/Qwen3-0.6B \
--datasets all \
--hypernet.target_modules q_proj v_proj \
--hypernet.hidden_dim  4096 \
--train.per_device_train_batch_size 2
