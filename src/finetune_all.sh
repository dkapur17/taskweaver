#!/bin/bash

python finetune.py --model EleutherAI/pythia-70m --lora.target_modules query_key_value --lora.rank 2 --lora.alpha 8
python finetune.py --model google/gemma-3-270m-it --lora.target_modules q_proj v_proj --lora.rank 2 --lora.alpha 8
python finetune.py --model Qwen/Qwen3-0.6B --lora.target_modules q_proj v_proj --lora.rank 2 --lora.alpha 8
