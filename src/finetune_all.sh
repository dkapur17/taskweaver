#!/bin/bash

python finetune.py --model EleutherAI/pythia-70m
python finetune.py --model google/gemma-3-270m-it
python finetune.py --model Qwen/Qwen3-0.6B
