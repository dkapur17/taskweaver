#!/bin/bash

echo "Evaluating Pythia 70M on all datasets"

python evaluate.py \
--model_path EleutherAI/pythia-70m \
--model_type base \
--datasets all \
--ignore_datasets openai/gsm8k.main ChilleD/SVAMP \
--device cuda \
--evaluator.batch_size 128 \
--evaluator.max_new_tokens 32 \

python evaluate.py \
--model_path EleutherAI/pythia-70m \
--model_type base \
--datasets openai/gsm8k.main ChilleD/SVAMP \
--device cuda \
--evaluator.batch_size 128 \
--evaluator.max_new_tokens 256 \

echo "Evaluating Gemma3 270M on all datasets"

python evaluate.py \
--model_path google/gemma-3-270m-it \
--model_type base \
--datasets all \
--ignore_datasets openai/gsm8k.main ChilleD/SVAMP \
--device cuda \
--evaluator.batch_size 64 \
--evaluator.max_new_tokens 32 \

python evaluate.py \
--model_path google/gemma-3-270m-it \
--model_type base \
--datasets openai/gsm8k.main ChilleD/SVAMP \
--device cuda \
--evaluator.batch_size 64 \
--evaluator.max_new_tokens 256 \

echo "Evaluating Qwen3 0.6B on all datasets"

python evaluate.py \
--model_path Qwen/Qwen3-0.6B \
--model_type base \
--datasets all \
--ignore_datasets openai/gsm8k.main ChilleD/SVAMP \
--device cuda \
--evaluator.batch_size 32 \
--evaluator.max_new_tokens 32 \

python evaluate.py \
--model_path Qwen/Qwen3-0.6B \
--model_type base \
--datasets openai/gsm8k.main ChilleD/SVAMP \
--device cuda \
--evaluator.batch_size 32 \
--evaluator.max_new_tokens 256 \
