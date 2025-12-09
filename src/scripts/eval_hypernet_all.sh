#!/bin/bash

echo "Evaluating TaskWeaver with Pythia 70M on all datasets"

python evaluate.py \
--model_path _models/hypernet/EleutherAI_pythia-70m/mix_8_d1024_r2_a8 \
--model_type hypernet \
--datasets all \
--ignore_datasets openai/gsm8k.main ChilleD/SVAMP google/boolq ehovy/race.middle \
--device cuda \
--evaluator.batch_size 64 \
--evaluator.max_new_tokens 32 \

python evaluate.py \
--model_path _models/hypernet/EleutherAI_pythia-70m/mix_8_d1024_r2_a8 \
--model_type hypernet \
--datasets google/boolq ehovy/race.middle \
--device cuda \
--evaluator.batch_size 64 \
--evaluator.max_new_tokens 32 \

python evaluate.py \
--model_path _models/hypernet/EleutherAI_pythia-70m/mix_8_d1024_r2_a8 \
--model_type hypernet \
--datasets openai/gsm8k.main ChilleD/SVAMP \
--device cuda \
--evaluator.batch_size 64 \
--evaluator.max_new_tokens 256 \


echo "Evaluating TaskWeaver with Gemma3 270M on all datasets"

python evaluate.py \
--model_path _models/hypernet/google_gemma-3-270m-it/mix_8_d1024_r2_a8 \
--model_type hypernet \
--datasets all \
--ignore_datasets openai/gsm8k.main ChilleD/SVAMP google/boolq ehovy/race.middle \
--device cuda \
--evaluator.batch_size 32 \
--evaluator.max_new_tokens 32 \

python evaluate.py \
--model_path _models/hypernet/google_gemma-3-270m-it/mix_8_d1024_r2_a8 \
--model_type hypernet \
--datasets google/boolq ehovy/race.middle \
--device cuda \
--evaluator.batch_size 8 \
--evaluator.max_new_tokens 32 \

python evaluate.py \
--model_path _models/hypernet/google_gemma-3-270m-it/mix_8_d1024_r2_a8 \
--model_type hypernet \
--datasets openai/gsm8k.main ChilleD/SVAMP \
--device cuda \
--evaluator.batch_size 32 \
--evaluator.max_new_tokens 256 \


echo "Evaluating TaskWeaver with Qwen3 0.6B on all datasets"

python evaluate.py \
--model_path _models/hypernet/Qwen_Qwen3-0.6B/mix_8_d1024_r2_a8 \
--model_type hypernet \
--datasets all \
--ignore_datasets openai/gsm8k.main ChilleD/SVAMP google/boolq ehovy/race.middle \
--device cuda \
--evaluator.batch_size 16 \
--evaluator.max_new_tokens 32 \

python evaluate.py \
--model_path _models/hypernet/Qwen_Qwen3-0.6B/mix_8_d1024_r2_a8 \
--model_type hypernet \
--datasets google/boolq ehovy/race.middle \
--device cuda \
--evaluator.batch_size 8 \
--evaluator.max_new_tokens 32 \

python evaluate.py \
--model_path _models/hypernet/Qwen_Qwen3-0.6B/mix_8_d1024_r2_a8 \
--model_type hypernet \
--datasets openai/gsm8k.main ChilleD/SVAMP \
--device cuda \
--evaluator.batch_size 16 \
--evaluator.max_new_tokens 256 \

