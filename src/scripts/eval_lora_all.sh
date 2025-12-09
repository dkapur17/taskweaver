#!/bin/bash

echo "Evaluating Pythia 70M's LoRA adapters"

python evaluate.py \
--model_path _models/lora/EleutherAI_pythia-70m/allenai_ai2_arc.ARC-Challenge \
--model_type lora \
--datasets allenai/ai2_arc.ARC-Challenge \
--device cuda \
--evaluator.batch_size 128 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/EleutherAI_pythia-70m/allenai_ai2_arc.ARC-Easy \
--model_type lora \
--datasets allenai/ai2_arc.ARC-Easy \
--device cuda \
--evaluator.batch_size 128 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/EleutherAI_pythia-70m/allenai_openbookqa.main \
--model_type lora \
--datasets allenai/openbookqa.main \
--device cuda \
--evaluator.batch_size 128 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/EleutherAI_pythia-70m/allenai_winogrande.winogrande_m \
--model_type lora \
--datasets allenai/winogrande.winogrande_m \
--device cuda \
--evaluator.batch_size 128 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/EleutherAI_pythia-70m/google_boolq \
--model_type lora \
--datasets google/boolq \
--device cuda \
--evaluator.batch_size 128 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/EleutherAI_pythia-70m/Rowan_hellaswag \
--model_type lora \
--datasets Rowan/hellaswag \
--device cuda \
--evaluator.batch_size 128 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/EleutherAI_pythia-70m/stanfordnlp_snli \
--model_type lora \
--datasets stanfordnlp/snli \
--device cuda \
--evaluator.batch_size 128 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/EleutherAI_pythia-70m/openai_gsm8k.main \
--model_type lora \
--datasets openai/gsm8k.main \
--device cuda \
--evaluator.batch_size 128 \
--evaluator.max_new_tokens 256 

python evaluate.py \
--model_path _models/lora/EleutherAI_pythia-70m/mix_8 \
--model_type lora \
--datasets all \
--ignore_datasets openai/gsm8k.main ChilleD/SVAMP \
--device cuda \
--evaluator.batch_size 128 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/EleutherAI_pythia-70m/mix_8 \
--model_type lora \
--datasets openai/gsm8k.main ChilleD/SVAMP \
--device cuda \
--evaluator.batch_size 128 \
--evaluator.max_new_tokens 256 

################################################################################


echo "Evaluating Gemma3 270M's LoRA adapters"

python evaluate.py \
--model_path _models/lora/google_gemma-3-270m-it/allenai_ai2_arc.ARC-Challenge \
--model_type lora \
--datasets allenai/ai2_arc.ARC-Challenge \
--device cuda \
--evaluator.batch_size 64 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/google_gemma-3-270m-it/allenai_ai2_arc.ARC-Easy \
--model_type lora \
--datasets allenai/ai2_arc.ARC-Easy \
--device cuda \
--evaluator.batch_size 64 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/google_gemma-3-270m-it/allenai_openbookqa.main \
--model_type lora \
--datasets allenai/openbookqa.main \
--device cuda \
--evaluator.batch_size 64 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/google_gemma-3-270m-it/allenai_winogrande.winogrande_m \
--model_type lora \
--datasets allenai/winogrande.winogrande_m \
--device cuda \
--evaluator.batch_size 64 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/google_gemma-3-270m-it/google_boolq \
--model_type lora \
--datasets google/boolq \
--device cuda \
--evaluator.batch_size 64 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/google_gemma-3-270m-it/Rowan_hellaswag \
--model_type lora \
--datasets Rowan/hellaswag \
--device cuda \
--evaluator.batch_size 64 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/google_gemma-3-270m-it/stanfordnlp_snli \
--model_type lora \
--datasets stanfordnlp/snli \
--device cuda \
--evaluator.batch_size 64 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/google_gemma-3-270m-it/openai_gsm8k.main \
--model_type lora \
--datasets openai/gsm8k.main \
--device cuda \
--evaluator.batch_size 64 \
--evaluator.max_new_tokens 256 

python evaluate.py \
--model_path _models/lora/google_gemma-3-270m-it/mix_8 \
--model_type lora \
--datasets all \
--ignore_datasets openai/gsm8k.main ChilleD/SVAMP \
--device cuda \
--evaluator.batch_size 64 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/google_gemma-3-270m-it/mix_8 \
--model_type lora \
--datasets openai/gsm8k.main ChilleD/SVAMP \
--device cuda \
--evaluator.batch_size 64 \
--evaluator.max_new_tokens 256 

################################################################################

echo "Evaluating Qwen3 0.6B's LoRA adapters"

python evaluate.py \
--model_path _models/lora/Qwen_Qwen3-0.6B/allenai_ai2_arc.ARC-Challenge \
--model_type lora \
--datasets allenai/ai2_arc.ARC-Challenge \
--device cuda \
--evaluator.batch_size 32 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/Qwen_Qwen3-0.6B/allenai_ai2_arc.ARC-Easy \
--model_type lora \
--datasets allenai/ai2_arc.ARC-Easy \
--device cuda \
--evaluator.batch_size 32 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/Qwen_Qwen3-0.6B/allenai_openbookqa.main \
--model_type lora \
--datasets allenai/openbookqa.main \
--device cuda \
--evaluator.batch_size 32 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/Qwen_Qwen3-0.6B/allenai_winogrande.winogrande_m \
--model_type lora \
--datasets allenai/winogrande.winogrande_m \
--device cuda \
--evaluator.batch_size 32 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/Qwen_Qwen3-0.6B/google_boolq \
--model_type lora \
--datasets google/boolq \
--device cuda \
--evaluator.batch_size 32 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/Qwen_Qwen3-0.6B/Rowan_hellaswag \
--model_type lora \
--datasets Rowan/hellaswag \
--device cuda \
--evaluator.batch_size 32 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/Qwen_Qwen3-0.6B/stanfordnlp_snli \
--model_type lora \
--datasets stanfordnlp/snli \
--device cuda \
--evaluator.batch_size 32 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/Qwen_Qwen3-0.6B/openai_gsm8k.main \
--model_type lora \
--datasets openai/gsm8k.main \
--device cuda \
--evaluator.batch_size 32 \
--evaluator.max_new_tokens 256 

python evaluate.py \
--model_path _models/lora/Qwen_Qwen3-0.6B/mix_8 \
--model_type lora \
--datasets all \
--ignore_datasets openai/gsm8k.main ChilleD/SVAMP \
--device cuda \
--evaluator.batch_size 32 \
--evaluator.max_new_tokens 32 

python evaluate.py \
--model_path _models/lora/Qwen_Qwen3-0.6B/mix_8 \
--model_type lora \
--datasets openai/gsm8k.main ChilleD/SVAMP \
--device cuda \
--evaluator.batch_size 32 \
--evaluator.max_new_tokens 256 
