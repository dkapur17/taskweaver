#! /bin/bash

train_gemma_model() {
    python train.py \
        --lora_rank=16 \
        --lora_alpha=32 \
        --lora_dropout=0.05 \
        --gradient_accumulation_steps=2 \
        --learning_rate=1e-4 \
        --lora_target_layers q_proj k_proj v_proj o_proj \
        --model_name google/gemma-7b &
}

train_pythia_model() {
    python train.py \
        --lora_rank=4 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --gradient_accumulation_steps=2 \
        --learning_rate=5e-5 \
        --model_name EleutherAI/pythia-70m-deduped &
}

# train_gemma_model
train_pythia_model