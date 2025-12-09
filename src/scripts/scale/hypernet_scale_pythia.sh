# Ablation 1: Model scale

python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--hypernet.hidden_dim 512 \
--hypernet.lora_rank 2 \
--hypernet.lora_alpha 8 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 8 \
--train.gradient_accumulation_steps 2 \
--hypernet.target_modules query_key_value
# 3.95% trainable parameters

python train_hypernet.py \
--model google/gemma-3-270m-it \
--datasets all \
--hypernet.hidden_dim 512 \
--hypernet.lora_rank 2 \
--hypernet.lora_alpha 8 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 4 \
--train.gradient_accumulation_steps 4 \
--hypernet.target_modules q_proj v_proj
# 1.29% trainable parameters


python train_hypernet.py \
--model Qwen/Qwen3-0.6B \
--datasets all \
--hypernet.hidden_dim 512 \
--hypernet.lora_rank 2 \
--hypernet.lora_alpha 8 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 2 \
--train.gradient_accumulation_steps 8 \
--hypernet.target_modules q_proj v_proj
# 1.05% trainable parameters

# Ablation 2: Hypernet Hidden Dimension

echo "Ablation 2: Hypernet Hidden Dimension"

echo "D=128"
python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--hypernet.hidden_dim 128 \
--hypernet.lora_rank 2 \
--hypernet.lora_alpha 8 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 8 \
--train.gradient_accumulation_steps 2 \
--hypernet.target_modules query_key_value

echo "D=256"
python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--hypernet.hidden_dim 256 \
--hypernet.lora_rank 2 \
--hypernet.lora_alpha 8 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 8 \
--train.gradient_accumulation_steps 2 \
--hypernet.target_modules query_key_value

echo "D=1024"
python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--hypernet.hidden_dim 1024 \
--hypernet.lora_rank 2 \
--hypernet.lora_alpha 8 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 8 \
--train.gradient_accumulation_steps 2 \
--hypernet.target_modules query_key_value

echo "D=2048"
python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--hypernet.hidden_dim 2048 \
--hypernet.lora_rank 2 \
--hypernet.lora_alpha 8 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 8 \
--train.gradient_accumulation_steps 2 \
--hypernet.target_modules query_key_value

# Ablation 3: LoRA Rank
echo "Ablation 3: LoRA Rank"

echo "r=1"
python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--hypernet.hidden_dim 512 \
--hypernet.lora_rank 1 \
--hypernet.lora_alpha 4 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 8 \
--train.gradient_accumulation_steps 2 \
--hypernet.target_modules query_key_value

echo "r=4"
python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--hypernet.hidden_dim 512 \
--hypernet.lora_rank 4 \
--hypernet.lora_alpha 16 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 8 \
--train.gradient_accumulation_steps 2 \
--hypernet.target_modules query_key_value

echo "r=8"
python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--hypernet.hidden_dim 512 \
--hypernet.lora_rank 8 \
--hypernet.lora_alpha 32 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 8 \
--train.gradient_accumulation_steps 2 \
--hypernet.target_modules query_key_value

echo "r=16"
python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--hypernet.hidden_dim 512 \
--hypernet.lora_rank 16 \
--hypernet.lora_alpha 64 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 8 \
--train.gradient_accumulation_steps 2 \
--hypernet.target_modules query_key_value

echo "r=32"
python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--hypernet.hidden_dim 512 \
--hypernet.lora_rank 32 \
--hypernet.lora_alpha 128 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 8 \
--train.gradient_accumulation_steps 2 \
--hypernet.target_modules query_key_value

# Ablation 4: LoRA Alpha
echo "Abalation 4: LoRA Alpha"

echo "a=0.5x"
python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--hypernet.hidden_dim 512 \
--hypernet.lora_rank 2 \
--hypernet.lora_alpha 1 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 8 \
--train.gradient_accumulation_steps 2 \
--hypernet.target_modules query_key_value

echo "a=1x"
python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--hypernet.hidden_dim 512 \
--hypernet.lora_rank 2 \
--hypernet.lora_alpha 2 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 8 \
--train.gradient_accumulation_steps 2 \
--hypernet.target_modules query_key_value

echo "a=2x"
python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--hypernet.hidden_dim 512 \
--hypernet.lora_rank 2 \
--hypernet.lora_alpha 4 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 8 \
--train.gradient_accumulation_steps 2 \
--hypernet.target_modules query_key_value

echo "a=8x"
python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--hypernet.hidden_dim 512 \
--hypernet.lora_rank 2 \
--hypernet.lora_alpha 16 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 8 \
--train.gradient_accumulation_steps 2 \
--hypernet.target_modules query_key_value

echo "a=16x"
python train_hypernet.py \
--model EleutherAI/pythia-70m \
--datasets all \
--hypernet.hidden_dim 512 \
--hypernet.lora_rank 2 \
--hypernet.lora_alpha 32 \
--train.num_train_epochs 1 \
--train.per_device_train_batch_size 8 \
--train.gradient_accumulation_steps 2 \
--hypernet.target_modules query_key_value

