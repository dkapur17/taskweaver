
from generate_weights import LORA_Finetune

def lora_ai2_arc_finetune(model_name, chat: bool = False):
    lora_finetuner = LORA_Finetune(
        model_name=model_name,
        dataset_name="allenai/ai2_arc",
        dataset_subset="ARC-Easy",
    )
    def process_sample(example):
        q = example["question"]
        labels = example["choices"]["label"]
        texts = example["choices"]["text"]
        mc_block = "\n".join([f"{label}. {txt}" for label, txt in zip(labels, texts)])
        answer = example["answerKey"]
        if chat:
            text = {
            "messages": [
                {
                    "role": "system",
                    "content": "Answer the multiple-choice question based on the provided options."
                },
                {
                    "role": "user",
                    "content": f"Question: {q}\n{mc_block}\n\nAnswer: "
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        }
        else:
            text = (
                f"Answer the multiple-choice question based on the provided options.\n\n"
                f"Question: {q}\n"
                f"{mc_block}\n\n"
                f"Answer: {answer}"
            )
        return {"text": text}
    
    lora_finetuner.load_model_and_tokenizer()
    lora_finetuner.load_dataset(formatting_func=process_sample)
    lora_finetuner.train()

def lora_openai_gsm8k(model_name, chat: bool = False):
    lora_finetuner = LORA_Finetune(
        model_name=model_name,
        dataset_name="openai/gsm8k",
        dataset_subset="main",
    )
    
    def process_sample(example):
        question = example["question"]
        answer = example["answer"]
        if chat:
            text = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Solve the given math problem by thinking step by step."
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nAnswer: "
                    },
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ]
            }
        else:
            text = (
                f"Solve the given math problem by thinking step by step.\n\n"
                f"Question: {question}\n"
                f"Answer: {answer}"
            )
        return {"text": text}
    lora_finetuner.load_model_and_tokenizer()
    lora_finetuner.load_dataset(formatting_func=process_sample)
    lora_finetuner.train()

def lora_hella_swag(model_name, chat: bool = False):
    lora_finetuner = LORA_Finetune(
        model_name=model_name,
        dataset_name="Rowan/hellaswag",
        dataset_subset="train",
    )
    
    def process_sample(example):
        context = example["ctx"]
        activity = example["activity_label"]
        endings = example["endings"]
        labels = ["0", "1", "2", "3"]
        mc_block = "\n".join([f"{label}. {txt}" for label, txt in zip(labels, endings)])
        label = example["label"]
        if chat:
            text = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Choose the most plausible continuation for the given context."
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\nOptions:\n{mc_block}\n\nAnswer: "
                    },
                    {
                        "role": "assistant",
                        "content": f"{label}."
                    }
                ]
            }
        else:
            text = (
                f"Choose the most plausible continuation for the given context.\n\n"
                f"Context: {context}\n"
                f"Options: {mc_block}\n\n"
                f"Answer: {label}."
            )
        return {"text": text}
    lora_finetuner.load_model_and_tokenizer()
    lora_finetuner.load_dataset(formatting_func=process_sample)
    lora_finetuner.train()

models = ["google/gemma-3-270m", "Qwen/Qwen3-0.6B-Base", "EleutherAI/pythia-70m"]

for model in models:
    lora_ai2_arc_finetune(model)
    lora_openai_gsm8k(model)
    lora_hella_swag(model)


chat_models = ["google/gemma-3-270m-it", "Qwen/Qwen3-0.6B"]

for model in chat_models:
    lora_ai2_arc_finetune(model, chat=True)
    lora_openai_gsm8k(model, chat=True)
    lora_hella_swag(model, chat=True)