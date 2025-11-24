import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

lora_weights_path = "lora_weights/google-gemma-3-270m-lora-finetuned-allenai-ai2_arc"

config = PeftConfig.from_pretrained(lora_weights_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
lora_model = PeftModel.from_pretrained(model, lora_weights_path)
merged_model = lora_model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)


def process_sample(example):
    q = example["question"]
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]

    # Build the multiple-choice block
    mc_block = "\n".join([f"{label}. {txt}" for label, txt in zip(labels, texts)])

    text = (
        f"Question: {q}\n"
        f"{mc_block}\n\n"
        f"Answer: "
    )
    return text

test_prompt = {
    "question": "Which of the following statements best explains why magnets usually stick to a refrigerator door?",
    "choices": {
        "text": [
            "The refrigerator door is smooth.",
            "The refrigerator door contains iron.",
            "The refrigerator door is a good conductor.",
            "The refrigerator door has electric wires in it."
        ],
        "label": [
            "A",
            "B",
            "C",
            "D"
        ]
    }
}

inputs = tokenizer(process_sample(test_prompt), return_tensors="pt").to(merged_model.device)
outputs = merged_model.generate(**inputs, max_new_tokens=10, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Model response:\n", response)
