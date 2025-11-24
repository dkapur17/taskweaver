from datasets import load_dataset
from ..task import Task
from ..eval_configs import MultipleChoiceConfig


def create_arc_easy_task(split: str = 'test[:5%]', is_chat_task: bool = True) -> Task:
    """
    Create ARC-Easy science QA task.
    
    Args:
        split: Dataset split (default: 'test[:5%]')
        is_chat_task: Whether to format for chat models (default: True)
    
    Returns:
        Task object configured for ARC-Easy evaluation
    """
    dataset = load_dataset('allenai/ai2_arc', 'ARC-Easy', split=split)
    
    # Process dataset to format choices
    def process_arc(example):
        labels = example['choices']['label']
        texts = example['choices']['text']
        
        # Build formatted choices string
        choices_str = "\n".join([f"{label}. {text}" for label, text in zip(labels, texts)])
        example['choices_formatted'] = choices_str
        
        return example
    
    dataset = dataset.map(process_arc)
    
    # Get all possible choice labels dynamically
    all_labels = set()
    for example in dataset:
        all_labels.update(example['choices']['label'])
    
    eval_config = MultipleChoiceConfig(choices=sorted(list(all_labels)))
    
    user_template = "Question: {question}\n{choices_formatted}\nAnswer:"
    assistant_template = "{answerKey}"
    system_prompt = "Answer the multiple choice science question by selecting the correct letter." if is_chat_task else None
    
    return Task(
        task_name='arc_easy',
        dataset=dataset,
        user_template=user_template,
        assistant_template=assistant_template,
        system_prompt=system_prompt,
        eval_config=eval_config,
        is_chat_task=is_chat_task
    )
