from datasets import load_dataset
from ..task import Task
from ..eval_configs import ExactMatchConfig


def create_squad_task(split: str = 'validation[:5%]', is_chat_task: bool = True) -> Task:
    """
    Create SQuAD v2 reading comprehension task.
    
    Args:
        split: Dataset split (default: 'validation[:5%]')
        is_chat_task: Whether to format for chat models (default: True)
    
    Returns:
        Task object configured for SQuAD evaluation
    """
    dataset = load_dataset('rajpurkar/squad_v2', split=split)
    
    # Process dataset to extract answer text
    def process_squad(example):
        if len(example['answers']['text']) > 0:
            example['answer_text'] = example['answers']['text'][0]
        else:
            example['answer_text'] = ""  # No answer for unanswerable questions
        return example
    
    dataset = dataset.map(process_squad)
    
    eval_config = ExactMatchConfig(normalize=True, case_sensitive=False)
    
    user_template = "Context: {context}\nQuestion: {question}\nAnswer:"
    assistant_template = "{answer_text}"
    system_prompt = "Answer the question based on the given context. If the question cannot be answered from the context, respond with an empty string." if is_chat_task else None
    
    return Task(
        task_name='squad_v2',
        dataset=dataset,
        user_template=user_template,
        assistant_template=assistant_template,
        system_prompt=system_prompt,
        eval_config=eval_config,
        is_chat_task=is_chat_task
    )
