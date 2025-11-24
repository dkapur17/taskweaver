from datasets import load_dataset
from ..task import Task
from ..eval_configs import NumericConfig


def create_gsm8k_task(split: str = 'test[:5%]', is_chat_task: bool = True) -> Task:
    """
    Create GSM8K math reasoning task.
    
    Args:
        split: Dataset split (default: 'test[:5%]')
        is_chat_task: Whether to format for chat models (default: True)
    
    Returns:
        Task object configured for GSM8K evaluation
    """
    dataset = load_dataset('openai/gsm8k', 'main', split=split)
    eval_config = NumericConfig(tolerance=1e-1)
    
    user_template = "{question}"
    assistant_template = "{answer}"
    system_prompt = "Solve the given math problem by thinking step by step." if is_chat_task else None
    
    return Task(
        task_name='gsm8k',
        dataset=dataset,
        user_template=user_template,
        assistant_template=assistant_template,
        system_prompt=system_prompt,
        eval_config=eval_config,
        is_chat_task=is_chat_task
    )
