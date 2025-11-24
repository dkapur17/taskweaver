from datasets import load_dataset
from ..task import Task
from ..eval_configs import MultipleChoiceConfig


def create_snli_task(split: str = 'test[:5%]', is_chat_task: bool = True) -> Task:
    """
    Create Stanford NLI entailment task.
    
    Args:
        split: Dataset split (default: 'test[:5%]')
        is_chat_task: Whether to format for chat models (default: True)
    
    Returns:
        Task object configured for SNLI evaluation
    """
    dataset = load_dataset('stanfordnlp/snli', split=split)
    
    # Filter out examples with label -1 (no gold label)
    dataset = dataset.filter(lambda x: x['label'] != -1)
    
    eval_config = MultipleChoiceConfig(choices=['0', '1', '2'])
    
    user_template = "Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail, contradict, or is neutral to the hypothesis? Answer with 0 (entailment), 1 (neutral), or 2 (contradiction)."
    assistant_template = "{label}"
    system_prompt = "You are evaluating logical relationships between statements." if is_chat_task else None
    
    return Task(
        task_name='snli',
        dataset=dataset,
        user_template=user_template,
        assistant_template=assistant_template,
        system_prompt=system_prompt,
        eval_config=eval_config,
        is_chat_task=is_chat_task
    )
