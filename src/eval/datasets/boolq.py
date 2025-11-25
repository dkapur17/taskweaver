from datasets import load_dataset
from ..task import Task
from ..eval_configs import MultipleChoiceConfig


def create_boolq_task(split: str = 'validation[:5%]', is_chat_task: bool = True) -> Task:
    """
    Create BoolQ (Boolean Questions) task.

    BoolQ is a question answering dataset for yes/no questions containing 15,942 examples.
    These questions are naturally occurring - they are generated in unprompted and unconstrained
    settings. Each example is a triplet of (question, passage, answer), with the title of the
    page as optional additional context.

    Args:
        split: Dataset split (default: 'validation[:5%]')
               Note: BoolQ only has 'train' and 'validation' splits, no 'test' split
        is_chat_task: Whether to format for chat models (default: True)

    Returns:
        Task object configured for BoolQ evaluation
    """
    # Load dataset - BoolQ has 'train' and 'validation' splits
    dataset = load_dataset('google/boolq', split=split)

    # BoolQ has boolean answers (True/False), we'll convert to Yes/No
    def process_boolq(example):
        # Convert boolean to string for consistency
        example['answer_str'] = 'Yes' if example['answer'] else 'No'
        return example

    dataset = dataset.map(process_boolq)

    # Configure as binary classification (Yes/No)
    eval_config = MultipleChoiceConfig(choices=['Yes', 'No'])

    user_template = """Passage: {passage}\nQuestion: {question}\nAnswer with 'Yes' or 'No'."""

    assistant_template = "{answer_str}"

    system_prompt = "You are answering yes/no questions based on the given passage. Answer with only 'Yes' or 'No'." if is_chat_task else None

    return Task(
        task_name='boolq',
        dataset=dataset,
        user_template=user_template,
        assistant_template=assistant_template,
        system_prompt=system_prompt,
        eval_config=eval_config,
        is_chat_task=is_chat_task
    )
