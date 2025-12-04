"""
Multi-Dataset Creator for TaskWeaver

This module provides utilities to create and combine multiple datasets
for training the TaskWeaver hypernetwork.
"""

from typing import Dict, List, Optional, Callable
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import PreTrainedTokenizer


class DatasetProcessor:
    """
    Base class for dataset-specific processing.

    Each dataset should have its own processor that defines how to
    extract prompts and completions from the dataset examples.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize the dataset processor.

        Args:
            tokenizer: Tokenizer to use for processing
        """
        self.tokenizer = tokenizer

    def process(self, examples: Dict) -> Dict:
        """
        Process a batch of examples from the dataset.

        This method should be overridden by subclasses to implement
        dataset-specific processing logic.

        Args:
            examples: Batch of examples from the dataset

        Returns:
            Dictionary with processed fields
        """
        raise NotImplementedError("Subclasses must implement process method")

    def _prepare_examples(
        self,
        prompts: List[str],
        completions: List[str]
    ) -> Dict:
        """
        Convert prompts and completions into tokenized examples.

        Args:
            prompts: List of prompt strings
            completions: List of completion strings

        Returns:
            Dictionary with tokenized fields
        """
        results = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
            'prompt_length': []
        }

        for prompt, completion in zip(prompts, completions):
            # Tokenize prompt and completion separately
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=True)
            completion_tokens = self.tokenizer(completion, add_special_tokens=False)

            # Concatenate the token IDs
            input_ids = prompt_tokens['input_ids'] + completion_tokens['input_ids']
            attention_mask = prompt_tokens['attention_mask'] + completion_tokens['attention_mask']

            # Create labels: mask prompt, keep completion
            prompt_length = len(prompt_tokens['input_ids'])
            labels = [-100] * prompt_length + completion_tokens['input_ids']

            results['input_ids'].append(input_ids)
            results['attention_mask'].append(attention_mask)
            results['labels'].append(labels)
            results['prompt_length'].append(prompt_length)

        return results


class GSM8KProcessor(DatasetProcessor):
    """Processor for GSM8K dataset."""

    def process(self, examples: Dict) -> Dict:
        """
        Process GSM8K examples.

        Args:
            examples: Batch with 'question' and 'answer' fields

        Returns:
            Processed examples
        """
        return self._prepare_examples(examples['question'], examples['answer'])


class ARCProcessor(DatasetProcessor):
    """Processor for ARC (AI2 Reasoning Challenge) datasets."""

    def process(self, examples: Dict) -> Dict:
        """
        Process ARC examples.

        Args:
            examples: Batch with 'question', 'choices', and 'answerKey' fields

        Returns:
            Processed examples
        """
        prompts = []
        completions = []

        for question, choices, answer_key in zip(
            examples['question'],
            examples['choices'],
            examples['answerKey']
        ):
            # Format choices
            choices_text = "\n".join([
                f"{label}: {text}"
                for label, text in zip(choices['label'], choices['text'])
            ])

            # Create prompt
            prompt = f"Question: {question}\nChoices:\n{choices_text}\nAnswer:"

            # Get the answer text
            answer_idx = choices['label'].index(answer_key)
            answer_text = choices['text'][answer_idx]
            completion = f" {answer_key}: {answer_text}"

            prompts.append(prompt)
            completions.append(completion)

        return self._prepare_examples(prompts, completions)


class BoolQProcessor(DatasetProcessor):
    """Processor for BoolQ dataset."""

    def process(self, examples: Dict) -> Dict:
        """
        Process BoolQ examples.

        Args:
            examples: Batch with 'passage', 'question', and 'answer' fields

        Returns:
            Processed examples
        """
        prompts = []
        completions = []

        for passage, question, answer in zip(
            examples['passage'], examples['question'], examples['answer']
        ):
            prompt = (
                "Instruction: Use the content in the passage to answer the question with either true or false only\n"
                f"Passage: {passage}\nQuestion: {question}\nAnswer: "
            )
            completion = str(answer).lower()
            prompts.append(prompt)
            completions.append(completion)

        return self._prepare_examples(prompts, completions)


class SNLIProcessor(DatasetProcessor):
    """Processor for SNLI dataset."""

    LABEL_MAP = {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    }

    def process(self, examples: Dict) -> Dict:
        """
        Process SNLI examples.

        Args:
            examples: Batch with 'premise', 'hypothesis', and 'label' fields

        Returns:
            Processed examples
        """
        prompts = []
        completions = []

        for premise, hypothesis, label in zip(
            examples['premise'], examples['hypothesis'], examples['label']
        ):
            # Skip unlabeled examples
            if label == -1:
                continue
            prompt = (
                "Instruction: Given a premise and a hypothesis, determine the relationship between them. "
                "Respond with 'entailment' if the hypothesis follows from the premise, 'contradiction' if the hypothesis contradicts the premise, "
                "or 'neutral' if the relationship is undetermined.\n"
                f"Premise: {premise}\nHypothesis: {hypothesis}\nRelationship: "
            )
            completion = self.LABEL_MAP.get(label, '')
            prompts.append(prompt)
            completions.append(completion)

        return self._prepare_examples(prompts, completions)


class WinograndeProcessor(DatasetProcessor):
    """Processor for Winogrande dataset."""

    def process(self, examples: Dict) -> Dict:
        """
        Process Winogrande examples.

        Args:
            examples: Batch with 'sentence', 'option1', 'option2', 'answer' fields

        Returns:
            Processed examples
        """
        prompts = []
        completions = []

        for sentence, option1, option2, answer in zip(
            examples['sentence'], examples['option1'], examples['option2'], examples['answer']
        ):
            question = f"{sentence}\n1. {option1}\n2. {option2}"
            prompt = (
                "Instruction: Fill in the blank with the correct option. Respond only with 1 or 2\n"
                f"{question}\nAnswer: "
            )
            completion = str(answer)
            prompts.append(prompt)
            completions.append(completion)

        return self._prepare_examples(prompts, completions)


class OpenBookQAProcessor(DatasetProcessor):
    """Processor for OpenBookQA dataset."""

    def process(self, examples: Dict) -> Dict:
        """
        Process OpenBookQA examples.

        Args:
            examples: Batch with 'question_stem', 'choices', and 'answerKey' fields

        Returns:
            Processed examples
        """
        prompts = []
        completions = []

        for stem, choices, answer_key in zip(
            examples['question_stem'], examples['choices'], examples['answerKey']
        ):
            choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choices['label'], choices['text'])])
            prompt = (
                "Instruction: Choose the most reasonable answer for the question from the given options. Respond only with A, B, C or D\n"
                f"{stem}\n{choices_text}\nAnswer: "
            )
            completion = answer_key
            prompts.append(prompt)
            completions.append(completion)

        return self._prepare_examples(prompts, completions)


class HellaSwagProcessor(DatasetProcessor):
    """Processor for HellaSwag dataset."""

    def process(self, examples: Dict) -> Dict:
        """
        Process HellaSwag examples.

        Args:
            examples: Batch with 'ctx', 'activity_label', 'endings', and 'label' fields

        Returns:
            Processed examples
        """
        prompts = []
        completions = []

        for ctx, activity_label, endings, label in zip(
            examples['ctx'], examples.get('activity_label', [None]*len(examples['ctx'])), examples['endings'], examples['label']
        ):
            context = f"{activity_label}: {ctx}" if activity_label else ctx
            options_text = "\n".join([f"{i}. {ending}" for i, ending in enumerate(endings)])
            prompt = (
                "Instruction: Choose the most reasonable continuation from the given options. Respond only with 0, 1, 2 or 3\n"
                f"{context}\n{options_text}\nAnswer: "
            )
            completion = str(label)
            prompts.append(prompt)
            completions.append(completion)

        return self._prepare_examples(prompts, completions)


class MultiDatasetCreator:
    """
    Creator for combining multiple datasets into a single training dataset.

    This class allows you to register dataset processors and combine
    multiple datasets into a single dataset for training.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize the multi-dataset creator.

        Args:
            tokenizer: Tokenizer to use for all datasets
        """
        self.tokenizer = tokenizer
        self.processors: Dict[str, DatasetProcessor] = {}

    def register_processor(
        self,
        name: str,
        processor: DatasetProcessor
    ) -> None:
        """
        Register a dataset processor.

        Args:
            name: Unique name for the dataset
            processor: Processor instance for the dataset
        """
        self.processors[name] = processor

    def load_and_process_dataset(
        self,
        name: str,
        dataset_path: str,
        dataset_config: Optional[str] = None,
        split: str = 'train',
        num_samples: Optional[int] = None
    ) -> Dataset:
        """
        Load and process a single dataset.

        Args:
            name: Name of the registered processor to use
            dataset_path: Path or name of the dataset on HuggingFace
            dataset_config: Dataset configuration name (optional)
            split: Dataset split to load (default: 'train')
            num_samples: Number of samples to load (optional, loads all if None)

        Returns:
            Processed dataset

        Raises:
            ValueError: If processor for the given name is not registered
        """
        if name not in self.processors:
            raise ValueError(
                f"Processor '{name}' not registered. "
                f"Available processors: {list(self.processors.keys())}"
            )

        # Load dataset
        if dataset_config:
            dataset = load_dataset(dataset_path, dataset_config, split=split)
        else:
            dataset = load_dataset(dataset_path, split=split)

        # Limit samples if specified
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        # Process dataset
        processor = self.processors[name]
        processed_dataset = dataset.map(
            processor.process,
            batched=True,
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
            desc=f'Processing {name}'
        )

        return processed_dataset

    def create_combined_dataset(
        self,
        dataset_configs: List[Dict]
    ) -> Dataset:
        """
        Create a combined dataset from multiple sources.

        Args:
            dataset_configs: List of dataset configuration dictionaries.
                Each dict should contain:
                - name: Name of the registered processor
                - dataset_path: Path to the dataset
                - dataset_config: Dataset configuration (optional)
                - split: Dataset split (optional, default: 'train')
                - num_samples: Number of samples to include (optional)

        Returns:
            Combined dataset

        Example:
            >>> creator = MultiDatasetCreator(tokenizer)
            >>> creator.register_processor('gsm8k', GSM8KProcessor(tokenizer))
            >>> creator.register_processor('arc', ARCProcessor(tokenizer))
            >>> dataset = creator.create_combined_dataset([
            ...     {
            ...         'name': 'gsm8k',
            ...         'dataset_path': 'openai/gsm8k',
            ...         'dataset_config': 'main',
            ...         'num_samples': 1000
            ...     },
            ...     {
            ...         'name': 'arc',
            ...         'dataset_path': 'allenai/ai2_arc',
            ...         'dataset_config': 'ARC-Easy',
            ...         'num_samples': 500
            ...     }
            ... ])
        """
        datasets = []

        for config in dataset_configs:
            dataset = self.load_and_process_dataset(
                name=config['name'],
                dataset_path=config['dataset_path'],
                dataset_config=config.get('dataset_config'),
                split=config.get('split', 'train'),
                num_samples=config.get('num_samples')
            )
            datasets.append(dataset)

        # Combine all datasets
        if len(datasets) == 1:
            return datasets[0]
        else:
            return concatenate_datasets(datasets)


def create_dataset(
    tokenizer: PreTrainedTokenizer,
    dataset_configs: List[Dict]
) -> Dataset:
    """
    Convenience function to create a combined dataset.

    Args:
        tokenizer: Tokenizer to use
        dataset_configs: List of dataset configurations

    Returns:
        Combined and processed dataset

    Example:
        >>> dataset = create_dataset(
        ...     tokenizer,
        ...     [
        ...         {
        ...             'name': 'gsm8k',
        ...             'dataset_path': 'openai/gsm8k',
        ...             'dataset_config': 'main'
        ...         }
        ...     ]
        ... )
    """
    creator = MultiDatasetCreator(tokenizer)

    # Register default processors
    creator.register_processor('gsm8k', GSM8KProcessor(tokenizer))
    creator.register_processor('arc-easy', ARCProcessor(tokenizer))
    creator.register_processor('arc-challenge', ARCProcessor(tokenizer))
    creator.register_processor('boolq', BoolQProcessor(tokenizer))
    creator.register_processor('snli', SNLIProcessor(tokenizer))
    creator.register_processor('winogrande', WinograndeProcessor(tokenizer))
    creator.register_processor('openbookqa', OpenBookQAProcessor(tokenizer))
    creator.register_processor('hellaswag', HellaSwagProcessor(tokenizer))

    return creator.create_combined_dataset(dataset_configs)
