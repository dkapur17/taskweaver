from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset, interleave_datasets
from typing import List, Literal, TypedDict, Dict, Type, Optional, Tuple, Union, Callable

class Message(TypedDict):
    role: Literal['system', 'user', 'assistant']
    content: str

class ChatOutput(TypedDict):
    messages: List[Message]

class NonChatOutput(TypedDict):
    prompt: List[str]
    completion: List[str]

class DatasetConfig(ABC):

    system_message: str = ''
    train_split: str = ''
    test_split: str = ''

    _registry: Dict[Tuple[str, Optional[str]], Type['DatasetConfig']] = {}

    def __new__(cls):
        raise TypeError(f"{cls.__name__} should not be instantiated")
    
    @classmethod
    def register(cls, dataset_path: str, dataset_name: Optional[str] = None):
        """Decorator to register a config class with its HF dataset path."""
        def decorator(config_cls: Type['DatasetConfig']):
            cls._registry[(dataset_path, dataset_name)] = config_cls
            config_cls.dataset_path = dataset_path  # Store path on the class too
            config_cls.dataset_name = dataset_name
            return config_cls
        return decorator
    
    @classmethod
    def from_dataset_path(cls, dataset_path: str, dataset_name: Optional[str] = None) -> Type['DatasetConfig']:
        """Factory method to get config class from HF dataset path."""
        if (dataset_path, dataset_name) not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown dataset: {dataset_path}. Available: {available}")
        return cls._registry[(dataset_path, dataset_name)]
    
    @classmethod
    def list_available(cls) -> List[tuple[str, Optional[str]]]:
        """List all registered dataset (path, name) pairs."""
        return list(cls._registry.keys())

    @staticmethod 
    @abstractmethod
    def chat_processor(batch: Dataset) -> ChatOutput:
        ...
    
    @staticmethod 
    @abstractmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:
        ...
    
    @classmethod
    def get_processor(cls, is_chat:bool) -> Union[Callable[[Dataset], ChatOutput], Callable[[Dataset], NonChatOutput]]:
        if is_chat:
            return cls.chat_processor
        return cls.non_chat_processor
    
    @classmethod
    def _build_chat_prompt_completion(cls, user_content:str, assistant_content:str) -> Tuple[List[Message], List[Message]]:
        prompt = [
            {'role': 'system', 'content': cls.system_message},
            {'role': 'user', 'content': user_content},
        ]
        completion = [{'role': 'assistant', 'content': assistant_content}]
        return prompt, completion
        
    @classmethod
    def _build_text_prompt(cls, question:str) -> str:
        return f"Instruction: {cls.system_message}\nQuestion: {question}\nAnswer:"
    
    @classmethod
    def load_and_process(cls, is_chat: bool, split: str = 'train', enable_thinking: Optional[bool] = None) -> Dataset:
        """Load the HF dataset and process it."""

        # Resolve split
        if split.startswith('train'):
            split = split.replace('train', cls.train_split)
        elif split.startswith('test'):
            split = split.replace('test', cls.test_split)
        else:
            raise AssertionError(f'Only train and test splits supported, got {split}')

        dataset = load_dataset(cls.dataset_path, cls.dataset_name, split=split)
        processor = cls.get_processor(is_chat)
        
        dataset = dataset.map(processor, batched=True, remove_columns=dataset.column_names)
        assert set(dataset.column_names) == set(['prompt', 'completion']), f"Only prompt completion training supported. Modify {cls.id()} processor to return prompt and competion. It has columns: {dataset.column_names}"
        
        # Injecting thinking argument if needed
        if enable_thinking is not None:
            dataset = dataset.map(lambda example: {**example, 'chat_template_kwargs': {'enable_thinking': enable_thinking}})
        
        return dataset

    @classmethod
    def id(cls) -> str:
        if cls.dataset_name is not None:
            return f"{cls.dataset_path}.{cls.dataset_name}"
        return cls.dataset_path
    
    @classmethod
    def to_eval_format(cls, processed_data: Dataset, is_chat: bool) -> Dict[str, List]:
        """Convert processor output to eval format (X, y_true, y_gt)."""
        if is_chat:
            # Chat processors return prompt (list of messages) and completion (list of messages)
            # Extract the prompt messages (system + user) and completion content (assistant)
            X = processed_data['prompt']  # Already list of message lists
            y = []
            for completion_msgs in processed_data['completion']:
                # Extract assistant content from completion messages
                if isinstance(completion_msgs, list) and len(completion_msgs) > 0:
                    y.append(completion_msgs[0]['content'])
                else:
                    y.append('')
            return {'X': X, 'y': y}
        else:
            # Non-chat: prompts are X, completions are y
            return {
                'X': processed_data['prompt'],
                'y': processed_data['completion'],
            }
    
    @classmethod
    @abstractmethod
    def get_eval_config(cls):
        """Return the appropriate EvaluationConfig for this dataset."""
        ...
    
    @classmethod
    def create_task(cls, is_chat: bool, split: str = 'test'):
        """Create an evaluation Task for this dataset."""
        from eval.task import Task

        # Load and process dataset
        processed_data = cls.load_and_process(is_chat=is_chat, split=split)
        eval_format = cls.to_eval_format(processed_data, is_chat=is_chat)
        
        # Create dataset with X, y columns
        dataset = Dataset.from_dict(eval_format)
        
        # Get eval config
        eval_config = cls.get_eval_config()
        
        return Task(
            task_name=cls.id(),
            dataset=dataset,
            eval_config=eval_config,
            is_chat_task=is_chat,
            skip_formatting=True
        )
    

############################################
### Per-Dataset Configs
############################################

class ArcConfig(DatasetConfig):
    system_message = "Choose the most reasonable answer for the question from the given options. Repond only with A, B, C or D"
    train_split = 'train'
    test_split = 'test'

    @staticmethod
    def chat_processor(batch: Dataset) -> ChatOutput:

        prompts = []
        completions = []

        for question, choices, answer in zip(batch['question'], batch['choices'], batch['answerKey']):
            question = f"{question}\n" + "\n".join([f"{label}.{text}" for label, text in zip(choices['label'], choices['text'])])
            prompt, completion = ArcConfig._build_chat_prompt_completion(question, answer)
            prompts.append(prompt)
            completions.append(completion)

        return {'prompt': prompts, 'completion': completions}
        
    @staticmethod 
    def non_chat_processor(batch: Dataset) -> NonChatOutput:
        
        prompts = []
        completions = []

        for question, choices, answer in zip(batch['question'], batch['choices'], batch['answerKey']):
            question = f"{question}\n" + "\n".join([f"{label}.{text}" for label, text in zip(choices['label'], choices['text'])])
            prompts.append(ArcEasyConfig._build_text_prompt(question))
            completions.append(f" {answer}")

        return {'prompt': prompts, 'completion': completions}
    
    @classmethod
    def get_eval_config(cls):
        from eval.eval_configs import MultipleChoiceConfig
        # Get all possible choice labels from first batch
        sample = load_dataset(cls.dataset_path, cls.dataset_name, split=f'{cls.test_split}[:10]')
        all_labels = set()
        for example in sample:
            all_labels.update(example['choices']['label'])
        return MultipleChoiceConfig(choices=sorted(all_labels))

@DatasetConfig.register('allenai/ai2_arc', 'ARC-Easy')
class ArcEasyConfig(ArcConfig):
    pass

@DatasetConfig.register('allenai/ai2_arc', 'ARC-Challenge')
class ArcChallengeConfig(ArcConfig):
    pass


@DatasetConfig.register('openai/gsm8k', 'main')
class GSM8KConfig(DatasetConfig):

    system_message = "Analyze the given math problem, reason through it step by step, and provide the final answer in a new line starting with ####, for example: #### 72"
    train_split = 'train'
    test_split = 'test'

    @staticmethod
    def chat_processor(batch: Dataset) -> ChatOutput:

        prompts = []
        completions = []

        for question, answer in zip(batch['question'], batch['answer']):
            prompt, completion = GSM8KConfig._build_chat_prompt_completion(question, answer)
            prompts.append(prompt)
            completions.append(completion)

        return {'prompt': prompts, 'completion': completions}
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:

        prompts = []
        completions = []

        for question, answer in zip(batch['question'], batch['answer']):
            prompts.append(GSM8KConfig._build_text_prompt(question))
            completions.append(f" {answer}")

        return {'prompt': prompts, 'completion': completions}
    
    @classmethod
    def get_eval_config(cls):
        from eval.eval_configs import NumericConfig
        return NumericConfig(tolerance=1e-1)
    

@DatasetConfig.register('google/boolq')
class BoolQConfig(DatasetConfig):

    system_message = "Use the content in the passage to answer the question with either true or false only"
    train_split = 'train'
    test_split = 'validation'

    @staticmethod
    def chat_processor(batch: Dataset) -> ChatOutput:

        prompts = []
        completions = []

        for passage, question, answer in zip(batch['passage'], batch['question'], batch['answer']):
            question = f"Passage: {passage}\n\nNow answer: {question}"
            prompt, completion = BoolQConfig._build_chat_prompt_completion(question, str(answer).lower())
            prompts.append(prompt)
            completions.append(completion)

        return {'prompt': prompts, 'completion': completions}
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:

        prompts, completions = [], []

        for passage, question, answer in zip(batch['passage'], batch['question'], batch['answer']):
            prompt = f"Instruction: {BoolQConfig.system_message}\nPassage: {passage}\nQuestion: {question}\nAnswer:"
            prompts.append(prompt)
            completions.append(f" {str(answer).lower()}")

        return {'prompt': prompts, 'completion': completions}
    
    @classmethod
    def get_eval_config(cls):
        from eval.eval_configs import BooleanConfig
        return BooleanConfig()


@DatasetConfig.register('stanfordnlp/snli')
class SNLIConfig(DatasetConfig):

    system_message = "Given a premise and a hypothesis, determine the relationship between them. Respond with 'entailment' if the hypothesis follows from the premise, 'contradiction' if the hypothesis contradicts the premise, or 'neutral' if the relationship is undetermined."
    train_split = 'train'
    test_split = 'test'
    label_map = {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    }

    @staticmethod
    def chat_processor(batch: Dataset) -> ChatOutput:

        prompts = []
        completions = []

        for premise, hypothesis, label in zip(batch['premise'], batch['hypothesis'], batch['label']):
            if label == -1:
                continue
            question = f"Premise: {premise}\nHypothesis: {hypothesis}"
            prompt, completion = SNLIConfig._build_chat_prompt_completion(question, SNLIConfig.label_map[label])
            prompts.append(prompt)
            completions.append(completion)
        
        return {'prompt': prompts, 'completion': completions}
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:
        prompts, completions = [], []

        for premise, hypothesis, label in zip(batch['premise'], batch['hypothesis'], batch['label']):
            if label == -1:
                continue
            prompt = f"Instruction: {SNLIConfig.system_message}\nPremise: {premise}\nHypothesis: {hypothesis}\nRelationship:"
            prompts.append(prompt)
            completions.append(f" {SNLIConfig.label_map[label]}")

        return {'prompt': prompts, 'completion': completions}
    
    @classmethod
    def get_eval_config(cls):
        from eval.eval_configs import ExactMatchConfig
        return ExactMatchConfig(case_sensitive=False, strip_think_tags=True, extract_last_word=True)


@DatasetConfig.register('allenai/winogrande', 'winogrande_m')
class WinograndeConfig(DatasetConfig):

    system_message = "Fill in the blank with the correct option. Respond only with 1 or 2"
    train_split = 'train'
    test_split = 'validation'

    @staticmethod
    def chat_processor(batch: Dataset) -> ChatOutput:
        
        prompts = []
        completions = []

        for sentence, option1, option2, answer in zip(batch['sentence'], batch['option1'], batch['option2'], batch['answer']):
            question = f"{sentence}\n1. {option1}\n2. {option2}"
            prompt, completion = WinograndeConfig._build_chat_prompt_completion(question, answer)
            prompts.append(prompt)
            completions.append(completion)

        return {'prompt': prompts, 'completion': completions}
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:
        prompts = []
        completions = []

        for sentence, option1, option2, answer in zip(batch['sentence'], batch['option1'], batch['option2'], batch['answer']):
            question = f"{sentence}\n1. {option1}\n2. {option2}"
            prompts.append(WinograndeConfig._build_text_prompt(question))
            completions.append(f" {answer}")

        return {'prompt': prompts, 'completion': completions}
    
    @classmethod
    def get_eval_config(cls):
        from eval.eval_configs import MultipleChoiceConfig
        return MultipleChoiceConfig(choices=['1', '2'])
    

@DatasetConfig.register('allenai/openbookqa', 'main')
class OpenBookQAConfig(DatasetConfig):

    system_message = "Choose the most reasonable answer for the question from the given options. Respond only with A, B, C or D"
    train_split = 'train'
    test_split = 'test'

    @staticmethod
    def chat_processor(batch: Dataset) -> ChatOutput:
        
        prompts = []
        completions = []

        for question_stem, choices, answer_key in zip(batch['question_stem'], batch['choices'], batch['answerKey']):
            question = f"{question_stem}\n" + "\n".join([f"{label}. {text}" for label, text in zip(choices['label'], choices['text'])])
            prompt, completion = OpenBookQAConfig._build_chat_prompt_completion(question, answer_key)
            prompts.append(prompt)
            completions.append(completion)

        return {'prompt': prompts, 'completion': completions}
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:

        prompts = []
        completions = []

        for question_stem, choices, answer in zip(batch['question_stem'], batch['choices'], batch['answerKey']):
            question = f"{question_stem}\n" + "\n".join([f"{label}. {text}" for label, text in zip(choices['label'], choices['text'])])
            prompts.append(OpenBookQAConfig._build_text_prompt(question))
            completions.append(f" {answer}")    

        return {'prompt': prompts, 'completion': completions}
    
    @classmethod
    def get_eval_config(cls):
        from eval.eval_configs import MultipleChoiceConfig
        sample = load_dataset(cls.dataset_path, cls.dataset_name, split=f'{cls.test_split}[:10]')
        all_labels = set()
        for example in sample:
            all_labels.update(example['choices']['label'])
        return MultipleChoiceConfig(choices=sorted(all_labels))


@DatasetConfig.register('Rowan/hellaswag')
class HellaSwagConfig(DatasetConfig):

    system_message = "Choose the most reasonable continuation from the given options. Respond only with 0, 1, 2 or 3"
    train_split = 'train'
    test_split = 'validation'

    @staticmethod
    def chat_processor(batch: Dataset) -> ChatOutput:
        
        prompts = []
        completions = []

        for ctx, activity_label, endings, label in zip(batch['ctx'], batch['activity_label'], batch['endings'], batch['label']):
            context = f"{activity_label}: {ctx}" if activity_label else ctx
            question = f"{context}\n" + "\n".join([f"{i}. {ending}" for i, ending in enumerate(endings)])
            prompt, completion = HellaSwagConfig._build_chat_prompt_completion(question, label)
            prompts.append(prompt)
            completions.append(completion)

        return {'prompt': prompts, 'completion': completions}
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:
        
        prompts = []
        completions = []

        for ctx, activity_label, endings, label in zip(batch['ctx'], batch['activity_label'], batch['endings'], batch['label']):
            context = f"{activity_label}: {ctx}" if activity_label else ctx
            question = f"{context}\n" + "\n".join([f"{i}. {ending}" for i, ending in enumerate(endings)])
            prompts.append(HellaSwagConfig._build_text_prompt(question))
            completions.append(f" {label}")

        return {'prompt': prompts, 'completion': completions}
    
    @classmethod
    def get_eval_config(cls):
        from eval.eval_configs import MultipleChoiceConfig
        return MultipleChoiceConfig(choices=['0', '1', '2', '3'])


@DatasetConfig.register('tau/commonsense_qa')
class CommonsenseQAConfig(DatasetConfig):

    system_message = "Answer the commonsense reasoning question by choosing the most appropriate option. Respond only with A, B, C, D, or E"
    train_split = 'train'
    test_split = 'validation'

    @staticmethod
    def chat_processor(batch: Dataset) -> ChatOutput:
        
        prompts = []
        completions = []

        for question, choices, answer_key in zip(batch['question'], batch['choices'], batch['answerKey']):
            full_question = f"{question}\n" + "\n".join([f"{label}. {text}" for label, text in zip(choices['label'], choices['text'])])
            prompt, completion = CommonsenseQAConfig._build_chat_prompt_completion(full_question, answer_key)
            prompts.append(prompt)
            completions.append(completion)

        return {'prompt': prompts, 'completion': completions}
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:
        
        prompts = []
        completions = []

        for question, choices, answer_key in zip(batch['question'], batch['choices'], batch['answerKey']):
            full_question = f"{question}\n" + "\n".join([f"{label}. {text}" for label, text in zip(choices['label'], choices['text'])])
            prompts.append(CommonsenseQAConfig._build_text_prompt(full_question))
            completions.append(f" {answer_key}")

        return {'prompt': prompts, 'completion': completions}
    
    @classmethod
    def get_eval_config(cls):
        from eval.eval_configs import MultipleChoiceConfig
        return MultipleChoiceConfig(choices=['A', 'B', 'C', 'D', 'E'])


@DatasetConfig.register('ChilleD/SVAMP')
class SVAMPConfig(DatasetConfig):

    system_message = "Solve the math word problem step by step and provide the final numerical answer"
    train_split = 'train'
    test_split = 'test'

    @staticmethod
    def chat_processor(batch: Dataset) -> ChatOutput:
        
        prompts = []
        completions = []

        for body, question, answer in zip(batch['Body'], batch['Question'], batch['Answer']):
            full_question = f"{body} {question}"
            prompt, completion = SVAMPConfig._build_chat_prompt_completion(full_question, str(answer))
            prompts.append(prompt)
            completions.append(completion)

        return {'prompt': prompts, 'completion': completions}
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:
        
        prompts = []
        completions = []

        for body, question, answer in zip(batch['Body'], batch['Question'], batch['Answer']):
            full_question = f"{body} {question}"
            prompts.append(SVAMPConfig._build_text_prompt(full_question))
            completions.append(f" {answer}")

        return {'prompt': prompts, 'completion': completions}
    
    @classmethod
    def get_eval_config(cls):
        from eval.eval_configs import NumericConfig
        return NumericConfig(tolerance=1e-1)


class RACEConfig(DatasetConfig):

    system_message = "Read the article and answer the question by choosing the most appropriate option. Respond only with A, B, C, or D"
    train_split = 'train'
    test_split = 'test'

    @staticmethod
    def chat_processor(batch: Dataset) -> ChatOutput:
        
        prompts = []
        completions = []

        for article, question, options, answer in zip(batch['article'], batch['question'], batch['options'], batch['answer']):
            full_question = f"Article: {article}\n\nQuestion: {question}\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            prompt, completion = RACEConfig._build_chat_prompt_completion(full_question, answer)
            prompts.append(prompt)
            completions.append(completion)

        return {'prompt': prompts, 'completion': completions}
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:
        
        prompts = []
        completions = []

        for article, question, options, answer in zip(batch['article'], batch['question'], batch['options'], batch['answer']):
            full_question = f"Article: {article}\n\nQuestion: {question}\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            prompts.append(RACEConfig._build_text_prompt(full_question))
            completions.append(f" {answer}")

        return {'prompt': prompts, 'completion': completions}
    
    @classmethod
    def get_eval_config(cls):
        from eval.eval_configs import MultipleChoiceConfig
        # RACE typically has 4 options (A, B, C, D)
        return MultipleChoiceConfig(choices=['A', 'B', 'C', 'D'])


@DatasetConfig.register('ehovy/race', 'middle')
class RACEMiddleConfig(RACEConfig):
    pass
    
class DatasetMixer(DatasetConfig):
    """Mix of multiple datasets that can be used anywhere a DatasetConfig is expected."""
    
    # Override to allow instantiation (parent raises TypeError)
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)
    
    def __init__(
        self, 
        datasets: Optional[List[Union[Tuple[str, Optional[str]], str, Type[DatasetConfig]]]] = None, # ('openai/gsm8k', 'main), 'openai/gsm8k.main', 'GSM8KConfig' all work
        probabilities: Optional[List[float]] = None,
        seed: Optional[int] = None,
        stopping_strategy: Literal['first_exhausted', 'all_exhausted'] = 'first_exhausted'
    ):
        """
        Args:
            datasets: List of dataset identifiers. Can be:
                     - (dataset_path, dataset_name) tuples
                     - "dataset_path.dataset_name" strings  
                     - DatasetConfig subclasses directly
                     If None, uses all registered datasets.
            probabilities: Sampling probabilities for each dataset. 
                          If None, samples uniformly.
            seed: Random seed for interleaving.
            stopping_strategy: 'first_exhausted' or 'all_exhausted'.
        """
        self._dataset_configs = self._resolve_datasets(datasets)
        self._probabilities = probabilities
        self._seed = seed
        self._stopping_strategy = stopping_strategy
        
        # Set attributes that parent class methods might expect
        self.train_split = 'train'
        self.test_split = 'test'
        self.system_message = ''
    
    def _resolve_datasets(
        self, 
        datasets: Optional[List[Union[Tuple[str, Optional[str]], str, Type[DatasetConfig]]]]
    ) -> List[Type[DatasetConfig]]:
        """Resolve dataset specifications to DatasetConfig classes."""
        if datasets is None:
            return [
                DatasetConfig.from_dataset_path(path, name) 
                for path, name in DatasetConfig.list_available()
            ]
        
        configs = []
        for ds in datasets:
            if isinstance(ds, tuple):
                path = ds[0]
                name = ds[1] if len(ds) > 1 else None
                configs.append(DatasetConfig.from_dataset_path(path, name))
            elif isinstance(ds, str):
                # Parse "path.name" format carefully with paths like "allenai/ai2_arc"
                if '/' in ds:
                    last_segment = ds.split('/')[-1]
                    if '.' in last_segment:
                        idx = ds.rfind('.')
                        path, name = ds[:idx], ds[idx+1:]
                    else:
                        path, name = ds, None
                else:
                    path, name = ds, None
                configs.append(DatasetConfig.from_dataset_path(path, name))
            elif isinstance(ds, type) and issubclass(ds, DatasetConfig):
                configs.append(ds)
            else:
                raise ValueError(f"Unknown dataset specification: {ds}")
        
        return configs
    
    # === Implement abstract methods (required by ABC) ===
    
    @staticmethod
    def chat_processor(batch: Dataset) -> ChatOutput:
        raise NotImplementedError(
            "DatasetMix doesn't have a single chat_processor. Use load_and_process() directly."
        )
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:
        raise NotImplementedError(
            "DatasetMix doesn't have a single non_chat_processor. Use load_and_process() directly."
        )
    
    def get_eval_config(self):
        raise NotImplementedError(
            "DatasetMix contains multiple eval configs. Access individual configs via .dataset_configs"
        )
    
    # === Override parent methods as instance methods ===
    
    def load_and_process(self, is_chat: bool, split: str = 'train') -> Dataset:
        """Load and process all datasets, then interleave them."""
        processed_datasets = []
        valid_configs = []
        
        for config in self._dataset_configs:
            try:
                processed = config.load_and_process(is_chat=is_chat, split=split)
                processed_datasets.append(processed)
                valid_configs.append(config)
            except Exception as e:
                print(f"Warning: Failed to load {config.id()}: {e}")
                continue
        
        if not processed_datasets:
            raise ValueError("No datasets were successfully loaded")
        
        self._dataset_configs = valid_configs
        
        if len(processed_datasets) == 1:
            return processed_datasets[0]
        
        probs = self._probabilities
        if probs is not None and len(probs) != len(processed_datasets):
            print(f"Warning: Adjusting probabilities from {len(probs)} to {len(processed_datasets)} datasets")
            probs = None
        
        return interleave_datasets(
            processed_datasets,
            probabilities=probs,
            seed=self._seed,
            stopping_strategy=self._stopping_strategy
        )
    
    def id(self) -> str:
        """Return identifier for this mix."""
        ids = [c.id().replace("/", "_").replace(".", "_") for c in self._dataset_configs]
        if len(ids) <= 3:
            return "mix_" + "_".join(ids)
        return f"mix_{len(ids)}_datasets"
    
    # === Additional utility methods ===
    
    @property 
    def dataset_configs(self) -> List[Type[DatasetConfig]]:
        """Return the list of dataset configs in this mix."""
        return self._dataset_configs
    
    @property
    def num_tasks(self) -> int:
        return len(self._dataset_configs)
    
    def __len__(self) -> int:
        return len(self._dataset_configs)
    
    def __repr__(self) -> str:
        if len(self._dataset_configs) <= 5:
            return f"DatasetMix({[c.id() for c in self._dataset_configs]})"
        return f"DatasetMix([{self._dataset_configs[0].id()}, ..., {self._dataset_configs[-1].id()}] ({len(self)} datasets))"
    