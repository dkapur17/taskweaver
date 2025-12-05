from datasets import Dataset, concatenate_datasets, interleave_datasets
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset
from typing import List, Literal, TypedDict, Dict, Type, Optional, Tuple, Union

class Message(TypedDict):
    role: Literal['system', 'user', 'assistant']
    content: str

class ChatOutput(TypedDict):
    messages: List[Message]

class NonChatOutput(TypedDict):
    prompt: List[str]
    completion: List[str]
    text: List[str]


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
    def get_processor(cls, is_chat:bool):
        if is_chat:
            return cls.chat_processor
        return cls.non_chat_processor
    
    @classmethod
    def _build_chat(cls, user_content:str, assistant_content:str) -> List[Message]:
        return [
            {'role': 'system', 'content': cls.system_message},
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': assistant_content}
        ]
        
    @classmethod
    def _build_prompt(cls, question:str) -> str:
        return f"Instruction: {cls.system_message}\nQuestion: {question}\nAnswer: "
    
    @classmethod
    def load_and_process(cls, is_chat: bool, split: str = 'train') -> Dataset:
        """Load the HF dataset and process it."""
        dataset = load_dataset(cls.dataset_path, cls.dataset_name, split=split)
        processor = cls.get_processor(is_chat)
        return dataset.map(processor, batched=True, remove_columns=dataset.column_names)
    
    @classmethod
    def id(cls) -> str:
        if cls.dataset_name is not None:
            return f"{cls.dataset_path}.{cls.dataset_name}"
        return cls.dataset_path
    
    @classmethod
    def to_eval_format(cls, processed_data: Dataset, is_chat: bool) -> Dict[str, List]:
        """Convert processor output to eval format (X, y_true, y_gt)."""
        if is_chat:
            # Extract prompts (system + user) and completions (assistant)
            X = []
            y_true = []
            for messages in processed_data['messages']:
                # Combine system and user messages for prompt
                prompt_msgs = [msg for msg in messages if msg['role'] in ['system', 'user']]
                X.append(prompt_msgs)
                # Extract assistant response
                assistant_msg = next((msg for msg in messages if msg['role'] == 'assistant'), None)
                y_true.append(assistant_msg['content'] if assistant_msg else '')
            return {'X': X, 'y_true': y_true, 'y_gt': y_true}
        else:
            # Non-chat: prompts are X, completions are y
            return {
                'X': processed_data['prompt'],
                'y_true': processed_data['completion'],
                'y_gt': processed_data['completion']
            }
    
    @classmethod
    @abstractmethod
    def get_eval_config(cls):
        """Return the appropriate EvaluationConfig for this dataset."""
        ...
    
    @classmethod
    def create_task(cls, split: str, is_chat: bool):
        """Create an evaluation Task for this dataset."""
        from eval.task import Task
        
        # Load and process dataset
        processed_data = cls.load_and_process(is_chat=is_chat, split=split)
        eval_format = cls.to_eval_format(processed_data, is_chat=is_chat)
        
        # Create dataset with X, y columns
        dataset = Dataset.from_dict({
            'X': eval_format['X'],
            'y': eval_format['y_true']
        })
        
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

    

@DatasetConfig.register('allenai/ai2_arc', 'ARC-Easy')
class ArcEasyConfig(DatasetConfig):

    system_message = "Choose the most reasonable answer for the question from the given options. Repond only with A, B, C or D"
    train_split = 'train'
    test_split = 'test'

    @staticmethod
    def chat_processor(batch: Dataset) -> ChatOutput:
        chats = []

        for question, choices, answer in zip(batch['question'], batch['choices'], batch['answerKey']):
            question = f"{question}\n" + "\n".join([f"{label}.{text}" for label, text in zip(choices['label'], choices['text'])])
            chats.append(ArcEasyConfig._build_chat(question, answer))

        return {'messages': chats}
    
    @staticmethod 
    def non_chat_processor(batch: Dataset) -> NonChatOutput:
        
        prompts = []

        for question, choices in zip(batch['question'], batch['choices']):
            question = f"{question}\n" + "\n".join([f"{label}.{text}" for label, text in zip(choices['label'], choices['text'])])
            prompts.append(ArcEasyConfig._build_prompt(question))

        completions = batch['answerKey']
        texts = [p + c for p, c in zip(prompts, completions)]
        return {'prompt': prompts, 'completion': completions, 'text': texts}
    
    @classmethod
    def get_eval_config(cls):
        from eval.eval_configs import MultipleChoiceConfig
        # Get all possible choice labels from first batch
        sample = load_dataset(cls.dataset_path, cls.dataset_name, split=f'{cls.test_split}[:10]')
        all_labels = set()
        for example in sample:
            all_labels.update(example['choices']['label'])
        return MultipleChoiceConfig(choices=sorted(all_labels))
    

@DatasetConfig.register('openai/gsm8k', 'main')
class GSM8KConfig(DatasetConfig):

    system_message = "Analyze the given math problem, reason through it step by step, and provide the final answer in a new line starting with ####, for example: #### 72"
    train_split = 'train'
    test_split = 'test'

    @staticmethod
    def chat_processor(batch: Dataset) -> ChatOutput:

        chats = []

        for question, answer in zip(batch['question'], batch['answer']):
            chats.append(GSM8KConfig._build_chat(question, answer))

        return {'messages': chats}
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:
        prompts = [GSM8KConfig._build_prompt(q) for q in batch['question']]
        completions = batch['answer']
        texts = [p + c for p, c in zip(prompts, completions)]
        return {'prompt': prompts, 'completion': completions, 'text': texts}
    
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
        chats = []

        for passage, question, answer in zip(batch['passage'], batch['question'], batch['answer']):
            question = f"Passage: {passage}\n\nNow answer: {question}"
            chats.append(BoolQConfig._build_chat(question, str(answer).lower()))

        return {'messages': chats}
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:

        prompts, completions = [], []

        for passage, question, answer in zip(batch['passage'], batch['question'], batch['answer']):
            prompt = f"Instruction: {BoolQConfig.system_message}\nPassage: {passage}\nQuestion: {question}\nAnswer: "
            prompts.append(prompt)
            completions.append(str(answer).lower())

        texts = [p + c for p, c in zip(prompts, completions)]
        return {'prompt': prompts, 'completion': completions, 'text': texts}
    
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
        chats = []

        for premise, hypothesis, label in zip(batch['premise'], batch['hypothesis'], batch['label']):
            if label == -1:
                continue
            question = f"Premise: {premise}\nHypothesis: {hypothesis}"
            chats.append(SNLIConfig._build_chat(question, SNLIConfig.label_map[label]))
        
        return {'messages': chats}
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:
        prompts, completions = [], []

        for premise, hypothesis, label in zip(batch['premise'], batch['hypothesis'], batch['label']):
            if label == -1:
                continue
            prompt = f"Instruction: {SNLIConfig.system_message}\nPremise: {premise}\nHypothesis: {hypothesis}\nRelationship: "
            prompts.append(prompt)
            completions.append(SNLIConfig.label_map[label])

        texts = [p + c for p, c in zip(prompts, completions)]
        return {'prompt': prompts, 'completion': completions, 'text': texts}
    
    @classmethod
    def get_eval_config(cls):
        from eval.eval_configs import ExactMatchConfig
        return ExactMatchConfig(case_sensitive=False)


@DatasetConfig.register('winogrande', 'winogrande_xl')
class WinograndeConfig(DatasetConfig):

    system_message = "Fill in the blank with the correct option. Respond only with 1 or 2"
    train_split = 'train'
    test_split = 'validation'

    @staticmethod
    def chat_processor(batch: Dataset) -> ChatOutput:
        chats = []

        for sentence, option1, option2, answer in zip(batch['sentence'], batch['option1'], batch['option2'], batch['answer']):
            question = f"{sentence}\n1. {option1}\n2. {option2}"
            chats.append(WinograndeConfig._build_chat(question, answer))

        return {'messages': chats}
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:
        prompts = []

        for sentence, option1, option2 in zip(batch['sentence'], batch['option1'], batch['option2']):
            question = f"{sentence}\n1. {option1}\n2. {option2}"
            prompts.append(WinograndeConfig._build_prompt(question))

        completions = batch['answer']
        texts = [p + c for p, c in zip(prompts, completions)]
        return {'prompt': prompts, 'completion': completions, 'text': texts}
    
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
        chats = []

        for question_stem, choices, answer_key in zip(batch['question_stem'], batch['choices'], batch['answerKey']):
            question = f"{question_stem}\n" + "\n".join([f"{label}. {text}" for label, text in zip(choices['label'], choices['text'])])
            chats.append(OpenBookQAConfig._build_chat(question, answer_key))

        return {'messages': chats}
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:
        prompts = []

        for question_stem, choices in zip(batch['question_stem'], batch['choices']):
            question = f"{question_stem}\n" + "\n".join([f"{label}. {text}" for label, text in zip(choices['label'], choices['text'])])
            prompts.append(OpenBookQAConfig._build_prompt(question))

        completions = batch['answerKey']
        texts = [p + c for p, c in zip(prompts, completions)]
        return {'prompt': prompts, 'completion': completions, 'text': texts}
    
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
        chats = []

        for ctx, activity_label, endings, label in zip(batch['ctx'], batch['activity_label'], batch['endings'], batch['label']):
            context = f"{activity_label}: {ctx}" if activity_label else ctx
            question = f"{context}\n" + "\n".join([f"{i}. {ending}" for i, ending in enumerate(endings)])
            chats.append(HellaSwagConfig._build_chat(question, label))

        return {'messages': chats}
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:
        prompts = []

        for ctx, activity_label, endings in zip(batch['ctx'], batch['activity_label'], batch['endings']):
            context = f"{activity_label}: {ctx}" if activity_label else ctx
            question = f"{context}\n" + "\n".join([f"{i}. {ending}" for i, ending in enumerate(endings)])
            prompts.append(HellaSwagConfig._build_prompt(question))

        completions = batch['label']
        texts = [p + c for p, c in zip(prompts, completions)]
        return {'prompt': prompts, 'completion': completions, 'text': texts}
    
    @classmethod
    def get_eval_config(cls):
        from eval.eval_configs import MultipleChoiceConfig
        return MultipleChoiceConfig(choices=['0', '1', '2', '3'])


############################################
### DatasetMixer - Subclass of DatasetConfig
############################################

class DatasetMixer(DatasetConfig):
    """Mix multiple datasets - works like any other DatasetConfig.
    
    Two strategies:
    1. 'shuffle': Concatenate all datasets (DataLoader shuffles)
    2. 'round_robin': Interleave samples from each dataset
    """
    
    def __init__(
        self,
        dataset_ids: List[str],
        strategy: Literal['shuffle', 'round_robin'] = 'shuffle',
        seed: Optional[int] = 42
    ):
        """Initialize mixer.
        
        Args:
            dataset_ids: List like ['openai/gsm8k.main', 'allenai/ai2_arc.ARC-Easy']
            strategy: 'shuffle' or 'round_robin'
            seed: Random seed for reproducibility
        """
        self.dataset_ids = dataset_ids
        self.strategy = strategy
        self.seed = seed
        # Required by parent class
        self.dataset_path = None
        self.dataset_name = None
    
    def __new__(cls, *args, **kwargs):
        # Override parent's __new__ which blocks instantiation
        return object.__new__(cls)
    
    def id(self) -> str:
        """Instance method override for unique identifier."""
        return f"mixed_{self.strategy}_" + "+".join([ds.replace('/', '-') for ds in self.dataset_ids])
    
    def load_and_process(self, is_chat: bool, split: str = 'train') -> Dataset:
        """Load and mix component datasets."""
        datasets = []
        
        # Load each dataset using its config
        for dataset_id in self.dataset_ids:
            if '.' in dataset_id:
                path, name = dataset_id.rsplit('.', 1)
            else:
                path, name = dataset_id, None
            
            config_cls = DatasetConfig.from_dataset_path(path, name)
            ds = config_cls.load_and_process(is_chat=is_chat, split=split)
            datasets.append(ds)
        
        # Mix according to strategy
        if self.strategy == 'shuffle':
            return concatenate_datasets(datasets)
        else:  # round_robin
            return interleave_datasets(
                datasets,
                seed=self.seed,
                stopping_strategy='all_exhausted'
            )
    
    # Implement abstract methods (not used for mixer)
    @staticmethod
    def chat_processor(batch: Dataset) -> ChatOutput:
        raise NotImplementedError("DatasetMixer processes datasets directly")
    
    @staticmethod
    def non_chat_processor(batch: Dataset) -> NonChatOutput:
        raise NotImplementedError("DatasetMixer processes datasets directly")
    
    @classmethod
    def get_eval_config(cls):
        raise NotImplementedError("Mixed datasets don't support direct evaluation")
