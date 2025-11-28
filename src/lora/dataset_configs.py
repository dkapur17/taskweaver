from datasets import Dataset
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset
from typing import List, Literal, TypedDict, Dict, Type, Optional, Tuple

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

        return {'prompt': prompts, 'completion': batch['answerKey']}
    

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
        return {'prompt': batch['question'], 'completion': batch['answer']}
    
    
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

        return {'prompt': prompts, 'completion': completions}
    

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

        return {'prompt': prompts, 'completion': completions}
