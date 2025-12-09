from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Union, Optional
import re
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    eval_type: str
    metrics: Optional[Dict[str, Union[int, float]]]
    inputs: List[Any]       # Input prompts or messages
    predictions: List[Union[str, List[str]]]  # Raw predictions (str for K=1, List[str] for K>1)
    references: List[str]   # Raw references
    parsed_predictions: Optional[List[Any]]  # Parsed predictions
    parsed_references: Optional[List[Any]]   # Parsed references
    num_samples: int
    num_pass: int = 1  # K value for pass@K
    
    def __repr__(self):
        metric_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                for k, v in self.metrics.items()])
        return f"EvaluationResult({self.eval_type}, {metric_str}, n={self.num_samples})"


class EvaluationConfig(ABC):
    """Base class for evaluation configurations"""
    
    eval_type: str = "BASE"  # Override in subclasses
    
    @abstractmethod
    def parse_prediction(self, pred: str) -> Any:
        """Parse the raw prediction string into a usable format"""
        pass
    
    @abstractmethod
    def parse_reference(self, ref: str) -> Any:
        """Parse the reference answer into a usable format"""
        pass
    
    @abstractmethod
    def is_correct(self, prediction: Any, reference: Any) -> bool:
        """Check if a single prediction matches the reference"""
        pass
    
    def compute_metrics(self, predictions: List[Any], references: List[Any]) -> Dict[str, Union[int, float]]:
        """Compute metrics given parsed predictions and references (shared logic)"""
        total = len(predictions)
        
        # Normalize to list format (K=1 becomes list of lists with single element)
        if predictions and not isinstance(predictions[0], list):
            predictions = [[p] for p in predictions]
        
        K = len(predictions[0]) if predictions else 1
        
        # Track at which k each sample first passes (0 means never passes)
        first_pass_at = []
        for p_list, r in zip(predictions, references):
            passed_at = 0
            for k, p in enumerate(p_list, 1):
                if p is not None and self.is_correct(p, r):
                    passed_at = k
                    break
            first_pass_at.append(passed_at)
        
        # Compute pass@k metrics from first_pass_at
        metrics = {}
        for k in range(1, K + 1):
            pass_at_k = sum(1 for passed_at in first_pass_at if 0 < passed_at <= k)
            metrics[f'pass_at_{k}_accuracy'] = pass_at_k / total if total > 0 else 0.0
            metrics[f'pass_at_{k}_correct'] = pass_at_k
        
        # Add answered rate for first attempt
        first_answered = sum(1 for p_list, _ in zip(predictions, references) if p_list[0] is not None)
        metrics['pass_at_1_answered_rate'] = first_answered / total if total > 0 else 0.0
        
        # For K=1, also add backward compatible keys
        if K == 1:
            metrics['accuracy'] = metrics['pass_at_1_accuracy']
            metrics['answered_rate'] = metrics['pass_at_1_answered_rate']
            metrics['correct'] = metrics['pass_at_1_correct']
            metrics['answered'] = first_answered
        
        metrics['total'] = total
        metrics['num_pass'] = K
        return metrics
    
    def __call__(self, inputs: List[Any], predictions: List[Union[str, List[str]]], references: List[str]) -> EvaluationResult:
        """
        Main evaluation method
        
        Args:
            inputs: List of input prompts or messages
            predictions: List of raw prediction strings (or lists of strings for K>1)
            references: List of raw reference strings
            
        Returns:
            EvaluationResult with metrics and parsed values
        """
        # Determine if this is K-pass evaluation
        num_pass = 1
        if predictions and isinstance(predictions[0], list):
            num_pass = len(predictions[0])
        
        # Parse predictions and references
        if num_pass == 1:
            # Single generation: predictions is List[str]
            parsed_predictions = [self.parse_prediction(p) for p in predictions]
        else:
            # K generations: predictions is List[List[str]]
            # Parse all K predictions for each sample
            parsed_predictions = [[self.parse_prediction(pred) for pred in p_list] for p_list in predictions]
        
        parsed_references = [self.parse_reference(r) for r in references]
        
        # Compute metrics
        metrics = self.compute_metrics(parsed_predictions, parsed_references)
        
        # Get eval_type (handles both class and instance attributes)
        eval_type = getattr(self, 'eval_type', self.__class__.eval_type)
        
        return EvaluationResult(
            eval_type=eval_type,
            metrics=metrics,
            inputs=inputs,
            predictions=predictions,
            references=references,
            parsed_predictions=parsed_predictions,
            parsed_references=parsed_references,
            num_samples=len(predictions),
            num_pass=num_pass
        )


class ExactMatchConfig(EvaluationConfig):
    """Simple exact string matching"""
    
    eval_type = "EXACT_MATCH"

    def __init__(self, normalize: bool = True, case_sensitive: bool = False, 
                 strip_think_tags: bool = False, extract_last_word: bool = False):
        self.normalize = normalize
        self.case_sensitive = case_sensitive
        self.strip_think_tags = strip_think_tags
        self.extract_last_word = extract_last_word
    
    def parse_prediction(self, pred: str) -> str:
        pred = pred.strip()
        
        # Remove think tags if enabled
        if self.strip_think_tags:
            pred = re.sub(r'<think>.*?</think>', '', pred, flags=re.DOTALL).strip()
        
        # Extract last word if enabled
        if self.extract_last_word:
            words = pred.split()
            pred = words[-1] if words else pred
        
        if self.normalize:
            pred = ' '.join(pred.split())  # Normalize whitespace
        if not self.case_sensitive:
            pred = pred.lower()
        return pred
    
    def parse_reference(self, ref: str) -> str:
        ref = ref.strip()
        if self.normalize:
            ref = ' '.join(ref.split())
        if not self.case_sensitive:
            ref = ref.lower()
        return ref
    
    def is_correct(self, prediction: str, reference: str) -> bool:
        return prediction == reference


import re
from typing import List, Optional


class MultipleChoiceConfig(EvaluationConfig):
    """Evaluation for multiple choice questions"""
    
    eval_type = "MULTIPLE_CHOICE"

    def __init__(
        self, 
        choices: List[str] = ['A', 'B', 'C', 'D', 'E'], 
        choice_is_index: bool = True,
        case_sensitive: bool = True,
    ):
        self.choices = list(choices)
        self.choice_is_index = choice_is_index
        self.case_sensitive = case_sensitive
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison if case-insensitive."""
        return text if self.case_sensitive else text.lower()
    
    def _choice_in_text(self, choice: str, text: str) -> bool:
        """Check if a choice appears in text."""
        choice_norm = self._normalize(choice)
        text_norm = self._normalize(text)
        
        if self.choice_is_index:
            # For index choices, check with common delimiters
            # This prevents 'A' from matching in 'Apple'
            delimiters = ['', ')', '.', ':', ',', ']', '"', "'", ' ']
            prefixes = ['', '(', '[', '"', "'", ' ']
            
            for prefix in prefixes:
                for suffix in delimiters:
                    pattern = f"{prefix}{choice_norm}{suffix}"
                    if pattern in text_norm:
                        return True
            return False
        else:
            # For word choices, use word boundary regex
            pattern = rf'\b{re.escape(choice_norm)}\b'
            return bool(re.search(pattern, text_norm))
    
    def parse_prediction(self, pred: str) -> Optional[str]:
        """Extract the choice from prediction."""
        pred = pred.strip()
        
        # Check exact match first
        pred_norm = self._normalize(pred)
        for choice in self.choices:
            choice_norm = self._normalize(choice)
            if pred_norm == choice_norm:
                return choice
            # Common exact formats for index choices
            if self.choice_is_index:
                if pred_norm in [f"({choice_norm})", f"{choice_norm})", f"{choice_norm}."]:
                    return choice
        
        # Find which choices appear in the prediction
        found = [choice for choice in self.choices if self._choice_in_text(choice, pred)]
        
        # Return only if exactly one choice found
        if len(found) == 1:
            return found[0]
        
        return None
    
    def parse_reference(self, ref: str) -> str:
        """Normalize reference answer."""
        ref = ref.strip()
        if not self.case_sensitive:
            for choice in self.choices:
                if choice.lower() == ref.lower():
                    return choice
        return ref
    
    def is_correct(self, prediction: Optional[str], reference: str) -> bool:
        if prediction is None:
            return False
        if self.case_sensitive:
            return prediction == reference
        return prediction.lower() == reference.lower()


class NumericConfig(EvaluationConfig):
    """Evaluation for numeric answers (e.g., math problems)"""

    eval_type = "NUMERICAL"
    
    def __init__(self, tolerance: float = 1e-3, extract_last: bool = True):
        self.tolerance = tolerance
        self.extract_last = extract_last
    
    def parse_prediction(self, pred: str) -> Optional[float]:
        """Extract numeric answer from prediction"""
        # Look for common patterns
        patterns = [
            r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',  # GSM8K format
            r'(?:answer|result|solution)\s*(?:is|=)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
            r'=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)'
        ]
        
        # for pattern in patterns:
        #     match = re.search(pattern, pred, re.IGNORECASE)
        #     if match:
        #         print("Matched 1= ", match.group(1), "\n")
        #         return float(match.group(1).replace(',', ''))
        
        # Fallback: extract all numbers and take last one
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', pred)
        if numbers:
            if self.extract_last:
                # print("Matched 2= ", numbers[-1], "\n")
                return float(numbers[-1].replace(',', ''))
            else:
                return float(numbers[0].replace(',', ''))
        
        return None
    
    def parse_reference(self, ref: str) -> float:
        """Parse reference number"""
        # Handle GSM8K format
        if '####' in ref:
            ref = ref.split('####')[-1].strip()
        return float(ref.replace(',', ''))
    
    def is_correct(self, prediction: Optional[float], reference: float) -> bool:
        if prediction is None:
            return False
        return abs(prediction - reference) <= self.tolerance


class BooleanConfig(EvaluationConfig):
    """Evaluation for yes/no or true/false questions"""
    
    eval_type = "BOOLEAN"

    def __init__(self, true_values: List[str] = ['yes', 'true', '1'],
                 false_values: List[str] = ['no', 'false', '0']):
        self.true_values = [v.lower() for v in true_values]
        self.false_values = [v.lower() for v in false_values]
    
    def parse_prediction(self, pred: str) -> Optional[bool]:
        """Extract boolean from prediction"""
        pred_lower = pred.strip().lower()
        
        # Check first 50 chars
        text = pred_lower[:50]
        
        # Count occurrences
        true_found = any(tv in text for tv in self.true_values)
        false_found = any(fv in text for fv in self.false_values)
        
        if true_found and not false_found:
            return True
        elif false_found and not true_found:
            return False
        
        return None  # Ambiguous
    
    def parse_reference(self, ref: Union[str, bool]) -> bool:
        """Parse reference boolean"""
        # Handle boolean directly
        if isinstance(ref, bool):
            return ref
        
        ref_lower = ref.strip().lower()
        
        if ref_lower in self.true_values or ref_lower == 'true':
            return True
        elif ref_lower in self.false_values or ref_lower == 'false':
            return False
        
        raise ValueError(f"Cannot parse reference as boolean: {ref}")
    
    def is_correct(self, prediction: Optional[bool], reference: bool) -> bool:
        return prediction == reference


class CustomConfig(EvaluationConfig):
    """Custom evaluation using user-provided functions"""
    
    def __init__(
        self,
        eval_type_name: str,
        parse_pred_fn: Callable[[str], Any],
        parse_ref_fn: Callable[[str], Any],
        metric_fn: Callable[[List[Any], List[Any]], Dict[str, Union[int, float]]]
    ):
        self.eval_type = eval_type_name  # Instance attribute
        self.parse_pred_fn = parse_pred_fn
        self.parse_ref_fn = parse_ref_fn
        self.metric_fn = metric_fn
    
    def parse_prediction(self, pred: str) -> Any:
        return self.parse_pred_fn(pred)
    
    def parse_reference(self, ref: str) -> Any:
        return self.parse_ref_fn(ref)
    
    def compute_metrics(self, predictions: List[Any], references: List[Any]) -> Dict[str, Union[int, float]]:
        return self.metric_fn(predictions, references)
