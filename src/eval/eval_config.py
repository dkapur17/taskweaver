from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Union, Optional
import re
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    eval_type: str
    metrics: Optional[Dict[str, Union[int, float]]]
    predictions: List[str]  # Raw predictions
    references: List[str]   # Raw references
    parsed_predictions: Optional[List[Any]]  # Parsed predictions
    parsed_references: Optional[List[Any]]   # Parsed references
    num_samples: int
    
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
    def compute_metrics(self, predictions: List[Any], references: List[Any]) -> Dict[str, Union[int, float]]:
        """Compute metrics given parsed predictions and references"""
        pass
    
    def __call__(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        """
        Main evaluation method
        
        Args:
            predictions: List of raw prediction strings
            references: List of raw reference strings
            
        Returns:
            EvaluationResult with metrics and parsed values
        """
        # Parse predictions and references
        parsed_predictions = [self.parse_prediction(p) for p in predictions]
        parsed_references = [self.parse_reference(r) for r in references]
        
        # Compute metrics
        metrics = self.compute_metrics(parsed_predictions, parsed_references)
        
        # Get eval_type (handles both class and instance attributes)
        eval_type = getattr(self, 'eval_type', self.__class__.eval_type)
        
        return EvaluationResult(
            eval_type=eval_type,
            metrics=metrics,
            predictions=predictions,
            references=references,
            parsed_predictions=parsed_predictions,
            parsed_references=parsed_references,
            num_samples=len(predictions)
        )


class ExactMatchConfig(EvaluationConfig):
    """Simple exact string matching"""
    
    eval_type = "EXACT_MATCH"

    def __init__(self, normalize: bool = True, case_sensitive: bool = False):
        self.normalize = normalize
        self.case_sensitive = case_sensitive
    
    def parse_prediction(self, pred: str) -> str:
        pred = pred.strip()
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
    
    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, Union[int, float]]:
        correct = sum(p == r for p, r in zip(predictions, references))
        total = len(predictions)
        
        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'exact_match': correct / total if total > 0 else 0.0,
            'correct': correct,
            'total': total
        }


class MultipleChoiceConfig(EvaluationConfig):
    """Evaluation for multiple choice questions"""
    
    eval_type = "MULTIPLE_CHOICE"

    def __init__(self, choices: List[str] = ['A', 'B', 'C', 'D', 'E']):
        self.choices = [c.upper() for c in choices]
    
    def parse_prediction(self, pred: str) -> Optional[str]:
        """Extract the choice letter from prediction"""
        pred = pred.strip().upper()
        
        # Try to find first valid choice letter
        for char in pred:
            if char in self.choices:
                return char
        
        # Try to find pattern like "A." or "A)"
        for choice in self.choices:
            if f"{choice}." in pred or f"{choice})" in pred:
                return choice
        
        return None  # No valid choice found
    
    def parse_reference(self, ref: str) -> str:
        """Reference should already be a letter"""
        return ref.strip().upper()
    
    def compute_metrics(self, predictions: List[Optional[str]], references: List[str]) -> Dict[str, Union[int, float]]:
        correct = sum(p == r for p, r in zip(predictions, references) if p is not None)
        answered = sum(p is not None for p in predictions)
        total = len(predictions)
        
        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'answered_rate': answered / total if total > 0 else 0.0,
            'correct': correct,
            'answered': answered,
            'total': total
        }


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
        
        for pattern in patterns:
            match = re.search(pattern, pred, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(',', ''))
        
        # Fallback: extract all numbers and take last one
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', pred)
        if numbers:
            if self.extract_last:
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
    
    def compute_metrics(self, predictions: List[Optional[float]], references: List[float]) -> Dict[str, Union[int, float]]:
        correct = 0
        answered = 0
        
        for pred, ref in zip(predictions, references):
            if pred is not None:
                answered += 1
                if abs(pred - ref) < self.tolerance:
                    correct += 1
        
        total = len(predictions)
        
        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'answered_rate': answered / total if total > 0 else 0.0,
            'correct': correct,
            'answered': answered,
            'total': total
        }


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
    
    def compute_metrics(self, predictions: List[Optional[bool]], references: List[bool]) -> Dict[str, Union[int, float]]:
        correct = sum(p == r for p, r in zip(predictions, references) if p is not None)
        answered = sum(p is not None for p in predictions)
        total = len(predictions)
        
        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'answered_rate': answered / total if total > 0 else 0.0,
            'correct': correct,
            'answered': answered,
            'total': total
        }


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