import sys
from typing import Optional, Tuple, List
from datasets import Dataset
from string import Formatter
from tqdm import tqdm
from .eval_configs import EvaluationConfig, EvaluationResult

class Task:
    def __init__(
            self,
            task_name: str,
            dataset: Dataset,
            user_template: str,
            assistant_template: str,
            system_prompt: Optional[str] = None,
            eval_config: Optional[EvaluationConfig] = None,
            is_chat_task: bool = True
    ):

        self.task_name = task_name  
        self.user_template = user_template
        self.assistant_template = assistant_template
        self.system_prompt = system_prompt
        self.eval_config = eval_config
        self.is_chat_task = is_chat_task

        self.dataset = dataset
        self._prepare_dataset()

    def _prepare_dataset(self) -> None:

        user_template_fields = [field_name for _, field_name, _, _ in Formatter().parse(self.user_template) if field_name is not None]
        assistant_template_fields = [field_name for _, field_name, _, _ in Formatter().parse(self.assistant_template) if field_name is not None]
        
        if set(user_template_fields + assistant_template_fields) != set(self.dataset.column_names):
            print(f'WARNING: fields {", ".join(set(self.dataset.column_names) - set(user_template_fields + assistant_template_fields))} are not used in any templates.', file=sys.stderr)

        def build_messages(examples):
            
            user_field_vals = {field:examples[field] for field in user_template_fields}
            assistant_field_vals = {field:examples[field] for field in assistant_template_fields}
            
            batch_size = len(user_field_vals[user_template_fields[0]])
            
            X_batch = []
            y_batch = []
            for i in range(batch_size):
                user_vals_row = {field:user_field_vals[field][i] for field in user_template_fields}
                assistant_vals_row = {field:assistant_field_vals[field][i] for field in assistant_template_fields}

                if self.is_chat_task:
                    X = [{"role": "user", "content": self.user_template.format_map(user_vals_row)}]
                    y = [{"role": "assistant", "content": self.assistant_template.format_map(assistant_vals_row)}]

                    if self.system_prompt:
                        X.insert(0, {"role": "system", "content": self.system_prompt})
                else:
                    prompt = self.user_template.format_map(user_vals_row)
                    if self.system_prompt:
                        prompt = f"{self.system_prompt}\n\n{prompt}"
                    X = prompt
                    y = self.assistant_template.format_map(assistant_vals_row)
                
                X_batch.append(X)
                y_batch.append(y)

            return {"X": X_batch, "y": y_batch}
        
        self.dataset = self.dataset.map(build_messages, batched=True, remove_columns=self.dataset.column_names)

    def _get_preds_and_refs(self, model, tokenizer, batch_size:int=64, max_new_tokens:int=32768, temperature:float=1.0, progress:bool=False) -> Tuple[List[str], List[str]]:
        if progress:
            pbar = tqdm(self.dataset.iter(batch_size), total=self.dataset.num_rows//batch_size + 1)
        else:
            pbar = self.dataset.iter(batch_size)

        refs = []
        preds = []
        for batch in pbar:
            if self.is_chat_task:
                model_input_texts = tokenizer.apply_chat_template(batch['X'], 
                                                            tokenize=False, 
                                                            add_generation_prompt=True, 
                                                            enable_thinking=False)
                y = [row[0]['content'] for row in batch['y']]
            else:
                model_input_texts = batch['X']
                y = batch['y']

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_inputs = tokenizer(model_input_texts, return_tensors='pt', padding=True).to(model.device)
            
            model_outputs = model.generate(**model_inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True)
            
            preds_batch = []
            for i, output in enumerate(model_outputs):
                generated_ids = output[len(model_inputs.input_ids[i]):]
                y_pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                preds_batch.append(y_pred)

            refs.extend(y)
            preds.extend(preds_batch)

            # print("\nInputs: ", model_input_texts)
            # print("\nReferences: ", refs)
            # print("\nPredictions: ", preds)

        return preds, refs

    def evaluate(self, model, tokenizer, batch_size:int=64, max_new_tokens:int=32768, temperature:float=1.0, progress:bool=False) -> EvaluationResult:
        
        preds, refs = self._get_preds_and_refs(model, tokenizer, batch_size, max_new_tokens, temperature, progress)

        if self.eval_config:
            return self.eval_config(preds, refs)
        else:
            return EvaluationResult(
                eval_type='NONE',
                metrics=None,
                predictions=preds,
                references=refs,
                parsed_predictions=None,
                parsed_references=None,
                num_samples=len(preds)
            )

    def __len__(self) -> int:
        return self.dataset.num_rows

    def __getitem__(self, idx:int) -> dict:
        return self.dataset[idx]
    
    def __repr__(self) -> str:
        return f"""Task(task_name={self.task_name}, num_samples={len(self)})"""

