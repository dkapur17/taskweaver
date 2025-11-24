from .gsm8k import create_gsm8k_task
from .snli import create_snli_task
from .squad import create_squad_task
from .arc_easy import create_arc_easy_task

__all__ = [
    'create_gsm8k_task',
    'create_snli_task',
    'create_squad_task',
    'create_arc_easy_task'
]
