from jsonargparse import Namespace
from pathlib import Path
from enum import Enum

def print_args_rec(args: Namespace, indent: int = 0, prefix: str = '') -> None:
    """Recursively print all arguments, including nested dataclasses."""
    indent_str = '  ' * indent
    
    # Get the dict representation
    if hasattr(args, '__dict__'):
        items = vars(args).items()
    elif isinstance(args, dict):
        items = args.items()
    else:
        return

    for key, value in items:
        full_key = f"{prefix}.{key}" if prefix else key
        
        # Check if value is a nested object (dataclass or Namespace)
        if hasattr(value, '__dataclass_fields__') or isinstance(value, Namespace):
            print(f"{indent_str}{key}:")
            print_args_rec(value, indent + 1, full_key)
        elif isinstance(value, dict):
            print(f"{indent_str}{key}:")
            print_args_rec(value, indent + 1, full_key)
        else:
            print(f"{indent_str}{key}: {value}")

def print_args(args, indent: int = 0, prefix: str = ''):
    print()
    print("="*30)
    print("Configuration")
    print("="*30)
    print_args_rec(args, indent, prefix)
    print("="*30)
    print()

def serialize_args(args) -> dict:
    """Recursively serialize arguments to a JSON-compatible dictionary."""
    if hasattr(args, '__dict__'):
        items = vars(args).items()
    elif isinstance(args, dict):
        items = args.items()
    else:
        return args
    
    result = {}
    for key, value in items:
        if hasattr(value, '__dataclass_fields__') or isinstance(value, Namespace):
            result[key] = serialize_args(value)
        elif isinstance(value, dict):
            result[key] = serialize_args(value)
        elif isinstance(value, (list, tuple)):
            result[key] = [
                serialize_args(v) if hasattr(v, '__dict__') else v 
                for v in value
            ]
        elif isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, Enum):
            result[key] = value.value
        else:
            result[key] = value
    
    return result