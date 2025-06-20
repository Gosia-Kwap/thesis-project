from enum import Enum

MODEL_MAP = {
    "gemma9b": "google/gemma-2-9b-it",
    "gemma27b": "google/gemma-2-27b-it",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}

class LEVEL(Enum):
    """Enum class for logging levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"

format_dict = {
    'logiqa': 'label',
    'SVAMP': 'int',
    'GSM8K': 'int',
    'CommonsenseQA': 'label',
    'ai2_arc': 'label',
    'ASDiv': 'int'

}