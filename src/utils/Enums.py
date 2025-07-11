from enum import Enum

MODEL_MAP = {
    "gemma9b": "google/gemma-2-9b-it",
    "gemma27b": "google/gemma-2-27b-it",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
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

LLAMA_MODEL_MAP = {
    "llama3": {
        "repo_id": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-v2",
        "file_prefix": "Meta-Llama-3-8B-Instruct-v2",
        "quant_format": "dot"
    },
    "gemma9b": {
        "repo_id": "bartowski/gemma-2-9b-it-GGUF",
        "file_prefix": "gemma-2-9b-it",
        "quant_format": "dash"
    },
    "gemma27b": {
        "repo_id": "bartowski/gemma-2-27b-it-GGUF",
        "file_prefix": "gemma-2-27b-it",
        "quant_format": "dash"
    },
    "deepseek": {

        "repo_id": "Qwen/Qwen3-8B-GGUF",
        "file_prefix": "Qwen3-8B",
        "quant_format": "dash"
    }
}


quantisation_map = {
    "6" : "Q6_K",
    "4" : "Q4_K_M"
}