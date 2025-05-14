from src.utils.Enums import LEVEL

def log_message(message, level = LEVEL.INFO):
    """Helper function for consistent logging format"""
    print(f"[{level}] {message}")