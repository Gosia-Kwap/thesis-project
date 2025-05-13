from datetime import datetime

def log_message(message):
    """Helper function for consistent logging format"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")