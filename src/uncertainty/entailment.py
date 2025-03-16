from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model
model_name = "potsawee/deberta-v3-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def get_entailment_score(premise, hypothesis):
    """
    Compute the entailment score between a premise and a hypothesis.

    Args:
        premise (str): The premise text.
        hypothesis (str): The hypothesis text.

    Returns:
        int: Entailment score (-1 for contradiction, 0 for neutral, 1 for entailment).
    """
    # Tokenize the input pair
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the same device as the model

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class (0: contradiction, 1: neutral, 2: entailment)
    predicted_class = torch.argmax(logits, dim=1).item()

    # Map the predicted class to the entailment score
    if predicted_class == 0:  # Contradiction
        return -1
    elif predicted_class == 1:  # Neutral
        return 0
    elif predicted_class == 2:  # Entailment
        return 1
    else:
        raise ValueError("Unexpected predicted class.")