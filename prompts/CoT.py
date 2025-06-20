generic_prompt = """Assign a confidence level to
 each step and the final answer. The output format is as follows:
 Step 1: [Your reasoning here], Confidence: [Your confidence here]%
 Step 2:
 ...
 Step 3:
 ...
 ...
 Step N:
 ...
 Final Answer and Overall Confidence(0-100): [Your answer as a number here], [Your confidence
 here]% Note: The confidence indicates the degree of certainty you have about your answer. For
 instance, if your confidence level is 80%, it means you are 80% certain that your answer is correct. 
 Question:
 """
trigger_phrases = ["Read the question, give your answer by analyzing step by step", "Let’s think step by step.", "Let’s think about question below logically step by step."
    , "Before we dive into the answer, think about the reasoning", "Before answering the question, let’s understand the input."]

prompt_single_choice = """
Assign a confidence level to each step and the final answer. The output format is as follows:

Step 1: [Your reasoning here], Confidence: [Your confidence here]%
Step 2: ...
...
Step N: ...
Final Answer: [Your final choice as A/B/C/D/E], Overall Confidence (0-100): [Your confidence here]%

Note: The final answer must be one of the multiple-choice options (A, B, C, D, or E). The confidence indicates how certain you are that your final choice is correct. 
For instance, if your confidence level is 80%, it means you are 80% certain that your selected option is the correct answer.

Question:
"""

prompt_dict = {
    "SVAMP": generic_prompt,
    "GSM8K": generic_prompt,
    "ASDiv": generic_prompt,
    "CommonsenseQA": prompt_single_choice,
    "ai2_arc": prompt_single_choice,
    "logiqa": prompt_single_choice,
}