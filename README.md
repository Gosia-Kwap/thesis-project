# **Chain-of-Thought Uncertainty Estimation in Quantized Large Language Models**

This repository contains the implementation and experiments for the final thesis project for Bachelor of Artificial Intelligence at RUG, titled **"Chain-of-Thought Uncertainty Estimation in Quantized Large Language Models"**. The project focuses on two main objectives:
1. Extending the Zero-shot Uncertainty-based Selection (ZEUS) method for uncertainty estimation in Chain of Thought (CoT) reasoning.
2. Investigating the effect of model quantization (using LoRa-Q) on model calibration.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)

---

## **Project Overview**
The project builds on the ZEUS uncertainty estimation method, which combines uncertainties from:
- Temperature-based perturbation
- Trigger phrase perturbation
- Rephrase perturbation

The extended ZEUS method is applied to Chain of Thought reasoning to produce uncertainty estimates for each step of the rationale. Additionally, the project investigates the impact of model quantization (using LoRa-Q) on model calibration.

### **Key Contributions**
- Implementation of ZEUS uncertainty estimation for Chain of Thought.
- Integration of LoRa-Q quantization for model compression.
- Evaluation of model calibration pre- and post-quantization.

---

## **Repository Structure**
```
.
├── data/                   # Datasets used for experiments
├── models/                 # Pre-trained and quantized models
├── notebooks/              # Jupyter notebooks for experiments and analysis
├── scripts/                # Utility scripts for data processing, evaluation, etc.
├── src/                    # Source code for the project
│   ├── uncertainty/        # Uncertainty estimation framework
│   ├── quantization/       # LoRa-Q quantization implementation
│   ├── evaluation/         # Calibration evaluation scripts
│   └── utils/              # Utility functions (e.g., data loading, preprocessing)
├── results/                # Experiment results (e.g., plots, tables)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

[//]: # (## **Usage**)

[//]: # (### **1. Uncertainty Estimation**)

[//]: # (To estimate uncertainty using the extended ZEUS method for Chain of Thought:)

[//]: # (```python)

[//]: # (from src.uncertainty.zeus_estimator import ZEUSUncertaintyEstimator)

[//]: # (from src.uncertainty.cot_uncertainty import ChainOfThoughtUncertainty)

[//]: # ()
[//]: # (# Initialize ZEUS estimator)

[//]: # (zeus_estimator = ZEUSUncertaintyEstimator&#40;model, tokenizer, temperature_range=[0.5, 1.5], trigger_phrases=["Let's think step by step"]&#41;)

[//]: # ()
[//]: # (# Initialize CoT uncertainty estimator)

[//]: # (cot_uncertainty = ChainOfThoughtUncertainty&#40;model, tokenizer, zeus_estimator&#41;)

[//]: # ()
[//]: # (# Estimate uncertainty for an input)

[//]: # (input_text = "What is the capital of France?")

[//]: # (uncertainty = cot_uncertainty.estimate_final_uncertainty&#40;input_text&#41;)

[//]: # (print&#40;f"Uncertainty: {uncertainty}"&#41;)

[//]: # (```)

[//]: # ()
[//]: # (### **2. Model Quantization**)

[//]: # (To quantize a model using LoRa-Q:)

[//]: # (```python)

[//]: # (from src.quantization.lora_quantizer import LoRaQuantizer)

[//]: # ()
[//]: # (# Initialize quantizer)

[//]: # (quantizer = LoRaQuantizer&#40;model, quantization_config={"bits": 4, "lora_rank": 8}&#41;)

[//]: # ()
[//]: # (# Quantize the model)

[//]: # (quantized_model = quantizer.quantize_model&#40;&#41;)

[//]: # ()
[//]: # (# Save the quantized model)

[//]: # (quantizer.save_quantized_model&#40;"models/quantized_model"&#41;)

[//]: # (```)

[//]: # ()
[//]: # (### **3. Calibration Evaluation**)

[//]: # (To evaluate model calibration:)

[//]: # (```python)

[//]: # (from src.evaluation.calibration_evaluator import CalibrationEvaluator)

[//]: # ()
[//]: # (# Initialize evaluator)

[//]: # (evaluator = CalibrationEvaluator&#40;model, tokenizer, cot_uncertainty&#41;)

[//]: # ()
[//]: # (# Evaluate calibration on a dataset)

[//]: # (results = evaluator.evaluate_calibration&#40;dataset&#41;)

[//]: # (print&#40;f"Expected Calibration Error &#40;ECE&#41;: {results['ece']}"&#41;)

[//]: # (```)

---

## **Experiments**
The experiments are organized into the following stages:
1. **Uncertainty Estimation**:
   - Compare ZEUS and extended ZEUS for CoT.
   - Evaluate uncertainty estimates on benchmark datasets.

2. **Model Quantization**:
   - Quantize models using LoRa-Q.
   - Compare pre- and post-quantization performance.

3. **Calibration Evaluation**:
   - Evaluate calibration for original and quantized models.
   - Compare results with baseline methods.

Detailed instructions for running experiments can be found in the `notebooks/` directory.

---
