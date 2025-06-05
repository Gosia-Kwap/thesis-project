from src.uncertainty.UncertaintyCalibrationAnalyser import UncertaintyCalibrationAnalyser

result_dir = "/Users/University/Library/CloudStorage/OneDrive-Personal/Dokumenty/studia/AI/Year3/ThesisAI/thesis-project/results/results-copy"

analyzer_all = UncertaintyCalibrationAnalyser(
    result_dir=result_dir,
    file_pattern="SVAMP_perturbed_outputs_gemma9b_*_uncertainty.json",
    num_bins=25,
    mode='all'
)
results_all = analyzer_all.analyze_calibration()

# Analyze only original answers
analyzer_orig = UncertaintyCalibrationAnalyser(
    result_dir=result_dir,
    file_pattern="SVAMP_perturbed_outputs_gemma9b_*_uncertainty.json",
    num_bins=25,
    mode='original'
)
results_orig = analyzer_orig.analyze_calibration()

# Compare results
print("All answers confidence ECE:", results_all['confidence']['ece'])
print("All answers uncertainty ECE:", results_all['uncertainty']['ece'])
print("Original-only confidence ECE:", results_orig['confidence']['ece'])
print("Original-only uncertainty ECE:", results_orig['uncertainty']['ece'])


