import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

import os
import glob


class UncertaintyCalibrationAnalyser:
    def __init__(self, result_dir, file_pattern="*_uncertainty.json", num_bins=5, mode='all'):
        """
        Initialize the analyzer with multiple result files

        Args:
            result_dir (str): Path to directory containing JSON results
            file_pattern (str): Pattern to match result files
            num_bins (int): Number of bins for uncertainty segmentation
            mode: 'all' (default) uses all answers, 'original' uses only original answers
        """
        self.result_files = self._find_result_files(result_dir, file_pattern)
        self.num_bins = num_bins
        self.df = self._load_and_preprocess_data()
        self.bin_edges = np.linspace(0, 1, num_bins + 1)
        self.mode = mode

    def _find_result_files(self, result_dir, pattern):
        """Find all result files matching the pattern"""
        search_path = os.path.join(result_dir, pattern)
        files = sorted(glob.glob(search_path))
        if not files:
            raise ValueError(f"No files found matching pattern: {search_path}")
        return files

    def _load_and_preprocess_data(self):
        """Load and concatenate all result files"""
        dfs = []
        for file in self.result_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)

                # Convert single file data to DataFrame format
                file_df = pd.DataFrame(data)
                dfs.append(file_df)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue

        if not dfs:
            raise ValueError("No valid data files could be loaded")

        # Combine all DataFrames and reset index
        combined_df = pd.concat(dfs, ignore_index=True)
        return self._preprocess_dataframe(combined_df)

    def _preprocess_dataframe(self, df):
        """Preprocess the combined DataFrame"""
        # Convert your specific format here
        processed = []
        for _, row in df.iterrows():
            entry = {
                'input': row['input'],
                'expected_output': row['expected_output'],
                'uncertainty': row['uncertainty'],
                'original_answer': row['generated_answers']['original_answer'][0],
            }
            for perturb_type, answers in row['generated_answers'].items():
                for i, answer in enumerate(answers):
                    if perturb_type =='original_answer':
                        key = 'original'
                    else:
                        key = f"{perturb_type}_{i}"
                    entry[f"{key}_answer"] = self._extract_final_answer(answer)
                    entry[f"{key}_confidence"] = self._extract_confidence(answer)
                    entry[f"{key}_correct"] = (
                        entry[f"{key}_answer"] == entry['expected_output']
                        if entry[f"{key}_answer"] is not None
                        else False
                    )

            processed.append(entry)

        return pd.DataFrame(processed)

    def _extract_final_answer(self, text):
        """Extract the final numerical answer from model output text"""
        # Priority 1: Look for "Final Answer" pattern
        final_answer_match = re.search(r'Final Answer[^:]*:\s*(\d+)', text, re.IGNORECASE)
        if final_answer_match:
            return int(final_answer_match.group(1))

        # Priority 2: Find last equation result
        equations = re.findall(r'(\d+\s*[+\-*/]\s*\d+\s*=\s*\d+)', text)
        if equations:
            last_eq = equations[-1].split('=')[-1].strip()
            return int(last_eq)

        # Priority 3: Find last standalone number (fallback)
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            return int(numbers[-1])

        return None

    def _extract_confidence(self, text):
        """Extract confidence percentage from model output text"""
        confidence_match = re.search(r'Overall Confidence[^:]*:\s*(\d+)%', text, re.IGNORECASE)
        return int(confidence_match.group(1)) / 100 if confidence_match else 0.5  # Default to 0.5 if not found

    def _get_uncertainty_bin(self, uncertainty):
        """Assign uncertainty value to a bin"""
        return pd.cut([uncertainty], bins=self.bin_edges, labels=range(self.num_bins))[0]

    def analyze_calibration(self):
        # Flatten the data
        flat_df = self._get_flattened_data()

        # 1. Confidence Calibration
        conf_bins = pd.cut(flat_df['confidence'], bins=self.num_bins)
        conf_stats = (
            flat_df.groupby(conf_bins)
            .agg(
                mean_confidence=('confidence', 'mean'),
                accuracy=('correct', 'mean'),
                count=('correct', 'size')
            )
            .reset_index()
            .dropna()
        )
        conf_ece = self._calculate_ece(conf_stats, 'mean_confidence', len(flat_df))

        # 2. Uncertainty Calibration (using 1 - uncertainty as "certainty")
        uncert_bins = pd.cut(flat_df['uncertainty'], bins=self.num_bins)
        uncert_stats = (
            flat_df.groupby(uncert_bins)
            .agg(
                mean_uncertainty=('uncertainty', 'mean'),
                mean_certainty=('uncertainty', lambda x: 1 - x.mean()),
                accuracy=('correct', 'mean'),
                count=('correct', 'size')
            )
            .reset_index()
            .dropna()
        )
        uncert_ece = self._calculate_ece(uncert_stats, 'mean_certainty', len(flat_df))

        return {
            'confidence': {
                'bin_stats': conf_stats,
                'ece': conf_ece,
                'calibration_curve': calibration_curve(
                    flat_df['correct'].astype(int),
                    flat_df['confidence'],
                    n_bins=self.num_bins
                )
            },
            'uncertainty': {
                'bin_stats': uncert_stats,
                'ece': uncert_ece,
                'calibration_curve': calibration_curve(
                    flat_df['correct'].astype(int),
                    1 - flat_df['uncertainty'],
                    n_bins=self.num_bins
                )
            }
        }

    def _get_flattened_data(self):
        flat_data = []
        for _, row in self.df.iterrows():
            if self.mode == 'original':
                # Only process original answer
                flat_data.append({
                    'uncertainty': row['uncertainty'],
                    'answer': row['original_answer'],
                    'confidence': row.get('original_confidence', 0.5),
                    'correct': row['original_correct'],
                    'perturbation': 'original',
                    'expected': row['expected_output']
                })
            else:
                # Process all answers (original mode)
                for col in [c for c in row.index if c.endswith('_answer')]:
                    prefix = col[:-len('_answer')]
                    flat_data.append({
                        'uncertainty': row['uncertainty'],
                        'answer': row[col],
                        'confidence': row[f'{prefix}_confidence'],
                        'correct': row[f'{prefix}_correct'],
                        'perturbation': prefix.split('_')[0],
                        'expected': row['expected_output'],

                    })
        return pd.DataFrame(flat_data)

    def _calculate_ece(self, bin_stats, pred_col, total_samples):
        """Generic ECE calculation that works for both metrics"""
        return np.sum(
            np.abs(bin_stats['accuracy'] - bin_stats[pred_col]) *
            (bin_stats['count'] / total_samples)
        )

    def save_results(self, analysis_results, output_path):
        # NEEDS ADJUSTMENT!!!!
        """Save analysis results to JSON file"""
        results = {
            'metrics': {
                'ece': analysis_results['ece'],
                'brier': analysis_results['brier']
            },
            'bin_stats': analysis_results['bin_stats'].to_dict(orient='records'),
            'calibration_curve': {
                'prob_true': analysis_results['calibration_curve'][0].tolist(),
                'prob_pred': analysis_results['calibration_curve'][1].tolist()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    def plot_comparison(self, analysis_results, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Confidence plot
        conf = analysis_results['confidence']
        ax1.plot(conf['calibration_curve'][1], conf['calibration_curve'][0], 's-')
        ax1.set_title(f'Confidence Calibration (ECE={conf["ece"]:.3f})')

        # Uncertainty plot (note: 1-uncertainty = "certainty")
        uncert = analysis_results['uncertainty']
        ax2.plot(uncert['bin_stats']['mean_certainty'],
                 uncert['bin_stats']['accuracy'], 'o-')
        ax2.set_title(f'Uncertainty Calibration (ECE={uncert["ece"]:.3f})')

        for ax in (ax1, ax2):
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Actual Accuracy')
            ax.grid(True)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()

