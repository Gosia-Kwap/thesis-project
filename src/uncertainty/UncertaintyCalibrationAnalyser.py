import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

import os
import glob


class UncertaintyCalibrationAnalyzer:
    def __init__(self, result_dir, file_pattern="*_uncertainty.json", num_bins=5):
        """
        Initialize the analyzer with multiple result files

        Args:
            result_dir (str): Path to directory containing JSON results
            file_pattern (str): Pattern to match result files
            num_bins (int): Number of bins for uncertainty segmentation
        """
        self.result_files = self._find_result_files(result_dir, file_pattern)
        self.num_bins = num_bins
        self.df = self._load_and_preprocess_data()
        self.bin_edges = np.linspace(0, 1, num_bins + 1)

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
                'uncertainty': row['uncertainty']
            }

            for perturb_type, answers in row['generated_answers'].items():
                for i, answer in enumerate(answers):
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
        confidence_match = re.search(r'Confidence[^:]*:\s*(\d+)%', text, re.IGNORECASE)
        return int(confidence_match.group(1)) / 100 if confidence_match else 0.5  # Default to 0.5 if not found

    def _preprocess_data(self):
        """Convert raw data into structured DataFrame for analysis"""
        processed = []

        for item in self.raw_data:
            entry = {
                'input': item['input'],
                'expected_output': item['expected_output'],
                'uncertainty': 1 - item['perceived_similarity']
            }

            for perturb_type, answers in item['generated_answers'].items():
                for i, answer in enumerate(answers):
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

    def _get_uncertainty_bin(self, uncertainty):
        """Assign uncertainty value to a bin"""
        return pd.cut([uncertainty], bins=self.bin_edges, labels=range(self.num_bins))[0]

    def analyze_calibration(self):
        """Perform full calibration analysis"""
        results = []

        # Flatten the dataframe for per-answer analysis
        flat_data = []
        for _, row in self.df.iterrows():
            for col in [c for c in row.index if c.endswith('_answer')]:  # Changed from '_answer in c'
                prefix = col[:-len('_answer')]
                flat_data.append({
                    'uncertainty': row['uncertainty'],
                    'answer': row[col],
                    'confidence': row[f'{prefix}_confidence'],
                    'correct': row[f'{prefix}_correct'],
                    'perturbation': prefix.split('_')[0],
                    'expected': row['expected_output'],

                })

        flat_df = pd.DataFrame(flat_data)
        flat_df['uncertainty_bin'] = flat_df['uncertainty'].apply(self._get_uncertainty_bin)

        # Calculate bin statistics
        bin_stats = flat_df.groupby('uncertainty_bin').agg(
            avg_confidence=('confidence', 'mean'),
            accuracy=('correct', 'mean'),
            count=('correct', 'size')
        ).reset_index()

        # Calculate calibration metrics
        prob_true, prob_pred = calibration_curve(
            flat_df['correct'].astype(int),
            flat_df['confidence'],
            n_bins=self.num_bins
        )

        ece = np.sum(
            np.abs(bin_stats['accuracy'] - bin_stats['avg_confidence']) *
            (bin_stats['count'] / len(flat_df))
        )

        brier = brier_score_loss(flat_df['correct'].astype(int), flat_df['confidence'])

        return {
            'flat_data': flat_df,
            'bin_stats': bin_stats,
            'calibration_curve': (prob_true, prob_pred),
            'ece': ece,
            'brier': brier
        }

    def plot_calibration(self, analysis_results, save_path=None):
        """Visualize calibration results"""
        plt.figure(figsize=(10, 6))

        # Reliability diagram
        plt.plot(
            analysis_results['bin_stats']['avg_confidence'],
            analysis_results['bin_stats']['accuracy'],
            's-',
            label='Model'
        )
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

        plt.xlabel('Mean predicted confidence')
        plt.ylabel('Fraction of correct answers')
        plt.title(f'Reliability Diagram\nECE = {analysis_results["ece"]:.3f}, Brier = {analysis_results["brier"]:.3f}')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()

    def save_results(self, analysis_results, output_path):
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

