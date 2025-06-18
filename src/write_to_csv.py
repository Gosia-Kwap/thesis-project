import os
import json
import pandas as pd


def convert_json_folder_to_csv_excel(folder_path, output_csv="output.csv", output_excel="output.xlsx"):
    all_data = []

    for filename in os.listdir(folder_path):
        if 'llama' in filename:
            if filename.endswith(".json"):
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        # Ensure it's a list of records
                        if isinstance(data, dict):
                            data = [data]
                        for item in data:
                            # Flatten nested lists into strings
                            for field in ['facts', 'decomposition']:
                                if field in item and isinstance(item[field], list):
                                    item[field] = "\n".join(item[field])
                            all_data.append(item)
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV to: {output_csv}")

    # Save to Excel
    df.to_excel(output_excel, index=False)
    print(f"Saved Excel to: {output_excel}")

path = r"/Users/University/Library/CloudStorage/OneDrive-Personal/Dokumenty/studia/AI/Year3/ThesisAI/thesis-project/results/uncertainty-new"
output_dir = r"/Users/University/Library/CloudStorage/OneDrive-Personal/Dokumenty/studia/AI/Year3/ThesisAI/thesis-project/results/csv"
convert_json_folder_to_csv_excel(path, output_csv=f"{output_dir}/output.csv", output_excel=f"{output_dir}/output.xlsx")