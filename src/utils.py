import pandas as pd
import os
import json
import datetime
import glob
import re

def save_experiment(cv_score, params_dict, description, submission_df=None, folder='experiments'):
    """
    Saves the experiment configuration (JSON) and the submission file if validated.
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"cv{cv_score:.4f}_{timestamp}"
    
    # 1. Save Config (JSON)
    config_filename = f"{folder}/conf_{base_name}.json"
    log_data = {
        "timestamp": timestamp,
        "cv_score": cv_score,
        "description": description,
        "parameters": params_dict,
        "status": "ABORTED" if submission_df is None else "COMPLETED"
    }
    
    with open(config_filename, 'w') as f:
        json.dump(log_data, f, indent=4)
    
    print(f"\n[INFO] Config saved: {config_filename}")

    # 2. Save CSV (If validated)
    if submission_df is not None:
        # Raw version
        csv_filename = f"{folder}/sub_{base_name}.csv"
        submission_df.to_csv(csv_filename, index=False)
        print(f"[INFO] Experiment CSV saved: {csv_filename}")
        
        # Incremental version for submission
        os.makedirs('submission', exist_ok=True)
        existing_files = glob.glob('submission/submission_V*.csv')
        version = 1
        if existing_files:
            versions = []
            for f in existing_files:
                match = re.search(r'submission_V(\d+)\.csv', f)
                if match:
                    versions.append(int(match.group(1)))
            if versions:
                version = max(versions) + 1
        
        final_filename = f'submission/submission_V{version}.csv'
        submission_df.to_csv(final_filename, index=False)
        print(f"[SUCCESS] Ready for submission: '{final_filename}'")
