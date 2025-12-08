import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import os

def load_data(
    features_path="voc_als_features.csv",
    targets_path="VOC-ALS.xlsx",
    test_size=0.2,
    random_state=42
):
    """
    Loads features and targets.
    Aggregates features per subject (Wide Format).
    Splits into train/val stratified by Sex.
    """
    
    # 1. Load Features
    print(f"Loading features from {features_path}...")
    df_features = pd.read_csv(features_path)
    
    # We need to pivot. 
    # Current columns: source_file, subject, sex, category, ... [features]
    # We want 1 row per subject.
    # To distinguish features from different audio files, we need a suffix.
    # 'source_file' is like 'CT001_phonationA.wav'. 
    # We can use the 'audio_type' or extract suffix from filename.
    # Let's use the 'source_file' name but stripped of the subject prefix if possible to be generic,
    # OR better: use the `task` name if available.
    # Looking at data: 'audio_type' is like 'vocal', 'ddk'. 'target_transcription' is '/a/'.
    # A cleaner key is probably the filename suffix.
    
    # Let's Create a 'TaskID' column. 
    # Example: CT001_phonationA.wav -> phonationA
    # We assume standard naming convention: {Subject}_{Task}.wav
    
    def extract_task(filename):
        # Remove extension
        base = str(filename).replace('.wav', '')
        # Remove subject prefix if present
        parts = base.split('_')
        if len(parts) > 1:
            return "_".join(parts[1:]) # e.g. phonationA
        return base
        
    df_features['task_id'] = df_features['source_file'].apply(extract_task)
    
    # Identify Feature Columns (numeric)
    # Exclude metadata
    metadata_cols = [
        'source_file', 'subject', 'sex', 'category', 'date', 'session', 
        'audio_type', 'item_id', 'target_transcription', 'task_id', 'duration'
    ]
    # Keep 'subject' for index, 'task_id' for pivoting
    feature_cols = [c for c in df_features.columns if c not in metadata_cols]
    
    # Filter only numeric columns just in case
    feature_cols = df_features[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Pivoting {len(feature_cols)} features across tasks...")
    
    # Pivot
    # index=subject, columns=task_id, values=feature_cols
    # Result columns will be MultiIndex (Feature, Task) -> Flatten
    pivot_df = df_features.pivot(index='subject', columns='task_id', values=feature_cols)
    
    # Flatten columns: {Feature}_{Task}
    pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
    pivot_df.reset_index(inplace=True)
    
    print(f"Wide format shape: {pivot_df.shape}")
    
    # 2. Load Targets
    print(f"Loading targets from {targets_path}...")
    # Header was found on row 1 (0-indexed) in preliminary analysis
    df_targets = pd.read_excel(targets_path, header=1)
    
    # Clean target columns
    df_targets.rename(columns={
        'ID': 'subject',
        'ALSFRS-R_TotalScore': 'ALSFRS_R',
        'FVC% ': 'FVC'
    }, inplace=True)
    
    # Robust renaming again
    for col in df_targets.columns:
        if "ALS" in str(col) and "TotalScore" in str(col):
            df_targets.rename(columns={col: 'ALSFRS_R'}, inplace=True)
        if "FVC" in str(col) and "%" in str(col):
            df_targets.rename(columns={col: 'FVC'}, inplace=True)
            
    # Need 'sex' for stratification. Assuming it's in df_features or df_targets.
    # df_features had 'sex'. df_targets might have it too.
    # Let's get 'sex' from df_features (first occurrence per subject)
    sex_map = df_features.drop_duplicates('subject').set_index('subject')['sex']
    
    # 3. Merge
    print("Merging data...")
    # Merge pivoted features with targets
    merged_df = pd.merge(pivot_df, df_targets[['subject', 'ALSFRS_R', 'FVC']], on='subject', how='inner')
    
    # Add sex back
    merged_df['sex'] = merged_df['subject'].map(sex_map)
    
    # Coerce targets
    merged_df['ALSFRS_R'] = pd.to_numeric(merged_df['ALSFRS_R'], errors='coerce')
    merged_df['FVC'] = pd.to_numeric(merged_df['FVC'], errors='coerce')
    
    # Drop NaNs in Targets (can't train without target)
    # NOTE: If we want to support training for FVC *or* ALSFRS_R separately, we should probably return the whole DF
    # and let the training loop drop NaNs for the specific target.
    # For splitting, we need valid data for stratification? 
    # Let's filter rows where at least one target is valid? 
    # Or just keep all and handle in training. 
    # But User requirement "for als patients".
    # We should perform the Split based on subjects available.
    
    # Remove rows with no Sex (cannot stratify)
    merged_df = merged_df.dropna(subset=['sex'])
    
    print(f"Total Subjects after merge: {len(merged_df)}")
    
    # 4. Stratified Split
    print("Performing Stratified Split by Sex...")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # We split based on the dataframe index
    # Stratify by 'sex'
    try:
        train_idx, val_idx = next(splitter.split(merged_df, merged_df['sex']))
    except ValueError as e:
        print(f"Stratification failed (maybe distinct classes < 2?): {e}")
        # Fallback to random if single sex or issue
        from sklearn.model_selection import ShuffleSplit
        splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, val_idx = next(splitter.split(merged_df))
    
    train_df = merged_df.iloc[train_idx].copy()
    val_df = merged_df.iloc[val_idx].copy()
    
    print(f"Train subjects: {len(train_df)} (Male/Female distribution kept)")
    print(f"Val subjects: {len(val_df)}")
    
    return train_df, val_df

if __name__ == "__main__":
    # Test run
    try:
        load_data()
    except Exception as e:
        print(f"Error: {e}")
