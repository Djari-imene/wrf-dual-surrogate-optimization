import pandas as pd
import numpy as np

# 1. Load your data
# Using the values from your screenshots
df = pd.read_csv('dataset_principale.csv')
observed = np.array([0, 0.79, 0, 6.97, 0, 2.26, 0.62, 6.65, 12.76, 10.33]) # day1, day2, day3, day4

def calculate_metrics(predicted, observed, threshold=1):
    """
    Calculates RMSE and ETS for a single row of predictions.
    Threshold is the rain/no-rain limit (default 0.1mm).
    """
    # --- RMSE Calculation ---
    rmse = np.sqrt(np.mean((predicted - observed)**2))
    
    # --- ETS Calculation ---
    # Convert to binary (Rain/No Rain) based on threshold
    pred_bin = predicted >= threshold
    obs_bin = observed >= threshold
    
    # Contingency Table elements
    hits = np.sum(pred_bin & obs_bin)
    misses = np.sum(~pred_bin & obs_bin)
    false_alarms = np.sum(pred_bin & ~obs_bin)
    correct_negatives = np.sum(~pred_bin & ~obs_bin)
    total = len(predicted)
    
    # Random hits (expected hits by chance)
    hits_random = ((hits + misses) * (hits + false_alarms)) / total
    
    # ETS Formula
    num = (hits - hits_random)
    den = (hits + misses + false_alarms - hits_random)
    
    # Handle division by zero for dry cases
    ets = num / den if den != 0 else 0
    
    return rmse, ets

# 2. Apply calculations to the dataset
results = []
for index, row in df.iterrows():
    # Extract just the precipitation columns (day1 to day4)
    pred_values = row[['day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'day7', 'day8', 'day9', 'day10']].values.astype(float)
    
    rmse, ets = calculate_metrics(pred_values, observed)
    results.append({'RMSE': rmse, 'ETS': ets})

# 3. Merge back to original dataframe
metrics_df = pd.DataFrame(results)
final_df = pd.concat([df, metrics_df], axis=1)

# Display result
print(final_df[['mp_physics', 'cu_physics', 'RMSE', 'ETS']].head())

# Save for your ML Surrogate training
final_df.to_csv('dataset_with_targets.csv', index=False)
