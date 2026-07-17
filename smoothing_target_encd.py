import pandas as pd
import numpy as np
import os

# ----------------------------
# CONFIGURATION
# ----------------------------
INPUT_CSV = "dataset_with_targets.csv"          
OUTPUT_ENCODING_TABLE = "encoding_lookup_table_rmse.csv"
OUTPUT_ENCODED_DATA = "dataset_encoded_target.csv"
SMOOTHING_ALPHA = 10                    # Smoothing parameter (α)

# Physics columns (must match your dataset)
PHYSICS_COLS = [
    'mp_physics',
    'cu_physics',
    'bl_pbl_physics',
    'ra_lw_physics',
    'ra_sw_physics',
    'sf_sfclay_physics',
    'sf_surface_physics'
]

TARGET_COL = 'RMSE'   #Change to 'ETS'

# ----------------------------
# LOAD DATA
# ----------------------------
print("Loading dataset...")
df = pd.read_csv(INPUT_CSV)

# Validate required columns
missing_cols = [col for col in PHYSICS_COLS + [TARGET_COL] if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ----------------------------
# COMPUTE GLOBAL MEAN
# ----------------------------
global_mean = df[TARGET_COL].mean()
print(f"Global mean RMSE: {global_mean:.4f}")

# ----------------------------
# BUILD ENCODING LOOKUP TABLE
# ----------------------------
all_encodings = []

for col in PHYSICS_COLS:
    print(f"\nProcessing column: {col}")
    
    # Group by category
    group_stats = df.groupby(col)[TARGET_COL].agg(['mean', 'count']).reset_index()
    group_stats.columns = [col, 'mean_rmse', 'count']
    
    # Apply smoothing: (n * mean + α * global_mean) / (n + α)
    group_stats['encoded_value'] = (
        (group_stats['count'] * group_stats['mean_rmse'] + SMOOTHING_ALPHA * global_mean) /
        (group_stats['count'] + SMOOTHING_ALPHA)
    )
    
    # Add column identifier
    group_stats['parameter'] = col
    
    # Reorder for clarity
    group_stats = group_stats[['parameter', col, 'count', 'mean_rmse', 'encoded_value']]
    group_stats.rename(columns={col: 'scheme_id'}, inplace=True)
    
    all_encodings.append(group_stats)
    
    print(f"  - Unique schemes: {len(group_stats)}")
    print(f"  - Min encoding: {group_stats['encoded_value'].min():.4f}")
    print(f"  - Max encoding: {group_stats['encoded_value'].max():.4f}")

# Combine all into one table
encoding_table = pd.concat(all_encodings, ignore_index=True)

# Add 'unseen' fallback for each parameter
unseen_rows = []
for col in PHYSICS_COLS:
    unseen_rows.append({
        'parameter': col,
        'scheme_id': 'unseen',
        'count': 0,
        'mean_rmse': np.nan,
        'encoded_value': global_mean
    })

unseen_df = pd.DataFrame(unseen_rows)
encoding_table = pd.concat([encoding_table, unseen_df], ignore_index=True)

# Sort for readability
encoding_table = encoding_table.sort_values(['parameter', 'scheme_id']).reset_index(drop=True)

# ----------------------------
# SAVE ENCODING TABLE
# ----------------------------
encoding_table.to_csv(OUTPUT_ENCODING_TABLE, index=False)
print(f"\n Encoding lookup table saved to: {OUTPUT_ENCODING_TABLE}")

# ----------------------------
# (OPTIONAL) ENCODE FULL DATASET
# ----------------------------
def encode_row(row, enc_dict):
    """Encode a single row using precomputed encoding dictionary"""
    encoded = []
    for col in PHYSICS_COLS:
        scheme = row[col]
        # Use encoding if known, else 'unseen'
        if scheme in enc_dict[col]:
            encoded.append(enc_dict[col][scheme])
        else:
            encoded.append(enc_dict[col]['unseen'])
    return encoded

# Build fast lookup dict: {param -> {scheme_id -> encoded_value}}
enc_dict = {}
for param in PHYSICS_COLS:
    sub = encoding_table[encoding_table['parameter'] == param]
    enc_dict[param] = dict(zip(sub['scheme_id'], sub['encoded_value']))

# Encode entire dataset
print("\nEncoding full dataset...")
encoded_data = []
for _, row in df.iterrows():
    encoded_row = encode_row(row, enc_dict)
    encoded_data.append(encoded_row)

X_encoded = pd.DataFrame(encoded_data, columns=PHYSICS_COLS)
X_encoded[TARGET_COL] = df[TARGET_COL].values  

X_encoded.to_csv(OUTPUT_ENCODED_DATA, index=False)
print(f" Encoded dataset saved to: {OUTPUT_ENCODED_DATA}")

# ----------------------------
# SUMMARY
# ----------------------------
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Input file: {INPUT_CSV}")
print(f"Smoothing alpha: {SMOOTHING_ALPHA}")
print(f"Global mean RMSE: {global_mean:.4f}")
print(f"Encoding table shape: {encoding_table.shape}")
print(f"Encoded data shape: {X_encoded.shape}")
print("\nSample of encoding table:")
print(encoding_table.head(10))
