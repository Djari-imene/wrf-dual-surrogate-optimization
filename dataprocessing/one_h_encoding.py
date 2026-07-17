# one_hot_encode_physics.py

import pandas as pd

# === 1. Load the dataset ===
file_path = 'dataset_with_targets.csv'
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# === 2. Define physics parameters and their possible values ===
physics_options = {
    "mp_physics": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17],
    "cu_physics": [1, 2, 3, 5, 6, 7, 10, 11, 14, 16, 93, 99],
    "bl_pbl_physics": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 99],
    "ra_lw_physics": [1, 3, 4, 5, 7, 99],
    "ra_sw_physics": [1, 2, 3, 4, 5, 7, 99],
    "sf_sfclay_physics": [1, 2, 3, 4, 10],
    "sf_surface_physics": [1, 2, 3, 4, 5,]
}

# === 3. Separate target variable ===
#if 'precipitation' in df.columns:
 #   y = df['precipitation']
  #  X = df.drop('precipitation', axis=1)
#else:
 #   X = df.copy()
  #  y = None

# === 3. Separate target variable ===
#if 'rmse_prec' in df.columns:
#    y = df['rmse_prec']
#    X = df.drop(['precipitation', 'rmse_prec'], axis=1)
#else:
#    X = df.copy()
#    y = None

if 'ETS' in df.columns:
    y = df['ETS']
    drop_cols = ['ETS']
    if 'RMSE' in df.columns:
        drop_cols.append('RMSE')
    X = df.drop(drop_cols, axis=1)
else:
    X = df.copy()
    y = None


# === 4. Apply one-hot encoding with explicit categories ===
X_encoded = pd.DataFrame(index=X.index)

for col in X.columns:
    if col in physics_options:
        # Convert to categorical with predefined categories
        cat_series = pd.Categorical(X[col], categories=physics_options[col])
        # Create one-hot dummies
        dummies = pd.get_dummies(cat_series, prefix=col)
        X_encoded = pd.concat([X_encoded, dummies], axis=1)
    else:
        # If not in physics list, just encode normally
        dummies = pd.get_dummies(X[col], prefix=col)
        X_encoded = pd.concat([X_encoded, dummies], axis=1)

print(f"\nEncoded features shape: {X_encoded.shape}")
print("Encoded feature columns sample:")
print(X_encoded.columns.tolist())

# === 5. Reattach precipitation if available ===
if y is not None:
    df_final = pd.concat([X_encoded, y.reset_index(drop=True)], axis=1)
    print(f"\nFinal dataset (with precipitation): {df_final.shape}")
else:
    df_final = X_encoded

# === 6. Save to CSV ===
output_file = 'dataset_one_hot_encoded.csv'
df_final.to_csv(output_file, index=False)
print(f"\n One-hot encoded dataset saved to '{output_file}'")

# Optional: Save feature list for reference
features_file = 'encoded_features.txt'
with open(features_file, 'w') as f:
    for feature in df_final.columns:
        f.write(feature + '\n')
print(f" List of encoded features saved to '{features_file}'")
