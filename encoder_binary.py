import pandas as pd
import numpy as np

# 1. Configuration: WRF Physics Suites
physics_options = {
    "mp_physics": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17],
    "cu_physics": [1, 2, 3, 5, 6, 7, 10, 11, 14, 16, 93, 99],
    "bl_pbl_physics": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 99],
    "ra_lw_physics": [1, 3, 4, 5, 7, 99],
    "ra_sw_physics": [1, 2, 3, 4, 5, 7, 99],
    "sf_sfclay_physics": [1, 2, 3, 4, 10],
    "sf_surface_physics": [1, 2, 3, 4, 5]
}

def generate_encoded_csv(input_file, output_file, options):
    # Load the original data
    df = pd.read_csv(input_file)
    
    # Create a copy to store results, keeping non-physics columns (like precipitation)
    encoded_df = df.copy()
    
    for col, values in options.items():
        if col not in df.columns:
            print(f"Warning: {col} not found in CSV. Skipping...")
            continue
            
        # Map WRF integers to zero-based indices (0, 1, 2...)
        mapping = {val: i for i, val in enumerate(values)}
        mapped_series = df[col].map(mapping)
        
        # Determine number of bits required for this category
        n_bits = int(np.ceil(np.log2(len(values))))
        
        # Create binary columns
        for i in range(n_bits):
            # Extract i-th bit
            bit_col_name = f"{col}_b{i}"
            encoded_df[bit_col_name] = mapped_series.apply(
                lambda x: (int(x) >> i) & 1 if pd.notnull(x) else 0
            )
        
        # Remove the original categorical integer column
        encoded_df.drop(columns=[col], inplace=True)
        
    # Save the final encoded dataset
    encoded_df.to_csv(output_file, index=False)
    print(f"Success! Encoded dataset saved as: {output_file}")
    print(f"Original columns: {len(df.columns)}")
    print(f"Encoded columns: {len(encoded_df.columns)}")

# --- EXECUTION ---
if __name__ == "__main__":
    # Ensure 'dataset.csv' is in your current folder
    generate_encoded_csv('for_binary.csv', 'dataset_encoded.csv', physics_options)
