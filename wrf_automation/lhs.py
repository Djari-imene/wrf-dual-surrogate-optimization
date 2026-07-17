import numpy as np
import pandas as pd
from scipy.stats import qmc

# Define the physics parameterizations
physics_params = {
    "mp_physics": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17],
    "cu_physics": [1, 2, 3, 5, 6, 7, 10, 11, 14, 16, 93, 99],
    "bl_pbl_physics": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 99],
    "ra_lw_physics": [1, 3, 4, 5, 7, 99],
    "ra_sw_physics": [1, 2, 3, 4, 5, 7, 99],
    "sf_sfclay_physics": [1, 2, 3, 4, 10],
    "sf_surface_physics": [1, 2, 3, 4, 5]
}

# Calculate total possible combinations
total_combinations = np.prod([len(options) for options in physics_params.values()])
print(f"Total possible combinations: {total_combinations}")

# Enhanced LHS method
def create_enhanced_lhs_design(physics_params, n_samples=700, seed=42):
    """
    Create enhanced LHS design ensuring better coverage of parameter space
    """
    np.random.seed(seed)
    
    param_names = list(physics_params.keys())
    n_params = len(param_names)
    
    # Create LHS samples
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    lhs_samples = sampler.random(n=n_samples)
    
    param_combinations = []
    
    for sample in lhs_samples:
        combination = {}
        for i, param_name in enumerate(param_names):
            options = physics_params[param_name]
            n_options = len(options)
            
            # More precise mapping to ensure even coverage
            # Map [0,1] to [0, n_options) then to integer indices
            continuous_index = sample[i] * n_options
            # Use rounding for better distribution
            index = int(np.round(continuous_index - 0.5))
            # Handle edge cases
            index = max(0, min(index, n_options - 1))
            combination[param_name] = options[index]
        
        param_combinations.append(combination)
    
    return param_combinations

# Generate the enhanced LHS design
print("Generating Enhanced LHS design...")

# You can adjust n_samples based on your computational budget
enhanced_lhs_combinations = create_enhanced_lhs_design(physics_params, n_samples=700, seed=42)

# Convert to DataFrame for easier handling
def combinations_to_dataframe(combinations, name):
    df = pd.DataFrame(combinations)
    df['design_type'] = name
    return df

# Create DataFrame
enhanced_lhs_df = combinations_to_dataframe(enhanced_lhs_combinations, 'enhanced_lhs')

print(f"\nGenerated {len(enhanced_lhs_combinations)} parameter combinations")

# Display sample of the design matrix
print(f"\nSample of parameter combinations:")
print(enhanced_lhs_df.head(15))

# Save to CSV
enhanced_lhs_df.to_csv('wrf_enhanced_lhs_design_matrix.csv', index=False)
print(f"\nDesign matrix saved to 'wrf_enhanced_lhs_design_matrix.csv'")

# Analysis functions
def analyze_design_coverage(df, physics_params):
    """
    Analyze how well the design covers the parameter space
    """
    print("\nParameter Coverage Analysis:")
    print("=" * 50)
    
    for param_name, options in physics_params.items():
        covered_options = set(df[param_name].unique())
        total_options = set(options)
        coverage_ratio = len(covered_options) / len(total_options)
        
        print(f"{param_name}:")
        print(f"  Options covered: {len(covered_options)}/{len(total_options)} ({coverage_ratio:.1%})")
        if len(total_options - covered_options) > 0:
            print(f"  Missing options: {sorted(total_options - covered_options)}")
        print()

# Analyze coverage
analyze_design_coverage(enhanced_lhs_df, physics_params)

# Function to validate combinations
def validate_combinations(df, physics_params):
    """
    Validate that all combinations use valid parameter values
    """
    invalid_combinations = []
    
    for idx, row in df.iterrows():
        for param_name, valid_options in physics_params.items():
            if row[param_name] not in valid_options:
                invalid_combinations.append((idx, param_name, row[param_name], valid_options))
    
    if invalid_combinations:
        print("Invalid combinations found:")
        for idx, param, value, valid in invalid_combinations[:5]:  # Show first 5
            print(f"  Row {idx}: {param} = {value} (valid: {valid})")
        return False
    else:
        print("All combinations are valid!")
        return True

# Validate the design
is_valid = validate_combinations(enhanced_lhs_df, physics_params)

# Generate WRF namelist snippets for each combination
def generate_namelist_snippets(df, filename='wrf_namelist_snippets.txt'):
    """
    Generate WRF namelist.input snippets for each parameter combination
    """
    with open(filename, 'w') as f:
        for idx, row in df.iterrows():
            f.write(f"! Enhanced LHS Combination {idx + 1}\n")
            f.write("&physics\n")
            for param_name in physics_params.keys():
                f.write(f"  {param_name} = {row[param_name]},\n")
            f.write("/\n\n")
    
    print(f"WRF namelist snippets saved to '{filename}'")

# Generate namelist snippets
generate_namelist_snippets(enhanced_lhs_df)

# Summary statistics
print("\nEnhanced LHS Design Summary:")
print("=" * 40)
print(f"Total parameter combinations: {len(enhanced_lhs_df)}")
print(f"Parameter dimensions: {len(physics_params)}")
print("\nParameter value ranges:")
for param, values in physics_params.items():
    print(f"  {param}: {len(values)} options (min: {min(values)}, max: {max(values)})")

# Check for duplicates
duplicates = enhanced_lhs_df.duplicated(subset=list(physics_params.keys()), keep=False)
n_duplicates = duplicates.sum()
print(f"\nDuplicate combinations: {n_duplicates}")

if n_duplicates > 0:
    print("Removing duplicates...")
    enhanced_lhs_df_clean = enhanced_lhs_df.drop_duplicates(subset=list(physics_params.keys())).reset_index(drop=True)
    print(f"Final combinations after removing duplicates: {len(enhanced_lhs_df_clean)}")
    
    # Save cleaned version
    enhanced_lhs_df_clean.to_csv('wrf_enhanced_lhs_design_matrix_clean.csv', index=False)
    print(f"Clean design matrix saved to 'wrf_enhanced_lhs_design_matrix_clean.csv'")
    
    # Generate namelist snippets for clean version
    generate_namelist_snippets(enhanced_lhs_df_clean, 'wrf_namelist_snippets_clean.txt')
