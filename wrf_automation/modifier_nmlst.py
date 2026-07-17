import re
import os
import pandas as pd

def modify_namelist(config, input_file, output_file):
    """
    Modifies WRF namelist.input with provided parameter values.
    
    Args:
        config (dict): Dictionary of parameter names and values.
        input_file (str): Path to original namelist.input.
        output_file (str): Path to save modified namelist.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input namelist file '{input_file}' not found.")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    str_config = {k: str(v) for k, v in config.items()}
    param_pattern = re.compile(r'^\s*([a-zA-Z0-9_]+)\s*=')

    new_lines = []
    in_physics_section = False
    
    for line in lines:
        # Check if we're entering or leaving the physics section
        if '&physics' in line:
            in_physics_section = True
        elif '/' in line and in_physics_section and not '=' in line:
            in_physics_section = False
            
        match = param_pattern.match(line)
        if match and in_physics_section:
            key = match.group(1).strip()
            if key in str_config:
                # Replace the line with the new value from config
                new_line = f" {key} = {str_config[key]},\n"
                new_lines.append(new_line)
                continue
        new_lines.append(line)

    with open(output_file, 'w') as f:
        f.writelines(new_lines)

    print(f" Updated namelist written to {output_file}")

def generate_namelists_from_population(
    population_csv='wrf_enhanced_lhs_design_matrix.csv',
    namelist_template='namelist.input',
    runs_dir='wrf_runs'
):
    """
    Generates a namelist.input file for each individual in the population.
    Compatible with your existing function signature.
    """
    # Load population data
    df = pd.read_csv(population_csv)

    # Remove design_type column if it exists
    if 'design_type' in df.columns:
        df = df.drop('design_type', axis=1)

    # Create main run directory if it doesn't exist
    os.makedirs(runs_dir, exist_ok=True)

    for index, row in df.iterrows():
        run_id = f"run_{index + 1:04d}"  # e.g., run_0001, run_0002, ...
        print(f" Generating namelist for {run_id}")

        # Create individual run directory
        run_dir = os.path.join(runs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Convert row to dict and keep only physics parameters
        physics_params = [
            "mp_physics", "cu_physics", "bl_pbl_physics", 
            "ra_lw_physics", "ra_sw_physics", "sf_surface_physics"
        ]
        
        config = {k: v for k, v in row.to_dict().items() if k in physics_params}

        # Add conditional logic for sf_sfclay_physics
        bl_pbl_value = config.get('bl_pbl_physics', 1)
        
        # If bl_pbl_physics is 2, 3, 4, or 10, set sf_sfclay_physics equal to bl_pbl_physics
        # Otherwise, set sf_sfclay_physics to 1
        if bl_pbl_value in [2, 3, 4, 10]:
            config['sf_sfclay_physics'] = bl_pbl_value
        else:
            config['sf_sfclay_physics'] = 1

        # Modify namelist
        output_namelist = os.path.join(run_dir, 'namelist.input')
        modify_namelist(config, namelist_template, output_namelist)

    print(f" All namelists generated under '{runs_dir}/'")

# Alternative version with more detailed logging
def generate_namelists_from_population_detailed(
    population_csv='wrf_enhanced_lhs_design_matrix.csv',
    namelist_template='namelist.input',
    runs_dir='wrf_runs'
):
    """
    Generates namelists with detailed logging of sf_sfclay_physics assignments.
    """
    # Load population data
    df = pd.read_csv(population_csv)

    # Remove design_type column if it exists
    if 'design_type' in df.columns:
        df = df.drop('design_type', axis=1)

    # Create main run directory if it doesn't exist
    os.makedirs(runs_dir, exist_ok=True)

    pbl_values_requiring_matching_sfclay = [2, 3, 4, 10]
    
    for index, row in df.iterrows():
        run_id = f"run_{index + 1:04d}"
        
        # Convert row to dict and keep only physics parameters
        physics_params = [
            "mp_physics", "cu_physics", "bl_pbl_physics", 
            "ra_lw_physics", "ra_sw_physics", "sf_surface_physics"
        ]
        
        config = {k: v for k, v in row.to_dict().items() if k in physics_params}
        bl_pbl_value = config.get('bl_pbl_physics', 1)

        # Apply conditional logic for sf_sfclay_physics
        if bl_pbl_value in pbl_values_requiring_matching_sfclay:
            config['sf_sfclay_physics'] = bl_pbl_value
            print(f" Generating namelist for {run_id} - bl_pbl_physics={bl_pbl_value}, sf_sfclay_physics={bl_pbl_value}")
        else:
            config['sf_sfclay_physics'] = 1
            print(f" Generating namelist for {run_id} - bl_pbl_physics={bl_pbl_value}, sf_sfclay_physics=1")

        # Create individual run directory
        run_dir = os.path.join(runs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Modify namelist
        output_namelist = os.path.join(run_dir, 'namelist.input')
        modify_namelist(config, namelist_template, output_namelist)

    # Summary statistics
    bl_pbl_values = df['bl_pbl_physics'].values
    matching_count = sum(1 for val in bl_pbl_values if val in pbl_values_requiring_matching_sfclay)
    default_count = len(bl_pbl_values) - matching_count
    
    print(f"\n sf_sfclay_physics Assignment Summary:")
    print(f"   Matching PBL values (2,3,4,10): {matching_count} runs")
    print(f"   Default value (1): {default_count} runs")
    print(f" All namelists generated under '{runs_dir}/'")

# Example usage
if __name__ == "__main__":
    generate_namelists_from_population(
        population_csv='wrf_enhanced_lhs_design_matrix.csv',  # Your LHS file
        namelist_template='namelist.input',
        runs_dir='wrf_runs'
    )
