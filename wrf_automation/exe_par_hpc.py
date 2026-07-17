import os
import subprocess
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import glob
import shutil
import multiprocessing
import time
import json

# --- Configuration ---
PATH = '/home/imene.std/WRF/GA/surrogate2/'
WPS_DIR = '/home/imene.std/WRF/GA/surrogate2/WPS-4.2/'
WRF_DIR = '/home/imene.std/WRF/GA/surrogate2/WRF-4.2.1/test/em_real/'
DATA_DIR = '/home/imene.std/WRF/DATA/2025/05/1/06Z/'
RUNS_DIR = os.path.join(PATH, 'wrf_runs')
OUTPUT_CSV = os.path.join(PATH, 'precipitation_results.csv')
# Log file to track execution status
STATUS_LOG = os.path.join(PATH, 'wrf_execution_status.json')

# --- Station coordinates ---
STATION_LAT = 35.75
STATION_LON = 6.17
STATION_ID = 'Batna_05'

# --- WRF Simulation Details ---
WRFOUT_PATTERN = 'wrfout_d01_2025-05-01_06:00:00'
NUM_PROCS = 28

slurm_ntasks = int(os.getenv('SLURM_NTASKS', 0))
if slurm_ntasks != 56:
    print(f" Warning: SLURM_NTASKS={slurm_ntasks}, expected 28")

# --- Status Tracking Functions ---

def update_status_log(run_index, run_dir_name, status, precip_value=None, start_time=None, end_time=None):
    """
    Update the status log with execution information.
    """
    if not os.path.exists(STATUS_LOG):
        status_data = {}
    else:
        try:
            with open(STATUS_LOG, 'r') as f:
                status_data = json.load(f)
        except:
            status_data = {}
    
    run_key = f"run_{run_index:04d}"
    
    if run_key not in status_data:
        status_data[run_key] = {
            'run_directory': run_dir_name,
            'status_history': []
        }
    
    # Create status entry
    status_entry = {
        'status': status,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if start_time:
        status_entry['start_time'] = start_time
    if end_time:
        status_entry['end_time'] = end_time
    if precip_value is not None:
        status_entry['precipitation_mm'] = precip_value
    
    status_data[run_key]['status_history'].append(status_entry)
    
    # Keep only the last 5 status entries
    if len(status_data[run_key]['status_history']) > 5:
        status_data[run_key]['status_history'] = status_data[run_key]['status_history'][-5:]
    
    # Update current status
    status_data[run_key]['current_status'] = status
    if precip_value is not None:
        status_data[run_key]['last_precipitation'] = precip_value
    
    with open(STATUS_LOG, 'w') as f:
        json.dump(status_data, f, indent=2)

def display_currently_executing(run_index, run_dir_name, step=None):
    """
    Display which WRF run is currently being executed.
    """
    timestamp = time.strftime('%H:%M:%S')
    if step:
        print(f"\n{'='*60}")
        print(f"[{timestamp}] EXECUTING: Run {run_index:04d} ({run_dir_name}) - {step}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"[{timestamp}] STARTING: Run {run_index:04d} ({run_dir_name})")
        print(f"{'='*60}")

def save_individual_result(run_index, run_dir_name, precip_value):
    """
    Save precipitation result for an individual run immediately after completion.
    """
    result = {
        'run_index': run_index,
        'run_directory': run_dir_name,
        'precipitation_mm': precip_value,
        'extraction_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'station_id': STATION_ID,
        'station_lat': STATION_LAT,
        'station_lon': STATION_LON
    }
    
    # Create DataFrame for this single result
    result_df = pd.DataFrame([result])
    
    # Check if output file exists
    if os.path.exists(OUTPUT_CSV):
        try:
            # Read existing results
            existing_df = pd.read_csv(OUTPUT_CSV)
            # Check if this run already exists in the results
            if existing_df[existing_df['run_index'] == run_index].empty:
                # Append new result
                combined_df = pd.concat([existing_df, result_df], ignore_index=True)
                combined_df.to_csv(OUTPUT_CSV, index=False)
                print(f"   Appended result for Run {run_index:04d} to {OUTPUT_CSV}")
            else:
                # Update existing result
                mask = existing_df['run_index'] == run_index
                existing_df.loc[mask, list(result.keys())] = list(result.values())
                existing_df.to_csv(OUTPUT_CSV, index=False)
                print(f"   Updated result for Run {run_index:04d} in {OUTPUT_CSV}")
        except Exception as e:
            print(f"   Could not append to existing CSV: {e}")
            # Save as new file
            result_df.to_csv(OUTPUT_CSV, index=False)
            print(f"   Created new results file: {OUTPUT_CSV}")
    else:
        # Create new file
        result_df.to_csv(OUTPUT_CSV, index=False)
        print(f"   Created results file: {OUTPUT_CSV}")

# --- Modified Functions ---

def link_vtable(wps_dir):
    """Create a symbolic link to Vtable.GFS."""
    vtable_source = os.path.join(wps_dir, 'ungrib', 'Variable_Tables', 'Vtable.GFS')
    vtable_target = os.path.join(wps_dir, 'Vtable')
    if os.path.islink(vtable_target):
        os.unlink(vtable_target)
    elif os.path.isfile(vtable_target):
        os.remove(vtable_target)
    os.symlink(vtable_source, vtable_target)
    print(f" Created symbolic link: {vtable_target} → {vtable_source}")

def link_wrf_input_files(source_dir, target_dir):
    """Create symbolic links to all files except namelist.input."""
    print(f" Linking all files (except namelist.input) from: {source_dir} → {target_dir}")
    for filename in os.listdir(source_dir):
        if filename == 'namelist.input':
            print(f"  Skipped: {filename} (will be generated separately)")
            continue
        src_path = os.path.join(source_dir, filename)
        if os.path.isdir(src_path):
            continue
        if filename.startswith('.'):
            continue
        dst_path = os.path.join(target_dir, filename)
        if os.path.islink(dst_path):
            os.unlink(dst_path)
        elif os.path.exists(dst_path):
            os.remove(dst_path)
        os.symlink(src_path, dst_path)
        print(f"🔗 Linked: {filename}")

def run_wps():
    """Run WPS steps."""
    os.chdir(WPS_DIR)
    print(" Running geogrid...")
    subprocess.run('./geogrid.exe', shell=True, check=True)
    print(" Linking GRIB files...")
    subprocess.run(f'./link_grib.csh {DATA_DIR}gfs*', shell=True, check=True)
    link_vtable(WPS_DIR)
    print(" Running ungrib...")
    subprocess.run('./ungrib.exe', shell=True, check=True)
    print(" Running metgrid...")
    subprocess.run('./metgrid.exe', shell=True, check=True)

def extract_precip_at_point(wrf_file, lat_station, lon_station, id_station, run_index):
    """Extracts total accumulated precipitation at a given point."""
    try:
        print(f"  Extracting precipitation from {os.path.basename(wrf_file)} for Run {run_index:03d}...")
        ds = Dataset(wrf_file)
        xlat = ds.variables['XLAT'][:]
        xlong = ds.variables['XLONG'][:]

        if len(xlat.shape) == 3:
            xlat = xlat[0, :, :]
            xlong = xlong[0, :, :]
        elif len(xlat.shape) == 4:
            xlat = xlat[0, 0, :, :]
            xlong = xlong[0, 0, :, :]

        dist = np.sqrt((xlat - lat_station)**2 + (xlong - lon_station)**2)
        i_point, j_point = np.unravel_index(np.argmin(dist), dist.shape)

        rainc = ds.variables['RAINC'][:, i_point, j_point]
        rainnc = ds.variables['RAINNC'][:, i_point, j_point]

        try:
            rainsh = ds.variables['RAINSH'][:, i_point, j_point]
        except KeyError:
            rainsh = np.zeros_like(rainc)

        precip_total = rainc + rainnc + rainsh
        precip_final_mm = precip_total[-1]
        extracted_lat = xlat[i_point, j_point]
        extracted_lon = xlong[i_point, j_point]

        ds.close()

        print(f"   Run {run_index:03d}: Extracted {precip_final_mm:.2f} mm at grid point (lat={extracted_lat:.4f}, lon={extracted_lon:.4f})")
        return precip_final_mm

    except Exception as e:
        print(f"   Error extracting precipitation for Run {run_index:03d}: {e}")
        return np.nan

def run_wrf_case(run_dir, run_index, run_dir_name):
    """
    Executes the WRF simulation for a single run directory.
    """
    original_cwd = os.getcwd()
    os.chdir(run_dir)
    
    # Update status log
    start_time = time.strftime('%Y-%m-%d %H:%M:%S')
    update_status_log(run_index, run_dir_name, 'starting', start_time=start_time)
    
    # Display execution info
    display_currently_executing(run_index, run_dir_name)

    # Link met_em files
    met_files = [os.path.join(WPS_DIR, f) for f in os.listdir(WPS_DIR) if f.startswith('met_em.')]
    for f in met_files:
        link_name = os.path.basename(f)
        if os.path.exists(link_name):
            os.remove(link_name)
        os.symlink(f, link_name)

    try:
        # --- Run real.exe ---
        display_currently_executing(run_index, run_dir_name, 'real.exe')
        update_status_log(run_index, run_dir_name, 'running_real.exe')
        
        result_real = subprocess.run(['./real.exe'], capture_output=True, text=True)
        if result_real.returncode != 0:
            print(f"   real.exe failed for Run {run_index:03d}!")
            update_status_log(run_index, run_dir_name, 'failed_real.exe')
            return {'run_index': run_index, 'run_directory': run_dir_name, 'precipitation_mm': None}

        # --- Run wrf.exe ---
        display_currently_executing(run_index, run_dir_name, 'wrf.exe')
        update_status_log(run_index, run_dir_name, 'running_wrf.exe')
        
        cmd_wrf = ['mpirun', '-np', str(NUM_PROCS), './wrf.exe']
        result_wrf = subprocess.run(cmd_wrf, capture_output=True, text=True)
        
        if result_wrf.returncode != 0:
            print(f"   wrf.exe failed for Run {run_index:04d}!")
            update_status_log(run_index, run_dir_name, 'failed_wrf.exe')
            return {'run_index': run_index, 'run_directory': run_dir_name, 'precipitation_mm': None}

        print(f"   WRF simulation completed successfully for Run {run_index:04d}.")
        update_status_log(run_index, run_dir_name, 'simulation_completed')

        # --- Find and process WRF output ---
        wrf_out_pattern_full = os.path.join(run_dir, WRFOUT_PATTERN)
        wrf_out_files = glob.glob(wrf_out_pattern_full)

        if not wrf_out_files:
            print(f"   WRF output file not found for Run {run_index:04d}.")
            update_status_log(run_index, run_dir_name, 'no_output_file')
            return {'run_index': run_index, 'run_directory': run_dir_name, 'precipitation_mm': None}

        wrf_out_file = wrf_out_files[0]
        print(f"  Found WRF output: {os.path.basename(wrf_out_file)}")

        # --- Extract precipitation ---
        display_currently_executing(run_index, run_dir_name, 'extracting_precipitation')
        update_status_log(run_index, run_dir_name, 'extracting_precipitation')
        
        precip_value = extract_precip_at_point(wrf_out_file, STATION_LAT, STATION_LON, STATION_ID, run_index)
        
        # Save result immediately
        if not np.isnan(precip_value):
            save_individual_result(run_index, run_dir_name, precip_value)
        
        # Update status log with final result
        end_time = time.strftime('%Y-%m-%d %H:%M:%S')
        update_status_log(run_index, run_dir_name, 'completed', 
                         precip_value=float(precip_value) if not np.isnan(precip_value) else None,
                         end_time=end_time)
        
        return {
            'run_index': run_index,
            'run_directory': run_dir_name,
            'precipitation_mm': precip_value if not np.isnan(precip_value) else None
        }

    except Exception as e:
        print(f"  Unexpected error during WRF run {run_index:03d}: {e}")
        update_status_log(run_index, run_dir_name, 'error', 
                         precip_value=None,
                         end_time=time.strftime('%Y-%m-%d %H:%M:%S'))
        return {'run_index': run_index, 'run_directory': run_dir_name, 'precipitation_mm': None}

    finally:
        os.chdir(original_cwd)
        print(f"  Run {run_index:04d} execution completed at {time.strftime('%H:%M:%S')}\n")

# --- Main Workflow ---
if __name__ == "__main__":
    # Initialize status log
    if os.path.exists(STATUS_LOG):
        print(f" Loading existing status log: {STATUS_LOG}")
    else:
        print(f" Creating new status log: {STATUS_LOG}")
        with open(STATUS_LOG, 'w') as f:
            json.dump({}, f)

    print(" WRF Batch Runner & Precipitation Extractor")
    print(f" Results will be saved to: {OUTPUT_CSV}")
    print(f" Execution status will be logged to: {STATUS_LOG}")
    
    # Ensure output directory exists
    os.makedirs(RUNS_DIR, exist_ok=True)
    
    # Step 1: Run WPS only once
    print("\n" + "="*60)
    print(" Initiating WPS run...")
    print("="*60)
    run_wps()
    print(" WPS run completed.\n")
    
    # Step 2: Find run directories
    print(f" Looking for run directories in: {RUNS_DIR}")
    run_dirs = [d for d in os.listdir(RUNS_DIR)
                if os.path.isdir(os.path.join(RUNS_DIR, d)) and d.startswith('run_')]
    run_dirs.sort()
    
    # Filter for runs >= run_0001
    run_dirs = [d for d in run_dirs if d >= 'run_0219']
    
    if not run_dirs:
        print(f" No directories matching 'run_*' found in {RUNS_DIR}. Exiting.")
        exit(1)
    
    num_runs = len(run_dirs)
    print(f" Found {num_runs} run directories.")
    print(f" Will process: {', '.join(run_dirs[:5])}{'...' if len(run_dirs) > 5 else ''}\n")
    
    # Step 3: Prepare run directories
    print("  Preparing run directories...")
    for i, run_dir_name in enumerate(run_dirs):
        run_index = i + 1
        run_dir_path = os.path.join(RUNS_DIR, run_dir_name)
        print(f"  Preparing {run_dir_name}...")
        
        real_exe_target = os.path.join(run_dir_path, 'real.exe')
        wrf_exe_target = os.path.join(run_dir_path, 'wrf.exe')
        
        if not os.path.exists(real_exe_target):
            shutil.copy2(os.path.join(WRF_DIR, 'real.exe'), real_exe_target)
        if not os.path.exists(wrf_exe_target):
            shutil.copy2(os.path.join(WRF_DIR, 'wrf.exe'), wrf_exe_target)
        
        link_wrf_input_files(WRF_DIR, run_dir_path)
        update_status_log(run_index, run_dir_name, 'prepared')
    
    print(" All run directories prepared.\n")
    
    # Step 4: Run WRF cases in parallel
    total_cores = 140
    max_parallel = total_cores // NUM_PROCS
    print(f" Running up to {max_parallel} WRF simulations in parallel...")
    print(f" Each run uses {NUM_PROCS} cores, total available: {total_cores}\n")
    
    # Prepare arguments for parallel execution
    args = []
    for i, run_dir_name in enumerate(run_dirs):
        run_index = i + 1
        run_dir_path = os.path.join(RUNS_DIR, run_dir_name)
        args.append((run_dir_path, run_index, run_dir_name))
    
    # Execute in parallel
    with multiprocessing.Pool(processes=max_parallel) as pool:
        results = pool.starmap(run_wrf_case, args)
    
    # Step 5: Final summary
    successful_runs = sum(1 for r in results if r['precipitation_mm'] is not None)
    failed_runs = len(results) - successful_runs
    
    print("\n" + "="*60)
    print(" ALL RUNS COMPLETED - FINAL SUMMARY")
    print("="*60)
    print(f" Successful runs: {successful_runs}")
    print(f" Failed runs:     {failed_runs}")
    print(f" Total runs:      {num_runs}")
    print(f"\n  Results saved to: {OUTPUT_CSV}")
    print(f"  Status log:       {STATUS_LOG}")
    
    # Display final results table
    if successful_runs > 0:
        print("\n Precipitation Results Summary:")
        print("-" * 50)
        print(f"{'Run':<10} {'Directory':<15} {'Precipitation (mm)':<20}")
        print("-" * 50)
        for result in results:
            if result['precipitation_mm'] is not None:
                print(f"{result['run_index']:<10} {result['run_directory']:<15} {result['precipitation_mm']:<20.2f}")
    
    print("\n Workflow completed successfully!")
