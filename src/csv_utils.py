from collections import deque
from bayes_opt.util import ensure_rng
import numpy as np
import glob
import re
import os
from circuit_dict import stages
from circuit_dict_utils import softmax_alpha, sigmoid_alpha

def read_best_voltage_csv(best_iteration, tb_date, folder):
    pattern = f"{folder}/*_voltage_data_{best_iteration}_{tb_date}.csv"
    matching_files = glob.glob(pattern)

    best_voltage_csv=[]
    if not matching_files:
        raise FileNotFoundError(f"No file found matching: {pattern}")
    
    if len(matching_files) == 6: # both omega_, alpha_ optimization found the best point in the same iteration
        for file in matching_files:
            if "alpha_" in file: # choose alpha point (since it is performed later)
                best_voltage_csv.append(file)
    elif len(matching_files) == 3:
        best_voltage_csv = matching_files
    else:
        raise ValueError(f"Unexpected number of files found matching: {pattern}. Expected 3 or 6, got {len(matching_files)}.")

    return best_voltage_csv

def abla2_read_best_voltage_csv(case, best_iteration, tb_date, folder):
    pattern = f"{folder}/{case}_voltage_data_{best_iteration}_{tb_date}.csv"
    matching_files = glob.glob(pattern)

    best_voltage_csv=[]
    if not matching_files:
        raise FileNotFoundError(f"No file found matching: {pattern}")
    
    if len(matching_files) == 1: # both omega_, alpha_ optimization found the best point in the same iteration
        best_voltage_csv = matching_files[0]
    else:
        raise ValueError(f"Unexpected number of files found matching: {pattern}. Expected 1, got {len(matching_files)}.")

    return best_voltage_csv

# for topology set case
def read_best_voltage_csv_manual(best_iteration, tb_date, folder):
    pattern = f"{folder}/*_voltage_data_{best_iteration}_{tb_date}.csv"
    matching_files = glob.glob(pattern)

    if not matching_files:
        raise FileNotFoundError(f"No file found matching: {pattern}")
    
    if len(matching_files) == 2: # both omega_, alpha_ optimization found the best point in the same iteration
        for file in matching_files:
            if "alpha_" in file: # choose alpha point (since it is performed later)
                best_voltage_csv = file
    elif len(matching_files) == 1:
        best_voltage_csv = matching_files[0]
    else:
        raise ValueError(f"Unexpected number of files found matching: {pattern}. Expected 3 or 6, got {len(matching_files)}.")

    return best_voltage_csv

def read_best_voltage_per_case_csv(case, best_outer_iteration, tb_date, folder):
    pattern = f"{folder}/*{case}_voltage_data_{best_outer_iteration}_*_{tb_date}.csv"
    matching_files = glob.glob(pattern)

    if not matching_files:
        raise FileNotFoundError(f"No file found matching: {pattern}")
    
    if len(matching_files) == 2: # both omega_, alpha_ optimization found the best point in the same iteration
        for file in matching_files:
            if "alpha_" in file: # choose alpha point (since it is performed later)
                best_voltage_csv = file
    elif len(matching_files) == 1:
        best_voltage_csv = matching_files[0]
    else:
        raise ValueError(f"Unexpected number of files found matching: {pattern}. Expected 3 or 6, got {len(matching_files)}.")

    return best_voltage_csv

def read_best_sim_output_csv(best_iteration, tb_date, folder):
    pattern_w_prefix = f"{folder}/*_sim_output_{best_iteration}_{tb_date}.csv"
    pattern_wo_prefix = f"{folder}/sim_output_{best_iteration}_{tb_date}.csv"
    matching_files = glob.glob(pattern_w_prefix) + glob.glob(pattern_wo_prefix)

    if "finalopt" in str(best_iteration):
        finalopt_pattern = f"{folder}/sim_output_{best_iteration}_{tb_date}.csv"
        matching_finalopt = glob.glob(finalopt_pattern)
        if len(matching_finalopt) == 1:
            return matching_finalopt[0]
        else:
            raise ValueError(f"Unexpected number of finalopt files found matching: {finalopt_pattern}. Expected 1, got {len(matching_finalopt)}.") 
    else:
        if not matching_files:
            raise FileNotFoundError(f"No file found matching: {pattern_w_prefix} or {pattern_wo_prefix}")

        if len(matching_files) == 2:
            for file in matching_files:
                if "alpha_" in file:
                    return file
                
        elif len(matching_files) == 1:
            return matching_files[0]
        
        else:
            raise ValueError(f"Unexpected number of files found matching: {pattern_w_prefix} or {pattern_wo_prefix}. Expected 1, got {len(matching_files)}.")

def extract_case_number(filename):
    match = re.search(r"_(case)(\d)_", filename)
    if match:
        return int(match.group(2))  # the digit after 'case'
    else:
        raise ValueError(f"Could not extract case number from filename: {filename}")
    
def remove_csv_after_iteration(iteration, tb_date):
    voltage_data_pattern = f"./*_voltage_data_{iteration}*_{tb_date}.csv"
    matching_files = glob.glob(voltage_data_pattern)

    sim_output_pattern = f"./sim_output_{iteration}*_{tb_date}.csv"
    matching_sim_output_files = glob.glob(sim_output_pattern)
    
    if not matching_files and not matching_sim_output_files:
        print(f"No files found matching: {voltage_data_pattern} and {sim_output_pattern}")
        return
    
    if not matching_files:
        print(f"No files found matching: {voltage_data_pattern}")

    # finalb4opt (hard selection) does not create sim_output files in "./"
    if ("finalb4opt" not in str(iteration)) and not matching_sim_output_files:
        print(f"No files found matching: {sim_output_pattern}")

    for file in matching_files + matching_sim_output_files:
        try:
            os.remove(file)
            # print(f"Removed file: {file}")
        except Exception as e:
            print(f"Error removing file {file}: {e}")