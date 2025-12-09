from collections import deque
from bayes_opt.util import ensure_rng
import numpy as np
import glob
import re
import os
from circuit_dict import stages
from circuit_dict_utils import softmax_alpha, sigmoid_alpha

def maximize_with_nan_control(bo, init_points, n_iter, objective_fn, crash_sentinel=None):
    queue = deque()
    z01_loss_history = []
    def is_duplicate(bo, new_point):
        new_array = np.array([new_point[k] for k in bo._space.keys])
        for res in bo.res:
            existing_array = np.array([res["params"][k] for k in bo._space.keys])
            if np.allclose(existing_array, new_array, atol=1e-8):
                return True
        return False

    def generate_unique_random_point():
        while True:
            new_point_values = bo._space.random_sample() # numpy array
            new_point_names = bo._space.keys # name list
            new_point = dict(zip(new_point_names, new_point_values))
            if not is_duplicate(bo, new_point):
                return new_point
            else:
                print("[WARN] Generated random point was duplicate. Retrying.")

    # Initialize with unique random points
    for _ in range(init_points):
        queue.append(generate_unique_random_point())

    iteration = 0
    while queue or iteration < n_iter:
        
        bo_idx = len(bo.res)
        # print("next idx of res to be written:", bo_idx)

        if queue:
            next_point = queue.popleft()
        else:
            next_point = bo.suggest()
            if is_duplicate(bo, next_point):
                print("[WARN] Duplicate point suggested by BO. Sampling a random point instead.")
                next_point = generate_unique_random_point()

        result = objective_fn(bo_idx, **next_point) # inner iteration

        if result == crash_sentinel:
            print("[WARN] NaN detected — not registering point. Sampling random point next.")
            queue.append(generate_unique_random_point())
            continue  # do not increment iteration

        if not isinstance(result, tuple) or len(result) != 2:
            print(f"[ERROR] Unexpected objective_fn return at iter {bo_idx}: {result}")
            queue.append(generate_unique_random_point())
            continue
        
        else: 
            target, weighted_z01_loss = result
            target = target-weighted_z01_loss

            param_bounds = bo._space._bounds 
            param_keys = bo._space.keys

            # Ensure all values are float and within bounds
            clean_point = {
                k: float(np.clip(v, *param_bounds[i]))
                for i, (k, v) in enumerate(next_point.items())
            }
            target = float(target)
            ### FOR DEBUGGING ############################################
            # print(f"[DEBUG] Registering point #{bo_idx}")
            # print(f"next_point: {next_point}")
            # print(f"target: {target}")
            # print(f"keys: {list(next_point.keys())}")
            # print(f"value types: {[type(v) for v in next_point.values()]}")
            ##############################################################

            bo.register(params=clean_point, target=target)
            z01_loss_history.append(weighted_z01_loss)

        if not queue:
            iteration += 1

    return bo, z01_loss_history

def get_max_fom_iteration(bo):
    max_target = bo.max["target"]
    for i, res in enumerate(bo.res):
        if np.isclose(res["target"], max_target, atol=1e-8):
            return i  
    return -1


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

def bo_early_stopping(current_best_alphas, always_selected_alphas, selected_case, alpha_groups, temp, best_iteration, current_iteration, fom_window=5, alpha_threshold=0.99):
    """
    Early stopping when
    1) There is one softmax alpha value greater than the threshold for every alpha in the alpha_groups of selected case.
    2) FoM does not improve for a certain number of iterations.
    """
    # Check condition 1
    condition_1 = check_alpha_convergence(selected_case=selected_case, current_best_alphas=current_best_alphas, always_selected_alphas=always_selected_alphas, alpha_groups=alpha_groups, temp=temp, alpha_threshold=alpha_threshold)

    # Check condition 2
    condition_2 = False
    if current_iteration >= fom_window:
        if best_iteration <= current_iteration-fom_window:
            condition_2 = True
        
    if condition_1 and condition_2:
        print(f"Early stopping triggered at iteration {current_iteration}.")
        return True

def check_alpha_convergence(selected_case, current_best_sm_alphas, always_selected_alphas, alpha_groups, temp, alpha_threshold=0.99):

    above_threshold_count = 0
    if current_best_sm_alphas[selected_case] > alpha_threshold:
        print(f"Alpha of Selected case {selected_case} is above threshold: {current_best_sm_alphas[selected_case]} > {alpha_threshold}")
        above_threshold_count += 1
        for case_stage, alpha_name_list in alpha_groups.items():
            if selected_case in case_stage:
                print("checking case_stage:", case_stage)
                max_alpha_name = max(alpha_name_list, key=lambda x: current_best_sm_alphas[x])
                if current_best_sm_alphas[max_alpha_name] > alpha_threshold:
                    above_threshold_count += 1
                    print(f"Maximum Alpha {max_alpha_name} in {case_stage} is above threshold: {current_best_sm_alphas[max_alpha_name]} > {alpha_threshold}")
                else:
                    print(f"Maximum Alpha {max_alpha_name} in {case_stage} is below threshold: {current_best_sm_alphas[max_alpha_name]} <= {alpha_threshold}")
        for always_selected_alpha in always_selected_alphas.keys():
            if selected_case in always_selected_alpha:
                above_threshold_count += 1
                print(f"Always selected Alpha {always_selected_alpha} = 1.0 ")
    print(f"Number of alphas above threshold: {above_threshold_count} / {len(stages)+1}")
    if above_threshold_count == len(stages)+1:
        print(f"All alphas are above threshold: {alpha_threshold}. Alpha Convergence achieved.")
        return True
    else:
        return False