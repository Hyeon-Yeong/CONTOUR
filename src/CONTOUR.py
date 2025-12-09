from pyswarms.single.global_best import GlobalBestPSO
import time
import os
import numpy as np
import pandas as pd
import matplotlib
import ast
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import csv
import shutil
from collections import defaultdict
import argparse
import re
import sys

import pdb

from circuit_dict import topology_dict, topology_per_case, pbounds_per_component, stages
from circuit_dict import step_sizes
from circuit_dict import simulations
from circuit_dict import fc1_target, fc2_target, fc1_bw_target, fc2_bw_target
from circuit_dict_utils import generate_pbounds_ver2
from circuit_dict_utils import generate_initial_params_dict, generate_initial_alphas_dict, generate_initial_alphas_dict_random, softmax_alpha
from circuit_dict_utils import convert_Vb_from_Vb_frac
from run_ads_simulation import get_simulation_output, run_ads_simulation_per_case
# from run_ads_simulation import run_ads_simulation
from rf_rx_bo_utils import read_best_voltage_per_case_csv, read_best_voltage_csv, read_best_sim_output_csv, remove_csv_after_iteration, extract_case_number, check_alpha_convergence
from circuit_dict_utils import pso_bounds

from rf_rx_pso_utils import pso_with_nan_control, pso_per_case_with_nan_control
from bo_fom import FoM_z01_manual, FoM_z01_per_case


parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=2, help="Maximum number of iterations for optimization")
parser.add_argument("--a_parts", type=int, default=1, help="Number of particles for alpha optimization")
parser.add_argument("--a_iters", type=int, default=2, help="Number of iterations for alpha optimization")
parser.add_argument("--w_parts", type=int, default=1, help="Number of particles for omega optimization")
parser.add_argument("--w_iters", type=int, default=2, help="Number of iterations for omega optimization")
parser.add_argument("--task", type=str, default="singleGain", choices=["dualBW", "dualGain", "dualGains", "singleBW", "singleGain", "singleGains"], help="Task type for optimization")
parser.add_argument("--opt_w_parts", type=int, default=1, help="Number of particles for omega optimization in the final stage") # 18
parser.add_argument("--opt_w_iters", type=int, default=2, help="Number of iterations for omega optimization in the final stage") # 15

parser.add_argument("--tb_date", type=str, default="0618", choices = ["0618", "a_0618", "b_0618", "a_0729", "b_0729", "c_0729", "d_0729"], help="Testbench type for simulation data")

args= parser.parse_args()

wrk_space = "/home/local/ace/hy7557/rf_rx_0306_wrk"
task = args.task
temp = 1.0
opt_type="PSO"
tb_date = args.tb_date
rng = np.random.default_rng(seed=42)

max_iterations = args.iters
alpha_particles = args.a_parts
alpha_iterations = args.a_iters
omega_iterations = args.w_iters
omega_particles = args.w_parts

opt_omega_particles = args.opt_w_parts
opt_omega_iterations = args.opt_w_iters

neg_z_fom = -1e6
nan_z_fom = -1e9

opt_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
folder_path = f"./{opt_type}_3cases_result/{task}_{opt_time}_{max_iterations}it_{alpha_iterations}alpha_{omega_iterations}omega"
current_best_path = f"./{opt_type}_3cases_current_best/{task}_{opt_time}"
os.makedirs(folder_path, exist_ok=True)
os.makedirs(current_best_path, exist_ok=True)

soft_case1_sim_output_csv = os.path.join(folder_path, f"{opt_time}_case1_best_sim_output.csv")
soft_case2_sim_output_csv = os.path.join(folder_path, f"{opt_time}_case2_best_sim_output.csv")
soft_case3_sim_output_csv = os.path.join(folder_path, f"{opt_time}_case3_best_sim_output.csv")
soft_case1_voltage_csv = os.path.join(folder_path, f"{opt_time}_case1_best_voltage.csv")
soft_case2_voltage_csv = os.path.join(folder_path, f"{opt_time}_case2_best_voltage.csv")
soft_case3_voltage_csv = os.path.join(folder_path, f"{opt_time}_case3_best_voltage.csv")

soft_selection_csv = os.path.join(folder_path, f"{opt_time}_soft_selection.csv")

hard_selection_csv = os.path.join(folder_path, f"{opt_time}_hard_selection.csv")
hard_selection_sim_output_csv = os.path.join(folder_path, f"{opt_time}_hard_selection_sim_output.csv")
hard_selection_voltage_csv = os.path.join(folder_path, f"{opt_time}_hard_selection_voltage_data.csv")

opt_selection_csv = os.path.join(folder_path, f"{opt_time}_omega_opt.csv")
opt_selection_sim_output_csv = os.path.join(folder_path, f"{opt_time}_omega_opt_sim_output.csv")
opt_selection_voltage_csv = os.path.join(folder_path, f"{opt_time}_omega_opt_voltage_data.csv")

param_pbounds, param_step_sizes, alpha_pbounds, alpha_groups, instance_list, always_selected_alphas, always_set_omega = generate_pbounds_ver2(topology_dict, topology_per_case, pbounds_per_component, step_sizes)
always_selected_alphas = {k: 1.0 for k in always_selected_alphas} 


for case in ["case1", "case2", "case3"]:
    del alpha_pbounds[case]

param_names, param_pso_bounds = pso_bounds(param_pbounds)
alpha_names, alpha_pso_bounds = pso_bounds(alpha_pbounds)

case1_param_names = []
case1_param_names_idx = []
case2_param_names = []
case2_param_names_idx = []
case3_param_names = []
case3_param_names_idx = []

for idx, param in enumerate(param_names):
    if param.startswith("case1"):
        case1_param_names.append(param)
        case1_param_names_idx.append(idx)
    elif param.startswith("case2"):
        case2_param_names.append(param)
        case2_param_names_idx.append(idx)
    elif param.startswith("case3"):
        case3_param_names.append(param)
        case3_param_names_idx.append(idx)

param_pso_lower_bounds, param_pso_upper_bounds = param_pso_bounds
case1_param_pso_bounds = (param_pso_lower_bounds[case1_param_names_idx], param_pso_upper_bounds[case1_param_names_idx])
case2_param_pso_bounds = (param_pso_lower_bounds[case2_param_names_idx], param_pso_upper_bounds[case2_param_names_idx])
case3_param_pso_bounds = (param_pso_lower_bounds[case3_param_names_idx], param_pso_upper_bounds[case3_param_names_idx])

case1_alpha_names = []
case1_alpha_names_idx = []
case2_alpha_names = []
case2_alpha_names_idx = []
case3_alpha_names = []
case3_alpha_names_idx = []

for idx, alpha in enumerate(alpha_names):
    if alpha.startswith("case1"):
        case1_alpha_names.append(alpha)
        case1_alpha_names_idx.append(idx)
    elif alpha.startswith("case2"):
        case2_alpha_names.append(alpha)
        case2_alpha_names_idx.append(idx)
    elif alpha.startswith("case3"):
        case3_alpha_names.append(alpha)
        case3_alpha_names_idx.append(idx)

alpha_pso_lower_bounds, alpha_pso_upper_bounds = alpha_pso_bounds
case1_alpha_pso_bounds = (alpha_pso_lower_bounds[case1_alpha_names_idx], alpha_pso_upper_bounds[case1_alpha_names_idx])
case2_alpha_pso_bounds = (alpha_pso_lower_bounds[case2_alpha_names_idx], alpha_pso_upper_bounds[case2_alpha_names_idx])
case3_alpha_pso_bounds = (alpha_pso_lower_bounds[case3_alpha_names_idx], alpha_pso_upper_bounds[case3_alpha_names_idx])

# 0721 TRIAL START FROM SM_ALPHA=0.5 #
alphas = generate_initial_alphas_dict(alphas_dict=alpha_pbounds, alpha_groups=alpha_groups)

def objective_omega(inner_iter, particle_idx, **omega_vars):

    global alphas 
    global outer_iter 

    case1_neg_z_found = False
    case1_nan_z_found = False
    case2_neg_z_found = False
    case2_nan_z_found = False
    case3_neg_z_found = False
    case3_nan_z_found = False

    if outer_iter == "finalopt":
        sm_alphas = alphas
        global final_selected_case
    else:
        sm_alphas = softmax_alpha(alphas, alpha_groups, temp)
    sm_alphas = {**sm_alphas, **always_selected_alphas}
    omegas = convert_Vb_from_Vb_frac(omega_vars)
    omegas = {**omegas, **always_set_omega} 

    # if len(omegas) != len(param_names)+len(always_set_omega): # error가 난 경우 혹은 retry의 경우임. 확인할 것!
    #     pdb.set_trace()

    try:
        final_stage_dsfile, case1_neg_z_found, case1_nan_z_found, case2_neg_z_found, case2_nan_z_found, case3_neg_z_found, case3_nan_z_found = run_ads_simulation_per_case(
            simulations=simulations,
            instance_list=instance_list,
            sm_alphas=sm_alphas,
            omegas=omegas,
            wrk_space=wrk_space,
            tb_date=tb_date,
        )
    except Exception as e:
        print(f"[SIM FAIL] Particle {particle_idx}: {e}")
        pdb.set_trace()
        return None

    if outer_iter == "finalopt":
        if case1_nan_z_found or case2_nan_z_found or case3_nan_z_found:
            print("[WARN] NaN impedance exists in final optimization.")
            return None
        if case1_neg_z_found or case2_neg_z_found or case3_neg_z_found:
            if final_selected_case == "case1" and case1_neg_z_found:
                print("[WARN] [CASE1] Negative impedance exists in final optimization. Skipping candidate.")
                fom = neg_z_fom
                weighted_z01_loss = 0.0
                weighted_penalty = 0.0
            elif final_selected_case == "case2" and case2_neg_z_found:
                print("[WARN] [CASE2] Negative impedance exists in final optimization. Skipping candidate.")
                fom = neg_z_fom
                weighted_z01_loss = 0.0
                weighted_penalty = 0.0
            elif final_selected_case == "case3" and case3_neg_z_found:
                print("[WARN] [CASE3] Negative impedance exists in final optimization. Skipping candidate.")
                fom = neg_z_fom
                weighted_z01_loss = 0.0
                weighted_penalty = 0.0
            else:
                print("[WARN] Neg Z occurred not due to selected case.")
                return None
            return fom, weighted_z01_loss, weighted_penalty
    else:
            
        if case1_nan_z_found or case2_nan_z_found or case3_nan_z_found:
            if case1_nan_z_found:
                print("[WARN] [CASE1] NaN impedance exists")
                case1_fom = nan_z_fom
                case1_weighted_z01_loss = 0.0
                case1_weighted_penalty = 0.0
            else:
                case1_fom = None
                case1_weighted_z01_loss = None
                case1_weighted_penalty = None
            if case2_nan_z_found:
                print("[WARN] [CASE2] NaN impedance exists")
                case2_fom = nan_z_fom
                case2_weighted_z01_loss = 0.0
                case2_weighted_penalty = 0.0
            else:
                case2_fom = None
                case2_weighted_z01_loss = None
                case2_weighted_penalty = None
            if case3_nan_z_found:
                print("[WARN] [CASE3] NaN impedance exists")
                case3_fom = nan_z_fom
                case3_weighted_z01_loss = 0.0
                case3_weighted_penalty = 0.0
            else:
                case3_fom = None
                case3_weighted_z01_loss = None
                case3_weighted_penalty = None
            return case1_fom, case2_fom, case3_fom, case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss, case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty

        if case1_neg_z_found or case2_neg_z_found or case3_neg_z_found:
            if case1_neg_z_found:
                print("[WARN] [CASE1] Penalize candidate due to negative impedance.")
                case1_fom = neg_z_fom
                case1_weighted_z01_loss = 0.0
                case1_weighted_penalty = 0.0
            else:
                case1_fom = None
                case1_weighted_z01_loss = None
                case1_weighted_penalty = None
            if case2_neg_z_found:
                print("[WARN] [CASE2] Penalize candidate due to negative impedance.")
                case2_fom = neg_z_fom
                case2_weighted_z01_loss = 0.0
                case2_weighted_penalty = 0.0
            else:
                case2_fom = None
                case2_weighted_z01_loss = None
                case2_weighted_penalty = None
            if case3_neg_z_found:
                print("[WARN] [CASE3] Penalize candidate due to negative impedance.")
                case3_fom = neg_z_fom
                case3_weighted_z01_loss = 0.0
                case3_weighted_penalty = 0.0
            else:
                case3_fom = None
                case3_weighted_z01_loss = None
                case3_weighted_penalty = None
            return case1_fom, case2_fom, case3_fom, case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss, case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty

    iteration_point = f"{outer_iter}_{inner_iter}_{particle_idx}"
    print("Will be written with file name (_iteration point):", iteration_point)
    sim_output_dict = get_simulation_output(final_stage_dsfile, sm_alphas, iteration_point, task=task, tb_date=tb_date)

    sim_output_df = pd.DataFrame([sim_output_dict])
    sim_output_df.to_csv(f"./sim_output_{iteration_point}_{tb_date}.csv", index=False)

    if outer_iter == "finalopt":
        fom, weighted_z01_loss, weighted_penalty = FoM_z01_manual(
            sim_output_dict=sim_output_dict,
            selected_case=final_selected_case,
            task=task,
            is_final=True
        )
        return fom, weighted_z01_loss, weighted_penalty

    else:
        case1_fom, case2_fom, case3_fom, case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss, case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty = FoM_z01_per_case(
            sim_output_dict=sim_output_dict,
            sm_alpha_dict=sm_alphas,
            alpha_groups=alpha_groups,
            task=task,
            is_final=False
        )

        
        return case1_fom, case2_fom, case3_fom, case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss, case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty


def objective_alpha(inner_iter, particle_idx, **alpha_vars):

    global omegas
    global outer_iter

    case1_neg_z_found = False
    case1_nan_z_found = False
    case2_neg_z_found = False
    case2_nan_z_found = False
    case3_neg_z_found = False
    case3_nan_z_found = False

    sm_alphas = softmax_alpha(alpha_vars, alpha_groups, temp)
    sm_alphas = {**sm_alphas, **always_selected_alphas}
    omegas = {**omegas, **always_set_omega}

    # if len(sm_alphas) != len(alpha_names)+len(always_selected_alphas): # error가 난 경우 혹은 retry의 경우임. 확인할 것!
    #     pdb.set_trace()

    try:
        final_stage_dsfile, case1_neg_z_found, case1_nan_z_found, case2_neg_z_found, case2_nan_z_found, case3_neg_z_found, case3_nan_z_found = run_ads_simulation_per_case(simulations=simulations, instance_list=instance_list, sm_alphas=sm_alphas, omegas=omegas, wrk_space=wrk_space, tb_date=tb_date)
    except Exception as e:
        print(f"[SIM FAIL] Particle {particle_idx}: {e}")
        pdb.set_trace()
        return None
    
    if case1_nan_z_found or case2_nan_z_found or case3_nan_z_found:
        if case1_nan_z_found:
            print("[WARN] [CASE1] NaN impedance exists")
            case1_fom = nan_z_fom
            case1_weighted_z01_loss = 0.0
            case1_weighted_penalty = 0.0
        else:
            case1_fom = None
            case1_weighted_z01_loss = None
            case1_weighted_penalty = None
        if case2_nan_z_found:
            print("[WARN] [CASE2] NaN impedance exists")
            case2_fom = nan_z_fom
            case2_weighted_z01_loss = 0.0
            case2_weighted_penalty = 0.0
        else:
            case2_fom = None
            case2_weighted_z01_loss = None
            case2_weighted_penalty = None
        if case3_nan_z_found:
            print("[WARN] [CASE3] NaN impedance exists")
            case3_fom = nan_z_fom
            case3_weighted_z01_loss = 0.0
            case3_weighted_penalty = 0.0
        else:
            case3_fom = None
            case3_weighted_z01_loss = None
            case3_weighted_penalty = None
        return case1_fom, case2_fom, case3_fom, case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss, case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty

    if case1_neg_z_found or case2_neg_z_found or case3_neg_z_found:
        if case1_neg_z_found:
            print("[WARN] [CASE1] Skipping candidate due to negative impedance.")
            case1_fom = -1e6
            case1_weighted_z01_loss = 0.0
            case1_weighted_penalty = 0.0
        else:
            case1_fom = None
            case1_weighted_z01_loss = None
            case1_weighted_penalty = None
        if case2_neg_z_found:
            print("[WARN] [CASE2] Skipping candidate due to negative impedance.")
            case2_fom = -1e6
            case2_weighted_z01_loss = 0.0
            case2_weighted_penalty = 0.0
        else:
            case2_fom = None
            case2_weighted_z01_loss = None
            case2_weighted_penalty = None
        if case3_neg_z_found:
            print("[WARN] [CASE3] Skipping candidate due to negative impedance.")
            case3_fom = -1e6
            case3_weighted_z01_loss = 0.0
            case3_weighted_penalty = 0.0
        else:
            case3_fom = None
            case3_weighted_z01_loss = None
            case3_weighted_penalty = None
        return case1_fom, case2_fom, case3_fom, case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss, case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty

    iteration_point = f"{outer_iter}_{inner_iter}_{particle_idx}"
    print("Will be written with file name (_iteration point):", iteration_point)
    sim_output_dict = get_simulation_output(final_stage_dsfile=final_stage_dsfile, sm_alphas=sm_alphas, iteration=iteration_point, task=task, tb_date=tb_date)

    sim_output_df = pd.DataFrame([sim_output_dict])
    sim_output_df.to_csv(f"./sim_output_{iteration_point}_{tb_date}.csv", index=False)

    case1_fom, case2_fom, case3_fom, case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss, case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty = FoM_z01_per_case(sim_output_dict=sim_output_dict, sm_alpha_dict=sm_alphas, alpha_groups=alpha_groups, task=task, is_final=False)

    return case1_fom, case2_fom, case3_fom, case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss, case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty

def custom_alpha_init(case, alpha_pbounds=alpha_pbounds, alpha_groups=alpha_groups): 
    # how to generate alphas the same as initialization inside pso_with_nan_control
    if case == "case1":
        alpha_pbounds = {k: v for k, v in alpha_pbounds.items() if k.startswith("case1")}
        alpha_groups = {k: v for k, v in alpha_groups.items() if k.startswith("case1")}
        alpha_names = case1_alpha_names
    elif case == "case2":
        alpha_pbounds = {k: v for k, v in alpha_pbounds.items() if k.startswith("case2")}
        alpha_groups = {k: v for k, v in alpha_groups.items() if k.startswith("case2")}
        alpha_names = case2_alpha_names
    elif case == "case3":
        alpha_pbounds = {k: v for k, v in alpha_pbounds.items() if k.startswith("case3")}
        alpha_groups = {k: v for k, v in alpha_groups.items() if k.startswith("case3")}
        alpha_names = case3_alpha_names
        
    alpha_dict = generate_initial_alphas_dict_random(alphas_dict=alpha_pbounds, alpha_groups=alpha_groups)
    alpha_vals = np.array([alpha_dict[name] for name in alpha_names])
    
    return alpha_vals


case1_alphas = {k:v for k,v in alphas.items() if k.startswith("case1")}
case2_alphas = {k:v for k,v in alphas.items() if k.startswith("case2")}
case3_alphas = {k:v for k,v in alphas.items() if k.startswith("case3")}

case1_current_best = {
    'outer_iter': None,
    'alphas': case1_alphas.copy(),
    'omegas': None,
    'omegas_w_vb_frac': None,
    'cost': float("inf")
}

case1_z01_loss_history = []
case1_penalty_history = []

case2_current_best = {
    'outer_iter': None,
    'alphas': case2_alphas.copy(),
    'omegas': None,
    'omegas_w_vb_frac': None,
    'cost': float("inf")
}

case2_z01_loss_history = []
case2_penalty_history = []

case3_current_best = {
    'outer_iter': None,
    'alphas': case3_alphas.copy(),
    'omegas': None,
    'omegas_w_vb_frac': None,
    'cost': float("inf"),
}

case3_z01_loss_history = []
case3_penalty_history = []

cases = ['case1', 'case2', 'case3']

alphas_history = []
omegas_history = []
cost_history = {
    'outer_iter': [],
    'case1_cost': [],
    'case2_cost': [],
    'case3_cost': [],
}
# alpha_zero_one_loss_history = {
#     "case1": [],
#     "case2": [],
#     "case3": []
# }

case1_omega_ver = 0
case2_omega_ver = 0
case3_omega_ver = 0
case1_alpha_ver = 0
case2_alpha_ver = 0
case3_alpha_ver = 0
case1_opt_omega_found = False
case2_opt_omega_found = False
case3_opt_omega_found = False
case1_opt_alpha_found = False
case2_opt_alpha_found = False
case3_opt_alpha_found = False

case1_current_best_alpha_swarm = None
case2_current_best_alpha_swarm = None
case3_current_best_alpha_swarm = None
case1_current_best_omega_swarm = None
case2_current_best_omega_swarm = None
case3_current_best_omega_swarm = None

print(f"=== Starting {opt_type} with alpha ({alpha_particles} particles and {alpha_iterations} iterations) omega ({omega_particles} particles and {omega_iterations} iterations) ===")

start_opt_time = time.time()
for i in range(max_iterations):
    outer_iter = i+1
    print(f"\n=== Omega Optimization Round {outer_iter}/{max_iterations} ===")
    print("Current Best (Alphas, Omegas) version:\n",
        f"[CASE1] Alpha {case1_alpha_ver}, Omega {case1_omega_ver}\n",
        f"[CASE2] Alpha {case2_alpha_ver}, Omega {case2_omega_ver}\n",
        f"[CASE3] Alpha {case3_alpha_ver}, Omega {case3_omega_ver}")

    if i == 0:
        case1_prev_best_omegas_w_vb_frac = None
        case2_prev_best_omegas_w_vb_frac = None
        case3_prev_best_omegas_w_vb_frac = None
        prev_best_cost = (None, None, None)
        prev_swarm = (None, None, None)
    elif i > 0:
        if case1_opt_alpha_found:  # use updated alphas. use the last best omegas
            print("[CASE1] Using updated alphas for omega optimization. Alpha ver:", case1_alpha_ver)
            case1_prev_best_omegas_w_vb_frac = np.array([case1_current_best['omegas_w_vb_frac'][k] for k in case1_param_names]) # ordered in param_names
            case1_prev_swarm = None
            case1_prev_best_cost = case1_current_best['cost']
        else:
            print("[CASE1] Continue previous pso for omega optimization.")
            case1_prev_best_omegas_w_vb_frac = None
            case1_prev_swarm = case1_current_best_omega_swarm
            case1_prev_best_cost = None
        if case2_opt_alpha_found:
            print("[CASE2] Using updated alphas for omega optimization. Alpha ver:", case2_alpha_ver)
            case2_prev_best_omegas_w_vb_frac = np.array([case2_current_best['omegas_w_vb_frac'][k] for k in case2_param_names])
            case2_prev_swarm = None
            case2_prev_best_cost = case2_current_best['cost']
        else:
            print("[CASE2] Continue previous pso for omega optimization.")
            case2_prev_best_omegas_w_vb_frac = None
            case2_prev_swarm = case2_current_best_omega_swarm
            case2_prev_best_cost = None
        if case3_opt_alpha_found:
            print("[CASE3] Using updated alphas for omega optimization. Alpha ver:", case3_alpha_ver)
            case3_prev_best_omegas_w_vb_frac = np.array([case3_current_best['omegas_w_vb_frac'][k] for k in case3_param_names])
            case3_prev_swarm = None
            case3_prev_best_cost = case3_current_best['cost']
        else:
            print("[CASE3] Continue previous pso for omega optimization.")
            case3_prev_best_omegas_w_vb_frac = None
            case3_prev_swarm = case3_current_best_omega_swarm
            case3_prev_best_cost = None

        prev_best_cost = (case1_current_best['cost'], case2_current_best['cost'], case3_current_best['cost'])
        prev_swarm = (case1_prev_swarm, case2_prev_swarm, case3_prev_swarm)

    # cost: list, pos: dict, cost_history: dict, z01_loss_history: dict, swarm: list
    best_omega_cost, best_omega_pos, omega_cost_history, omega_z01_loss_history, omega_penalty_history, omega_best_iteration, omega_swarm = pso_per_case_with_nan_control(objective_fn=objective_omega, n_particles=omega_particles, n_iterations=omega_iterations,
    case1_bounds=case1_param_pso_bounds, case2_bounds=case2_param_pso_bounds, case3_bounds=case3_param_pso_bounds,
    case1_param_names=case1_param_names, case2_param_names=case2_param_names, case3_param_names=case3_param_names,
    crash_sentinel=None,
    neg_z_fom=neg_z_fom, nan_z_fom=nan_z_fom,
    max_retry_per_particle=5,
    case1_prev_init_pos=case1_prev_best_omegas_w_vb_frac, 
    case2_prev_init_pos=case2_prev_best_omegas_w_vb_frac,
    case3_prev_init_pos=case3_prev_best_omegas_w_vb_frac,
    prev_best_cost= prev_best_cost, custom_initializer=None, prev_swarm=prev_swarm)
    
    case1_current_best_omegas_w_vb_frac = dict(zip(case1_param_names, best_omega_pos["case1"]))
    case1_current_best_omegas = convert_Vb_from_Vb_frac(case1_current_best_omegas_w_vb_frac)
    case2_current_best_omegas_w_vb_frac = dict(zip(case2_param_names, best_omega_pos["case2"]))
    case2_current_best_omegas = convert_Vb_from_Vb_frac(case2_current_best_omegas_w_vb_frac)
    case3_current_best_omegas_w_vb_frac = dict(zip(case3_param_names, best_omega_pos["case3"]))
    case3_current_best_omegas = convert_Vb_from_Vb_frac(case3_current_best_omegas_w_vb_frac)

    case1_best_omega_cost, case2_best_omega_cost, case3_best_omega_cost = best_omega_cost

    case1_current_omega_z01_loss_history = omega_z01_loss_history["case1"]
    case2_current_omega_z01_loss_history = omega_z01_loss_history["case2"]
    case3_current_omega_z01_loss_history = omega_z01_loss_history["case3"]

    case1_current_omega_penalty_history = omega_penalty_history["case1"]
    case2_current_omega_penalty_history = omega_penalty_history["case2"]
    case3_current_omega_penalty_history = omega_penalty_history["case3"]

    if i == 0:
        case1_z01_loss_history.append(case1_current_omega_z01_loss_history[-1])
        case2_z01_loss_history.append(case2_current_omega_z01_loss_history[-1])
        case3_z01_loss_history.append(case3_current_omega_z01_loss_history[-1])

    case1_penalty_history.append(case1_current_omega_penalty_history[-1])
    case2_penalty_history.append(case2_current_omega_penalty_history[-1])
    case3_penalty_history.append(case3_current_omega_penalty_history[-1])

    case1_omega_best_inner, case1_omega_best_particle = omega_best_iteration["case1"]
    case2_omega_best_inner, case2_omega_best_particle = omega_best_iteration["case2"]
    case3_omega_best_inner, case3_omega_best_particle = omega_best_iteration["case3"]

    case1_current_omega_best_iteration = f"{outer_iter}_{case1_omega_best_inner}_{case1_omega_best_particle}"
    case2_current_omega_best_iteration = f"{outer_iter}_{case2_omega_best_inner}_{case2_omega_best_particle}"
    case3_current_omega_best_iteration = f"{outer_iter}_{case3_omega_best_inner}_{case3_omega_best_particle}"

    # better omega found for case1
    if i == 0 or case1_best_omega_cost < case1_current_best['cost']: 
        print(f"[CASE1] Better omega found in iteration {outer_iter}. Current Best Cost: {case1_best_omega_cost:.2f}, Prev Best Cost: {case1_current_best['cost']:.2f}")
        case1_opt_omega_found = True
        case1_omega_ver += 1
        print(f"[CASE1] Updated Omega Version: {case1_omega_ver}")
        case1_current_best['outer_iter'] = outer_iter
        case1_current_best['cost'] = case1_best_omega_cost
        case1_current_best['alphas'] = case1_current_best['alphas'].copy()  # alphas remain the same
        case1_current_best['omegas'] = case1_current_best_omegas.copy()
        case1_current_best['omegas_w_vb_frac'] = case1_current_best_omegas_w_vb_frac.copy()
        case1_current_best_omega_swarm = omega_swarm[0]
        case1_omegas = case1_current_best_omegas.copy()

        case1_current_best_voltage_csv = f"case1_voltage_data_{case1_current_omega_best_iteration}_{tb_date}.csv"
        shutil.copy(src=case1_current_best_voltage_csv, dst=os.path.join(current_best_path, f"omega_{case1_current_best_voltage_csv}"))
        case1_current_best_sim_output_csv = f"sim_output_{case1_current_omega_best_iteration}_{tb_date}.csv"
        shutil.copy(src=case1_current_best_sim_output_csv, dst=os.path.join(current_best_path, f"omega_{case1_current_best_sim_output_csv}"))

    else:
        print(f"[CASE1] No better omega found in iteration {outer_iter}. Current best cost: {case1_current_best['cost']:.2f}")
        case1_opt_omega_found = False
        case1_omegas = case1_current_best['omegas'].copy()  # use the last best omegas
    
    # better omega found for case2
    if i == 0 or case2_best_omega_cost < case2_current_best['cost']: 
        print(f"[CASE2] Better omega found in iteration {outer_iter}. Current Best Cost: {case2_best_omega_cost:.2f}, Prev Best Cost: {case2_current_best['cost']:.2f}")
        case2_opt_omega_found = True
        case2_omega_ver += 1
        print(f"[CASE2] Updated Omega Version: {case2_omega_ver}")
        case2_current_best['outer_iter'] = outer_iter
        case2_current_best['cost'] = case2_best_omega_cost
        case2_current_best['alphas'] = case2_current_best['alphas'].copy()  # alphas remain the same
        case2_current_best['omegas'] = case2_current_best_omegas.copy()
        case2_current_best['omegas_w_vb_frac'] = case2_current_best_omegas_w_vb_frac.copy()
        case2_current_best_omega_swarm = omega_swarm[1]
        case2_omegas = case2_current_best_omegas.copy()

        case2_current_best_voltage_csv = f"case2_voltage_data_{case2_current_omega_best_iteration}_{tb_date}.csv"
        shutil.copy(src=case2_current_best_voltage_csv, dst=os.path.join(current_best_path, f"omega_{case2_current_best_voltage_csv}"))
        case2_current_best_sim_output_csv = f"sim_output_{case2_current_omega_best_iteration}_{tb_date}.csv"
        shutil.copy(src=case2_current_best_sim_output_csv, dst=os.path.join(current_best_path, f"omega_{case2_current_best_sim_output_csv}"))
    else:
        print(f"[CASE2] No better omega found in iteration {outer_iter}. Current best cost: {case2_current_best['cost']:.2f}")
        case2_opt_omega_found = False
        case2_omegas = case2_current_best['omegas'].copy()  # use the last best omegas
    
    # better omega found for case3
    if i == 0 or case3_best_omega_cost < case3_current_best['cost']: 
        print(f"[CASE3] Better omega found in iteration {outer_iter}. Current Best Cost: {case3_best_omega_cost:.2f}, Prev Best Cost: {case3_current_best['cost']:.2f}")
        case3_opt_omega_found = True
        case3_omega_ver += 1
        print(f"[CASE3] Updated Omega Version: {case3_omega_ver}")
        case3_current_best['outer_iter'] = outer_iter
        case3_current_best['cost'] = case3_best_omega_cost
        case3_current_best['alphas'] = case3_current_best['alphas'].copy()
        case3_current_best['omegas'] = case3_current_best_omegas.copy()
        case3_current_best['omegas_w_vb_frac'] = case3_current_best_omegas_w_vb_frac.copy()
        case3_current_best_omega_swarm = omega_swarm[2]
        case3_omegas = case3_current_best_omegas.copy()

        case3_current_best_voltage_csv = f"case3_voltage_data_{case3_current_omega_best_iteration}_{tb_date}.csv"
        shutil.copy(src=case3_current_best_voltage_csv, dst=os.path.join(current_best_path, f"omega_{case3_current_best_voltage_csv}"))
        case3_current_best_sim_output_csv = f"sim_output_{case3_current_omega_best_iteration}_{tb_date}.csv"
        shutil.copy(src=case3_current_best_sim_output_csv, dst=os.path.join(current_best_path, f"omega_{case3_current_best_sim_output_csv}"))
    else:
        print(f"[CASE3] No better omega found in iteration {outer_iter}. Current best cost: {case3_current_best['cost']:.2f}")
        case3_opt_omega_found = False
        case3_omegas = case3_current_best['omegas'].copy()

    omegas = {**case1_omegas, **case2_omegas, **case3_omegas} 

    omegas_history.append({
        'outer_iter': outer_iter,
        'case1_omegas': case1_omegas,
        'case1_cost': case1_current_best['cost'],
        'case2_omegas': case2_current_best['omegas'],
        'case2_cost': case2_current_best['cost'],
        'case3_omegas': case3_current_best['omegas'],
        'case3_cost': case3_current_best['cost'],
    })

    if i == 0: # Add initial alphas to history
        alphas_history.append({
        'outer_iter': 0,
        'case1_alphas': case1_current_best['alphas'],
        'case1_cost': case1_current_best['cost'],
        'case2_alphas': case2_current_best['alphas'],
        'case2_cost': case2_current_best['cost'],
        'case3_alphas': case3_current_best['alphas'],
        'case3_cost': case3_current_best['cost'],
        })

    cost_history['outer_iter'].append(outer_iter)
    cost_history['case1_cost'].append(case1_current_best['cost'])
    cost_history['case2_cost'].append(case2_current_best['cost'])
    cost_history['case3_cost'].append(case3_current_best['cost'])

    remove_csv_after_iteration(iteration=outer_iter, tb_date=tb_date)

    print(f"\n=== Alpha Optimization Round {outer_iter}/{max_iterations} ===")
    print("Current Best (Alphas, Omegas) version:\n ",
         f"[CASE1] Alpha {case1_alpha_ver}, Omega {case1_omega_ver}\n ",
         f"[CASE2] Alpha {case2_alpha_ver}, Omega {case2_omega_ver}\n ",
         f"[CASE3] Alpha {case3_alpha_ver}, Omega {case3_omega_ver}")

    if i == 0:
        case1_prev_best_alphas = np.array([case1_current_best['alphas'][k] for k in case1_alpha_names])  # ordered in alpha_names
        case2_prev_best_alphas = np.array([case2_current_best['alphas'][k] for k in case2_alpha_names])
        case3_prev_best_alphas = np.array([case3_current_best['alphas'][k] for k in case3_alpha_names])
        prev_best_cost = (case1_current_best['cost'], case2_current_best['cost'], case3_current_best['cost'])
        prev_swarm = (None, None, None)  # no previous swarm for the first iteration
    elif i > 0:
        if case1_opt_omega_found:
            print("[CASE1] Using updated omegas for alpha optimization. Omega ver:", case1_omega_ver)
            case1_prev_best_alphas = np.array([case1_current_best['alphas'][k] for k in case1_alpha_names]) # ordered in alpha_names
            case1_prev_swarm = None
            case1_prev_best_cost = case1_current_best['cost']
        else:
            print("[CASE1] Continue previous pso for alpha optimization.")
            case1_prev_best_alphas = None
            case1_prev_swarm = case1_current_best_alpha_swarm
            case1_prev_best_cost = None

        if case2_opt_omega_found:
            print("[CASE2] Using updated omegas for alpha optimization. Omega ver:", case2_omega_ver)
            case2_prev_best_alphas = np.array([case2_current_best['alphas'][k] for k in case2_alpha_names])
            case2_prev_swarm = None
            case2_prev_best_cost = case2_current_best['cost']
        else:
            print("[CASE2] Continue previous pso for alpha optimization.")
            case2_prev_best_alphas = None
            case2_prev_swarm = case2_current_best_alpha_swarm
            case2_prev_best_cost = None

        if case3_opt_omega_found:
            print("[CASE3] Using updated omegas for alpha optimization. Omega ver:", case3_omega_ver)
            case3_prev_best_alphas = np.array([case3_current_best['alphas'][k] for k in case3_alpha_names])
            case3_prev_swarm = None
            case3_prev_best_cost = case3_current_best['cost']
        else:
            print("[CASE3] Continue previous pso for alpha optimization.")
            case3_prev_best_alphas = None
            case3_prev_swarm = case3_current_best_alpha_swarm
            case3_prev_best_cost = None
        
        prev_best_cost = (case1_prev_best_cost, case2_prev_best_cost, case3_prev_best_cost)
        prev_swarm = (case1_prev_swarm, case2_prev_swarm, case3_prev_swarm)

    best_alpha_cost, best_alpha_pos, alpha_cost_history, alpha_z01_loss_history, alpha_penalty_history, alpha_best_iteration, alpha_swarm = pso_per_case_with_nan_control(objective_fn=objective_alpha, n_particles=alpha_particles, n_iterations=alpha_iterations, 
    case1_bounds=case1_alpha_pso_bounds, case2_bounds=case2_alpha_pso_bounds, case3_bounds=case3_alpha_pso_bounds,
    case1_param_names=case1_alpha_names, case2_param_names=case2_alpha_names, case3_param_names=case3_alpha_names,
    crash_sentinel=None,
    neg_z_fom=neg_z_fom, nan_z_fom=nan_z_fom,
    max_retry_per_particle=5,
    case1_prev_init_pos=case1_prev_best_alphas,
    case2_prev_init_pos=case2_prev_best_alphas,
    case3_prev_init_pos=case3_prev_best_alphas,
    prev_best_cost=prev_best_cost, custom_initializer=custom_alpha_init, prev_swarm=prev_swarm)

    case1_current_alpha_z01_loss_history = alpha_z01_loss_history["case1"]
    case2_current_alpha_z01_loss_history = alpha_z01_loss_history["case2"]
    case3_current_alpha_z01_loss_history = alpha_z01_loss_history["case3"]

    # alpha_zero_one_loss_history["case1"].append(case1_current_alpha_z01_loss_history[-1])
    # alpha_zero_one_loss_history["case2"].append(case2_current_alpha_z01_loss_history[-1])
    # alpha_zero_one_loss_history["case3"].append(case3_current_alpha_z01_loss_history[-1])

    case1_current_best_alphas = dict(zip(case1_alpha_names, best_alpha_pos["case1"]))
    case2_current_best_alphas = dict(zip(case2_alpha_names, best_alpha_pos["case2"]))
    case3_current_best_alphas = dict(zip(case3_alpha_names, best_alpha_pos["case3"]))

    case1_best_alpha_cost, case2_best_alpha_cost, case3_best_alpha_cost = best_alpha_cost

    case1_current_alpha_penalty_history = alpha_penalty_history["case1"]
    case2_current_alpha_penalty_history = alpha_penalty_history["case2"]
    case3_current_alpha_penalty_history = alpha_penalty_history["case3"]

    case1_z01_loss_history.append(case1_current_alpha_z01_loss_history[-1])
    case2_z01_loss_history.append(case2_current_alpha_z01_loss_history[-1])
    case3_z01_loss_history.append(case3_current_alpha_z01_loss_history[-1])

    case1_penalty_history.append(case1_current_alpha_penalty_history[-1])
    case2_penalty_history.append(case2_current_alpha_penalty_history[-1])
    case3_penalty_history.append(case3_current_alpha_penalty_history[-1])

    case1_alpha_best_inner, case1_alpha_best_particle = alpha_best_iteration["case1"]
    case2_alpha_best_inner, case2_alpha_best_particle = alpha_best_iteration["case2"]
    case3_alpha_best_inner, case3_alpha_best_particle = alpha_best_iteration["case3"]

    case1_current_alpha_best_iteration = f"{outer_iter}_{case1_alpha_best_inner}_{case1_alpha_best_particle}"
    case2_current_alpha_best_iteration = f"{outer_iter}_{case2_alpha_best_inner}_{case2_alpha_best_particle}"
    case3_current_alpha_best_iteration = f"{outer_iter}_{case3_alpha_best_inner}_{case3_alpha_best_particle}"

    # better alpha found for case1
    if case1_best_alpha_cost < case1_current_best['cost']: 
        print(f"[CASE1] Better alpha found in iteration {outer_iter}. Current Best Cost: {case1_best_alpha_cost:.2f}, Prev Best Cost: {case1_current_best['cost']:.2f}")
        case1_opt_alpha_found = True
        case1_alpha_ver += 1
        print(f"[CASE1] Updated Alpha Version: {case1_alpha_ver}")
        case1_current_best['outer_iter'] = outer_iter
        case1_current_best['cost'] = case1_best_alpha_cost
        case1_current_best['alphas'] = case1_current_best_alphas.copy()
        case1_current_best['omegas'] = case1_current_best['omegas'].copy()  # omegas remain the same
        case1_current_best['omegas_w_vb_frac'] = case1_current_best['omegas_w_vb_frac'].copy() # remaining the same
        case1_current_best_alpha_swarm = alpha_swarm[0]
        case1_alphas = case1_current_best_alphas.copy()

        case1_current_best_voltage_csv = f"case1_voltage_data_{case1_current_alpha_best_iteration}_{tb_date}.csv"
        shutil.copy(src=case1_current_best_voltage_csv, dst=os.path.join(current_best_path, f"alpha_{case1_current_best_voltage_csv}"))
        case1_current_best_sim_output_csv = f"sim_output_{case1_current_alpha_best_iteration}_{tb_date}.csv"
        shutil.copy(src=case1_current_best_sim_output_csv, dst=os.path.join(current_best_path, f"alpha_{case1_current_best_sim_output_csv}"))
    else:
        print(f"[CASE1] No better alpha found in iteration {outer_iter}. Current best cost: {case1_current_best['cost']:.2f}")
        case1_opt_alpha_found = False
        case1_alphas = case1_current_best['alphas'].copy()  # use the last best alphas
    
    if case2_best_alpha_cost < case2_current_best['cost']:
        print(f"[CASE2] Better alpha found in iteration {outer_iter}. Current Best Cost: {case2_best_alpha_cost:.2f}, Prev Best Cost: {case2_current_best['cost']:.2f}")
        case2_opt_alpha_found = True
        case2_alpha_ver += 1
        print(f"[CASE2] Updated Alpha Version: {case2_alpha_ver}")
        case2_current_best['outer_iter'] = outer_iter
        case2_current_best['cost'] = case2_best_alpha_cost
        case2_current_best['alphas'] = case2_current_best_alphas.copy()
        case2_current_best['omegas'] = case2_current_best['omegas'].copy() # omegas remain the same
        case2_current_best['omegas_w_vb_frac'] = case2_current_best['omegas_w_vb_frac'].copy() # remaining the same
        case2_current_best_alpha_swarm = alpha_swarm[1]
        case2_alphas = case2_current_best_alphas.copy()

        case2_current_best_voltage_csv = f"case2_voltage_data_{case2_current_alpha_best_iteration}_{tb_date}.csv"
        shutil.copy(src=case2_current_best_voltage_csv, dst=os.path.join(current_best_path, f"alpha_{case2_current_best_voltage_csv}"))
        case2_current_best_sim_output_csv = f"sim_output_{case2_current_alpha_best_iteration}_{tb_date}.csv"
        shutil.copy(src=case2_current_best_sim_output_csv, dst=os.path.join(current_best_path, f"alpha_{case2_current_best_sim_output_csv}"))
    else:
        print(f"[CASE2] No better alpha found in iteration {outer_iter}. Current best cost: {case2_current_best['cost']:.2f}")
        case2_opt_alpha_found = False
        case2_alphas = case2_current_best['alphas'].copy()
    
    if case3_best_alpha_cost < case3_current_best['cost']:
        print(f"[CASE3] Better alpha found in iteration {outer_iter}. Current Best Cost: {case3_best_alpha_cost:.2f}, Prev Best Cost: {case3_current_best['cost']:.2f}")
        case3_opt_alpha_found = True
        case3_alpha_ver += 1
        print(f"[CASE3] Updated Alpha Version: {case3_alpha_ver}")
        case3_current_best['outer_iter'] = outer_iter
        case3_current_best['cost'] = case3_best_alpha_cost
        case3_current_best['alphas'] = case3_current_best_alphas.copy()
        case3_current_best['omegas'] = case3_current_best['omegas'].copy()
        case3_current_best['omegas_w_vb_frac'] = case3_current_best['omegas_w_vb_frac'].copy()
        case3_current_best_alpha_swarm = alpha_swarm[2]
        case3_alphas = case3_current_best_alphas.copy()

        case3_current_best_voltage_csv = f"case3_voltage_data_{case3_current_alpha_best_iteration}_{tb_date}.csv"
        shutil.copy(src=case3_current_best_voltage_csv, dst=os.path.join(current_best_path, f"alpha_{case3_current_best_voltage_csv}"))
        case3_current_best_sim_output_csv = f"sim_output_{case3_current_alpha_best_iteration}_{tb_date}.csv"
        shutil.copy(src=case3_current_best_sim_output_csv, dst=os.path.join(current_best_path, f"alpha_{case3_current_best_sim_output_csv}"))
    
    else:
        print(f"[CASE3] No better alpha found in iteration {outer_iter}. Current best cost: {case3_current_best['cost']:.2f}")
        case3_opt_alpha_found = False
        case3_alphas = case3_current_best['alphas'].copy()

    alphas = {**case1_alphas, **case2_alphas, **case3_alphas}

    alphas_history.append({
        'outer_iter': outer_iter,
        'case1_alphas': case1_current_best['alphas'],
        'case1_cost': case1_current_best['cost'],
        'case2_alphas': case2_current_best['alphas'],
        'case2_cost': case2_current_best['cost'],
        'case3_alphas': case3_current_best['alphas'],
        'case3_cost': case3_current_best['cost'],
    })

    remove_csv_after_iteration(iteration=outer_iter, tb_date=tb_date)

end_opt_time = time.time()
opt_elapsed = end_opt_time - start_opt_time

case1_final_cost = case1_current_best['cost']
case2_final_cost = case2_current_best['cost']
case3_final_cost = case3_current_best['cost']

case1_final_best_outer_iter = case1_current_best['outer_iter']
case2_final_best_outer_iter = case2_current_best['outer_iter']
case3_final_best_outer_iter = case3_current_best['outer_iter']

def forward_fill_nan(values):
    values = np.array(values, dtype=float)
    for i in range(1, len(values)):
        if np.isnan(values[i]):
            values[i] = values[i-1]
    return values

case1_z01_loss_history = forward_fill_nan(case1_z01_loss_history)
case2_z01_loss_history = forward_fill_nan(case2_z01_loss_history)
case3_z01_loss_history = forward_fill_nan(case3_z01_loss_history)
case1_penalty_history = forward_fill_nan(case1_penalty_history)
case2_penalty_history = forward_fill_nan(case2_penalty_history)
case3_penalty_history = forward_fill_nan(case3_penalty_history)

# select between cases based on final costs
case_costs = {
    "case1": case1_final_cost,
    "case2": case2_final_cost,
    "case3": case3_final_cost
}

case_bests = {
    "case1": case1_current_best,
    "case2": case2_current_best,
    "case3": case3_current_best
}

final_selected_case = min(case_costs, key=case_costs.get)
final_expected_cost = case_costs[final_selected_case]
final_current_best = case_bests[final_selected_case]

final_alphas = final_current_best['alphas']
final_alpha_groups = {k: v for k, v in alpha_groups.items() if k.startswith(final_selected_case)}
final_sm_alphas = softmax_alpha(final_alphas, final_alpha_groups, temp)

case_alphas = {}
case_sm_alphas = {}
for case in ["case1", "case2", "case3"]:
    if case == final_selected_case:
        case_alphas[case] = 10.0
        case_sm_alphas[case] = 1.0
    else:
        case_alphas[case] = 0.0
        case_sm_alphas[case] = 0.0

final_omegas = final_current_best['omegas']
final_omegas_w_vb_frac = final_current_best['omegas_w_vb_frac']
final_best_outer_iter = final_current_best['outer_iter']

print(f"=== {opt_time} Optimization Summary ===")
print(f"Alpha-Omega Optimization Elapsed Time: {opt_elapsed:.2f}s")
print(f"Alpha-Omega Optimization Best Cost: {final_expected_cost:.4f}")
print(f"Final Selected {final_selected_case} from final cost (case1, case2, case3): {case1_final_cost:.2f}, {case2_final_cost:.2f}, {case3_final_cost:.2f}")

final_sm_alphas_w_case = {**final_sm_alphas, **case_sm_alphas}
alpha_convergence = check_alpha_convergence(selected_case=final_selected_case, current_best_sm_alphas=final_sm_alphas_w_case, always_selected_alphas=always_selected_alphas, alpha_groups=alpha_groups, temp=temp)
if not alpha_convergence:
    print(f"[Warn] Alpha convergence condition not met. Hard Selection Result may not be similar to Soft Selection Result.")

# 1) voltage_data
case1_best_voltage_csv = read_best_voltage_per_case_csv(case="case1", best_outer_iteration=f"{case1_final_best_outer_iter}", tb_date=tb_date, folder=current_best_path)
case1_voltage_filename = os.path.basename(case1_best_voltage_csv)
case2_best_voltage_csv = read_best_voltage_per_case_csv(case="case2", best_outer_iteration=f"{case2_final_best_outer_iter}", tb_date=tb_date, folder=current_best_path)
case2_voltage_filename = os.path.basename(case2_best_voltage_csv)
case3_best_voltage_csv = read_best_voltage_per_case_csv(case="case3", best_outer_iteration=f"{case3_final_best_outer_iter}", tb_date=tb_date, folder=current_best_path)
case3_voltage_filename = os.path.basename(case3_best_voltage_csv)
print(f"Best Voltage CSVs: {case1_best_voltage_csv}, {case2_best_voltage_csv}, {case3_best_voltage_csv}")
for i, filename in enumerate([case1_voltage_filename, case2_voltage_filename, case3_voltage_filename]):
    match = re.search(rf'_voltage_data_(\d+_\d+_\d+)_({tb_date})\.csv', filename)
    if match:
        if i == 0:
            case1_outer_inner_part = match.group(1)
            outer_iter, inner_iter, particle = case1_outer_inner_part.split('_')
            case1_final_best_sim = f"{case1_final_best_outer_iter}_{inner_iter}_{particle}"
        elif i == 1:
            case2_outer_inner_part = match.group(1)
            outer_iter, inner_iter, particle = case2_outer_inner_part.split('_')
            case2_final_best_sim = f"{case2_final_best_outer_iter}_{inner_iter}_{particle}"
        elif i == 2:
            case3_outer_inner_part = match.group(1)
            outer_iter, inner_iter, particle = case3_outer_inner_part.split('_')
            case3_final_best_sim = f"{case3_final_best_outer_iter}_{inner_iter}_{particle}"
    if not match:
        raise ValueError(f"Filename {filename} does not match expected pattern for voltage data.")

shutil.copy(src=case1_best_voltage_csv, dst=soft_case1_voltage_csv)
shutil.copy(src=case2_best_voltage_csv, dst=soft_case2_voltage_csv)
shutil.copy(src=case3_best_voltage_csv, dst=soft_case3_voltage_csv)
case1_voltage_df = pd.read_csv(case1_best_voltage_csv)
case2_voltage_df = pd.read_csv(case2_best_voltage_csv)
case3_voltage_df = pd.read_csv(case3_best_voltage_csv)

# 2) sim_output
case1_best_sim_output_csv = read_best_sim_output_csv(best_iteration=f"{case1_final_best_sim}", tb_date= tb_date, folder=current_best_path)
case2_best_sim_output_csv = read_best_sim_output_csv(best_iteration=f"{case2_final_best_sim}", tb_date= tb_date, folder=current_best_path)
case3_best_sim_output_csv = read_best_sim_output_csv(best_iteration=f"{case3_final_best_sim}", tb_date= tb_date, folder=current_best_path)
print(f"Best Simulation Output CSVs: {case1_best_sim_output_csv}, {case2_best_sim_output_csv}, {case3_best_sim_output_csv}")
shutil.copy(src=case1_best_sim_output_csv, dst=soft_case1_sim_output_csv)
shutil.copy(src=case2_best_sim_output_csv, dst=soft_case2_sim_output_csv)
shutil.copy(src=case3_best_sim_output_csv, dst=soft_case3_sim_output_csv)

if final_selected_case == "case1":
    soft_selection_voltage_df = pd.read_csv(case1_best_voltage_csv)
    soft_selection_sim_output_df = pd.read_csv(case1_best_sim_output_csv)
    soft_selection_sim_output_dict = soft_selection_sim_output_df.iloc[0].to_dict()
elif final_selected_case == "case2":
    soft_selection_voltage_df = pd.read_csv(case2_best_voltage_csv)
    soft_selection_sim_output_df = pd.read_csv(case2_best_sim_output_csv)
    soft_selection_sim_output_dict = soft_selection_sim_output_df.iloc[0].to_dict()
elif final_selected_case == "case3":
    soft_selection_voltage_df = pd.read_csv(case3_best_voltage_csv)
    soft_selection_sim_output_df = pd.read_csv(case3_best_sim_output_csv)
    soft_selection_sim_output_dict = soft_selection_sim_output_df.iloc[0].to_dict()

if task in ["singleBW", "dualBW"]:
    soft_selection_bw_fc1 = soft_selection_sim_output_dict[f"{final_selected_case}_cg_bw_fc1"]
    if isinstance(soft_selection_bw_fc1, str):
        soft_selection_bw_fc1 = ast.literal_eval(soft_selection_bw_fc1)
    if len(soft_selection_bw_fc1) == 2:
        soft_selection_bw_fc1_start, soft_selection_bw_fc1_end = soft_selection_bw_fc1
        if soft_selection_bw_fc1_start != soft_selection_bw_fc1_end:
            soft_selection_bw_fc1_exists = True
        else:
            soft_selection_bw_fc1_exists = False
    else:
        soft_selection_bw_fc1_exists = False

    if task in ["dualBW"]:
        soft_selection_bw_fc2 = soft_selection_sim_output_dict[f"{final_selected_case}_cg_bw_fc2"]
        if isinstance(soft_selection_bw_fc2, str):
            soft_selection_bw_fc2 = ast.literal_eval(soft_selection_bw_fc2)
        if len(soft_selection_bw_fc2) == 2:
            soft_selection_bw_fc2_start, soft_selection_bw_fc2_end = soft_selection_bw_fc2
            if soft_selection_bw_fc2_start != soft_selection_bw_fc2_end:
                soft_selection_bw_fc2_exists = True
            else:
                soft_selection_bw_fc2_exists = False
        else:
            soft_selection_bw_fc2_exists = False

# 1. Plot Soft-Selection: CG_dB vs freq
plt.figure(figsize=(12, 6))
x_freq = soft_selection_voltage_df['freq']
case1_y_cg_dB = case1_voltage_df['cg_dB']
case2_y_cg_dB = case2_voltage_df['cg_dB']
case3_y_cg_dB = case3_voltage_df['cg_dB']
soft_selected_cg_dB = soft_selection_voltage_df['cg_dB']
plt.plot(x_freq, case1_y_cg_dB, label=f"Case 1 (cost:{case1_final_cost:.2f})", marker='o', markersize=5)
plt.plot(x_freq, case2_y_cg_dB, label=f"Case 2 (cost:{case2_final_cost:.2f})", marker='o', markersize=5)
plt.plot(x_freq, case3_y_cg_dB, label=f"Case 3 (cost:{case3_final_cost:.2f})", marker='o', markersize=5)
plt.plot(x_freq, soft_selected_cg_dB, label=f"Selected Case-{final_selected_case}", color='black', linestyle='--', linewidth=2)
if task in ["singleBW", "dualBW"]:
    if soft_selection_bw_fc1_exists:
        plt.axvspan(soft_selection_bw_fc1_start, soft_selection_bw_fc1_end, color='skyblue', alpha=0.3, label=f"BW1 ({soft_selection_bw_fc1_start:.2e}, {soft_selection_bw_fc1_end:.2e})")
    if task == "dualBW":
        if soft_selection_bw_fc2_exists:
            plt.axvspan(soft_selection_bw_fc2_start, soft_selection_bw_fc2_end, color='lightcoral', alpha=0.3, label=f"BW2 ({soft_selection_bw_fc2_start:.2e}, {soft_selection_bw_fc2_end:.2e})")
elif task in ["singleGains", "dualGains"]:
    fc1_bw_target_lower, fc1_bw_target_upper = fc1_bw_target
    plt.axvspan(fc1_bw_target_lower, fc1_bw_target_upper, color='yellow', alpha=0.3, label=f"Freq range of interest ({fc1_bw_target_lower:.2e}, {fc1_bw_target_upper:.2e})")
    if task == "dualGains":
        fc2_bw_target_lower, fc2_bw_target_upper = fc2_bw_target
        plt.axvspan(fc2_bw_target_lower, fc2_bw_target_upper, color='green', alpha=0.3, label=f"Freq range of interest ({fc2_bw_target_lower:.2e}, {fc2_bw_target_upper:.2e})")
plt.axvline(x=fc1_target, color='green', linestyle='--', label=f'fc1_target({fc1_target:.2e})')
if task in ["dualBW", "dualGains", "dualGain"]:
    plt.axvline(x=fc2_target, color='orange', linestyle='--', label=f'fc2_target({fc2_target:.2e})')
plt.title("Soft-Selection: CG_dB vs Frequency")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Conversion Gain (dB)")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(folder_path, f"{opt_time}_soft_selection_cg_dB_vs_freq.png"), bbox_inches='tight')
plt.close()

# 2. alphas, omegas, and cost history to csv
alphas_history_df = pd.DataFrame([
    {
        'outer_iter': entry['outer_iter'],
        'case1_cost': entry['case1_cost'], **entry['case1_alphas'],
        'case2_cost': entry['case2_cost'], **entry['case2_alphas'],
        'case3_cost': entry['case3_cost'], **entry['case3_alphas']
    } for entry in alphas_history
]).set_index('outer_iter')

omegas_history_df = pd.DataFrame([
    {
        'outer_iter': entry['outer_iter'],
        'case1_cost': entry['case1_cost'], **entry['case1_omegas'],
        'case2_cost': entry['case2_cost'], **entry['case2_omegas'],
        'case3_cost': entry['case3_cost'], **entry['case3_omegas']
    } for entry in omegas_history
]).set_index('outer_iter')

alphas_history_df.to_csv(os.path.join(folder_path, f"{opt_time}_alphas_history.csv"))
omegas_history_df.to_csv(os.path.join(folder_path, f"{opt_time}_omegas_history.csv"))
cost_history_df = pd.DataFrame(cost_history)
cost_history_df.to_csv(os.path.join(folder_path, f"{opt_time}_cost_history.csv"))

# 3. Save soft-selection results (alpha)
with open(soft_selection_csv, 'w') as f:
    f.write(f"Selected case, Best Cost found at iteration, Best Cost (final_expected_cost)\n")
    f.write(f"{final_selected_case},{final_best_outer_iter}, {final_expected_cost}\n")
    f.write("Total Entire Iterations, Alpha Particles, Alpha Iterations, Omega Particles, Omega Iterations\n")
    f.write(f"{max_iterations}, {alpha_particles}, {alpha_iterations}, {omega_particles}, {omega_iterations}\n")
    f.write("Alpha Convergence Condition, Softmax Temperature\n")
    f.write(f"{alpha_convergence}, {temp}\n")
    f.write("Alpha Parameter, Alpha Value\n")
    for alpha_name, alpha_value in final_sm_alphas.items():
        f.write(f"{alpha_name}, {alpha_value}\n")
f.close()

# 4. Plot cost history with z01 loss history, penalty history
x_z01 = np.arange(len(case1_z01_loss_history)) # start from 0
plt.figure(figsize=(12, 6))
plt.plot(cost_history_df['outer_iter'], cost_history_df['case1_cost'], marker='o', markersize=5, label='Case 1', color='blue')
plt.plot(x_z01, case1_z01_loss_history, marker='x', markersize=5, label='Case 1 Z01 Loss', color='#66B2FF', linestyle='--')
plt.plot(cost_history_df['outer_iter'], cost_history_df['case2_cost'], marker='o', markersize=5, label='Case 2', color='orange')
plt.plot(x_z01, case2_z01_loss_history, marker='x', markersize=5, label='Case 2 Z01 Loss', color='#FFB84D', linestyle='--')
plt.plot(cost_history_df['outer_iter'], cost_history_df['case3_cost'], marker='o', markersize=5, label='Case 3', color='green')
plt.plot(x_z01, case3_z01_loss_history, marker='x', markersize=5, label='Case 3 Z01 Loss', color='#66CC99', linestyle='--')
if task in ["dualBW", "dualGain"]:
    n_outer = len(cost_history_df['outer_iter'])
    x_half = np.arange(0.5, n_outer+0.5, 0.5)
    if len(case1_penalty_history) != x_half.shape[0]:
        pdb.set_trace()
    plt.plot(x_half, case1_penalty_history, marker='s', markersize=5, label='Case 1 Penalty', color='#004C99', linestyle=':')
    plt.plot(x_half, case2_penalty_history, marker='s', markersize=5, label='Case 2 Penalty', color='#994C00', linestyle=':')
    plt.plot(x_half, case3_penalty_history, marker='s', markersize=5, label='Case 3 Penalty', color='#006633', linestyle=':')

plt.axhline(y=final_expected_cost, color='r', linestyle='--', label=f'Final Cost ({final_selected_case}): {final_expected_cost:.2f}')

handles, labels = plt.gca().get_legend_handles_labels()
label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
desired = []
for case in ["Case 1", "Case 2", "Case 3"]:
    for suffix in ["", " Z01 Loss", " Penalty"]:
        lbl = f"{case}{suffix}"
        if lbl in label_to_idx:
            desired.append(lbl)
finals = [lbl for lbl in labels if lbl.startswith("Final Cost")]
desired += finals
order = [label_to_idx[lbl] for lbl in desired]
plt.legend([handles[i] for i in order], [labels[i] for i in order], loc="best")

plt.title("Best Cost per Case vs Outer Iteration")
plt.xlabel("Outer Iteration")
plt.ylabel("Best Cost")
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(folder_path, f"{opt_time}_global_best_cost_history.png"), bbox_inches='tight')
plt.close()


cases = ["case1", "case2", "case3"]
z01_histories = {
    "case1": case1_z01_loss_history,
    "case2": case2_z01_loss_history,
    "case3": case3_z01_loss_history,
}

for case in cases:
    out_path = os.path.join(folder_path, f"{opt_time}_{case}_alpha_zero_one_loss_history.csv")
    with open(out_path, "w") as f:  
        writer = csv.writer(f)
        writer.writerow(["outer_iter", "z01_loss"])  
        for outer_iter, z01_value in enumerate(z01_histories[case], start=1):
            if z01_value is None:
                writer.writerow([f"Outer_{outer_iter}", "NaN"])
            else:
                writer.writerow([f"Outer_{outer_iter}", float(z01_value)])

# ####################### FINAL SELECTION & SIMULATION ##########################
# Set the selected case alpha to 1.0
hard_selected_sm_alphas = {sm_alphas:0.0 for sm_alphas in final_sm_alphas.keys()}
selected_topologies_alpha = []
for group, alpha_name_list in alpha_groups.items():
    case, stage = group.split('_')
    if case == final_selected_case:
        hard_selected_topology_alpha = max(alpha_name_list, key=lambda x: final_sm_alphas[x])
        hard_selected_sm_alphas[hard_selected_topology_alpha] = 1.0
        print(f"Selected topology for {stage}: {hard_selected_topology_alpha} with alpha {final_sm_alphas[hard_selected_topology_alpha]:.4f}")

        selected_topologies_alpha.append(hard_selected_topology_alpha)

for always_selected_alpha in always_selected_alphas.keys():
    case = always_selected_alpha.split("_")[0]
    if case == final_selected_case:
        selected_topologies_alpha.append(always_selected_alpha)

# ############ MAYBE NOT NEEDED? #########################
# # at least one alpha per group should be selected to run the simulation
# for alpha_group, alpha_list in alpha_groups.items():
#     selected = False
#     for alpha in alpha_list:
#         if hard_selected_sm_alphas[alpha] == 1.0:
#             selected = True
#             break
#     if not selected:
#         hard_selected_sm_alphas[alpha] = 1.0
# #########################################################

# 1. Run simulation with hard selected alphas
print("=== Hard Selection Simulation ===")
hard_selection_iteration_name = "finalb4opt"
start_time = time.time()
hard_selected_sm_alpha_vars = {**hard_selected_sm_alphas, **always_selected_alphas}
final_omegas_vars = {**final_omegas, **always_set_omega}
hard_selected_stage_dsfile, case1_neg_z_found, case1_nan_z_found, case2_neg_z_found, case2_nan_z_found, case3_neg_z_found, case3_nan_z_found = run_ads_simulation_per_case(simulations=simulations, instance_list=instance_list, sm_alphas=hard_selected_sm_alpha_vars, omegas=final_omegas_vars, wrk_space=wrk_space, tb_date=tb_date)
end_time = time.time()
sim_elapse = end_time - start_time
print(f"Final Simulation Elapsed: {sim_elapse:.2f}s")

sim_err_found = False
if final_selected_case == "case1":
    sim_err_found = case1_neg_z_found or case1_nan_z_found
elif final_selected_case == "case2":
    sim_err_found = case2_neg_z_found or case2_nan_z_found
elif final_selected_case == "case3":
    sim_err_found = case3_neg_z_found or case3_nan_z_found

if not sim_err_found:
    # save ds file
    shutil.copy(src=hard_selected_stage_dsfile, dst=os.path.join(folder_path, f"{opt_time}_hard_selected_sim_output.ds"))
    hard_selected_sim_output_dict = get_simulation_output(final_stage_dsfile=hard_selected_stage_dsfile, sm_alphas=hard_selected_sm_alphas, iteration=hard_selection_iteration_name,task=task, tb_date=tb_date)

    hard_selected_sim_output_df = pd.DataFrame([hard_selected_sim_output_dict])
    hard_selected_sim_output_df.to_csv(hard_selection_sim_output_csv, index=False)

    hard_selected_fom, weighted_z01_loss, weighted_penalty = FoM_z01_manual(
    sim_output_dict=hard_selected_sim_output_dict,
    selected_case=final_selected_case,
    task=task,
    is_final=False
    )

    hard_selected_cost = - (hard_selected_fom - weighted_z01_loss - weighted_penalty)
    print(f"Hard Selection Cost: {hard_selected_cost:.2f}")
    print(f"Expected Cost (Soft Selection): {final_expected_cost:.2f}")

    if abs(hard_selected_cost - final_expected_cost) > 5.0:
        print("[WARN] Hard Selected Cost after hard selection differs significantly from expected cost.")

    # 1) voltage data
    hard_selected_voltage_csv_files = read_best_voltage_csv(best_iteration=hard_selection_iteration_name, tb_date = tb_date, folder=".")
    for voltage_csv_file in hard_selected_voltage_csv_files:
        filename = os.path.basename(voltage_csv_file)
        if final_selected_case in filename:
            hard_selected_voltage_csv_file = voltage_csv_file
            shutil.copy(src=voltage_csv_file, dst=hard_selection_voltage_csv)
            break
    hard_selected_voltage_df = pd.read_csv(hard_selected_voltage_csv_file)

    # 2) sim output
    if task in ["singleBW", "dualBW"]:
        hard_selection_bw_fc1 = hard_selected_sim_output_dict[f"{final_selected_case}_cg_bw_fc1"]
        if len(hard_selection_bw_fc1) == 2:
            hard_selection_bw_fc1_start, hard_selection_bw_fc1_end = hard_selection_bw_fc1
            if hard_selection_bw_fc1_start != hard_selection_bw_fc1_end:
                hard_selection_bw_fc1_exists = True
            else:
                hard_selection_bw_fc1_exists = False
        else:
            hard_selection_bw_fc1_exists = False

        if task in ["dualBW"]:
            hard_selection_bw_fc2 = hard_selected_sim_output_dict[f"{final_selected_case}_cg_bw_fc2"]
            if len(hard_selection_bw_fc2) == 2:
                hard_selection_bw_fc2_start, hard_selection_bw_fc2_end = hard_selection_bw_fc2
                if hard_selection_bw_fc2_start != hard_selection_bw_fc2_end:
                    hard_selection_bw_fc2_exists = True
                else:
                    hard_selection_bw_fc2_exists = False
            else:
                hard_selection_bw_fc2_exists = False

    # 2. Plot CG_dB vs. frequency for the hard selected case
    plt.figure(figsize=(12, 6))
    x_freq = hard_selected_voltage_df["freq"]
    y_cg_dB = hard_selected_voltage_df["cg_dB"]
    plt.plot(x_freq, y_cg_dB, label=f"Hard-Selected(cost:{hard_selected_cost:.2f})", marker='o', markersize=5)
    plt.plot(x_freq, soft_selected_cg_dB, label=f"Soft-Selected (cost:{final_expected_cost:.2f})", linestyle='--', color='black')
    if task in ["singleBW", "dualBW"]:
        if soft_selection_bw_fc1_exists:
            plt.axvspan(soft_selection_bw_fc1_start, soft_selection_bw_fc1_end, color='skyblue', alpha=0.3, label=f"Soft-Selection BW1 ({soft_selection_bw_fc1_start:.2e}, {soft_selection_bw_fc1_end:.2e})")
        if hard_selection_bw_fc1_exists:
            plt.axvspan(hard_selection_bw_fc1_start, hard_selection_bw_fc1_end, color='royalblue', alpha=0.3, label=f"Hard-Selection BW1 ({hard_selection_bw_fc1_start:.2e}, {hard_selection_bw_fc1_end:.2e})")
        if task == "dualBW":
            if soft_selection_bw_fc2_exists:
                plt.axvspan(soft_selection_bw_fc2_start, soft_selection_bw_fc2_end, color='lightcoral', alpha=0.3, label=f"Soft-Selection BW2 ({soft_selection_bw_fc2_start:.2e}, {soft_selection_bw_fc2_end:.2e})")
            if hard_selection_bw_fc2_exists:
                plt.axvspan(hard_selection_bw_fc2_start, hard_selection_bw_fc2_end, color='indianred', alpha=0.3, label=f"Hard-Selection BW2 ({hard_selection_bw_fc2_start:.2e}, {hard_selection_bw_fc2_end:.2e})")
    elif task in ["singleGains", "dualGains"]:
        fc1_bw_target_lower, fc1_bw_target_upper = fc1_bw_target
        plt.axvspan(fc1_bw_target_lower, fc1_bw_target_upper, color='yellow', alpha=0.3, label=f"Freq range of interest ({fc1_bw_target_lower:.2e}, {fc1_bw_target_upper:.2e})")
        if task == "dualGains":
            fc2_bw_target_lower, fc2_bw_target_upper = fc2_bw_target
            plt.axvspan(fc2_bw_target_lower, fc2_bw_target_upper, color='green', alpha=0.3, label=f"Freq range of interest ({fc2_bw_target_lower:.2e}, {fc2_bw_target_upper:.2e})")
    plt.axvline(x=fc1_target, color='green', linestyle='--', label=f'fc1_target({fc1_target:.2e})')
    if task in ["dualBW", "dualGains", "dualGain"]:
        plt.axvline(x=fc2_target, color='orange', linestyle='--', label=f'fc2_target({fc2_target:.2e})')
    plt.title(f"Hard Selected vs Soft-Selected: CG_dB vs Frequency (Cost: {hard_selected_cost:.2f})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Conversion Gain (dB)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt_save_dir = os.path.join(folder_path, f"{opt_time}_hard_selected_cg_vs_freq.png")
    plt.savefig(plt_save_dir, bbox_inches='tight')
    plt.close()

else:
    print("[ERROR] Negative Z values found or NaN found in the final hard selection simulation. Exiting.")
    hard_selected_cost = 1e6
    sys.exit(1)
    
# 3. Store the final selection in csv file
final_omegas_selected_only = defaultdict(float) # for omegas optimization (should not include always_set_omega)
with open(hard_selection_csv, "w") as f:
    f.write("[CASE1] Best Cost found at iteration, [CASE2] Best Cost found at iteration, [CASE3] Best Cost found at iteration, Final_Cost, Expected_Cost, Difference_Cost\n")
    f.write(f"{case1_current_best['outer_iter']}, {case2_current_best['outer_iter']}, {case3_current_best['outer_iter']}, {hard_selected_cost:.2f}, {final_expected_cost:.2f}, {abs(hard_selected_cost - final_expected_cost):.2f}\n")
    f.write("Total Entire Iterations, Alpha Iterations, Alpha Particles, Omega Iterations, Omega Particles, Elapsed time for optimization (s)\n")
    f.write(f"{len(cost_history['outer_iter'])}, {alpha_iterations}, {alpha_particles}, {omega_iterations}, {omega_particles}, {opt_elapsed:.2f}\n")
    f.write(f"Selected Case, MN1, LNA, MN2, ADD, MN3, MX\n")
    f.write(f"{final_selected_case},")
    for stage in stages:
        for topology_alpha in selected_topologies_alpha:
            if topology_alpha.startswith(f"{final_selected_case.lower()}_{stage.lower()}"):
                f.write(f"{topology_alpha},")
    f.write("\n")

    f.write("Omega Parameter,Value\n")
    for topo_component, value in final_omegas_vars.items(): # including always_set_omega
        if "." in topo_component:
            selected_case_topology, selected_topology_component = topo_component.split(".")
            if selected_case_topology.lower() in selected_topologies_alpha:
                if "MN" in selected_case_topology and "_P" in selected_case_topology:
                    f.write(f"{topo_component},{value}\n") # parameters in _P

                    selected_n_case_topology = selected_case_topology.replace("_P","_N")
                    n_topo_component = f"{selected_n_case_topology}.{selected_topology_component}"
                    n_topo_value = final_omegas_vars[n_topo_component]
                    f.write(f"{n_topo_component},{n_topo_value}\n") # parameters in _N

                    if topo_component not in always_set_omega.keys(): # always_set_omega not in final_omegas
                        final_omegas_selected_only[topo_component] = value
                        final_omegas_selected_only[n_topo_component] = n_topo_value

                else:
                    f.write(f"{topo_component},{value}\n")
                    if topo_component not in always_set_omega.keys():
                        final_omegas_selected_only[topo_component] = value

        else: # Voltage parameters
            if topo_component.startswith(final_selected_case):
                selected_case, vol_type, instance = topo_component.split("_", maxsplit=2)
                if "mnlo" in instance.lower():
                    instance = instance.lower().replace("mnlo_","") # remove mnlo_ prefix
                if f"{selected_case}_{instance}" in selected_topologies_alpha:
                    f.write(f"{topo_component},{value}\n")

                    if topo_component not in always_set_omega.keys(): 
                        vol_vb_frac = final_omegas_w_vb_frac[topo_component]
                        final_omegas_selected_only[topo_component] = vol_vb_frac # store vb_frac version of voltages
    
    f.write("Alpha Parameter,Value\n")
    for alpha_name, value in final_sm_alphas.items():
        f.write(f"{alpha_name},{value}\n")

f.close()

remove_csv_after_iteration(iteration=hard_selection_iteration_name, tb_date=tb_date)

# ####################### FINAL OMEGA BO OPTIMIZATION ##########################
outer_iter = "finalopt"
print("=== Final Hard Selection w/ Optimized Omega===")
alphas = hard_selected_sm_alphas # should be the softmax value # always selected alphas will be added inside the objective function
param_pbounds_selected_only = {k: v for k, v in param_pbounds.items() if k in final_omegas_selected_only.keys()} # contains omegas_w_vb_frac
param_names_selected_only, param_pso_bounds_selected_only = pso_bounds(param_pbounds_selected_only)
opt_start_time = time.time()
prev_best_omegas_w_vb_frac = np.array([final_omegas_selected_only[k] for k in param_names_selected_only])
prev_best_cost = hard_selected_cost
opt_best_omega_cost, opt_best_omega_pos, opt_cost_history, opt_z01_loss_history, opt_penalty_history, opt_best_iteration, opt_swarm = pso_with_nan_control(
    objective_fn=objective_omega,
    bounds=param_pso_bounds_selected_only,
    n_particles=opt_omega_particles,
    n_iterations=opt_omega_iterations,
    param_names=param_names_selected_only,
    crash_sentinel=None,
    max_retry_per_particle=5,
    prev_init_pos=prev_best_omegas_w_vb_frac,
    prev_best_cost=prev_best_cost,
    custom_initializer=None,
    prev_swarm=None
)
opt_end_time = time.time()
opt_elapsed = opt_end_time - opt_start_time
print(f"Final Omega Optimization Elapsed Time: {opt_elapsed:.2f}s")

# if better cost found
if opt_best_omega_cost < hard_selected_cost:
    opt_best_omegas_w_vb_frac = dict(zip(param_names_selected_only, opt_best_omega_pos))
    opt_best_omegas = convert_Vb_from_Vb_frac(opt_best_omegas_w_vb_frac)

    opt_best_inner, opt_best_particle = opt_best_iteration
    opt_best_iteration = f"{outer_iter}_{opt_best_inner}_{opt_best_particle}"
    print(f"Best omega found at {opt_best_iteration}")

    print(f"Final Optimized Omega Cost: {opt_best_omega_cost:.2f}")
    print(f"Expected Cost (Soft Selection): {final_expected_cost:.2f}")

    # 1) voltage data
    opt_voltage_csv_files = read_best_voltage_csv(best_iteration=opt_best_iteration, tb_date = tb_date, folder=".")
    for voltage_csv_file in opt_voltage_csv_files:
        filename = os.path.basename(voltage_csv_file)
        if final_selected_case in filename:
            opt_voltage_csv_file = voltage_csv_file
            shutil.copy(src=voltage_csv_file, dst=opt_selection_voltage_csv)
            break
    opt_voltage_df = pd.read_csv(opt_voltage_csv_file)

    # 2) sim_output
    opt_sim_output_csv = read_best_sim_output_csv(best_iteration=opt_best_iteration, tb_date = tb_date, folder=".")
    shutil.copy(src=opt_sim_output_csv, dst=opt_selection_sim_output_csv)
    opt_sim_output_df = pd.read_csv(opt_sim_output_csv)
    opt_sim_output_dict = opt_sim_output_df.iloc[0].to_dict()

    if task in ["singleBW", "dualBW"]:
        opt_bw_fc1 = opt_sim_output_dict[f"{final_selected_case}_cg_bw_fc1"]
        if isinstance(opt_bw_fc1, str):
            opt_bw_fc1 = ast.literal_eval(opt_bw_fc1)
        if len(opt_bw_fc1) == 2:
            opt_bw_fc1_start, opt_bw_fc1_end = opt_bw_fc1
            if opt_bw_fc1_start != opt_bw_fc1_end:
                opt_bw_fc1_exists = True
            else:
                opt_bw_fc1_exists = False
        else:
            opt_bw_fc1_exists = False

        if task in ["dualBW"]:
            opt_bw_fc2 = opt_sim_output_dict[f"{final_selected_case}_cg_bw_fc2"]
            if isinstance(opt_bw_fc2, str):
                opt_bw_fc2 = ast.literal_eval(opt_bw_fc2)
            if len(opt_bw_fc2) == 2:
                opt_bw_fc2_start, opt_bw_fc2_end = opt_bw_fc2
                if opt_bw_fc2_start != opt_bw_fc2_end:
                    opt_bw_fc2_exists = True
                else:
                    opt_bw_fc2_exists = False
            else:
                opt_bw_fc2_exists = False

    # 1. Plot CG_dB vs. frequency for the hard selected case vs. w/ optimized omega
    x_freq = opt_voltage_df["freq"]
    y_opt_cg_dB = opt_voltage_df["cg_dB"]

    plt.figure(figsize=(12, 6))
    plt.plot(x_freq, y_cg_dB, label=f"Hard-Selected (cost: {hard_selected_cost:.4f})", color='m', marker='o', markersize=5)
    plt.plot(x_freq, soft_selected_cg_dB, label=f"Soft-Selected (cost: {final_expected_cost:.4f})", linestyle='--', color='black')
    plt.plot(x_freq, y_opt_cg_dB, label=f"Hard-Selected w/ Optimized Omega (cost: {opt_best_omega_cost:.4f})", color='blue', marker='x', markersize=5)
    if task in ["singleBW", "dualBW"]:
        if soft_selection_bw_fc1_exists:
            plt.axvspan(soft_selection_bw_fc1_start, soft_selection_bw_fc1_end, color='skyblue', alpha=0.3, label=f"Soft-Selection BW1 ({soft_selection_bw_fc1_start:.2e}, {soft_selection_bw_fc1_end:.2e})")
        if hard_selection_bw_fc1_exists:
            plt.axvspan(hard_selection_bw_fc1_start, hard_selection_bw_fc1_end, color='royalblue', alpha=0.3, label=f"Hard-Selection BW1 ({hard_selection_bw_fc1_start:.2e}, {hard_selection_bw_fc1_end:.2e})")
        if opt_bw_fc1_exists:
            plt.axvspan(opt_bw_fc1_start, opt_bw_fc1_end, color='deepskyblue', alpha=0.3, label=f"Optimized Omega BW1 ({opt_bw_fc1_start:.2e}, {opt_bw_fc1_end:.2e})")
        if task == "dualBW":
            if soft_selection_bw_fc2_exists:
                plt.axvspan(soft_selection_bw_fc2_start, soft_selection_bw_fc2_end, color='lightcoral', alpha=0.3, label=f"Soft-Selection BW2 ({soft_selection_bw_fc2_start:.2e}, {soft_selection_bw_fc2_end:.2e})")
            if hard_selection_bw_fc2_exists:
                plt.axvspan(hard_selection_bw_fc2_start, hard_selection_bw_fc2_end, color='indianred', alpha=0.3, label=f"Hard-Selection BW2 ({hard_selection_bw_fc2_start:.2e}, {hard_selection_bw_fc2_end:.2e})")
            if opt_bw_fc2_exists:
                plt.axvspan(opt_bw_fc2_start, opt_bw_fc2_end, color='orangered', alpha=0.3, label=f"Optimized Omega BW2 ({opt_bw_fc2_start:.2e}, {opt_bw_fc2_end:.2e})")
    elif task in ["singleGains", "dualGains"]:
        fc1_bw_target_lower, fc1_bw_target_upper = fc1_bw_target
        plt.axvspan(fc1_bw_target_lower, fc1_bw_target_upper, color='yellow', alpha=0.3, label=f"Freq range of interest ({fc1_bw_target_lower:.2e}, {fc1_bw_target_upper:.2e})")
        if task == "dualGains":
            fc2_bw_target_lower, fc2_bw_target_upper = fc2_bw_target
            plt.axvspan(fc2_bw_target_lower, fc2_bw_target_upper, color='green', alpha=0.3, label=f"Freq range of interest ({fc2_bw_target_lower:.2e}, {fc2_bw_target_upper:.2e})")
    plt.axvline(x=fc1_target, color='green', linestyle='--', label=f'fc1_target({fc1_target:.2e})')
    if task in ["dualBW", "dualGains", "dualGain"]:
        plt.axvline(x=fc2_target, color='orange', linestyle='--', label=f'fc2_target({fc2_target:.2e})')
    plt.title("Hard Selected Case w/ Optimized Omega: CG_dB vs Frequency")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Conversion Gain (dB)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt_save_dir = os.path.join(folder_path, f"{opt_time}_hard_selected_w_optimized_omega_cg_vs_freq.png")
    plt.savefig(plt_save_dir, bbox_inches='tight')
    plt.close()

    #2. Store the final selection with optimization in csv file
    opt_best_omegas_vars = {**opt_best_omegas, **always_set_omega} # include always_set_omega
    with open(opt_selection_csv, "w") as f:
        f.write("Final_Optimized_Cost, Final_Cost, Difference_Cost\n")
        f.write(f"{opt_best_omega_cost:.2f}, {hard_selected_cost:.2f}, {abs(hard_selected_cost - opt_best_omega_cost):.2f}\n")
        f.write("Additional Omega Iterations, Additional Omega Particles, Elapsed time for optimization (s)\n")
        f.write(f"{opt_omega_iterations}, {opt_omega_particles}, {opt_elapsed:.2f}\n")
        f.write(f"Selected Case, MN1, LNA, MN2, ADD, MN3, MX\n")
        f.write(f"{final_selected_case},")
        for stage in stages:
            for topology in selected_topologies_alpha:
                if topology.startswith(f"{final_selected_case}_{stage}"):
                    f.write(f"{topology},")
        f.write("\n")

        f.write("Omega Parameter,Value\n")
        for topo_component, value in opt_best_omegas_vars.items():
            if "." in topo_component:
                selected_case_topology, selected_topology_component = topo_component.split(".")
                if selected_case_topology.lower() in selected_topologies_alpha:
                    if "MN" in selected_case_topology and "_P" in selected_case_topology:
                        f.write(f"{topo_component},{value}\n") # parameters in _P
                        selected_n_case_topology = selected_case_topology.replace("_P","_N")
                        n_topo_component = f"{selected_n_case_topology}.{selected_topology_component}"
                        n_topo_value = opt_best_omegas_vars[n_topo_component]
                        f.write(f"{n_topo_component},{n_topo_value}\n") # parameters in _N
                    else:
                        f.write(f"{topo_component},{value}\n")
            else: # Voltage parameters
                if topo_component.startswith(final_selected_case):
                    f.write(f"{topo_component},{value}\n")
    f.close()

else:
    print("Cost did not improve after final optimization. Remain best cost:", hard_selected_cost)

remove_csv_after_iteration(iteration=outer_iter, tb_date=tb_date)