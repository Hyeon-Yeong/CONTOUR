# omega optimization for Manual 6
# manual_circuit = "Manual6"

import time
import os
import numpy as np
import pandas as pd
import matplotlib
import ast
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
from collections import defaultdict
import torch
import pdb
import argparse

from circuit_dict import fc1_target, fc2_target, fc1_bw_target, fc2_bw_target
from ads_sim_test_utils import alpha_omega_settings, manual_alpha_omega_settings
from circuit_dict_utils import generate_pbounds, generate_pbounds_ver2, convert_Vb_from_Vb_frac, revert_Vb_to_Vb_frac
from circuit_dict_utils import generate_initial_params_dict
from circuit_dict_utils import pso_bounds
from circuit_dict import topology_dict, topology_per_case, pbounds_per_component, stages, step_sizes
from run_ads_simulation import run_ads_simulation_manual, get_simulation_output_manual
from bo_fom import FoM_z01_manual
from rf_rx_bo_utils import read_best_voltage_csv_manual,read_best_sim_output_csv, remove_csv_after_iteration
from rf_rx_pso_utils import pso_with_nan_control

parser = argparse.ArgumentParser()
parser.add_argument("--omega_particles", type=int, default=18, help="Number of particles for omega optimization")
parser.add_argument("--omega_iterations", type=int, default=20, help="Number of iterations for omega optimization")
parser.add_argument("--task", type=str, default="singleBW", choices=["dualBW", "dualGain", "dualGains", "dualGainsBW","singleBW", "singleGain", "singleGains"], help="Task type for optimization")
parser.add_argument("--base_needed", type=bool, default=False, help="Whether to run base simulation before optimization")
parser.add_argument("--Manual_circuit", type=str, default="Manual6", choices=["Manual3", "Manual6", "Manual1"], help="Manual circuit name")
parser.add_argument("--tb_date", type=str, default="0618", choices = ["0618", "a_0618", "b_0618", "a_0729", "b_0729", "c_0729", "d_0729"], help="Testbench type for simulation data")
args= parser.parse_args()

wrk_space = "/home/local/ace/hy7557/rf_rx_0306_wrk"
task = args.task
temp = 1.0
opt_type="PSO"
manual_circuit = args.Manual_circuit
outer_iter = f"{manual_circuit}{opt_type}"
tb_date = args.tb_date

omega_iterations = args.omega_iterations
omega_particles = args.omega_particles

base_needed = args.base_needed

opt_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
folder_path = f"./{manual_circuit}_{opt_type}_result/{manual_circuit}/{task}_{opt_time}_{omega_particles}particles_{omega_iterations}iter"
os.makedirs(folder_path, exist_ok=True)

manual_opt_sim_output_csv = os.path.join(folder_path, f"{opt_time}_{manual_circuit}_sim_output.csv")
manual_opt_voltage_csv = os.path.join(folder_path, f"{opt_time}_{manual_circuit}_voltage_data.csv")
manual_opt_selection_csv = os.path.join(folder_path, f"{opt_time}_{manual_circuit}_selection.csv")

selected_alphas_list, selected_omegas = manual_alpha_omega_settings(test_circuit=manual_circuit)
param_pbounds, param_step_sizes, alpha_pbounds, alpha_groups, instance_list, always_selected_alphas, always_set_omega = generate_pbounds_ver2(topology_dict, topology_per_case, pbounds_per_component, step_sizes)

alphas = {k: 1.0 for k in selected_alphas_list}
sm_alphas = alphas.copy()

param_pbounds_selected_only = {}
for k, v in param_pbounds.items():
    if k in always_set_omega.keys():
        print(f"Skipping {k} as it is always set to {v}. Not included in PSO bounds.")
        continue
    elif k in selected_omegas.keys():
        param_pbounds_selected_only[k] = v

param_names, param_pso_bounds = pso_bounds(param_pbounds_selected_only)

cases = ["case1", "case2", "case3"]
for case in cases:
    if case in selected_alphas_list:
        selected_case = case
        break

if len(selected_alphas_list) != (len(stages)+1):
    raise ValueError("Number of selected alphas does not match the number of stages. Please check your input.")


if base_needed:
    print(f"=== Manual Selection {manual_circuit} Baseline ===")
    manual_base_sim_output_csv = os.path.join(f"./{manual_circuit}_{opt_type}_result/{manual_circuit}", f"{manual_circuit}_sim_output.csv")
    manual_base_voltage_csv = os.path.join(f"./{manual_circuit}_{opt_type}_result/{manual_circuit}", f"{manual_circuit}_voltage_data.csv")

    sim_start_time = time.time()
    base_final_stage_dsfile, neg_z_found, nan_z_found = run_ads_simulation_manual(tb_name=manual_circuit, tb_date=tb_date, instance_list=instance_list, wrk_space=wrk_space, sm_alphas=sm_alphas, omegas=selected_omegas)

    base_sim_output_dict = get_simulation_output_manual(final_stage_dsfile=base_final_stage_dsfile, selected_case=selected_case, iteration=manual_circuit, task=task, tb_date=tb_date)

    base_fom, weighted_z01_loss, weighted_penalty = FoM_z01_manual(sim_output_dict=base_sim_output_dict, selected_case=selected_case, task=task, is_final=True)
    base_cost = -(base_fom - weighted_z01_loss - weighted_penalty)
    sim_end_time = time.time()
    sim_elapsed = sim_end_time - sim_start_time
    print(f"Base Simulation Elapsed Time: {sim_elapsed:.2f}s")
    print(f"Base Cost: {base_cost}, Base FoM: {base_fom}, Weighted Z01 Loss: {weighted_z01_loss}, Weighted Penalty: {weighted_penalty}")

    base_sim_output_df = pd.DataFrame([base_sim_output_dict], columns=base_sim_output_dict.keys())
    base_sim_output_df.to_csv(f"./sim_output_{manual_circuit}_{tb_date}.csv", index=False)
    shutil.copy(src=f"./sim_output_{manual_circuit}_{tb_date}.csv", dst=manual_base_sim_output_csv)

    base_voltage_data_csv = read_best_voltage_csv_manual(best_iteration=manual_circuit, tb_date=tb_date, folder=".")
    shutil.copy(src=base_voltage_data_csv, dst=manual_base_voltage_csv)
    base_voltage_df = pd.read_csv(manual_base_voltage_csv)

    remove_csv_after_iteration(iteration=manual_circuit, tb_date=tb_date)

print(f"=== Starting {opt_type} for {manual_circuit} with {omega_particles} particles and {omega_iterations} iterations ===")


def objective_omega(inner_iter, particle_idx, **omega_vars):

    global sm_alphas 
    global outer_iter 

    neg_z_found = False
    nan_z_found = False

    omegas = convert_Vb_from_Vb_frac(omega_vars)
    omegas = {**omegas, **always_set_omega}  # Always set omega values

    try:
        final_stage_dsfile, neg_z_found, nan_z_found = run_ads_simulation_manual(
            tb_name=manual_circuit,
            tb_date=tb_date,
            instance_list=instance_list,
            wrk_space=wrk_space,
            sm_alphas=sm_alphas,
            omegas=omegas
        )
    except Exception as e:
        print(f"[SIM FAIL] Particle {particle_idx}: {e}")
        print("omega_vars:", omega_vars)
        print("omegas:", omegas)
        pdb.set_trace()
        return None

    if nan_z_found:
        print("[WARN] NaN impedance exists")
        return None  
    if neg_z_found:
        print("[WARN] Skipping candidate due to negative impedance.")
        fom = -1e6
        weighted_z01_loss = 0.0
        return fom, weighted_z01_loss

    iteration_point = f"{outer_iter}_{inner_iter}_{particle_idx}"
    print("Will be written with file name (_iteration point):", iteration_point)
    sim_output_dict = get_simulation_output_manual(final_stage_dsfile=final_stage_dsfile, selected_case=selected_case, iteration=iteration_point, task=task, tb_date=tb_date)

    sim_output_df = pd.DataFrame([sim_output_dict])
    sim_output_df.to_csv(f"./sim_output_{iteration_point}_{tb_date}.csv", index=False)

    fom, weighted_z01_loss, weighted_penalty = FoM_z01_manual(
        sim_output_dict=sim_output_dict,
        selected_case=selected_case,
        task=task,
        is_final=False
    )

    return fom, weighted_z01_loss, weighted_penalty

opt_start_time = time.time()
print("Number of parameters to optimize:", len(param_names))
opt_best_omega_cost, opt_best_omega_pos, opt_cost_history, opt_z01_loss_history, opt_penalty_history, opt_best_iteration, opt_swarm = pso_with_nan_control(
    objective_fn=objective_omega,
    bounds=param_pso_bounds,
    n_particles=omega_particles,
    n_iterations=omega_iterations,
    param_names=param_names,
    crash_sentinel=None,
    max_retry_per_particle=5,
    prev_init_pos=None,
    prev_best_cost=None,
    custom_initializer=None,
    prev_swarm=None
)
opt_end_time = time.time()
opt_elapsed = opt_end_time - opt_start_time
print(f"Final Omega Optimization Elapsed Time: {opt_elapsed:.2f}s")
print(f"Best Omega Cost: {opt_best_omega_cost}")

opt_best_omegas_w_vb_frac = dict(zip(param_names, opt_best_omega_pos))
opt_best_omegas = convert_Vb_from_Vb_frac(opt_best_omegas_w_vb_frac)
opt_best_inner, opt_best_particle = opt_best_iteration
opt_best_iteration = f"{outer_iter}_{opt_best_inner}_{opt_best_particle}"
print(f"Final Omega Optimization Elapsed Time: {opt_elapsed:.2f}s")
print(f"Best Omega Cost: {opt_best_omega_cost}")
print(f"Best Omegas: {opt_best_omegas}")

# 1) voltage_data
opt_voltage_csv_file = read_best_voltage_csv_manual(best_iteration=opt_best_iteration, tb_date=tb_date, folder=".")
shutil.copy(src=opt_voltage_csv_file, dst=manual_opt_voltage_csv)
opt_voltage_df = pd.read_csv(manual_opt_voltage_csv)

# 2) sim_output
opt_sim_output_csv = read_best_sim_output_csv(best_iteration=opt_best_iteration, tb_date=tb_date, folder=".")
shutil.copy(src=opt_sim_output_csv, dst=manual_opt_sim_output_csv)
opt_sim_output_df = pd.read_csv(manual_opt_sim_output_csv)
opt_sim_output_dict = opt_sim_output_df.iloc[0].to_dict()
if task in ["singleBW", "dualBW", "dualGainsBW"]:
    if base_needed:
        base_bw_fc1 = base_sim_output_dict[f"{selected_case}_cg_bw_fc1"]
        if len(base_bw_fc1) == 2:
            base_bw_fc1_start, base_bw_fc1_end = base_bw_fc1
            if base_bw_fc1_start != base_bw_fc1_end:
                base_bw_fc1_exists = True
            else:
                base_bw_fc1_exists = False
        else:
            base_bw_fc1_exists = False

    opt_bw_fc1 = opt_sim_output_dict[f"{selected_case}_cg_bw_fc1"]
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

    if task in ["dualBW", "dualGainsBW"]:
        if base_needed:
            base_bw_fc2 = opt_sim_output_dict[f"{selected_case}_cg_bw_fc2"]
            if len(base_bw_fc2) == 2:
                base_bw_fc2_start, base_bw_fc2_end = base_bw_fc2
                if base_bw_fc2_start != base_bw_fc2_end:
                    base_bw_fc2_exists = True
                else:
                    base_bw_fc2_exists = False
            else:
                base_bw_fc2_exists = False

        opt_bw_fc2 = opt_sim_output_dict[f"{selected_case}_cg_bw_fc2"]
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

# 3. Manual vs. PSO Comparison
# 1) CG_dB vs. Freq
plt.figure(figsize=(12, 6))
x_freq = opt_voltage_df["freq"]
if base_needed:
    y_cg_base = base_voltage_df["cg_dB"]
    plt.plot(x_freq, y_cg_base, label=f"Base CG_dB (cost:{base_cost:.4f})", linestyle='--', marker="x", markersize=5, color='black')
y_cg_opt = opt_voltage_df["cg_dB"]
if task in ["singleBW", "dualBW", "dualGainsBW"]:
    if base_needed and base_bw_fc1_exists:
        plt.axvspan(base_bw_fc1_start, base_bw_fc1_end, color='skyblue', alpha=0.3, label=f"Base BW1 ({base_bw_fc1_start:.2e}, {base_bw_fc1_end:.2e})")
    if opt_bw_fc1_exists:
        plt.axvspan(opt_bw_fc1_start, opt_bw_fc1_end, color='royalblue', alpha=0.3, label=f"Opt BW1 ({opt_bw_fc1_start:.2e}, {opt_bw_fc1_end:.2e})")
    if task == "dualBW":
        if base_needed and base_bw_fc2_exists:
            plt.axvspan(base_bw_fc2_start, base_bw_fc2_end, color='lightcoral', alpha=0.3, label=f"Base BW2 ({base_bw_fc2_start:.2e}, {base_bw_fc2_end:.2e})")
        if opt_bw_fc2_exists:
            plt.axvspan(opt_bw_fc2_start, opt_bw_fc2_end, color='indianred', alpha=0.3, label=f"Opt BW2 ({opt_bw_fc2_start:.2e}, {opt_bw_fc2_end:.2e})")
elif task in ["singleGains", "dualGains", "dualGainsBW"]:
    fc1_bw_target_lower, fc1_bw_target_upper = fc1_bw_target
    plt.axvspan(fc1_bw_target_lower, fc1_bw_target_upper, color='yellow', alpha=0.3, label=f"Freq range of interest ({fc1_bw_target_lower:.2e}, {fc1_bw_target_upper:.2e})")
    if task == "dualGains":
        fc2_bw_target_lower, fc2_bw_target_upper = fc2_bw_target
        plt.axvspan(fc2_bw_target_lower, fc2_bw_target_upper, color='green', alpha=0.3, label=f"Freq range of interest ({fc2_bw_target_lower:.2e}, {fc2_bw_target_upper:.2e})")
plt.axvline(x=fc1_target, color='green', linestyle='--', label=f'fc1_target({fc1_target:.2e})')
if task in ["dualBW", "dualGains", "dualGain", "dualGainsBW"]:
    plt.axvline(x=fc2_target, color='orange', linestyle='--', label=f'fc2_target({fc2_target:.2e})')
plt.plot(x_freq, y_cg_opt, label=f"Opt CG_dB (cost:{opt_best_omega_cost:.4f})",  marker='o', markersize=5, linestyle='-', color='blue')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Conversion Gain (dB)")
plt.title(f"Conversion Gain Comparison - {manual_circuit} ({task}, {opt_type})")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(folder_path, f"{manual_circuit}_{opt_type}_omega_opt.png"))
plt.close()

# 2) save the best omega selection
with open(manual_opt_selection_csv, 'w') as f:
    if base_needed:
        f.write("Best Cost, Baseline Cost, Cost Improvement\n")
        f.write(f"{opt_best_omega_cost},{base_cost},{base_cost - opt_best_omega_cost}\n")
        f.write("Base Cost, Base FoM, Base Weighted Z01 Loss\n")
        f.write(f"{base_cost},{base_fom},{weighted_z01_loss}\n")
    else:
        f.write("Best Cost\n")
        f.write(f"{opt_best_omega_cost}\n")
    f.write("Best Inner Iteration, Elapsed time (s), Omega Inner Iterations, Omega Particles\n")
    f.write(f"{opt_best_inner},{opt_elapsed},{omega_iterations},{omega_particles}\n")
    f.write("fc1_target,fc2_target,fc1_bw_target,fc2_bw_target\n")
    if task in ["dualBW","dualGains", "dualGainsBW"]:
        f.write(f"{fc1_target},{fc2_target},{fc1_bw_target},{fc2_bw_target}\n")
    elif task in ["singleBW","singleGains"]:
        f.write(f"{fc1_target},, {fc1_bw_target},\n")
    elif task == "dualGain":
        f.write(f"{fc1_target},{fc2_target},,,\n")
    elif task == "singleGain":
        f.write(f"{fc1_target},,,\n")
    f.write("Omega Parameter,Value\n")
    for param, value in opt_best_omegas.items():
        f.write(f"{param},{value}\n")
    f.write("Always Set Omega Parameter,Value\n")
    for param, value in always_set_omega.items():
        f.write(f"{param},{value}\n")
f.close()

# 3) Best Cost vs. Iteration
plt.figure(figsize=(12, 6))
plt.plot(opt_cost_history, label='Global Best Cost', marker='o', markersize=5, color='blue')
if task in ["dualBW", "dualGain"]:
    plt.plot(opt_penalty_history, label='Global Best Penalty', marker='x', linestyle='--', markersize=5, color='green')
plt.xlabel("Iteration")
plt.ylabel("Global Best Cost")
plt.title(f"Global Best Cost History - {manual_circuit} ({task}, {opt_type})")
if base_needed:
    plt.axhline(y=base_cost, color='black', linestyle='--', label=f"Base Cost: {base_cost:.2f}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(folder_path, f"{manual_circuit}_{opt_type}_omega_cost_history.png"))
plt.close()

remove_csv_after_iteration(iteration=outer_iter, tb_date=tb_date)