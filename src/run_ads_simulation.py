import subprocess
import json
import sys
from collections import defaultdict
from sim_utils import modify_netlist, run_simulation, modify_netlist_manual
import os
import time
import traceback
from tqdm import tqdm

import pdb

#######################################################################
ads_python = "/usr/local/packages/ADS2023/tools/python/bin/python3.10" # USE YOUR LOCATION OF ADS PYTHON HERE
#######################################################################

def run_ads_simulation_per_case(simulations, instance_list, wrk_space, tb_date, sm_alphas, omegas):
    try:
        tot_sum = 0
        for simulation in tqdm(simulations, desc="ADS Simulations"):

            neg_z_found = False
            nan_z_found = False

            # tqdm.write(f"Running simulation: {tb_name}")
            modified_netlist = modify_netlist(
                simulation_name = simulation,
                instance_list = instance_list, 
                sm_alphas=sm_alphas,
                omegas= omegas,
                wrk_space= wrk_space,
                tb_date= tb_date
                )
            wait_until_file_stable(modified_netlist, timeout=60, interval=0.5)
            # tqdm.write(f"DONE Modified netlist file: {modified_netlist}")

            tb_name = f"{simulation}_{tb_date}"
            # tb_name = f"{simulation}_0618" ################################################################# HARD CODED FOR NOW

            dsfile = run_simulation(modified_netlistfile_dir=modified_netlist, tb_name=tb_name, wrk_space=wrk_space)
            wait_until_file_stable(dsfile, timeout=60, interval=0.5)
            # tqdm.write(f"DONE Simulation dsfile: {dsfile}")

            if "HB_Z" in simulation:
                cmd = [ads_python, "/home/local/ace/hy7557/rf_rx_darts/catch_neg_Z_per_case.py", dsfile]
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    case1_neg_z_found, case1_nan_z_found, case2_neg_z_found, case2_nan_z_found, case3_neg_z_found, case3_nan_z_found = json.loads(result.stdout.strip())

                    if case1_neg_z_found or case2_neg_z_found or case3_neg_z_found:
                        print(f"[ERROR] [CASE1] Negative Z values found in simulation {simulation} : {case1_neg_z_found}")
                        print(f"[ERROR] [CASE2] Negative Z values found in simulation {simulation} : {case2_neg_z_found}")
                        print(f"[ERROR] [CASE3] Negative Z values found in simulation {simulation} : {case3_neg_z_found}")
                    if case1_nan_z_found or case2_nan_z_found or case3_nan_z_found:
                        print(f"[ERROR] [CASE1] NaN Z values found in simulation {simulation} : {case1_nan_z_found}")
                        print(f"[ERROR] [CASE2] NaN Z values found in simulation {simulation} : {case2_nan_z_found}")
                        print(f"[ERROR] [CASE3] NaN Z values found in simulation {simulation} : {case3_nan_z_found}")

                        return None, case1_neg_z_found, case1_nan_z_found, case2_neg_z_found, case2_nan_z_found, case3_neg_z_found, case3_nan_z_found

                except subprocess.CalledProcessError as e:
                    print("[ERROR] Negative Z check script crashed.")
                    print("Command:", " ".join(cmd))
                    print("Return code:", e.returncode)
                    print("STDERR:\n", e.stderr)
                    print("STDOUT:\n", e.stdout)
                    raise RuntimeError("Negative Z check script failed.") from e
                
                # JUST FOR VERIFICATION ##########################
                # with open(f"ads_simulation_{tb_name}_debug.json", "w") as errfile:
                #     errfile.write(result.stderr)
                #     print("simulation STDERR written to a file")
            tot_sum += 1
        return dsfile, case1_neg_z_found, case1_nan_z_found, case2_neg_z_found, case2_nan_z_found, case3_neg_z_found, case3_nan_z_found
    
    except Exception as e:
        print(f"[ERROR] Unexpected error in run_ads_simulation: {e}")
        traceback.print_exc()
        pdb.set_trace()
        raise

def get_simulation_output(final_stage_dsfile, sm_alphas, iteration, task, tb_date):

    output_dict = defaultdict(dict)
    # Raw output of ADS simulation e.g. Conversion Gain (CG) at 2.4GHz and 5GHz, S11 bandwidth, etc.
    iteration = str(iteration)
    if "0618" in tb_date:
        cmd = [ads_python, "/home/local/ace/hy7557/rf_rx_darts/read_ads_dsfile_0618.py", final_stage_dsfile, iteration, task, tb_date]
        try: 
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print("[ERROR] Simulation script crashed.")
            print("Command:", " ".join(cmd))
            print("Return code:", e.returncode)
            print("STDERR:\n", e.stderr)
            print("STDOUT:\n", e.stdout)
            raise RuntimeError("Simulation script failed.") from e
    elif "0729" in tb_date:
        cmd = [ads_python, "/home/local/ace/hy7557/rf_rx_darts/read_ads_dsfile_0729.py", final_stage_dsfile, iteration, task, tb_date]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print("[ERROR] Simulation script crashed.")
            print("Command:", " ".join(cmd))
            print("Return code:", e.returncode)
            print("STDERR:\n", e.stderr)
            print("STDOUT:\n", e.stdout)
            raise RuntimeError("Simulation script failed.") from e
    # JUST FOR VERIFICATION ##########################
    # with open(f"final_stage_dsfile_read_debug.json", "w") as errfile:
    #     errfile.write(result.stderr)
    #     print("get_simulation_output STDERR written to a file")

    # print("STDERR:\n", result.stderr)
    ##################################################

    try:
        output_dict = json.loads(result.stdout)
        # output_dict["case1"] = sm_alphas["case1"]
        # output_dict["case2"] = sm_alphas["case2"]
        # output_dict["case3"] = sm_alphas["case3"]

        # JUST FOR VERIFICATION ##########################
        # print("Parsed ADS output:", output_dict)
        ##################################################

    except Exception as e:
        print("Error parsing output:", e)
        print("Raw output:", result.stdout)
        pdb.set_trace()
        raise
    
    return output_dict

def wait_until_file_stable(filepath, timeout, interval):
    start_time = time.time()
    last_mtime = None
    last_size = None

    while True:
        if not os.path.exists(filepath):
            time.sleep(interval)
            continue

        stat = os.stat(filepath)
        current_mtime = stat.st_mtime
        current_size = stat.st_size

        if last_mtime == current_mtime and last_size == current_size:
            return  # file is stable
        else:
            last_mtime = current_mtime
            last_size = current_size

        if time.time() - start_time > timeout:
            raise TimeoutError(f"{filepath} did not stabilize within {timeout} seconds.")

        time.sleep(interval)

# for topology set case
def run_ads_simulation_manual(tb_name, tb_date, instance_list, wrk_space, sm_alphas, omegas):
    try:

        neg_z_found = False
        nan_z_found = False

        netlist_name = f"{tb_name}_{tb_date}"
        # print(f"Running simulation: {tb_name}")
        modified_netlist = modify_netlist_manual(
            netlist_name = netlist_name,
            instance_list = instance_list,
            sm_alphas=sm_alphas,
            omegas=omegas,
            wrk_space=wrk_space
            )
        wait_until_file_stable(modified_netlist, timeout=60, interval=0.5)
        # print("DONE Modified netlist file:", modified_netlist)
        dsfile = run_simulation(modified_netlistfile_dir=modified_netlist, tb_name=netlist_name, wrk_space=wrk_space)
        wait_until_file_stable(dsfile, timeout=60, interval=0.5)
        # print("DONE Simulation dsfile:", dsfile)

        return dsfile, neg_z_found, nan_z_found
    
    except Exception as e:
        print(f"[ERROR] Unexpected error in run_ads_simulation: {e}")
        traceback.print_exc()
        pdb.set_trace()
        raise

def get_simulation_output_manual(final_stage_dsfile, selected_case, iteration, task, tb_date):

    output_dict = defaultdict(dict)
    # Raw output of ADS simulation e.g. Conversion Gain (CG) at 2.4GHz and 5GHz, S11 bandwidth, etc.
    iteration = str(iteration)
    cmd = [ads_python, "/home/local/ace/hy7557/rf_rx_darts/read_ads_dsfile_manual.py", final_stage_dsfile, selected_case, iteration, task, tb_date]
    try: 
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("[ERROR] READ script crashed.")
        print("Command:", " ".join(cmd))
        print("Return code:", e.returncode)
        print("STDERR:\n", e.stderr)
        print("STDOUT:\n", e.stdout)
        # JUST FOR VERIFICATION ##########################
        with open(f"final_stage_dsfile_read_debug.json", "w") as errfile:
            errfile.write(e.stderr)
            print("STDERR written to a file")

        print("STDERR:\n", e.stderr)
        ##################################################
        raise RuntimeError("Simulation script failed.") from e



    try:
        output_dict = json.loads(result.stdout)
        # JUST FOR VERIFICATION ##########################
        # print("Parsed ADS output:", output_dict)
        ##################################################

    except Exception as e:
        print("Error parsing output:", e)
        print("Raw output:", result.stdout)
    
    return output_dict