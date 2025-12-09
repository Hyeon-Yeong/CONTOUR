import numpy as np
from collections import defaultdict
import pdb

def generate_pbounds(topology_dict, topology_per_case, pbounds_per_component, step_sizes):
    alpha_low = 0.0
    alpha_high = 10.0
    alpha_pbounds = {"case1": (alpha_low, alpha_high), "case2": (alpha_low, alpha_high), "case3": (alpha_low, alpha_high)}
    param_pbounds = {}
    param_step_sizes = {}
    instance_list = []
    always_selected_alpha=[]

    alpha_groups = defaultdict(list) 
    alpha_groups ["per_case"] = ["case1", "case2", "case3"]
    
    for case_stage, topology_list in topology_per_case.items():
        case, stage, IOtype = case_stage.split("_")
        for topology in topology_list:
            topology_type, topology_num = topology.split("_")
            topo_w_case = f"{case}_{stage}_{topology_num}"

            # Extract bounds for each component in the topology
            component_dict = topology_dict[topology]
            for component in component_dict.keys():
                matched = False
                if component.startswith("L") and component[1:].isdigit(): # Length value
                        param_pbound = pbounds_per_component["Length"]
                        step_size = step_sizes["Length"]
                        matched = True
                elif component.startswith("W") and component[1:].isdigit(): # Width value
                        param_pbound = pbounds_per_component["Width"]
                        step_size = step_sizes["Width"]
                        matched = True
                else:
                    for component_type in sorted(pbounds_per_component.keys(), key=lambda x: len(x), reverse=True):
                        if component.startswith(component_type):
                            matched = True
                            param_pbound = pbounds_per_component[component_type]
                            step_size = step_sizes[component_type]
                            break
            
                if matched:
                    if ("MN" in topology) and (IOtype.endswith("d")):
                        if "Vb" in component: # MN does not have Vdd
                            voltage_P_name=f"{case}_{component}_{stage.lower()}_{topology_num}_p"
                            # voltage_N_name=f"{case}_{component}_{stage.lower()}_{topology_num}_n" # Vb_n = Vb_p
                            param_pbounds[f"{voltage_P_name}"] = param_pbound
                            # param_pbounds[f"{voltage_N_name}"] = param_pbound
                            param_step_sizes[f"{voltage_P_name}"] = step_size
                        else:
                            param_P_name = f"{topo_w_case}_P"
                            param_N_name = f"{topo_w_case}_N"

                            param_pbounds[f"{param_P_name}.{component}"] = param_pbound
                            param_pbounds[f"{param_N_name}.{component}"] = param_pbound
                            param_step_sizes[f"{param_P_name}.{component}"] = step_size
                            param_step_sizes[f"{param_N_name}.{component}"] = step_size

                            instance_list.append(param_P_name)
                            instance_list.append(param_N_name)

                    elif (topology == "ADD_sd1") and "Vb1" in component:
                        vb1_add_sd1_bound = pbounds_per_component["Vb1_add_sd1"]
                        voltage_name = f"{case}_{component}_{stage.lower()}_{topology_num}"
                        param_pbounds[f"{voltage_name}"] = vb1_add_sd1_bound
                        param_step_sizes[f"{voltage_name}"] = step_size

                    else:
                        if "Vdd" in component or "Vb" in component:
                            voltage_name=f"{case}_{component}_{stage.lower()}_{topology_num}"
                            param_pbounds[f"{voltage_name}"] = param_pbound
                            param_step_sizes[f"{voltage_name}"] = step_size
                        else:
                            param_name = f"{topo_w_case}"
                            param_pbounds[f"{param_name}.{component}"] = param_pbound
                            param_step_sizes[f"{param_name}.{component}"] = step_size

                            instance_list.append(param_name)
            
            if topology_num == "0": # topology_num 0 doesn't have any components (just wire)
                instance_list.append(topo_w_case) 

            # Alpha parameters for each topology
            alpha_group = f"{case}_{stage}"
            if len(topology_list) > 1: # stage with one topology is set to "1.0"
                if ("MN" in topology) and (IOtype.endswith("d")):
                    alpha_name = f"{topo_w_case}_P"
                    alpha_name = alpha_name.lower()
                else:
                    alpha_name = f"{topo_w_case}"
                    alpha_name = alpha_name.lower()
                alpha_pbounds[f"{alpha_name}"] = (alpha_low, alpha_high)
                alpha_groups[alpha_group].append(alpha_name)
            else:
                alpha_name = f"{topo_w_case}"
                alpha_name = alpha_name.lower()
                always_selected_alpha.append(alpha_name)

    instance_list = list(set(instance_list))  # Remove duplicates from instance_list

    return param_pbounds, param_step_sizes, alpha_pbounds, alpha_groups, instance_list, always_selected_alpha

def generate_pbounds_ver2(topology_dict, topology_per_case, pbounds_per_component, step_sizes):
    alpha_low = 0.0
    alpha_high = 10.0
    alpha_pbounds = {"case1": (alpha_low, alpha_high), "case2": (alpha_low, alpha_high), "case3": (alpha_low, alpha_high)}
    param_pbounds = {}
    param_step_sizes = {}
    instance_list = []
    always_selected_alpha=[]
    always_set_omega={}

    alpha_groups = defaultdict(list) 
    # alpha_groups ["per_case"] = ["case1", "case2", "case3"]
    
    for case_stage, topology_list in topology_per_case.items():
        case, stage, IOtype = case_stage.split("_")
        for topology in topology_list:
            topology_type, topology_num = topology.split("_")
            topo_w_case = f"{case}_{stage}_{topology_num}"

            # Extract bounds for each component in the topology
            component_dict = topology_dict[topology]

            for component in component_dict.keys():
                matched = False
                if topology == "MN_4" and component == "MN_C":
                        param_pbound = pbounds_per_component["MN_4_C"]
                        matched = True
                elif component.startswith("L") and component[1:].isdigit(): # Length value
                        param_pbound = pbounds_per_component["Length"]
                        step_size = step_sizes["Length"]
                        matched = True
                elif component.startswith("W") and component[1:].isdigit(): # Width value
                        param_pbound = pbounds_per_component["Width"]
                        step_size = step_sizes["Width"]
                        matched = True
                else:
                    for component_type in sorted(pbounds_per_component.keys(), key=lambda x: len(x), reverse=True):
                        if component.startswith(component_type):
                            matched = True
                            param_pbound = pbounds_per_component[component_type]
                            step_size = step_sizes[component_type]
                            break
            
                if matched:
                    if ("MN" in topology) and (IOtype.endswith("d")):
                        if "Vb" in component: # MN does not have Vdd
                            voltage_P_name=f"{case}_{component}_{stage.lower()}_{topology_num}_p"
                            # voltage_N_name=f"{case}_{component}_{stage.lower()}_{topology_num}_n" # Vb_n = Vb_p
                            param_pbounds[f"{voltage_P_name}"] = param_pbound
                            # param_pbounds[f"{voltage_N_name}"] = param_pbound
                            param_step_sizes[f"{voltage_P_name}"] = step_size
                        else:
                            param_P_name = f"{topo_w_case}_P"
                            param_N_name = f"{topo_w_case}_N"

                            param_pbounds[f"{param_P_name}.{component}"] = param_pbound
                            param_pbounds[f"{param_N_name}.{component}"] = param_pbound
                            param_step_sizes[f"{param_P_name}.{component}"] = step_size
                            param_step_sizes[f"{param_N_name}.{component}"] = step_size

                            instance_list.append(param_P_name)
                            instance_list.append(param_N_name)

                    elif (topology == "ADD_sd1") and "Vb1" in component:
                        vb1_add_sd1_bound = pbounds_per_component["Vb1_add_sd1"]
                        voltage_name = f"{case}_{component}_{stage.lower()}_{topology_num}"
                        param_pbounds[f"{voltage_name}"] = vb1_add_sd1_bound
                        param_step_sizes[f"{voltage_name}"] = step_size

                    else:
                        if "Vdd" in component or "Vb" in component:
                            voltage_name=f"{case}_{component}_{stage.lower()}_{topology_num}"
                            param_pbounds[f"{voltage_name}"] = param_pbound
                            param_step_sizes[f"{voltage_name}"] = step_size
                        else:
                            param_name = f"{topo_w_case}"
                            param_pbounds[f"{param_name}.{component}"] = param_pbound
                            param_step_sizes[f"{param_name}.{component}"] = step_size

                            instance_list.append(param_name)
            
            if topology_num == "0": # topology_num 0 doesn't have any components (just wire)
                instance_list.append(topo_w_case) 

            # Alpha parameters for each topology
            alpha_group = f"{case}_{stage}"
            if len(topology_list) > 1: # stage with one topology is set to "1.0"
                if ("MN" in topology) and (IOtype.endswith("d")):
                    alpha_name = f"{topo_w_case}_P"
                    alpha_name = alpha_name.lower()
                else:
                    alpha_name = f"{topo_w_case}"
                    alpha_name = alpha_name.lower()
                alpha_pbounds[f"{alpha_name}"] = (alpha_low, alpha_high)
                alpha_groups[alpha_group].append(alpha_name)
            else:
                alpha_name = f"{topo_w_case}"
                alpha_name = alpha_name.lower()
                always_selected_alpha.append(alpha_name)

    # print("param_pbounds components before removing always_set_omega:", len(param_pbounds.keys()))
    always_set_omega_names = []
    for component, pbounds in param_pbounds.items():
        lower_bound, upper_bound = pbounds
        if lower_bound == upper_bound:
            always_set_omega[component] = lower_bound
            always_set_omega_names.append(component)
            # print(f"Removing {component} from param_pbounds as it is always set to {lower_bound}.")
    for component in always_set_omega_names:
        del param_pbounds[component]
        del param_step_sizes[component]
    # print("param_pbounds components after removing always_set_omega:", len(param_pbounds.keys()))

    instance_list = list(set(instance_list))  # Remove duplicates from instance_list

    return param_pbounds, param_step_sizes, alpha_pbounds, alpha_groups, instance_list, always_selected_alpha, always_set_omega

# BO format -> PSO format
def pso_bounds(bounds_dict):
    keys = list(bounds_dict.keys())
    bounds_array = np.array([bounds_dict[k] for k in keys])
    lower_bounds = bounds_array[:, 0]
    upper_bounds = bounds_array[:, 1]
    return keys, (lower_bounds, upper_bounds)

def sade_bounds(bounds_dict):
    bounds = []
    keys = list(bounds_dict.keys())
    for key in keys:
        lower, upper = bounds_dict[key]
        bounds.append([lower, upper])
    return keys, np.array(bounds)

## TO DO: THIS ONLY WORKS WHEN VDD IS FIXED (in always_set_omega)
# Vb_frac bounds to Vb bounds
def make_pbounds_with_vb(bounds_dict, always_set_omega):
    new_bounds = {}
    for key, (low, high) in bounds_dict.items():
        if "Vb" in key:
            vb_component_name_list = key.split("_")
            for i, vb_component_name in enumerate(vb_component_name_list):
                if "Vb" in vb_component_name:
                    voltage_component_idx = i
                    vb_name = vb_component_name_list[voltage_component_idx]
                    break
            vdd_component_name = key.replace(vb_name, "Vdd")
            if vdd_component_name in always_set_omega:
                vdd_value = always_set_omega[vdd_component_name]
            else:
                print(f"Warning: {vdd_component_name} not found in always_set_omega. Using default Vdd value of 2.5.")
                vdd_value = 2.5
            new_bounds[key] = (low * vdd_value, high * vdd_value)
        else:
            new_bounds[key] = (low, high)
    return new_bounds

def generate_initial_params_dict(pbounds_dict, rng=None):
    if rng is None:
        rng = np.random.default_rng()  # Random if no seed

    initial_params = {}
    for case_w_topology, (low, high) in pbounds_dict.items():
        val = rng.uniform(low, high)
        initial_params[case_w_topology] = val
    return initial_params


def generate_initial_alphas_dict_biased(alphas_dict, alpha_groups):
    initial_alphas = {}

    for case_stage, alpha_name_list in alpha_groups.items():
        rand_high_idx = np.random.randint(0, len(alpha_name_list))
        rand_high_alpha = alpha_name_list[rand_high_idx]
        low, high = alphas_dict[rand_high_alpha]
        rand_high_alpha_val = np.random.uniform(max(high-4.0, low), high)
        initial_alphas[rand_high_alpha] = rand_high_alpha_val

        for alpha_name in alpha_name_list:
            if alpha_name != rand_high_alpha:
                low, high = alphas_dict[alpha_name]
                rand_alpha_val = np.random.uniform(low, min(low + 4.0, rand_high_alpha_val))
                initial_alphas[alpha_name] = rand_alpha_val
    
    ################ For debugging purposes ################
    if len(initial_alphas.keys()) != len(alphas_dict.keys()):
        print("Warning: The number of parameters in initial_alphas_dict and alphas_dict are different.")
        print(f"Initial alphas keys: {list(initial_alphas.keys())}")
        print(f"Alphas dict keys: {list(alphas_dict.keys())}")
    ########################################################

    return initial_alphas

def generate_initial_alphas_dict(alphas_dict, alpha_groups): # 0721 TRIAL
    initial_alphas = {}

    for alpha_name, (low, high) in alphas_dict.items():
        # initial_alphas[alpha_name] = np.random.uniform(low, high)
        initial_alphas[alpha_name] = 5.0
    
    ################ For debugging purposes ################
    if len(initial_alphas.keys()) != len(alphas_dict.keys()):
        print("Warning: The number of parameters in initial_alphas_dict and alphas_dict are different.")
        print(f"Initial alphas keys: {list(initial_alphas.keys())}")
        print(f"Alphas dict keys: {list(alphas_dict.keys())}")
    ########################################################

    return initial_alphas

def generate_initial_alphas_dict_random(alphas_dict, alpha_groups):
    initial_alphas = {}

    for alpha_name, (low, high) in alphas_dict.items():
        initial_alphas[alpha_name] = np.random.uniform(4.0, 6.0)
        # initial_alphas[alpha_name] = np.random.uniform(low, high)

    
    ################ For debugging purposes ################
    if len(initial_alphas.keys()) != len(alphas_dict.keys()):
        print("Warning: The number of parameters in initial_alphas_dict and alphas_dict are different.")
        print(f"Initial alphas keys: {list(initial_alphas.keys())}")
        print(f"Alphas dict keys: {list(alphas_dict.keys())}")
    ########################################################

    return initial_alphas

def softmax_alpha(alpha_dict, alpha_groups, temp):
    softmax_alphas = {}
    
    for case_stage, alpha_name_list in alpha_groups.items():
        vals_list = []
        for alpha_name in alpha_name_list:
            vals_list.append(alpha_dict[alpha_name])
        vals_list = np.array(vals_list)
        softmax_vals = np.exp(vals_list / temp) / np.sum(np.exp(vals_list/temp))
        for alpha_name, sm_val in zip(alpha_name_list, softmax_vals):
            softmax_alphas[alpha_name] = sm_val
        
    ################ For debugging purposes ################
    if softmax_alphas.keys() != alpha_dict.keys():
        print("Warning: The keys in softmax_alphas and alpha_dict do not match.")
        print(f"length of softmax_alphas: {len(softmax_alphas.keys())}, length of alpha_dict: {len(alpha_dict.keys())}")
        print(f"softmax_alphas keys: {list(softmax_alphas.keys())}")
        print(f"alpha_dict keys: {list(alpha_dict.keys())}")
    ########################################################

    return softmax_alphas

def softmax_alpha_w_temp(alpha_dict, alpha_groups, temp):
    softmax_alphas = {}
    t_opt_candidates = []
    alpha_groups_vals = []
    alpha_groups_names = []

    for case_stage, alpha_name_list in alpha_groups.items():
        vals_list = []
        for alpha_name in alpha_name_list:
            vals_list.append(alpha_dict[alpha_name])
        vals_list = np.array(vals_list)
        alpha_groups_vals.append(vals_list)
        alpha_groups_names.append(alpha_name_list)

        t_opt = adjust_temp(vals_list, temp)
        t_opt_candidates.append(t_opt)

    t_opt = np.min(t_opt_candidates)
    if abs(t_opt - temp) > 1e-5:
        print(f"Adjusted temperature from {temp} to {t_opt} for softmax calculation.")

    for vals_list, alpha_name_list in zip(alpha_groups_vals, alpha_groups_names):
        shift_vals_list = vals_list - np.max(vals_list)
        softmax_vals = np.exp(shift_vals_list / t_opt) / np.sum(np.exp(shift_vals_list / t_opt))
        for alpha_name, sm_val in zip(alpha_name_list, softmax_vals):
            softmax_alphas[alpha_name] = sm_val

    ################ For debugging purposes ################
    if softmax_alphas.keys() != alpha_dict.keys():
        print("Warning: The keys in softmax_alphas and alpha_dict do not match.")
        print(f"softmax_alphas keys: {list(softmax_alphas.keys())}")
        print(f"alpha_dict keys: {list(alpha_dict.keys())}")
    ########################################################

    return softmax_alphas, t_opt

from scipy.optimize import brentq
def adjust_temp(alphas_per_group, current_temp, target_sm_max=0.95):
    alphas_list = np.array(alphas_per_group)
    d = alphas_list-np.max(alphas_list)
    def f(T):
        return 1/np.sum(np.exp(d/T))-target_sm_max
    
    if f(0.001) < 0: # solution is not in the range [0.001, current_temp]
        return 0.001 
    
    if f(current_temp) > 0: # current_temp already satisfies the condition
        return current_temp
    
    t_opt = brentq(f, 0.001, current_temp)
    return t_opt

def sigmoid_alpha(alpha_dict):
    sigmoid_alphas = {}
    for alpha_name, alpha_value in alpha_dict.items():
        sigmoid_alphas[alpha_name] = 1 / (1 + np.exp(-alpha_value))
    return sigmoid_alphas

def convert_Vb_from_Vb_frac (omega_dict):
    """
    Vb value after optimization is a "fraction" of Vdd (in range [0.0, 1.0]).
    Therefore, convert Vb = Vb_frac * Vdd. (range [0.0, Vdd]).
    """
    new_omega_dict = omega_dict.copy()
    # Process Vb parameters
    for param, value in omega_dict.items():
        if "add_sd1" in param and "Vb1" in param:
            add_sd1_Vdd_param = param.replace("Vb1", "Vdd")
            if value > omega_dict[add_sd1_Vdd_param]:
                value = omega_dict[add_sd1_Vdd_param]  # Ensure Vb1_add_sd1 <= Vdd_add_sd1
            new_omega_dict[param] = value
            
        elif "Vb" in param:
            if "Vb_mnlo" in param:
                Vb_frac_value = value
                Vdd_param = param.replace("Vb_mnlo", "Vdd")
            else:
                Vb_frac_value = value
                Vb_parts = param.split("_")
                Vb_parts[1] = "Vdd"
                Vdd_param = "_".join(Vb_parts)

            if Vdd_param not in omega_dict.keys(): # MN topology may not have Vdd
                vdd_val = 2.5
            else:
                vdd_val = omega_dict[Vdd_param]
            
            vb_val = Vb_frac_value * vdd_val  # Scale Vb to be a fraction of Vdd. Vb should be less than or equal to Vdd
            new_omega_dict[param] = vb_val

    ################ For debugging purposes ################
    if len(omega_dict.keys()) != len(new_omega_dict.keys()):
        print("Warning: The number of parameters in the original omega_dict and new_omega_dict are different.")
        print(f"Original omega_dict keys: {list(omega_dict.keys())}")
        print(f"New omega_dict keys: {list(new_omega_dict.keys())}")
    ########################################################

    return new_omega_dict

def revert_Vb_to_Vb_frac(omega_dict):
    """
    Revert Vb values back to their fractional form: Vb_frac = Vb / Vdd.
    """
    reverted_dict = omega_dict.copy()

    for param, value in omega_dict.items():
        if "add_sd1" in param and "Vb1" in param:
            # Skip conversion: this Vb1 is already in absolute voltage
            continue

        elif "Vb" in param:
            if "Vb_mnlo" in param:
                Vdd_param = param.replace("Vb_mnlo", "Vdd")
            else:
                Vb_parts = param.split("_")
                Vb_parts[1] = "Vdd"
                Vdd_param = "_".join(Vb_parts)

            if Vdd_param not in omega_dict:
                vdd_val = 2.5
            else:
                vdd_val = omega_dict[Vdd_param]

            vb_frac = value / vdd_val if vdd_val != 0 else 0.0
            reverted_dict[param] = vb_frac

    return reverted_dict
