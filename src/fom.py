import math
import numpy as np
import pdb
from circuit_dict import fc1_target, fc2_target
gamma = 500

# for topology set case
def FoM_z01_manual(sim_output_dict, selected_case, task, is_final=False):
    ghz_scale = 1e9  # 1GHz
    global gamma
    lambda_ = 0.05
    beta_ = 1.0

    def convert_cg_to_dB(cg_linear):
        return 20 * np.log10(cg_linear) if cg_linear > 0 else float("nan")
    
    def convert_bw_to_fbw_dB(bw_linear, fc_target):
        if bw_linear <= 0:
            return 10 * np.log10(1e-6 / fc_target)
        else:
            return 10 * np.log10(bw_linear / fc_target)
        
    def convert_fbw_to_dB(fbw_linear):
        return 10 * np.log10(fbw_linear) if fbw_linear > 0 else float("nan")
    
    def penalty_for_equalization(cg_linear1, cg_linear2, lambda_):
        cg_dB1 = convert_cg_to_dB(cg_linear1)
        cg_dB2 = convert_cg_to_dB(cg_linear2)
        return lambda_ * (cg_dB1 - cg_dB2) ** 2

    if task == "dualBW":

        # case1_cg_bw = sim_output_dict["case1_cg_bw"]
        cg_bw_fc1 = sim_output_dict[f"{selected_case}_cg_bw_fc1"]
        cg_bw_fc2 = sim_output_dict[f"{selected_case}_cg_bw_fc2"]
        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))
        cg_fc2 = sim_output_dict.get(f"{selected_case}_cg_fc2", float("nan"))

        # cg
        cg_bw_fc1_mag = cg_bw_fc1[1] - cg_bw_fc1[0] if len(cg_bw_fc1) == 2 else 0.0
        cg_bw_fc2_mag = cg_bw_fc2[1] - cg_bw_fc2[0] if len(cg_bw_fc2) == 2 else 0.0

        if cg_bw_fc1_mag < 0 or cg_bw_fc2_mag < 0:
            print("[WARN] Invalid bandwidth detected (negative). Setting FoM to 0.")
            return 0.0

        # avg_cg = (cg_fc1 + cg_fc2) / 2 if cg_fc1 and cg_fc2 else float("nan")
        # avg_cg_dB = convert_cg_to_dB(avg_cg)

        # avg_fbw = (cg_fbw_fc1 + cg_fbw_fc2) / 2 if cg_fbw_fc1 and cg_fbw_fc2 else float("nan")
        # avg_fbw_dB = convert_fbw_to_dB(avg_fbw)

        geometric_avg_cg = np.sqrt(cg_fc1 * cg_fc2) if cg_fc1 and cg_fc2 else float("nan")
        avg_cg_dB = convert_cg_to_dB(geometric_avg_cg)

        cg_fbw_fc1 = cg_bw_fc1_mag / fc1_target if cg_bw_fc1_mag > 0 else float("nan")
        cg_fbw_fc2 = cg_bw_fc2_mag / fc2_target if cg_bw_fc2_mag > 0 else float("nan")

        geometric_avg_fbw = np.sqrt(cg_fbw_fc1 * cg_fbw_fc2) if cg_fbw_fc1 and cg_fbw_fc2 else float("nan")
        avg_fbw_dB = convert_fbw_to_dB(geometric_avg_fbw)

        fom = avg_cg_dB + beta_ * avg_fbw_dB

        penalty = penalty_for_equalization(cg_fc1, cg_fc2, lambda_)


    elif task == "dualGain":

        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))
        cg_fc2 = sim_output_dict.get(f"{selected_case}_cg_fc2", float("nan"))

        # avg_cg = (cg_fc1 + cg_fc2) / 2 if cg_fc1 and cg_fc2 else float("nan")
        # fom = convert_cg_to_dB(avg_cg)

        geometric_avg_cg = np.sqrt(cg_fc1 * cg_fc2) if cg_fc1 and cg_fc2 else float("nan")
        fom = convert_cg_to_dB(geometric_avg_cg)

        penalty = penalty_for_equalization(cg_fc1, cg_fc2, lambda_)

    
    elif task == "singleBW":
        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))
        cg_bw_fc1 = sim_output_dict.get(f"{selected_case}_cg_bw_fc1", [])
        cg_bw_fc1_mag = cg_bw_fc1[1] - cg_bw_fc1[0] if len(cg_bw_fc1) == 2 else 0.0

        cg_fbw_fc1_dB = convert_bw_to_fbw_dB(cg_bw_fc1_mag, fc1_target)
        cg_fc1_dB = convert_cg_to_dB(cg_fc1)

        fom = cg_fc1_dB + beta_ * cg_fbw_fc1_dB


    elif task == "singleGain":

        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))

        fom = convert_cg_to_dB(cg_fc1)


    if is_final:
        print("Overall FoM:", fom)
        if task in ["dualBW", "dualGain"]:
            print(f"cg_{fc1_target/ghz_scale}GHz (linear):", cg_fc1)
            print(f"cg_{fc1_target/ghz_scale}GHz (dB):", 20 * np.log10(cg_fc1))
            print(f"cg_{fc2_target/ghz_scale}GHz (linear):", cg_fc2)
            print(f"cg_{fc2_target/ghz_scale}GHz (dB):", 20 * np.log10(cg_fc2))
            if task == "dualBW":
                print(f"cg_bw_{fc1_target/ghz_scale}GHz (GHz):", cg_bw_fc1_mag/ghz_scale)
                print(f"cg_bw_{fc2_target/ghz_scale}GHz (GHz):", cg_bw_fc2_mag/ghz_scale)
        
    
        if task in ["singleBW", "singleGain"]:
            print(f"cg_{fc1_target/ghz_scale}GHz (linear):", cg_fc1)
            print(f"cg_{fc1_target/ghz_scale}GHz (dB):", 20 * np.log10(cg_fc1))
            if task == "singleBW":
                print(f"cg_bw_{fc1_target/ghz_scale}GHz (GHz):", cg_bw_fc1_mag/ghz_scale)

    z01_loss = 0.0
    weighted_z01_loss = gamma * z01_loss

    if task in ["dualGain", "dualBW"]:
        weighted_penalty = penalty
    else:
        weighted_penalty = 0.0

    return fom, weighted_z01_loss, weighted_penalty

def FoM_z01_manual_penalty(sim_output_dict, selected_case, task, lambda_, is_final=False):
    ghz_scale = 1e9  # 1GHz
    # gamma = 50.0 # weight for zero-one loss
    global gamma

    def convert_cg_to_dB(cg_linear):
        return 20 * np.log10(cg_linear) if cg_linear > 0 else float("nan")
    
    def convert_bw_to_fbw_dB(bw_linear, fc_target):
        if bw_linear <= 0:
            return 10 * np.log10(1e-6 / fc_target)
        else:
            return 10 * np.log10(bw_linear / fc_target)
        
    def convert_fbw_to_dB(fbw_linear):
        return 10 * np.log10(fbw_linear) if fbw_linear > 0 else float("nan")
    
    def penalty_for_equalization(cg_linear1, cg_linear2, lambda_):
        cg_dB1 = convert_cg_to_dB(cg_linear1)
        cg_dB2 = convert_cg_to_dB(cg_linear2)
        return lambda_ * (cg_dB1 - cg_dB2) ** 2

    if task == "dualBW":
        # output_dict = {
        # "case1_cg_bw_fc1" : [],  # Hz
        # "case1_cg_bw_fc2" : [],  # Hz
        # "case1_cg_fc1" : float("nan"), # linear
        # "case1_cg_fc2" : float("nan"), # linear

        # "case2_cg_bw_fc1" : [], # Hz
        # "case2_cg_bw_fc2" : [], # Hz
        # "case2_cg_fc1" : float("nan"), # linear
        # "case2_cg_fc2" : float("nan"), # linear

        # "case3_cg_bw_fc1" : [], # Hz
        # "case3_cg_bw_fc2" : [], # Hz
        # "case3_cg_fc1" : float("nan"), # linear
        # "case3_cg_fc2" : float("nan") # linear
        # }

        # case1_cg_bw = sim_output_dict["case1_cg_bw"]
        cg_bw_fc1 = sim_output_dict[f"{selected_case}_cg_bw_fc1"]
        cg_bw_fc2 = sim_output_dict[f"{selected_case}_cg_bw_fc2"]
        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))
        cg_fc2 = sim_output_dict.get(f"{selected_case}_cg_fc2", float("nan"))

        # cg
        cg_bw_fc1_mag = cg_bw_fc1[1] - cg_bw_fc1[0] if len(cg_bw_fc1) == 2 else 0.0
        cg_bw_fc2_mag = cg_bw_fc2[1] - cg_bw_fc2[0] if len(cg_bw_fc2) == 2 else 0.0

        if cg_bw_fc1_mag < 0 or cg_bw_fc2_mag < 0:
            print("[WARN] Invalid bandwidth detected (negative). Setting FoM to 0.")
            return 0.0

        # avg_cg = (cg_fc1 + cg_fc2) / 2 if cg_fc1 and cg_fc2 else float("nan")
        # avg_cg_dB = convert_cg_to_dB(avg_cg)

        # avg_fbw = (cg_fbw_fc1 + cg_fbw_fc2) / 2 if cg_fbw_fc1 and cg_fbw_fc2 else float("nan")
        # avg_fbw_dB = convert_fbw_to_dB(avg_fbw)

        geometric_avg_cg = np.sqrt(cg_fc1 * cg_fc2) if cg_fc1 and cg_fc2 else float("nan")
        avg_cg_dB = convert_cg_to_dB(geometric_avg_cg)

        cg_fbw_fc1 = cg_bw_fc1_mag / fc1_target if cg_bw_fc1_mag > 0 else float("nan")
        cg_fbw_fc2 = cg_bw_fc2_mag / fc2_target if cg_bw_fc2_mag > 0 else float("nan")

        geometric_avg_fbw = np.sqrt(cg_fbw_fc1 * cg_fbw_fc2) if cg_fbw_fc1 and cg_fbw_fc2 else float("nan")
        avg_fbw_dB = convert_fbw_to_dB(geometric_avg_fbw)

        fom = avg_cg_dB + avg_fbw_dB

        penalty = penalty_for_equalization(cg_fc1, cg_fc2, lambda_)

    elif task == "dualGains":

        cg_fc1_list = sim_output_dict.get(f"{selected_case}_cg_fc1", [])
        cg_fc2_list = sim_output_dict.get(f"{selected_case}_cg_fc2", [])

        avg_cg = np.mean(cg_fc1_list + cg_fc2_list) if cg_fc1_list and cg_fc2_list else float("nan")

        fom = convert_cg_to_dB(avg_cg)

    elif task == "dualGain":

        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))
        cg_fc2 = sim_output_dict.get(f"{selected_case}_cg_fc2", float("nan"))

        # avg_cg = (cg_fc1 + cg_fc2) / 2 if cg_fc1 and cg_fc2 else float("nan")
        # fom = convert_cg_to_dB(avg_cg)

        geometric_avg_cg = np.sqrt(cg_fc1 * cg_fc2) if cg_fc1 and cg_fc2 else float("nan")
        fom = convert_cg_to_dB(geometric_avg_cg)

        penalty = penalty_for_equalization(cg_fc1, cg_fc2, lambda_)
    
    elif task == "singleBW":
        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))
        cg_bw_fc1 = sim_output_dict.get(f"{selected_case}_cg_bw_fc1", [])
        cg_bw_fc1_mag = cg_bw_fc1[1] - cg_bw_fc1[0] if len(cg_bw_fc1) == 2 else 0.0

        cg_fbw_fc1_dB = convert_bw_to_fbw_dB(cg_bw_fc1_mag, fc1_target)
        cg_fc1_dB = convert_cg_to_dB(cg_fc1)

        fom = cg_fc1_dB + cg_fbw_fc1_dB

    elif task == "singleGains":
        
        cg_fc1_list = sim_output_dict.get(f"{selected_case}_cg_fc1", [])
        avg_cg = np.mean(cg_fc1_list) if cg_fc1_list else float("nan")

        fom = convert_cg_to_dB(avg_cg)

    elif task == "singleGain":
        # output_dict = {
        #     "case1_cg_fc1": float("nan"),  # linear
        #     "case2_cg_fc1": float("nan"),  # linear
        #     "case3_cg_fc1": float("nan")   # linear
        # }

        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))

        fom = convert_cg_to_dB(cg_fc1)


    if is_final:
        print("Overall FoM:", fom)
        if task in ["dualBW", "dualGain"]:
            print(f"cg_{fc1_target/ghz_scale}GHz (linear):", cg_fc1)
            print(f"cg_{fc1_target/ghz_scale}GHz (dB):", 20 * np.log10(cg_fc1))
            print(f"cg_{fc2_target/ghz_scale}GHz (linear):", cg_fc2)
            print(f"cg_{fc2_target/ghz_scale}GHz (dB):", 20 * np.log10(cg_fc2))
            print("penalty for equalization:", penalty)
            if task == "dualBW":
                print(f"cg_bw_{fc1_target/ghz_scale}GHz (GHz):", cg_bw_fc1_mag/ghz_scale)
                print(f"cg_bw_{fc2_target/ghz_scale}GHz (GHz):", cg_bw_fc2_mag/ghz_scale)
        
    
        if task in ["singleBW", "singleGain"]:
            print(f"cg_{fc1_target/ghz_scale}GHz (linear):", cg_fc1)
            print(f"cg_{fc1_target/ghz_scale}GHz (dB):", 20 * np.log10(cg_fc1))
            if task == "singleBW":
                print(f"cg_bw_{fc1_target/ghz_scale}GHz (GHz):", cg_bw_fc1_mag/ghz_scale)

    z01_loss = 0.0
    weighted_z01_loss = gamma * z01_loss

    if task in ["dualGain", "dualBW"]:
        weighted_penalty = penalty
    else:
        weighted_penalty = 0

    return fom, weighted_z01_loss, weighted_penalty

def FoM_z01_manual_alpha(sim_output_dict, selected_case, task, alpha_, is_final=False):
    ghz_scale = 1e9  # 1GHz
    # gamma = 50.0 # weight for zero-one loss
    lambda_ = 0.05
    global gamma

    def convert_cg_to_dB(cg_linear):
        return 20 * np.log10(cg_linear) if cg_linear > 0 else float("nan")
    
    def convert_bw_to_fbw_dB(bw_linear, fc_target):
        if bw_linear <= 0:
            return 10 * np.log10(1e-6 / fc_target)
        else:
            return 10 * np.log10(bw_linear / fc_target)
        
    def convert_fbw_to_dB(fbw_linear):
        return 10 * np.log10(fbw_linear) if fbw_linear > 0 else float("nan")
    
    def penalty_for_equalization(cg_linear1, cg_linear2, lambda_):
        cg_dB1 = convert_cg_to_dB(cg_linear1)
        cg_dB2 = convert_cg_to_dB(cg_linear2)
        return lambda_ * (cg_dB1 - cg_dB2) ** 2

    if task == "dualBW":

        # case1_cg_bw = sim_output_dict["case1_cg_bw"]
        cg_bw_fc1 = sim_output_dict[f"{selected_case}_cg_bw_fc1"]
        cg_bw_fc2 = sim_output_dict[f"{selected_case}_cg_bw_fc2"]
        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))
        cg_fc2 = sim_output_dict.get(f"{selected_case}_cg_fc2", float("nan"))

        # cg
        cg_bw_fc1_mag = cg_bw_fc1[1] - cg_bw_fc1[0] if len(cg_bw_fc1) == 2 else 0.0
        cg_bw_fc2_mag = cg_bw_fc2[1] - cg_bw_fc2[0] if len(cg_bw_fc2) == 2 else 0.0

        if cg_bw_fc1_mag < 0 or cg_bw_fc2_mag < 0:
            print("[WARN] Invalid bandwidth detected (negative). Setting FoM to 0.")
            return 0.0

        # avg_cg = (cg_fc1 + cg_fc2) / 2 if cg_fc1 and cg_fc2 else float("nan")
        # avg_cg_dB = convert_cg_to_dB(avg_cg)

        # avg_fbw = (cg_fbw_fc1 + cg_fbw_fc2) / 2 if cg_fbw_fc1 and cg_fbw_fc2 else float("nan")
        # avg_fbw_dB = convert_fbw_to_dB(avg_fbw)

        geometric_avg_cg = np.sqrt(cg_fc1 * cg_fc2) if cg_fc1 and cg_fc2 else float("nan")
        avg_cg_dB = convert_cg_to_dB(geometric_avg_cg)

        cg_fbw_fc1 = cg_bw_fc1_mag / fc1_target if cg_bw_fc1_mag > 0 else float("nan")
        cg_fbw_fc2 = cg_bw_fc2_mag / fc2_target if cg_bw_fc2_mag > 0 else float("nan")

        geometric_avg_fbw = np.sqrt(cg_fbw_fc1 * cg_fbw_fc2) if cg_fbw_fc1 and cg_fbw_fc2 else float("nan")
        avg_fbw_dB = convert_fbw_to_dB(geometric_avg_fbw)

        fom = alpha_ * avg_cg_dB + avg_fbw_dB

        penalty = penalty_for_equalization(cg_fc1, cg_fc2, lambda_)

    elif task == "dualGains":

        cg_fc1_list = sim_output_dict.get(f"{selected_case}_cg_fc1", [])
        cg_fc2_list = sim_output_dict.get(f"{selected_case}_cg_fc2", [])

        avg_cg = np.mean(cg_fc1_list + cg_fc2_list) if cg_fc1_list and cg_fc2_list else float("nan")

        fom = convert_cg_to_dB(avg_cg)

    elif task == "dualGain":

        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))
        cg_fc2 = sim_output_dict.get(f"{selected_case}_cg_fc2", float("nan"))

        # avg_cg = (cg_fc1 + cg_fc2) / 2 if cg_fc1 and cg_fc2 else float("nan")
        # fom = convert_cg_to_dB(avg_cg)

        geometric_avg_cg = np.sqrt(cg_fc1 * cg_fc2) if cg_fc1 and cg_fc2 else float("nan")
        fom = alpha_*convert_cg_to_dB(geometric_avg_cg)

        penalty = penalty_for_equalization(cg_fc1, cg_fc2, lambda_)
    
    elif task == "singleBW":
        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))
        cg_bw_fc1 = sim_output_dict.get(f"{selected_case}_cg_bw_fc1", [])
        cg_bw_fc1_mag = cg_bw_fc1[1] - cg_bw_fc1[0] if len(cg_bw_fc1) == 2 else 0.0

        cg_fbw_fc1_dB = convert_bw_to_fbw_dB(cg_bw_fc1_mag, fc1_target)
        cg_fc1_dB = convert_cg_to_dB(cg_fc1)

        fom = cg_fc1_dB + cg_fbw_fc1_dB

    elif task == "singleGains":
        
        cg_fc1_list = sim_output_dict.get(f"{selected_case}_cg_fc1", [])
        avg_cg = np.mean(cg_fc1_list) if cg_fc1_list else float("nan")

        fom = convert_cg_to_dB(avg_cg)

    elif task == "singleGain":
        # output_dict = {
        #     "case1_cg_fc1": float("nan"),  # linear
        #     "case2_cg_fc1": float("nan"),  # linear
        #     "case3_cg_fc1": float("nan")   # linear
        # }

        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))

        fom = convert_cg_to_dB(cg_fc1)


    if is_final:
        print("Overall FoM:", fom)
        if task in ["dualBW", "dualGain"]:
            print(f"cg_{fc1_target/ghz_scale}GHz (linear):", cg_fc1)
            print(f"cg_{fc1_target/ghz_scale}GHz (dB):", 20 * np.log10(cg_fc1))
            print(f"cg_{fc2_target/ghz_scale}GHz (linear):", cg_fc2)
            print(f"cg_{fc2_target/ghz_scale}GHz (dB):", 20 * np.log10(cg_fc2))
            print("penalty for equalization:", penalty)
            if task == "dualBW":
                print(f"cg_bw_{fc1_target/ghz_scale}GHz (GHz):", cg_bw_fc1_mag/ghz_scale)
                print(f"cg_bw_{fc2_target/ghz_scale}GHz (GHz):", cg_bw_fc2_mag/ghz_scale)
        
    
        if task in ["singleBW", "singleGain"]:
            print(f"cg_{fc1_target/ghz_scale}GHz (linear):", cg_fc1)
            print(f"cg_{fc1_target/ghz_scale}GHz (dB):", 20 * np.log10(cg_fc1))
            if task == "singleBW":
                print(f"cg_bw_{fc1_target/ghz_scale}GHz (GHz):", cg_bw_fc1_mag/ghz_scale)

    z01_loss = 0.0
    weighted_z01_loss = gamma * z01_loss

    if task in ["dualGain", "dualBW"]:
        weighted_penalty = penalty
    else:
        weighted_penalty = 0.0

    return fom, weighted_z01_loss, weighted_penalty

def FoM_z01_manual_beta(sim_output_dict, selected_case, task, beta_, is_final=False):
    ghz_scale = 1e9  # 1GHz
    # gamma = 50.0 # weight for zero-one loss
    lambda_ = 0.05
    global gamma

    def convert_cg_to_dB(cg_linear):
        return 20 * np.log10(cg_linear) if cg_linear > 0 else float("nan")
    
    def convert_bw_to_fbw_dB(bw_linear, fc_target):
        if bw_linear <= 0:
            return 10 * np.log10(1e-6 / fc_target)
        else:
            return 10 * np.log10(bw_linear / fc_target)
        
    def convert_fbw_to_dB(fbw_linear):
        return 10 * np.log10(fbw_linear) if fbw_linear > 0 else float("nan")
    
    def penalty_for_equalization(cg_linear1, cg_linear2, lambda_):
        cg_dB1 = convert_cg_to_dB(cg_linear1)
        cg_dB2 = convert_cg_to_dB(cg_linear2)
        return lambda_ * (cg_dB1 - cg_dB2) ** 2

    if task == "dualBW":
        # output_dict = {
        # "case1_cg_bw_fc1" : [],  # Hz
        # "case1_cg_bw_fc2" : [],  # Hz
        # "case1_cg_fc1" : float("nan"), # linear
        # "case1_cg_fc2" : float("nan"), # linear

        # "case2_cg_bw_fc1" : [], # Hz
        # "case2_cg_bw_fc2" : [], # Hz
        # "case2_cg_fc1" : float("nan"), # linear
        # "case2_cg_fc2" : float("nan"), # linear

        # "case3_cg_bw_fc1" : [], # Hz
        # "case3_cg_bw_fc2" : [], # Hz
        # "case3_cg_fc1" : float("nan"), # linear
        # "case3_cg_fc2" : float("nan") # linear
        # }

        # case1_cg_bw = sim_output_dict["case1_cg_bw"]
        cg_bw_fc1 = sim_output_dict[f"{selected_case}_cg_bw_fc1"]
        cg_bw_fc2 = sim_output_dict[f"{selected_case}_cg_bw_fc2"]
        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))
        cg_fc2 = sim_output_dict.get(f"{selected_case}_cg_fc2", float("nan"))

        # cg
        cg_bw_fc1_mag = cg_bw_fc1[1] - cg_bw_fc1[0] if len(cg_bw_fc1) == 2 else 0.0
        cg_bw_fc2_mag = cg_bw_fc2[1] - cg_bw_fc2[0] if len(cg_bw_fc2) == 2 else 0.0

        if cg_bw_fc1_mag < 0 or cg_bw_fc2_mag < 0:
            print("[WARN] Invalid bandwidth detected (negative). Setting FoM to 0.")
            return 0.0

        # avg_cg = (cg_fc1 + cg_fc2) / 2 if cg_fc1 and cg_fc2 else float("nan")
        # avg_cg_dB = convert_cg_to_dB(avg_cg)

        # avg_fbw = (cg_fbw_fc1 + cg_fbw_fc2) / 2 if cg_fbw_fc1 and cg_fbw_fc2 else float("nan")
        # avg_fbw_dB = convert_fbw_to_dB(avg_fbw)

        geometric_avg_cg = np.sqrt(cg_fc1 * cg_fc2) if cg_fc1 and cg_fc2 else float("nan")
        avg_cg_dB = convert_cg_to_dB(geometric_avg_cg)

        cg_fbw_fc1 = cg_bw_fc1_mag / fc1_target if cg_bw_fc1_mag > 0 else float("nan")
        cg_fbw_fc2 = cg_bw_fc2_mag / fc2_target if cg_bw_fc2_mag > 0 else float("nan")

        geometric_avg_fbw = np.sqrt(cg_fbw_fc1 * cg_fbw_fc2) if cg_fbw_fc1 and cg_fbw_fc2 else float("nan")
        avg_fbw_dB = convert_fbw_to_dB(geometric_avg_fbw)

        fom = avg_cg_dB + beta_*avg_fbw_dB

        penalty = penalty_for_equalization(cg_fc1, cg_fc2, lambda_)

    elif task == "dualGains":

        cg_fc1_list = sim_output_dict.get(f"{selected_case}_cg_fc1", [])
        cg_fc2_list = sim_output_dict.get(f"{selected_case}_cg_fc2", [])

        avg_cg = np.mean(cg_fc1_list + cg_fc2_list) if cg_fc1_list and cg_fc2_list else float("nan")

        fom = convert_cg_to_dB(avg_cg)

    elif task == "dualGain":

        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))
        cg_fc2 = sim_output_dict.get(f"{selected_case}_cg_fc2", float("nan"))

        # avg_cg = (cg_fc1 + cg_fc2) / 2 if cg_fc1 and cg_fc2 else float("nan")
        # fom = convert_cg_to_dB(avg_cg)

        geometric_avg_cg = np.sqrt(cg_fc1 * cg_fc2) if cg_fc1 and cg_fc2 else float("nan")
        fom = convert_cg_to_dB(geometric_avg_cg)

        penalty = penalty_for_equalization(cg_fc1, cg_fc2, lambda_)
    
    elif task == "singleBW":
        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))
        cg_bw_fc1 = sim_output_dict.get(f"{selected_case}_cg_bw_fc1", [])
        cg_bw_fc1_mag = cg_bw_fc1[1] - cg_bw_fc1[0] if len(cg_bw_fc1) == 2 else 0.0

        cg_fbw_fc1_dB = convert_bw_to_fbw_dB(cg_bw_fc1_mag, fc1_target)
        cg_fc1_dB = convert_cg_to_dB(cg_fc1)

        fom = cg_fc1_dB + beta_*cg_fbw_fc1_dB

    elif task == "singleGains":
        
        cg_fc1_list = sim_output_dict.get(f"{selected_case}_cg_fc1", [])
        avg_cg = np.mean(cg_fc1_list) if cg_fc1_list else float("nan")

        fom = convert_cg_to_dB(avg_cg)

    elif task == "singleGain":
        # output_dict = {
        #     "case1_cg_fc1": float("nan"),  # linear
        #     "case2_cg_fc1": float("nan"),  # linear
        #     "case3_cg_fc1": float("nan")   # linear
        # }

        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))

        fom = convert_cg_to_dB(cg_fc1)


    if is_final:
        print("Overall FoM:", fom)
        if task in ["dualBW", "dualGain"]:
            print(f"cg_{fc1_target/ghz_scale}GHz (linear):", cg_fc1)
            print(f"cg_{fc1_target/ghz_scale}GHz (dB):", 20 * np.log10(cg_fc1))
            print(f"cg_{fc2_target/ghz_scale}GHz (linear):", cg_fc2)
            print(f"cg_{fc2_target/ghz_scale}GHz (dB):", 20 * np.log10(cg_fc2))
            print("penalty for equalization:", penalty)
            if task == "dualBW":
                print(f"cg_bw_{fc1_target/ghz_scale}GHz (GHz):", cg_bw_fc1_mag/ghz_scale)
                print(f"cg_bw_{fc2_target/ghz_scale}GHz (GHz):", cg_bw_fc2_mag/ghz_scale)
        
    
        if task in ["singleBW", "singleGain"]:
            print(f"cg_{fc1_target/ghz_scale}GHz (linear):", cg_fc1)
            print(f"cg_{fc1_target/ghz_scale}GHz (dB):", 20 * np.log10(cg_fc1))
            if task == "singleBW":
                print(f"cg_bw_{fc1_target/ghz_scale}GHz (GHz):", cg_bw_fc1_mag/ghz_scale)

    z01_loss = 0.0
    weighted_z01_loss = gamma * z01_loss

    if task in ["dualGain", "dualBW"]:
        weighted_penalty = penalty
    else:
        weighted_penalty = 0.0

    return fom, weighted_z01_loss, weighted_penalty

def FoM_z01_manual_weight_test(sim_output_dict, selected_case, task, is_final=False): 

    ghz_scale = 1e9 # 1GHz

    if task == "dualBW":

        cg_bw_fc1 = sim_output_dict[f"{selected_case}_cg_bw_fc1"]
        cg_bw_fc2 = sim_output_dict[f"{selected_case}_cg_bw_fc2"]
        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))
        cg_fc2 = sim_output_dict.get(f"{selected_case}_cg_fc2", float("nan"))

        # cg
        cg_bw_fc1_mag = cg_bw_fc1[1] - cg_bw_fc1[0] if len(cg_bw_fc1) == 2 else 0.0
        cg_bw_fc2_mag = cg_bw_fc2[1] - cg_bw_fc2[0] if len(cg_bw_fc2) == 2 else 0.0

        if cg_bw_fc1_mag < 0 or cg_bw_fc2_mag < 0:
            print("[WARN] Invalid bandwidth detected (negative). Setting FoM to 0.")
            return 0.0

        cg_fc1_dB = 20 * np.log10(cg_fc1) if cg_fc1 > 0 else float("nan")
        cg_fc2_dB = 20 * np.log10(cg_fc2) if cg_fc2 > 0 else float("nan")
        if cg_bw_fc1_mag <= 0:
            cg_fbw_fc1_dB = 10*np.log10(1e-6/ fc1_target)
        else: 
            cg_fbw_fc1_dB = 10 * np.log10(cg_bw_fc1_mag / fc1_target)
        if cg_bw_fc2_mag <= 0:
            cg_fbw_fc2_dB = 10*np.log10(1e-6/ fc2_target)
        else:
            cg_fbw_fc2_dB = 10 * np.log10(cg_bw_fc2_mag / fc2_target)

        # 1. dB scale
        fom1 = cg_fc1_dB + cg_fc2_dB
        fom2 = cg_fbw_fc1_dB + cg_fbw_fc2_dB

        # 2. linear scale
        # fom1 = cg_fc1 +  cg_fc2
        # fom2 = cg_bw_fc1_mag/fc1_target + cg_bw_fc2_mag/fc2_target
    
    elif task == "singleBW":
        cg_fc1 = sim_output_dict.get(f"{selected_case}_cg_fc1", float("nan"))
        cg_bw_fc1 = sim_output_dict.get(f"{selected_case}_cg_bw_fc1", [])
        cg_bw_fc1_mag = cg_bw_fc1[1] - cg_bw_fc1[0] if len(cg_bw_fc1) == 2 else 0.0

        cg_fc1_dB = 20 * np.log10(cg_fc1) if cg_fc1 > 0 else float("nan")
        if cg_bw_fc1_mag <= 0:
            cg_fbw_fc1_dB = 10*np.log10(1e-6/ fc1_target)
        else:
            cg_fbw_fc1_dB = 10 * np.log10(cg_bw_fc1_mag / fc1_target)

        # 1. dB scale
        fom1 = cg_fc1_dB
        fom2 = cg_fbw_fc1_dB

        # 2. linear scale
        # fom1 = cg_fc1
        # fom2 = cg_bw_fc1_mag/fc1_target

    if is_final:
        print(f"cg_{fc1_target/ghz_scale}GHz (linear):", cg_fc1)
        print(f"cg_{fc1_target/ghz_scale}GHz (dB):", 20 * np.log10(cg_fc1))
        print(f"cg_bw_{fc1_target/ghz_scale}GHz (GHz):", cg_bw_fc1_mag/ghz_scale)
        if task == "dualBW":
            print(f"cg_{fc2_target/ghz_scale}GHz (linear):", cg_fc2)
            print(f"cg_{fc2_target/ghz_scale}GHz (dB):", 20 * np.log10(cg_fc2))
            print(f"cg_bw_{fc2_target/ghz_scale}GHz (GHz):", cg_bw_fc2_mag/ghz_scale)

    z01_loss = 0.0

    return fom1, fom2, z01_loss

########## still used in per_case ###########
####### REVISED 0829 - gamma_, lambda_, beta_ included #######
def FoM_z01_per_case(sim_output_dict, sm_alpha_dict, alpha_groups, task, is_final=False):

    ghz_scale = 1e9 # 1GHz
    global gamma
    lambda_= 0.05
    beta_ = 1.0

    def convert_cg_to_dB(cg_linear):
        return 20 * np.log10(cg_linear) if cg_linear > 0 else float("nan")
    
    def convert_bw_to_fbw_dB(bw_linear, fc_target):
        if bw_linear <= 0:
            return 10 * np.log10(1e-6 / fc_target)
        else:
            return 10 * np.log10(bw_linear / fc_target)
        
    def convert_fbw_to_dB(fbw_linear):
        return 10 * np.log10(fbw_linear) if fbw_linear > 0 else float("nan")
    
    def penalty_for_equalization(cg_linear1, cg_linear2, lambda_):
        cg_dB1 = convert_cg_to_dB(cg_linear1)
        cg_dB2 = convert_cg_to_dB(cg_linear2)
        return lambda_ * (cg_dB1 - cg_dB2) ** 2

    def zero_one_loss_per_case(sm_alpha_dict, alpha_groups):
        case1_loss_vals = []
        case2_loss_vals = []
        case3_loss_vals = []
        for group, alpha_list in alpha_groups.items():
            if group == "per_case":
                continue
            alpha_vals = [sm_alpha_dict[alpha] for alpha in alpha_list]
            max_alpha_val = max(alpha_vals)
            loss = (1.0-max_alpha_val) ** 2
            if group.startswith("case1"):
                case1_loss_vals.append(loss)
            elif group.startswith("case2"):
                # print(f"Group: {group}, Max Alpha: {alpha_list[np.argmax(alpha_vals)]}, Max Alpha Value: {max_alpha_val}")
                # print(f"Loss: {loss}")
                case2_loss_vals.append(loss)
            elif group.startswith("case3"):
                case3_loss_vals.append(loss)

            if max_alpha_val < 0.9:
                print(f"[WARN] {group} max alpha {alpha_list[np.argmax(alpha_vals)]}:{max_alpha_val} < 0.9")
                
        case1_z01_loss = np.mean(case1_loss_vals) if case1_loss_vals else 0.0
        case2_z01_loss = np.mean(case2_loss_vals) if case2_loss_vals else 0.0
        case3_z01_loss = np.mean(case3_loss_vals) if case3_loss_vals else 0.0
        return case1_z01_loss, case2_z01_loss, case3_z01_loss

    if task == "dualBW":
        # output_dict = {
        # "case1_cg_bw_fc1" : [],  # Hz
        # "case1_cg_bw_fc2" : [],  # Hz
        # "case1_cg_fc1" : float("nan"), # linear
        # "case1_cg_fc2" : float("nan"), # linear

        # "case2_cg_bw_fc1" : [], # Hz
        # "case2_cg_bw_fc2" : [], # Hz
        # "case2_cg_fc1" : float("nan"), # linear
        # "case2_cg_fc2" : float("nan"), # linear

        # "case3_cg_bw_fc1" : [], # Hz
        # "case3_cg_bw_fc2" : [], # Hz
        # "case3_cg_fc1" : float("nan"), # linear
        # "case3_cg_fc2" : float("nan") # linear
        # }

        # case1_cg_bw = sim_output_dict["case1_cg_bw"]
        case1_cg_bw_fc1 = sim_output_dict["case1_cg_bw_fc1"]
        case1_cg_bw_fc2 = sim_output_dict["case1_cg_bw_fc2"]
        case1_cg_fc1 = sim_output_dict.get("case1_cg_fc1", float("nan"))
        case1_cg_fc2 = sim_output_dict.get("case1_cg_fc2", float("nan"))
        # case2_cg_bw = sim_output_dict["case2_cg_bw"]
        case2_cg_bw_fc1 = sim_output_dict["case2_cg_bw_fc1"]
        case2_cg_bw_fc2 = sim_output_dict["case2_cg_bw_fc2"]
        case2_cg_fc1 = sim_output_dict.get("case2_cg_fc1", float("nan"))
        case2_cg_fc2 = sim_output_dict.get("case2_cg_fc2", float("nan"))
        # case3_cg_bw = sim_output_dict["case3_cg_bw"]
        case3_cg_bw_fc1 = sim_output_dict["case3_cg_bw_fc1"]
        case3_cg_bw_fc2 = sim_output_dict["case3_cg_bw_fc2"]
        case3_cg_fc1 = sim_output_dict.get("case3_cg_fc1", float("nan"))
        case3_cg_fc2 = sim_output_dict.get("case3_cg_fc2", float("nan"))

        # cg
        case1_cg_bw_fc1_mag = case1_cg_bw_fc1[1] - case1_cg_bw_fc1[0] if len(case1_cg_bw_fc1) == 2 else 0.0
        case1_cg_bw_fc2_mag = case1_cg_bw_fc2[1] - case1_cg_bw_fc2[0] if len(case1_cg_bw_fc2) == 2 else 0.0
        case2_cg_bw_fc1_mag = case2_cg_bw_fc1[1] - case2_cg_bw_fc1[0] if len(case2_cg_bw_fc1) == 2 else 0.0
        case2_cg_bw_fc2_mag = case2_cg_bw_fc2[1] - case2_cg_bw_fc2[0] if len(case2_cg_bw_fc2) == 2 else 0.0
        case3_cg_bw_fc1_mag = case3_cg_bw_fc1[1] - case3_cg_bw_fc1[0] if len(case3_cg_bw_fc1) == 2 else 0.0
        case3_cg_bw_fc2_mag = case3_cg_bw_fc2[1] - case3_cg_bw_fc2[0] if len(case3_cg_bw_fc2) == 2 else 0.0

        if case1_cg_bw_fc1_mag < 0 or case1_cg_bw_fc2_mag < 0 or case2_cg_bw_fc1_mag < 0 or case2_cg_bw_fc2_mag < 0 or case3_cg_bw_fc1_mag < 0 or case3_cg_bw_fc2_mag < 0:
            print("[WARN] Invalid bandwidth detected (negative). Setting FoM to 0.")
            return 0.0

        case1_cg_fbw_fc1 = case1_cg_bw_fc1_mag / fc1_target if case1_cg_bw_fc1_mag > 0 else float("nan")
        case1_cg_fbw_fc2 = case1_cg_bw_fc2_mag / fc2_target if case1_cg_bw_fc2_mag > 0 else float("nan")
        case2_cg_fbw_fc1 = case2_cg_bw_fc1_mag / fc1_target if case2_cg_bw_fc1_mag > 0 else float("nan")
        case2_cg_fbw_fc2 = case2_cg_bw_fc2_mag / fc2_target if case2_cg_bw_fc2_mag > 0 else float("nan")
        case3_cg_fbw_fc1 = case3_cg_bw_fc1_mag / fc1_target if case3_cg_bw_fc1_mag > 0 else float("nan")
        case3_cg_fbw_fc2 = case3_cg_bw_fc2_mag / fc2_target if case3_cg_bw_fc2_mag > 0 else float("nan")

        # case1_avg_cg = (case1_cg_fc1 + case1_cg_fc2) / 2 if case1_cg_fc1 and case1_cg_fc2 else float("nan")
        # case2_avg_cg = (case2_cg_fc1 + case2_cg_fc2) / 2 if case2_cg_fc1 and case2_cg_fc2 else float("nan")
        # case3_avg_cg = (case3_cg_fc1 + case3_cg_fc2) / 2 if case3_cg_fc1 and case3_cg_fc2 else float("nan")

        # case1_avg_fbw = (case1_cg_fbw_fc1 + case1_cg_fbw_fc2) / 2 if case1_cg_fbw_fc1 and case1_cg_fbw_fc2 else float("nan")
        # case2_avg_fbw = (case2_cg_fbw_fc1 + case2_cg_fbw_fc2) / 2 if case2_cg_fbw_fc1 and case2_cg_fbw_fc2 else float("nan")
        # case3_avg_fbw = (case3_cg_fbw_fc1 + case3_cg_fbw_fc2) / 2 if case3_cg_fbw_fc1 and case3_cg_fbw_fc2 else float("nan")

        # case1_avg_cg_dB = convert_cg_to_dB(case1_avg_cg)
        # case2_avg_cg_dB = convert_cg_to_dB(case2_avg_cg)
        # case3_avg_cg_dB = convert_cg_to_dB(case3_avg_cg)

        # case1_avg_fbw_dB = convert_fbw_to_dB(case1_avg_fbw)
        # case2_avg_fbw_dB = convert_fbw_to_dB(case2_avg_fbw)
        # case3_avg_fbw_dB = convert_fbw_to_dB(case3_avg_fbw)

        case1_geometric_avg_cg = np.sqrt(case1_cg_fc1 * case1_cg_fc2) if case1_cg_fc1 and case1_cg_fc2 else float("nan")
        case2_geometric_avg_cg = np.sqrt(case2_cg_fc1 * case2_cg_fc2) if case2_cg_fc1 and case2_cg_fc2 else float("nan")
        case3_geometric_avg_cg = np.sqrt(case3_cg_fc1 * case3_cg_fc2) if case3_cg_fc1 and case3_cg_fc2 else float("nan")
        case1_geometric_avg_fbw = np.sqrt(case1_cg_fbw_fc1 * case1_cg_fbw_fc2) if case1_cg_fbw_fc1 and case1_cg_fbw_fc2 else float("nan")
        case2_geometric_avg_fbw = np.sqrt(case2_cg_fbw_fc1 * case2_cg_fbw_fc2) if case2_cg_fbw_fc1 and case2_cg_fbw_fc2 else float("nan")
        case3_geometric_avg_fbw = np.sqrt(case3_cg_fbw_fc1 * case3_cg_fbw_fc2) if case3_cg_fbw_fc1 and case3_cg_fbw_fc2 else float("nan")

        case1_avg_cg_dB = convert_cg_to_dB(case1_geometric_avg_cg)
        case2_avg_cg_dB = convert_cg_to_dB(case2_geometric_avg_cg)
        case3_avg_cg_dB = convert_cg_to_dB(case3_geometric_avg_cg)
        case1_avg_fbw_dB = convert_fbw_to_dB(case1_geometric_avg_fbw)
        case2_avg_fbw_dB = convert_fbw_to_dB(case2_geometric_avg_fbw)
        case3_avg_fbw_dB = convert_fbw_to_dB(case3_geometric_avg_fbw)

        case1_fom = case1_avg_cg_dB + beta_ * case1_avg_fbw_dB
        case2_fom = case2_avg_cg_dB + beta_ * case2_avg_fbw_dB
        case3_fom = case3_avg_cg_dB + beta_ * case3_avg_fbw_dB

        case1_penalty = penalty_for_equalization(case1_cg_fc1, case1_cg_fc2, lambda_)
        case2_penalty = penalty_for_equalization(case2_cg_fc1, case2_cg_fc2, lambda_)
        case3_penalty = penalty_for_equalization(case3_cg_fc1, case3_cg_fc2, lambda_)


    elif task == "dualGain":

        case1_cg_fc1 = sim_output_dict.get("case1_cg_fc1", float("nan"))
        case1_cg_fc2 = sim_output_dict.get("case1_cg_fc2", float("nan"))
        case2_cg_fc1 = sim_output_dict.get("case2_cg_fc1", float("nan"))
        case2_cg_fc2 = sim_output_dict.get("case2_cg_fc2", float("nan"))
        case3_cg_fc1 = sim_output_dict.get("case3_cg_fc1", float("nan"))
        case3_cg_fc2 = sim_output_dict.get("case3_cg_fc2", float("nan"))

        case1_geometric_avg_cg = np.sqrt(case1_cg_fc1 * case1_cg_fc2) if case1_cg_fc1 and case1_cg_fc2 else float("nan")
        case2_geometric_avg_cg = np.sqrt(case2_cg_fc1 * case2_cg_fc2) if case2_cg_fc1 and case2_cg_fc2 else float("nan")
        case3_geometric_avg_cg = np.sqrt(case3_cg_fc1 * case3_cg_fc2) if case3_cg_fc1 and case3_cg_fc2 else float("nan")

        case1_fom = convert_cg_to_dB(case1_geometric_avg_cg)
        case2_fom = convert_cg_to_dB(case2_geometric_avg_cg)
        case3_fom = convert_cg_to_dB(case3_geometric_avg_cg)

        case1_penalty = penalty_for_equalization(case1_cg_fc1, case1_cg_fc2, lambda_)
        case2_penalty = penalty_for_equalization(case2_cg_fc1, case2_cg_fc2, lambda_)
        case3_penalty = penalty_for_equalization(case3_cg_fc1, case3_cg_fc2, lambda_)

    elif task == "singleBW":

        case1_cg_fc1 = sim_output_dict.get("case1_cg_fc1", float("nan"))
        case2_cg_fc1 = sim_output_dict.get("case2_cg_fc1", float("nan"))
        case3_cg_fc1 = sim_output_dict.get("case3_cg_fc1", float("nan"))

        case1_cg_bw_fc1 = sim_output_dict["case1_cg_bw_fc1"]
        case2_cg_bw_fc1 = sim_output_dict["case2_cg_bw_fc1"]
        case3_cg_bw_fc1 = sim_output_dict["case3_cg_bw_fc1"]

        case1_cg_bw_fc1_mag = case1_cg_bw_fc1[1] - case1_cg_bw_fc1[0] if len(case1_cg_bw_fc1) == 2 else 0.0
        case2_cg_bw_fc1_mag = case2_cg_bw_fc1[1] - case2_cg_bw_fc1[0] if len(case2_cg_bw_fc1) == 2 else 0.0
        case3_cg_bw_fc1_mag = case3_cg_bw_fc1[1] - case3_cg_bw_fc1[0] if len(case3_cg_bw_fc1) == 2 else 0.0

        case1_cg_fc1_dB = convert_cg_to_dB(case1_cg_fc1)
        case2_cg_fc1_dB = convert_cg_to_dB(case2_cg_fc1)
        case3_cg_fc1_dB = convert_cg_to_dB(case3_cg_fc1)

        case1_fbw_fc1_dB = convert_bw_to_fbw_dB(case1_cg_bw_fc1_mag, fc1_target)
        case2_fbw_fc1_dB = convert_bw_to_fbw_dB(case2_cg_bw_fc1_mag, fc1_target)
        case3_fbw_fc1_dB = convert_bw_to_fbw_dB(case3_cg_bw_fc1_mag, fc1_target)

        case1_fom = case1_cg_fc1_dB + beta_ * case1_fbw_fc1_dB
        case2_fom = case2_cg_fc1_dB + beta_ * case2_fbw_fc1_dB
        case3_fom = case3_cg_fc1_dB + beta_ * case3_fbw_fc1_dB


    elif task == "singleGain":

        case1_cg_fc1 = sim_output_dict.get("case1_cg_fc1", float("nan"))
        case2_cg_fc1 = sim_output_dict.get("case2_cg_fc1", float("nan"))
        case3_cg_fc1 = sim_output_dict.get("case3_cg_fc1", float("nan"))

        case1_fom = convert_cg_to_dB(case1_cg_fc1)
        case2_fom = convert_cg_to_dB(case2_cg_fc1)
        case3_fom = convert_cg_to_dB(case3_cg_fc1)

    if is_final:
        print("[CASE1] Overall FoM:", case1_fom)
        print("[CASE2] Overall FoM:", case2_fom)
        print("[CASE3] Overall FoM:", case3_fom)
        if task in ["dualBW", "dualGain"]:
            print(f"[CASE1] cg_{fc1_target/ghz_scale}GHz (linear):", case1_cg_fc1)
            print(f"[CASE1] cg_{fc1_target/ghz_scale}GHz (dB):", 20 * np.log10(case1_cg_fc1))
            print(f"[CASE1] cg_{fc2_target/ghz_scale}GHz (linear):", case1_cg_fc2)
            print(f"[CASE1] cg_{fc2_target/ghz_scale}GHz (dB):", 20 * np.log10(case1_cg_fc2))
            print(f"[CASE2] cg_{fc1_target/ghz_scale}GHz (linear):", case2_cg_fc1)
            print(f"[CASE2] cg_{fc1_target/ghz_scale}GHz (dB):", 20 * np.log10(case2_cg_fc1))
            print(f"[CASE2] cg_{fc2_target/ghz_scale}GHz (linear):", case2_cg_fc2)
            print(f"[CASE2] cg_{fc2_target/ghz_scale}GHz (dB):", 20 * np.log10(case2_cg_fc2))
            print(f"[CASE3] cg_{fc1_target/ghz_scale}GHz (linear):", case3_cg_fc1)
            print(f"[CASE3] cg_{fc1_target/ghz_scale}GHz (dB):", 20 * np.log10(case3_cg_fc1))
            print(f"[CASE3] cg_{fc2_target/ghz_scale}GHz (linear):", case3_cg_fc2)
            print(f"[CASE3] cg_{fc2_target/ghz_scale}GHz (dB):", 20 * np.log10(case3_cg_fc2))
            if task == "dualBW":
                print(f"[CASE1] cg_bw_{fc1_target/ghz_scale}GHz (GHz):", case1_cg_bw_fc1_mag/ghz_scale)
                print(f"[CASE1] cg_bw_{fc2_target/ghz_scale}GHz (GHz):", case1_cg_bw_fc2_mag/ghz_scale)
                print(f"[CASE2] cg_bw_{fc1_target/ghz_scale}GHz (GHz):", case2_cg_bw_fc1_mag/ghz_scale)
                print(f"[CASE2] cg_bw_{fc2_target/ghz_scale}GHz (GHz):", case2_cg_bw_fc2_mag/ghz_scale)
                print(f"[CASE3] cg_bw_{fc1_target/ghz_scale}GHz (GHz):", case3_cg_bw_fc1_mag/ghz_scale)
                print(f"[CASE3] cg_bw_{fc2_target/ghz_scale}GHz (GHz):", case3_cg_bw_fc2_mag/ghz_scale)

        if task in ["singleBW", "singleGain"]:
            print(f"[CASE1] cg_{fc1_target/ghz_scale}GHz (linear):", case1_cg_fc1)
            print(f"[CASE1] cg_{fc1_target/ghz_scale}GHz (dB):", 20 * np.log10(case1_cg_fc1))
            print(f"[CASE2] cg_{fc1_target/ghz_scale}GHz (linear):", case2_cg_fc1)
            print(f"[CASE2] cg_{fc1_target/ghz_scale}GHz (dB):", 20 * np.log10(case2_cg_fc1))
            print(f"[CASE3] cg_{fc1_target/ghz_scale}GHz (linear):", case3_cg_fc1)
            print(f"[CASE3] cg_{fc1_target/ghz_scale}GHz (dB):", 20 * np.log10(case3_cg_fc1))
            if task == "singleBW":
                print(f"[CASE1] cg_bw_{fc1_target/ghz_scale}GHz (GHz):", case1_cg_bw_fc1_mag/ghz_scale)
                print(f"[CASE2] cg_bw_{fc1_target/ghz_scale}GHz (GHz):", case2_cg_bw_fc1_mag/ghz_scale)
                print(f"[CASE3] cg_bw_{fc1_target/ghz_scale}GHz (GHz):", case3_cg_bw_fc1_mag/ghz_scale)

    case1_z01_loss, case2_z01_loss, case3_z01_loss = zero_one_loss_per_case(sm_alpha_dict, alpha_groups)
    case1_weighted_z01_loss = gamma * case1_z01_loss
    case2_weighted_z01_loss = gamma * case2_z01_loss
    case3_weighted_z01_loss = gamma * case3_z01_loss

    if task in ["dualGain", "dualBW"]:
        case1_weighted_penalty = case1_penalty
        case2_weighted_penalty = case2_penalty
        case3_weighted_penalty = case3_penalty
    else:
        case1_weighted_penalty = 0.0
        case2_weighted_penalty = 0.0
        case3_weighted_penalty = 0.0

    return case1_fom, case2_fom, case3_fom, case1_weighted_z01_loss, case2_weighted_z01_loss, case3_weighted_z01_loss, case1_weighted_penalty, case2_weighted_penalty, case3_weighted_penalty
