import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict
# from circuit_dict import fc1_bw_target as FC1_BW_TARGET
# from circuit_dict import fc1_target, fc2_target, fc2_bw_target, bw_target_single

from circuit_dict import fc1_target, fc2_target, fc2_bw_target, fc1_bw_target

# freq_step_thres = 0.15e9 # Since freq step size is 0.1GHz, we set threshold to 0.15GHz
freq_step_thres = 0.025e9  # Hz, since freq step size is 0.02GHz, we set threshold to 0.025GHz

try:
    import keysight.ads.dataset as dataset
    from keysight import pwdatatools as pwdt
except Exception as e:
    print("Import error:", e, file=sys.stderr)
    sys.exit(1)

def read_dsfile(final_stage_dsfile, iteration, task, tb_date):

    if task == "dualBW": #target freq: fc1, fc2 & wideband BW
        output_dict = {
        "case1_cg_bw_fc1" : [],  # Hz
        "case1_cg_bw_fc2" : [],  # Hz
        "case1_cg_fc1" : float("nan"), # linear
        "case1_cg_fc2" : float("nan"), # linear

        "case2_cg_bw_fc1" : [], # Hz
        "case2_cg_bw_fc2" : [], # Hz
        "case2_cg_fc1" : float("nan"), # linear
        "case2_cg_fc2" : float("nan"), # linear

        "case3_cg_bw_fc1" : [], # Hz
        "case3_cg_bw_fc2" : [], # Hz
        "case3_cg_fc1" : float("nan"), # linear
        "case3_cg_fc2" : float("nan") # linear
        }

    elif task == "dualGains": # maximize gain across all frequencies
        output_dict = {
            "case1_cg_fc1": [], # list of linear value
            "case1_cg_fc2": [], # list of linear value
            "case2_cg_fc1": [], 
            "case2_cg_fc2": [],
            "case3_cg_fc1": [],
            "case3_cg_fc2": []  
        }
    
    elif task == "dualGain":
        output_dict = {
            "case1_cg_fc1": float("nan"),  # linear
            "case1_cg_fc2": float("nan"),  # linear
            "case2_cg_fc1": float("nan"),  # linear
            "case2_cg_fc2": float("nan"),  # linear
            "case3_cg_fc1": float("nan"),  # linear
            "case3_cg_fc2": float("nan")   # linear
        }
    
    elif task == "singleBW":
        output_dict = {
            "case1_cg_bw_fc1": [],  # Hz
            "case1_cg_fc1": float("nan"),  # linear
            "case2_cg_bw_fc1": [],  # Hz
            "case2_cg_fc1": float("nan"),  # linear
            "case3_cg_bw_fc1": [],  # Hz
            "case3_cg_fc1": float("nan")   # linear
        }

    elif task == "singleGains": # maximize gain across all frequencies
        output_dict = {
            "case1_cg_fc1": [], # list of linear value
            "case2_cg_fc1": [],
            "case3_cg_fc1": []
        }
    
    elif task == "singleGain":
        output_dict = {
            "case1_cg_fc1": float("nan"),  
            "case2_cg_fc1": float("nan"), 
            "case3_cg_fc1": float("nan")  
        }

    # ################# ADDED 0826 ##############
    # if task == "singleBW":
    #     fc1_bw_target = bw_target_single
    # else:
    #     fc1_bw_target = FC1_BW_TARGET
    # ############################################
    
    try:
        dsfile = final_stage_dsfile
        output_data = dataset.open(dsfile)
        # output_into_group = pwdt.read_file(dsfile)

        for member in list(output_data.keys()):
            block_data = output_data[member].to_dataframe().reset_index()
            block_cols = block_data.columns
            # print(f"block_data:{block_data}", file=sys.stderr)
            # print(f"block_cols: {block_cols}", file=sys.stderr)

            if ("case1_Vin_fund" in block_cols): # ALL VIN_FUND data are in the same block
                freq_vin_fund = (block_data["RFfreq"]*1e9).round(2) # Hz
                case1_vin_fund = block_data["case1_Vin_fund"] # V
                case2_vin_fund = block_data["case2_Vin_fund"] # V
                case3_vin_fund = block_data["case3_Vin_fund"] # V

                vin_fund_data = pd.DataFrame({
                    "freq": freq_vin_fund,
                    "case1_Vin_fund": case1_vin_fund,
                    "case2_Vin_fund": case2_vin_fund,
                    "case3_Vin_fund": case3_vin_fund
                })

            elif ("case1_Vout_fund" in block_cols):
                freq_vout_fund = (block_data["RFfreq"]*1e9).round(2) # Hz
                case1_vout_fund = block_data["case1_Vout_fund"] # V
            
            elif ("case2_Vout_fund" in block_cols):
                case2_vout_fund = block_data["case2_Vout_fund"] # V
            
            elif ("case3_Vout_fund" in block_cols):
                case3_vout_fund = block_data["case3_Vout_fund"] # V

        vout_fund_data= pd.DataFrame({
            "freq": freq_vout_fund,
            "case1_Vout_fund": case1_vout_fund,
            "case2_Vout_fund": case2_vout_fund,
            "case3_Vout_fund": case3_vout_fund
        })

        voltage_data_per_case = defaultdict(dict)

        try:
            for case in ["case1", "case2", "case3"]:
                print(f"{case}----", file=sys.stderr)

                voltage_data = pd.DataFrame({
                    "freq": vin_fund_data["freq"],
                    "Vout_fund": vout_fund_data[f"{case}_Vout_fund"],
                    "Vin_fund": vin_fund_data[f"{case}_Vin_fund"]
                })

                voltage_data["cg_linear"] = abs(voltage_data["Vout_fund"]/voltage_data["Vin_fund"])
                voltage_data["cg_dB"] = 20 * np.log10(voltage_data["cg_linear"])

                cg_fc1 = voltage_data["cg_linear"][voltage_data["freq"].round(2) == fc1_target] # linear
                cg_fc1 = float(cg_fc1.values[0]) if not cg_fc1.empty else float("nan")
                print("cg simulation result: \n", voltage_data, file=sys.stderr)
                print("cg at fc1: ", cg_fc1, file=sys.stderr)

                if task in ["dualBW", "dualGain", "dualGains"]:
                    cg_fc2 = voltage_data["cg_linear"][voltage_data["freq"].round(2) == fc2_target] # linear
                    cg_fc2 = float(cg_fc2.values[0]) if not cg_fc2.empty else float("nan")
                    print("cg at fc2: ", cg_fc2, file=sys.stderr)

                cg_fc1_db = voltage_data["cg_dB"][voltage_data["freq"].round(2) == fc1_target] # dB
                cg_fc1_db = float(cg_fc1_db.values[0]) if not cg_fc1_db.empty else float("nan")

                if task in ["dualBW", "dualGain", "dualGains", "dualGainsBW"]:
                    cg_fc2_db = voltage_data["cg_dB"][voltage_data["freq"].round(2) == fc2_target] # dB
                    cg_fc2_db = float(cg_fc2_db.values[0]) if not cg_fc2_db.empty else float("nan")

                # OUTPUT 1: # cg at fc1, fc2
                if task == "dualBW"or task == "dualGain":
                    output_dict[f"{case}_cg_fc1"] = cg_fc1
                    output_dict[f"{case}_cg_fc2"] = cg_fc2
                    
                elif task in ["singleBW", "singleGain"]:
                    output_dict[f"{case}_cg_fc1"] = cg_fc1

                voltage_data_per_case[case] = voltage_data

                # NEED FREQ MASK
                if task in ["singleBW", "dualBW", "singleGains", "dualGains"]:
                    epsilon = 1e-6
                    cg_bw_fc1_lower_thres = cg_fc1_db - 3 # dB
                    cg_bw_fc1_upper_thres = cg_fc1_db + epsilon # change 0703
                    cg_mask_fc1_lower = voltage_data["cg_dB"] >= cg_bw_fc1_lower_thres
                    cg_mask_fc1_upper = voltage_data["cg_dB"] <= cg_bw_fc1_upper_thres

                    fc1_bw_target_lower = fc1_bw_target[0] # Hz
                    fc1_bw_target_upper = fc1_bw_target[1] # Hz

                    cg_mask_fc1_freq_lower = voltage_data["freq"].round(2) >= fc1_bw_target_lower # fc1 BW target
                    cg_mask_fc1_freq_upper = voltage_data["freq"].round(2) <= fc1_bw_target_upper # fc1 BW target

                    if task in ["dualBW", "dualGains"]:
                        cg_bw_fc2_lower_thres = cg_fc2_db - 3 # dB
                        cg_bw_fc2_upper_thres = cg_fc2_db + epsilon # change 0703
                        cg_mask_fc2_lower = voltage_data["cg_dB"] >= cg_bw_fc2_lower_thres
                        cg_mask_fc2_upper = voltage_data["cg_dB"] <= cg_bw_fc2_upper_thres

                        fc2_bw_target_lower = fc2_bw_target[0] # Hz
                        fc2_bw_target_upper = fc2_bw_target[1] # Hz

                        # cg_mask_fc1_freq = voltage_data["freq"].round(2) >= fc1_target # BW towards fc2
                        # cg_mask_fc2_freq = voltage_data["freq"].round(2) <= fc2_target # BW towards fc1

                        cg_mask_fc2_freq_lower = voltage_data["freq"].round(2) >= fc2_bw_target_lower # fc2 BW target
                        cg_mask_fc2_freq_upper = voltage_data["freq"].round(2) <= fc2_bw_target_upper # fc2 BW target

                if task in ["dualBW", "dualGains"]:
                    freq_filtered_cg_fc1 = voltage_data[cg_mask_fc1_freq_lower & cg_mask_fc1_freq_upper].copy() #dualGain_fc1
                    freq_filtered_cg_fc2 = voltage_data[cg_mask_fc2_freq_lower & cg_mask_fc2_freq_upper].copy() #dualGain_fc2
                    print(f"freq_filtered_cg_fc1{fc1_target/1e9}GHz:", freq_filtered_cg_fc1, file=sys.stderr)
                    print(f"freq_filtered_cg_fc2{fc2_target/1e9}GHz:", freq_filtered_cg_fc2, file=sys.stderr)
                
                # NEED CG_DB 3dB MASK
                    if task == "dualBW":
                        filtered_cg_fc1 = freq_filtered_cg_fc1[cg_mask_fc1_lower & cg_mask_fc1_upper]
                        filtered_cg_fc2 = freq_filtered_cg_fc2[cg_mask_fc2_lower & cg_mask_fc2_upper]
                        print("filtered_cg_fc1:", filtered_cg_fc1, file=sys.stderr)
                        print("filtered_cg_fc2:", filtered_cg_fc2, file=sys.stderr)
                
                elif task in ["singleBW", "singleGains"]:
                    freq_filtered_cg = voltage_data[cg_mask_fc1_freq_lower & cg_mask_fc1_freq_upper].copy()
                    filtered_cg_fc1 = freq_filtered_cg[cg_mask_fc1_lower & cg_mask_fc1_upper].copy()
                    print(f"filtered_cg_fc1 [{fc1_target/1e9}GHz]:", filtered_cg_fc1, file=sys.stderr)

                # Extract Outputs
                if task == "dualBW":
                    filtered_cg_dict = {"fc1": (fc1_target, filtered_cg_fc1), "fc2": (fc2_target, filtered_cg_fc2)}

                    for fc, (target_freq, filtered_cg_fc) in filtered_cg_dict.items():

                        if not filtered_cg_fc.empty:
                            filtered_cg = filtered_cg_fc.copy()
                            filtered_cg["freq_diff"] = filtered_cg["freq"].diff().fillna(0)
                            filtered_cg["freq_diff"] = filtered_cg["freq_diff"].abs()
                            filtered_cg["groupID"] = (filtered_cg["freq_diff"] > freq_step_thres).cumsum()
                            print(f"for target freq {target_freq/1e9} GHz:", file=sys.stderr)
                            print(f"cg > -3 dB frequencies: \n", filtered_cg, file=sys.stderr)

                            target_freq_row = filtered_cg[filtered_cg["freq"].round(2) == target_freq]
                            target_freq_groupID = target_freq_row["groupID"].values[0]
                            print(f"target_freq_row: {target_freq_row}", file=sys.stderr)
                            # fc1_row = filtered_cg[filtered_cg["freq"].round(2) == fc1_target]
                            # fc2_row = filtered_cg[filtered_cg["freq"].round(2) == fc2_target]
                            # print(f"fc1_row: {fc1_row}", file=sys.stderr)
                            # print(f"fc2_row: {fc2_row}", file=sys.stderr)

                            # if not fc1_row.empty and not fc2_row.empty:
                            #     fc1_groupID = fc1_row["groupID"].values[0]
                            #     fc2_groupID = fc2_row["groupID"].values[0]

                            if not target_freq_row.empty:

                                # if (fc1_groupID == fc2_groupID):
                                #     bw_group_with_target = filtered_cg[filtered_cg["groupID"] == fc1_groupID]["freq"]
                                    # Extract min and max frequency within that band
                                bw_group_with_target = filtered_cg[filtered_cg["groupID"] == target_freq_groupID]["freq"]
                                f_start = bw_group_with_target.min()
                                f_stop = bw_group_with_target.max()
                                bandwidth = f_stop - f_start
                                print(f"Start freq: {f_start/1e9:.2f} GHz, Stop freq: {f_stop/1e9:.2f} GHz", file=sys.stderr)

                                cg_bw = [f_start, f_stop]
                                if target_freq == fc1_target:
                                    output_dict[f"{case}_cg_bw_fc1"] = cg_bw
                                elif target_freq == fc2_target:
                                    output_dict[f"{case}_cg_bw_fc2"] = cg_bw
                                # output_dict[f"{case}_cg_bw"] = cg_bw
                                if bandwidth > 0:
                                    print(f"cg > -3 dB Bandwidth: {bandwidth/1e9:.2f} GHz", file=sys.stderr)
                                
                            else: # freq group with target frequencies not found
                                # print("No frequencies found where cg > cg_max-3 dB including target frequencies", file=sys.stderr)
                                print(f"No frequencies found where cg > cg_max-3 dB including target frequency {target_freq/1e9} GHz", file=sys.stderr)
                        
                        else:
                            print("No frequencies found where cg > cg_max-3 dB", file=sys.stderr)
                
                if task == "dualGains":
                    output_dict[f"{case}_cg_fc1"] = freq_filtered_cg_fc1["cg_linear"].to_list()
                    output_dict[f"{case}_cg_fc2"] = freq_filtered_cg_fc2["cg_linear"].to_list()
                
                if task == "singleBW":
                        target_freq = fc1_target
                        filtered_cg = filtered_cg_fc1.copy()
                        filtered_cg["freq_diff"] = filtered_cg["freq"].diff().fillna(0)
                        filtered_cg["freq_diff"] = filtered_cg["freq_diff"].abs()
                        filtered_cg["groupID"] = (filtered_cg["freq_diff"] > freq_step_thres).cumsum()
                        print(f"for target freq {target_freq/1e9} GHz:", file=sys.stderr)
                        print(f"cg > -3 dB frequencies: \n", filtered_cg, file=sys.stderr)

                        target_freq_row = filtered_cg[filtered_cg["freq"].round(2) == target_freq]
                        target_freq_groupID = target_freq_row["groupID"].values[0]
                        print(f"target_freq_row: {target_freq_row}", file=sys.stderr)
                        if not target_freq_row.empty:
                            bw_group_with_target = filtered_cg[filtered_cg["groupID"] == target_freq_groupID]["freq"]
                            f_start = bw_group_with_target.min()
                            f_stop = bw_group_with_target.max()
                            bandwidth = f_stop - f_start
                            print(f"Start freq: {f_start/1e9:.2f} GHz, Stop freq: {f_stop/1e9:.2f} GHz", file=sys.stderr)
                            cg_bw = [f_start, f_stop]

                            # OUTPUT 2: cg_bw at fc1
                            output_dict[f"{case}_cg_bw_fc1"] = cg_bw
                            if bandwidth > 0:
                                print(f"cg > -3 dB Bandwidth: {bandwidth/1e9:.2f} GHz", file=sys.stderr)
                            
                        else: # freq group with target frequencies not found
                            print(f"No target frequency found where cg > cg_max-3 dB: target frequency {target_freq/1e9} GHz", file=sys.stderr)
                
                if task == "singleGains":
                    output_dict[f"{case}_cg_fc1"] = freq_filtered_cg["cg_linear"].to_list()

        except NameError:
            print("Vin_fund_data or Vout_fund_data not found", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print("Read error:", e, file=sys.stderr)
    
    for case, df in voltage_data_per_case.items():
        df.to_csv(f"{case}_voltage_data_{iteration}_{tb_date}.csv", index=False)
    
    return output_dict

if __name__ == "__main__":

    dsfile = sys.argv[1]
    iteration = sys.argv[2]
    task = sys.argv[3]
    tb_date = sys.argv[4]
    output = read_dsfile(final_stage_dsfile=dsfile, iteration=iteration, task=task, tb_date=tb_date)

    print(json.dumps(output))  # stdout
