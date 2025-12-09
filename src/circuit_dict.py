fc1_target = 2.4e9 # Hz 2.4e9
fc2_target = 5.0e9 # Hz 5.0e9
fc1_bw_target = [1.6e9, 3.2e9] # Hz 2.0e9, 2.8e9
fc2_bw_target = [3.4e9, 6.5e9] # Hz 4.2e9, 5.8e9
# bw_target_single = [0.5e9, 6.5e9]

stages = ["MN1", "LNA", "MN2", "ADD", "MN3", "MX"]

simulations = ["DC_VDC1", "DC_VDC2",
               "HB_Z1", "HB_Z2", "HB_Z3", "HB_Z4", "HB_Z5",
               "HB_MN1", "HB_LNA", "HB_MN2", "HB_ADD", "HB_MN3", "HB_MX"]

topology_per_case = {
    "case1_MN1_ss": ["MN_0", "MN_1", "MN_2", "MN_3", "MN_4", "MN_5"],
    "case1_LNA_ss": ["LNA_ss0", "LNA_ss1", "LNA_ss2"], 
    "case1_MN2_ss": ["MN_0", "MN_1", "MN_2", "MN_3", "MN_4", "MN_5"],
    "case1_ADD_ss": ["ADD_ss0"],
    # "case1_MN3_ss": ["MN_0", "MN_1", "MN_2", "MN_3", "MN_4", "MN_5"],
    "case1_MN3_ss": ["MN_0", "MN_1", "MN_2", "MN_4"],
    "case1_MX_sd": ["MX_sd1"],

    "case2_MN1_ss": ["MN_0", "MN_1", "MN_2", "MN_3", "MN_4", "MN_5"],
    "case2_LNA_ss": ["LNA_ss0", "LNA_ss1", "LNA_ss2"],
    "case2_MN2_ss": ["MN_0", "MN_1", "MN_2", "MN_3", "MN_4", "MN_5"],
    "case2_ADD_sd": ["ADD_sd1", "ADD_sd2"],
    # "case2_MN3_dd": ["MN_0", "MN_1", "MN_2", "MN_3", "MN_4", "MN_5"],
    "case2_MN3_dd": ["MN_1", "MN_2", "MN_4"],
    "case2_MX_dd": ["MX_dd1", "MX_dd2"],

    "case3_MN1_ss": ["MN_0", "MN_1", "MN_2", "MN_3", "MN_4", "MN_5"],
    "case3_LNA_sd": ["LNA_sd1"],
    "case3_MN2_dd": ["MN_0", "MN_1", "MN_2", "MN_3", "MN_4", "MN_5"],
    "case3_ADD_dd": ["ADD_dd0", "ADD_dd1"],
    # "case3_MN3_dd": ["MN_0", "MN_1", "MN_2", "MN_3", "MN_4", "MN_5"],
    "case3_MN3_dd": ["MN_1", "MN_2", "MN_4"],
    "case3_MX_dd": ["MX_dd1", "MX_dd2"]
}

# Revised 0707
# pbounds_per_component = {
#     "Length": (0.13, 0.13), # always_set 0.13um
#     "Width": (5.0, 160.0), 
#     "Vdd": (1.5, 2.5), 
#     "Vb" : (0.2, 1.0), 
#     "Vb1_add_sd1": (0.5, 1.8), 
#     "R_f" : (20000, 20000), # always_set 20k
#     "R" : (100,2000), 
#     "C" : (0.1, 5.0), 
#     "L": (0.1, 10.0),
#     "MN_C": (0.1, 10.0),
#     "MN_4_C": (1000.0, 1000.0), # always_set 1nF
#     "MN_L": (0.1, 15.0) 
# }

# for Manual 6
pbounds_per_component = {
    "Length": (0.13, 0.18), 
    "Width": (5.0, 25.0), 
    "Vdd": (2.5, 2.5), # not tunable
    "Vb" : (0.32, 0.8), # 2.5*0.32=0.8, 2.5*0.8=2.0
    "Vb1_add_sd1": (1.8, 1.8), 
    "R_f" : (20000, 20000), # always_set
    "R" : (100,2020), 
    "C" : (0.1, 1.5), 
    "L": (0.8, 10.0), 
    "MN_C": (0.1, 10.0), 
    "MN_4_C": (1000.0, 1000.0), # always_set 1nF
    "MN_L": (1.0, 10.0) 
}

#always_selected_alphas
#{'case1_add_ss0': 1.0, 'case1_mx_sd1': 1.0, 'case3_lna_sd1': 1.0}

#always_set_omega
#{'case1_MN1_4.MN_C': 1000.0, 'case1_LNA_ss1.R_f': 20000, 'case1_Vdd_lna_ss1': 2.5, 'case1_Vdd_lna_ss2': 2.5, 'case1_MN2_4.MN_C': 1000.0, 'case1_MN3_4.MN_C': 1000.0, 'case1_Vdd_mx_sd1': 2.5,
# 
# 'case2_MN1_4.MN_C': 1000.0, 
# 'case2_LNA_ss1.R_f': 20000, 'case2_Vdd_lna_ss1': 2.5, 'case2_Vdd_lna_ss2': 2.5, 
# 'case2_MN2_4.MN_C': 1000.0, 
# 'case2_Vdd_add_sd1': 2.5, 'case2_Vb1_add_sd1': 1.8, 'case2_Vdd_add_sd2': 2.5, 
# 'case2_MN3_4_P.MN_C': 1000.0, 'case2_MN3_4_N.MN_C': 1000.0, 
# 'case2_Vdd_mx_dd2': 2.5, 
# 
# 'case3_MN1_4.MN_C': 1000.0, 'case3_Vdd_lna_sd1': 2.5, 'case3_MN2_4_P.MN_C': 1000.0, 'case3_MN2_4_N.MN_C': 1000.0, 'case3_Vdd_add_dd1': 2.5, 'case3_MN3_4_P.MN_C': 1000.0, 'case3_MN3_4_N.MN_C': 1000.0, 'case3_Vdd_mx_dd2': 2.5}

# param_pbounds
# {'case1_MN1_1.MN_C': (0.1, 10.0), 'case1_Vb_mn1_1': (0.32, 0.8), 'case1_MN1_2.MN_C': (0.1, 10.0), 'case1_MN1_2.MN_L': (1.0, 10.0), 'case1_Vb_mn1_2': (0.32, 0.8), 'case1_MN1_3.MN_C': (0.1, 10.0), 'case1_MN1_3.MN_L': (1.0, 10.0), 'case1_MN1_5.MN_L': (1.0, 10.0), 'case1_LNA_ss1.L1': (0.13, 0.18), 'case1_LNA_ss1.L2': (0.13, 0.18), 'case1_LNA_ss1.L3': (0.13, 0.18), 'case1_LNA_ss1.L4': (0.13, 0.18), 'case1_LNA_ss1.W1': (5.0, 25.0), 'case1_LNA_ss1.W2': (5.0, 25.0), 'case1_LNA_ss1.W3': (5.0, 25.0), 'case1_LNA_ss1.W4': (5.0, 25.0), 'case1_LNA_ss1.R1': (100, 2020), 'case1_LNA_ss1.C1': (0.1, 1.5), 'case1_Vb_lna_ss1': (0.32, 0.8), 'case1_LNA_ss2.L1': (0.13, 0.18), 'case1_LNA_ss2.L2': (0.13, 0.18), 'case1_LNA_ss2.L3': (0.13, 0.18), 'case1_LNA_ss2.L4': (0.13, 0.18), 'case1_LNA_ss2.W1': (5.0, 25.0), 'case1_LNA_ss2.W2': (5.0, 25.0), 'case1_LNA_ss2.W3': (5.0, 25.0), 'case1_LNA_ss2.W4': (5.0, 25.0), 'case1_LNA_ss2.R_l': (100, 2020), 'case1_LNA_ss2.C_l': (0.1, 1.5), 'case1_LNA_ss2.L_l': (0.8, 10.0), 'case1_LNA_ss2.R_s': (100, 2020), 'case1_LNA_ss2.C_s': (0.1, 1.5), 'case1_LNA_ss2.L_s': (0.8, 10.0), 'case1_LNA_ss2.C1': (0.1, 1.5), 'case1_Vb_lna_ss2': (0.32, 0.8), 'case1_MN2_1.MN_C': (0.1, 10.0), 'case1_Vb_mn2_1': (0.32, 0.8), 'case1_MN2_2.MN_C': (0.1, 10.0), 'case1_MN2_2.MN_L': (1.0, 10.0), 'case1_Vb_mn2_2': (0.32, 0.8), 'case1_MN2_3.MN_C': (0.1, 10.0), 'case1_MN2_3.MN_L': (1.0, 10.0), 'case1_MN2_5.MN_L': (1.0, 10.0), 'case1_MN3_1.MN_C': (0.1, 10.0), 'case1_Vb_mn3_1': (0.32, 0.8), 'case1_MN3_2.MN_C': (0.1, 10.0), 'case1_MN3_2.MN_L': (1.0, 10.0), 'case1_Vb_mn3_2': (0.32, 0.8), 'case1_MX_sd1.L1': (0.13, 0.18), 'case1_MX_sd1.L2': (0.13, 0.18), 'case1_MX_sd1.L3': (0.13, 0.18), 'case1_MX_sd1.W1': (5.0, 25.0), 'case1_MX_sd1.W2': (5.0, 25.0), 'case1_MX_sd1.W3': (5.0, 25.0), 'case1_MX_sd1.R_l': (100, 2020), 'case1_Vb_mx_sd1': (0.32, 0.8), 'case1_Vb_mnlo_mx_sd1': (0.32, 0.8),
# 'case2_MN1_1.MN_C': (0.1, 10.0), 'case2_Vb_mn1_1': (0.32, 0.8), 'case2_MN1_2.MN_C': (0.1, 10.0), 'case2_MN1_2.MN_L': (1.0, 10.0), 'case2_Vb_mn1_2': (0.32, 0.8),
# 'case2_MN1_3.MN_C': (0.1, 10.0), 'case2_MN1_3.MN_L': (1.0, 10.0), 
# 'case2_MN1_5.MN_L': (1.0, 10.0), 'case2_LNA_ss1.L1': (0.13, 0.18), 'case2_LNA_ss1.L2': (0.13, 0.18), 'case2_LNA_ss1.L3': (0.13, 0.18), 'case2_LNA_ss1.L4': (0.13, 0.18), 'case2_LNA_ss1.W1': (5.0, 25.0), 'case2_LNA_ss1.W2': (5.0, 25.0), 'case2_LNA_ss1.W3': (5.0, 25.0), 'case2_LNA_ss1.W4': (5.0, 25.0), 'case2_LNA_ss1.R1': (100, 2020), 'case2_LNA_ss1.C1': (0.1, 1.5), 'case2_Vb_lna_ss1': (0.32, 0.8), 
# 'case2_LNA_ss2.L1': (0.13, 0.18), 'case2_LNA_ss2.L2': (0.13, 0.18), 'case2_LNA_ss2.L3': (0.13, 0.18), 'case2_LNA_ss2.L4': (0.13, 0.18), 'case2_LNA_ss2.W1': (5.0, 25.0), 'case2_LNA_ss2.W2': (5.0, 25.0), 'case2_LNA_ss2.W3': (5.0, 25.0), 'case2_LNA_ss2.W4': (5.0, 25.0), 'case2_LNA_ss2.R_l': (100, 2020), 'case2_LNA_ss2.C_l': (0.1, 1.5), 'case2_LNA_ss2.L_l': (0.8, 10.0), 'case2_LNA_ss2.R_s': (100, 2020), 'case2_LNA_ss2.C_s': (0.1, 1.5), 'case2_LNA_ss2.L_s': (0.8, 10.0), 'case2_LNA_ss2.C1': (0.1, 1.5), 'case2_Vb_lna_ss2': (0.32, 0.8), 
# 'case2_MN2_1.MN_C': (0.1, 10.0), 'case2_Vb_mn2_1': (0.32, 0.8), 
# 'case2_MN2_2.MN_C': (0.1, 10.0), 'case2_MN2_2.MN_L': (1.0, 10.0), 'case2_Vb_mn2_2': (0.32, 0.8), 
# 'case2_MN2_3.MN_C': (0.1, 10.0), 'case2_MN2_3.MN_L': (1.0, 10.0), 'case2_MN2_5.MN_L': (1.0, 10.0), 'case2_ADD_sd1.L1': (0.13, 0.18), 'case2_ADD_sd1.L2': (0.13, 0.18), 'case2_ADD_sd1.L3': (0.13, 0.18), 'case2_ADD_sd1.W1': (5.0, 25.0), 'case2_ADD_sd1.W2': (5.0, 25.0), 'case2_ADD_sd1.W3': (5.0, 25.0), 'case2_Vb3_add_sd1': (0.32, 0.8),
# 'case2_ADD_sd2.L1': (0.13, 0.18), 'case2_ADD_sd2.L2': (0.13, 0.18), 'case2_ADD_sd2.L3': (0.13, 0.18), 'case2_ADD_sd2.W1': (5.0, 25.0), 'case2_ADD_sd2.W2': (5.0, 25.0), 'case2_ADD_sd2.W3': (5.0, 25.0), 'case2_ADD_sd2.R_l': (100, 2020), 'case2_Vb_add_sd2': (0.32, 0.8),
# 'case2_MN3_1_P.MN_C': (0.1, 10.0), 'case2_MN3_1_N.MN_C': (0.1, 10.0), 'case2_Vb_mn3_1_p': (0.32, 0.8), 'case2_MN3_2_P.MN_C': (0.1, 10.0), 'case2_MN3_2_N.MN_C': (0.1, 10.0), 'case2_MN3_2_P.MN_L': (1.0, 10.0), 'case2_MN3_2_N.MN_L': (1.0, 10.0), 'case2_Vb_mn3_2_p': (0.32, 0.8), 
# 'case2_MX_dd1.L1': (0.13, 0.18), 'case2_MX_dd1.W1': (5.0, 25.0), 'case2_MX_dd2.L1': (0.13, 0.18), 'case2_MX_dd2.L2': (0.13, 0.18), 'case2_MX_dd2.L3': (0.13, 0.18), 'case2_MX_dd2.W1': (5.0, 25.0), 'case2_MX_dd2.W2': (5.0, 25.0), 'case2_MX_dd2.W3': (5.0, 25.0), 'case2_MX_dd2.R_l': (100, 2020), 'case2_Vb_mx_dd2': (0.32, 0.8), 'case2_Vb_mnlo_mx_dd2': (0.32, 0.8),
# 'case3_MN1_1.MN_C': (0.1, 10.0), 'case3_Vb_mn1_1': (0.32, 0.8), 'case3_MN1_2.MN_C': (0.1, 10.0), 'case3_MN1_2.MN_L': (1.0, 10.0), 'case3_Vb_mn1_2': (0.32, 0.8), 'case3_MN1_3.MN_C': (0.1, 10.0), 'case3_MN1_3.MN_L': (1.0, 10.0), 'case3_MN1_5.MN_L': (1.0, 10.0), 'case3_LNA_sd1.L1': (0.13, 0.18), 'case3_LNA_sd1.L2': (0.13, 0.18), 'case3_LNA_sd1.W1': (5.0, 25.0), 'case3_LNA_sd1.W2': (5.0, 25.0), 'case3_LNA_sd1.R_l': (100, 2020), 'case3_LNA_sd1.R_s': (100, 2020), 'case3_LNA_sd1.R1': (100, 2020), 'case3_LNA_sd1.C1': (0.1, 1.5), 'case3_Vb1_lna_sd1': (0.32, 0.8), 'case3_Vb2_lna_sd1': (0.32, 0.8), 'case3_MN2_1_P.MN_C': (0.1, 10.0), 'case3_MN2_1_N.MN_C': (0.1, 10.0), 'case3_Vb_mn2_1_p': (0.32, 0.8), 'case3_MN2_2_P.MN_C': (0.1, 10.0), 'case3_MN2_2_N.MN_C': (0.1, 10.0), 'case3_MN2_2_P.MN_L': (1.0, 10.0), 'case3_MN2_2_N.MN_L': (1.0, 10.0), 'case3_Vb_mn2_2_p': (0.32, 0.8), 'case3_MN2_3_P.MN_C': (0.1, 10.0), 'case3_MN2_3_N.MN_C': (0.1, 10.0), 'case3_MN2_3_P.MN_L': (1.0, 10.0), 'case3_MN2_3_N.MN_L': (1.0, 10.0), 'case3_MN2_5_P.MN_L': (1.0, 10.0), 'case3_MN2_5_N.MN_L': (1.0, 10.0), 'case3_ADD_dd1.L1': (0.13, 0.18), 'case3_ADD_dd1.W1': (5.0, 25.0), 'case3_ADD_dd1.R_l': (100, 2020), 'case3_MN3_1_P.MN_C': (0.1, 10.0), 'case3_MN3_1_N.MN_C': (0.1, 10.0), 'case3_Vb_mn3_1_p': (0.32, 0.8), 'case3_MN3_2_P.MN_C': (0.1, 10.0), 'case3_MN3_2_N.MN_C': (0.1, 10.0), 'case3_MN3_2_P.MN_L': (1.0, 10.0), 'case3_MN3_2_N.MN_L': (1.0, 10.0), 'case3_Vb_mn3_2_p': (0.32, 0.8), 'case3_MX_dd1.L1': (0.13, 0.18), 'case3_MX_dd1.W1': (5.0, 25.0), 'case3_MX_dd2.L1': (0.13, 0.18), 'case3_MX_dd2.L2': (0.13, 0.18), 'case3_MX_dd2.L3': (0.13, 0.18), 'case3_MX_dd2.W1': (5.0, 25.0), 'case3_MX_dd2.W2': (5.0, 25.0), 'case3_MX_dd2.W3': (5.0, 25.0), 'case3_MX_dd2.R_l': (100, 2020), 'case3_Vb_mx_dd2': (0.32, 0.8), 'case3_Vb_mnlo_mx_dd2': (0.32, 0.8)}


step_sizes = {
    "Length": 0.01,
    "Width": 1.0,
    "Vdd": 0.01,
    "Vb": 0.004,  # continuous
    "Vb1_add_sd1": 0.01,
    "R_f": 100.0,
    "R": 1.0,
    "C": 0.1,
    "L": 0.1,
    "MN_C": 0.1,
    "MN_L": 0.1
}

# pbounds_per_component = {
#     "Length": (0.13, 0.65), #(0.13, 0.65) um
#     "Width": (5.0, 80.0,), #(5.0, 180.0)#um
#     "Vdd": (1.5, 2.5), # (1.5, 2.5) #V
#     "Vb" : (0.2, 1.0), #V, Maximum value = Vdd. (0.2, 1.0) -> will scale to (0.5, Vdd) Vb=Vb_frac*Vdd
#     "Vb1_add_sd1": (0.5, 1.8), # (0.5, 1.8) # V # Small Vb1 of ADD_sd1 makes negative Zin_add_sd1
#     "R_f" : (10000, 20000), # (10000, 20000) #Ohm
#     "R" : (100,800), # (100,2000) #Ohm
#     "C" : (0.1, 5.0), #pF
#     "L": (0.1, 20.0), #nH
#     "MN_C": (0.1, 1000.0), #pF
#     "MN_L": (0.1, 20.0) #nH
# }

topology_dict = {
    #MN
    "MN_0":{
        
    },

    "MN_1": {
        # "MN_R": 20000, # Ohm
        "MN_C": None,
        "Vb": None
    },

    "MN_2": {
        "MN_C": None, # Ohm
        "MN_L": None,
        "Vb": None
    },

    "MN_3": {
        "MN_C": None, # Ohm
        "MN_L": None,
        # "Vb": 0
    },

    "MN_4": {
        "MN_C": None
    },

    "MN_5": {
        "MN_L": None, # nH
    },

    #LNA
    "LNA_ss0":{
        
    },

    "LNA_ss1": {
          "L1": None,
          "L2": None,
          "L3": None,
          "L4": None,
          "W1": None,
          "W2": None,
          "W3": None,
          "W4": None,
          "R_f": None,
          "R1": None,
          "C1": None,
          "Vdd": None,
          "Vb": None
                },

    "LNA_ss2": {
        "L1": None,
        "L2": None,
        "L3": None,
        "L4": None,
        "W1": None,
        "W2": None,
        "W3": None,
        "W4": None,
        "R_l": None,
        "C_l": None,
        "L_l": None,
        "R_s": None,
        "C_s": None,
        "L_s": None,
        "C1": None,
        "Vdd": None,
        "Vb": None
    },

    "LNA_sd1": {
        "L1": None,
        "L2": None,
        "W1": None,
        "W2": None,
        "R_l": None,
        "R_s": None,
        "R1": None,
        "C1": None,
        "Vdd": None,
        "Vb1": None,
        "Vb2": None
    },

    #ADD
    "ADD_ss0": {
    },

    "ADD_sd1": {
        "L1": None,
        "L2": None,
        "L3": None,
        "W1": None,
        "W2": None,
        "W3": None,
        "Vdd": None,
        "Vb1": None,
        # "Vb2": None, # Set to be same as input VDC
        "Vb3": None
    },

    "ADD_sd2": {
        "L1": None,
        "L2": None,
        "L3": None,
        "W1": None,
        "W2": None,
        "W3": None,
        "R_l": None,
        "Vdd": None,
        "Vb": None
    },

    "ADD_dd0": {
        
    },

    "ADD_dd1": {
        "L1": None,
        "W1": None,
        "R_l": None,
        "Vdd": None
    },


  #MX
  "MX_sd1":{
    "L1": None,
    "L2": None,
    "L3": None,
    "W1": None,
    "W2": None,
    "W3": None,
    "R_l": None,
    "Vdd": None,
    "Vb": None,

    "Vb_mnlo": None
  },

  "MX_dd1": {
      "L1": None,
      "W1": None,

    #   "Vb_mnlo": None

      },

  "MX_dd2": {
      "L1": None,
      "L2": None,
      "L3": None,
      "W1": None,
      "W2": None,
      "W3": None,
      "R_l": None,
      "Vdd": None,
      "Vb": None,

      "Vb_mnlo": None
            }

}
