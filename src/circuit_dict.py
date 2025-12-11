fc1_target = 2.4e9 # Hz
fc2_target = 5.0e9 # Hz
fc1_bw_target = [1.6e9, 3.2e9] # Hz
fc2_bw_target = [3.4e9, 6.5e9] # Hz

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
    "case2_MN3_dd": ["MN_1", "MN_2", "MN_4"],
    "case2_MX_dd": ["MX_dd1", "MX_dd2"],

    "case3_MN1_ss": ["MN_0", "MN_1", "MN_2", "MN_3", "MN_4", "MN_5"],
    "case3_LNA_sd": ["LNA_sd1"],
    "case3_MN2_dd": ["MN_0", "MN_1", "MN_2", "MN_3", "MN_4", "MN_5"],
    "case3_ADD_dd": ["ADD_dd0", "ADD_dd1"],
    "case3_MN3_dd": ["MN_1", "MN_2", "MN_4"],
    "case3_MX_dd": ["MX_dd1", "MX_dd2"]
}

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
