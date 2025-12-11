# CONTOUR
Continuous Topology Search for Unified RF Receiver Front-End Design Optimization, Hyeon-Yeong Yeo, Master's Thesis @ UT Austin ECE Fall 2025

CONTOUR is a research framework for **automatic optimization and design of RF receiver (RX) front-ends**.  
It couples PSO, **Python-based optimization**,  with **Keysight ADS netlist simulations** for evaluation. In this framework, we are able to search over both circuit **topologies** and **device-level parameters**.

---

## Overview

Modern analog/RF design often requires iterating manually over many circuit topologies and parameter settings.  
CONTOUR automates this process for a dual-band RX front-end by:

- Representing the RX as a composition of functional blocks (LNA, matching network, mixer).
- Using continuous “architecture parameters” to approximate topology selection.
- Using an optimizer to jointly tune architecture parameters and circuit-level parameters. The problem is formulated as bi-level optimization.
- Driving ADS simulations via netlists and parsing voltage results back into Python. Conversion gain and its BW is calculated based on the voltage results.
- Cost is defined with Figure of Merit (FoM) per task. 
- **Sequential, loading-aware simulation flow**: The evaluation will go through big three steps: DC simulation -> HB simulation -> HB simulation. First two simulations are for extracting the DC output voltages and Zin/Zout of each topology and stage. These information will be used as the termination condition of running HB simulation at the last stage, which is the actual evaulation. After the last HB simulation, the voltage output would be extracted. 

---

## Repository Structure

```text
CONTOUR/
├─ README.md                # Explanation on files in this repository
├─ src/                     # Python source code
│  ├─ CONTOUR.py            # Main optimization script
│  ├─ circuit_dict.py       # Settings of the target, circuit topologies, tuning range.
│  ├─ circuit_dict_utils.py # Functions to generate/modify the dictionaries for running the script
│  ├─ csv_utils.py          # Functions to extract/find the output csv files
│  ├─ fom.py                # Functions to calculate FoM and Cost
│  ├─ rf_rx_pso_utils.py    # Functions for running pso. It also checks whether there is a NaN value in the output. If so, it 
│  │                        retries with a new particle near the evaluation-failed particle.
│  ├─ run_ads_simualtion.py # Functions to run ADS simualtions. It will run the simulation in sequence (**Sequential,           
│  │                        loading-aware simulation flow**): DC_VDC1 -> DC_VDC2 -> HB_Z1 -> HB_Z2 -> HB_Z3 -> HB_Z4 -> HB_Z5 
│  │                        -> HB_MN1 -> HB_LNA -> HB_MN2 -> HB_ADD -> HB_MN3 -> HB_MX
│  │                        If there is any case where we see negative impedance, the simulation flow stops.
│  ├─ catch_neg_Z_per_case.py # Read the impedances from HB_Z1, ..., HB_Z5 and check if there is negative Z value.
│  └─read_ads_dsfile_*.py  # Read the final output (HB_MX) and extract Vin, Vout per frequency. The output file is in .ds       
│                          format, and the python from ADS is used to read it. .ds file is in ASCII, so you cannot read it by 
│                          just opening it. 
│                            0618: simulation in 0.1GHz step, 0729: simulation in 0.02GHz, manual: used for 
│                          reading the result of manual circuits (e.g. Manual 6). The type should follow the netlist name as well.
│
├─ netlists/                # ADS netlist files. These should be under the ADS workspace (e.g. rf_rx_0306_wrk), as it may use the
│  │                        PDK/library of the circuit you made. (e.g. "/home/local/ace/hy7557/rf_rx_0306_wrk/130nm_bulk.net)
│  ├─ DC_VDC1_*.log         # Loading-aware DC simulation 1
│  ├─ DC_VDC2_*.log         # Loading-aware DC simulation 2
│  ├─ HB_Z1_*.log           # Loading-aware HB simulation 1 (Zin/Zout)
│  ├─ HB_Z2_*.log           # Loading-aware HB simulation 2 (Zin/Zout)
│  ├─ HB_Z3_*.log           # Loading-aware HB simulation 3 (Zin/Zout)
│  ├─ HB_Z4_*.log           # Loading-aware HB simulation 4 (Zin/Zout)
│  ├─ HB_Z5_*.log           # Loading-aware HB simulation 5 (Zin/Zout)
│  ├─ HB_MN1_*.log          # Actual evaluation. Will be run in the order defined in **circuit_dict.py** stages. The information 
│  │                        needed to run the simulation (termination conditions, output voltage of previous stage) will be used 
│  │                        by DAC(Data Access Component) component in ADS. You can assign the file name and the parameter name 
│  │                        to read.
│  ├─ ...
│  └─ HB_MX_*.log           # Last Actual evaluation. The vin voltage and vout voltage of this file will be used to calculation 
│  │                        conversion gain and BW. (by read_ads_dsfile_*.py)
├─ experiments/             # Other experiments for comparison. Refer to the thesis for each experiment.
│  ├─ bothOpt_pso.py
│  ├─ manual_pso.py
│  └─ ...
├─ results/                 # Example of result folder generated after running CONTOUR.py / bothOpt_pso.py / manual_pso.py
│  ├─ contour/
│  ├─ bothOpt/
│  └─ manaul/
└─ docs/                    # Additional documentation
   └─ How to run schematic level simulation in Keysight ADS using Python.pdf # Guide on running ADS with netlists
