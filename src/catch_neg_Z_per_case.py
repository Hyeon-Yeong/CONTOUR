import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict

try:
    import keysight.ads.dataset as dataset
    from keysight import pwdatatools as pwdt
except Exception as e:
    print("Import error:", e, file=sys.stderr)
    sys.exit(1)

def catch_neg_z(dsfile):
    case1_neg_z_found = False
    case1_nan_z_found = False
    case2_neg_z_found = False
    case2_nan_z_found = False
    case3_neg_z_found = False
    case3_nan_z_found = False

    try:
        output_data = dataset.open(dsfile)


        for member in output_data:
            block_data = output_data[member].to_dataframe().reset_index()

            for col in block_data.columns:
                if "Zin" in col or "Zout" in col:
                    z_values = block_data[col]

                    if "_org" not in col and (np.real(z_values) < -0.1).any():
                        print(f"Negative Z values found in {col} of block {member}.", file=sys.stderr)
                        if "case1" in col:
                            case1_neg_z_found = True
                        elif "case2" in col:
                            case2_neg_z_found = True
                        elif "case3" in col:
                            case3_neg_z_found = True
                        break

                    if z_values.isna().any():
                        print(f"NaN Z values found in {col} of block {member}.", file=sys.stderr)
                        if "case1" in col:
                            case1_nan_z_found = True
                        elif "case2" in col:
                            case2_nan_z_found = True
                        elif "case3" in col:
                            case3_nan_z_found = True
                        break

            if case1_neg_z_found or case1_nan_z_found or case2_neg_z_found or case2_nan_z_found or case3_neg_z_found or case3_nan_z_found:
                break

    except Exception as e:
        print("Read error:", e, file=sys.stderr)

    return (case1_neg_z_found, case1_nan_z_found, 
            case2_neg_z_found, case2_nan_z_found, 
            case3_neg_z_found, case3_nan_z_found)

if __name__ == "__main__":
    dsfile = sys.argv[1]
    output = catch_neg_z(dsfile)
    print(json.dumps(output))  # stdout
