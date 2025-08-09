#%%
import itertools as it
# import multiprocessing as mp # Not needed for a single case
from pathlib import Path
import time

import cantera as ct
import numpy as np
# import pandas as pd # Not needed for this specific ROP analysis output, unless you want to save tabular ROP data
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # For formatting ticks
#%%
# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

MECH                = "reactionsokafar.yaml"
P_ATM               = ct.one_atm
T_FUEL              = 300.0
T_AIR               = 300.0
TEMP_THRESHOLD      = 1400.0

SINGLE_CRACK_FRACTION = 0.28
SINGLE_UF             = 0.2
SINGLE_UA             = 0.2
SINGLE_WIDTH          = 0.01

FIG_DIR             = Path("figures_single_case_rop")
FIG_DIR.mkdir(exist_ok=True)
#%%
# ---------------------------------------------------------------------------
# COMPOSITION FUNCTIONS
# ---------------------------------------------------------------------------

def cracked_fuel_X(f):
    return f"NH3:{1 - f}, H2:{1.5 * f}, N2:{0.5 * f}"

AIR_X = "O2:0.21, N2:0.79"

def fix_velocity(u):
    return max(u, 1e-9)
#%%
# ---------------------------------------------------------------------------
# SIMULATION AND ROP FUNCTION
# ---------------------------------------------------------------------------

def run_single_flame_and_analyze_rop(f_crack, uf, ua, width):
    gas = ct.Solution(MECH)

    gas.TPX = T_FUEL, P_ATM, cracked_fuel_X(f_crack)
    rho_f = gas.density
    mdot_f = rho_f * fix_velocity(uf)

    gas.TPX = T_AIR, P_ATM, AIR_X
    rho_a = gas.density
    mdot_a = rho_a * fix_velocity(ua)

    flame = ct.CounterflowDiffusionFlame(gas, width=width)
    flame.P = P_ATM

    flame.fuel_inlet.mdot = mdot_f
    flame.fuel_inlet.T = T_FUEL
    flame.fuel_inlet.X = cracked_fuel_X(f_crack)

    flame.oxidizer_inlet.mdot = mdot_a
    flame.oxidizer_inlet.T = T_AIR
    flame.oxidizer_inlet.X = AIR_X

    flame.set_initial_guess()

    print(f"Solving flame for: crack_fraction={f_crack}, u_fuel={uf} m/s, u_air={ua} m/s, width={width*1000} mm...")
    try:
        flame.solve(loglevel=0, auto=True, refine_grid=True)
        Tmax = float(np.max(flame.T))
        print(f"Flame solved successfully. Maximum Temperature = {Tmax:.2f} K")
        if Tmax < TEMP_THRESHOLD:
            print(f"Warning: Maximum temperature ({Tmax:.2f} K) is below the threshold ({TEMP_THRESHOLD} K).")
            print("The flame might be weak or extinguished. ROP analysis will proceed, but interpret results with caution.")
    except Exception as e:
        print(f"Error solving flame: {e}")
        print("Cannot proceed with ROP analysis.")
        return
# %%
