# ---------------------------------------------------------------------------
# COUNTERFLOW DIFFUSION FLAME SIMULATION SCRIPT
#
# Description:
# This script simulates the extinction characteristics of a counterflow
# diffusion flame for partially cracked ammonia combustion using Cantera.
# It iterates over various cracking fractions, fuel/air velocities, and
# burner widths, running the simulations in parallel to save time.
#
# Output:
# 1. An Excel file ('extinction_data.xlsx') containing the complete dataset
#    of results, ready for machine learning, as well as separate sheets
#    for each specific case.
# 2. A series of PNG plot files ('figures/*.png') showing the 2D
#    extinction maps for each cracking fraction and width combination.
# ---------------------------------------------------------------------------

import itertools as it
import multiprocessing as mp
from pathlib import Path
import time

# Cantera is the core library for chemical kinetics and flame simulation
import cantera as ct

# NumPy and Pandas are used for numerical operations and data management
import numpy as np
import pandas as pd

# Matplotlib is used for plotting the results
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# USER SETTINGS AND GLOBAL CONSTANTS
# ---------------------------------------------------------------------------

# --- SIMULATION SETTINGS ---
MECH           = "reactionsokafar.yaml"  # Chemical mechanism file
P_ATM          = ct.one_atm              # Pressure [Pa]
T_FUEL         = 300.0                   # Fuel inlet temperature [K]
T_AIR          = 300.0                   # Air inlet temperature [K]
TEMP_THRESHOLD = 1273.0                  # Extinction threshold temperature [K]

# --- PARALLEL PROCESSING ---
# Use one less than the total number of CPU cores, or at least 1
CPUS = max(mp.cpu_count() - 1, 1)

# --- PARAMETER RANGES TO SIMULATE ---
# Cracking fractions (f=0 is pure NH3, f=1 is fully cracked)
crack_fractions = [0.28, 0.50, 0.75]
# Range of fuel velocities [m/s]
uf_range        = np.arange(0.0, 30.0 + 1e-9, 1.0)
# Range of air velocities [m/s]
ua_range        = np.arange(0.0, 18.0 + 1e-9, 1.0)
# Burner widths [meters]
widths          = [0.01, 0.02, 0.03]      # Corresponds to 1 cm, 2 cm, 3 cm

# --- OUTPUT CONFIGURATION ---
OUT_XLSX = Path("extinction_data.xlsx")
FIG_DIR  = Path("figures")


# ---------------------------------------------------------------------------
# HELPER AND COMPOSITION FUNCTIONS
# ---------------------------------------------------------------------------

def cracked_fuel_X(f):
    """
    Returns the mole fraction string for partially cracked ammonia fuel.
    The reaction is: NH3 -> (1-f)NH3 + 1.5*f*H2 + 0.5*f*N2
    """
    return f"NH3:{1 - f}, H2:{1.5 * f}, N2:{0.5 * f}"

AIR_X = "O2:0.21, N2:0.79"

def fix_velocity(u):
    """
    Replaces a velocity of 0 with a very small number (1e-9) for the simulation,
    as Cantera's mass flow controllers cannot handle zero velocity.
    """
    return max(u, 1e-9)

def print_progress(count, total, f, uf, ua, w, Tmax, elapsed):
    """Formats and prints a dynamic progress line to the console."""
    progress_percent = (count / total) * 100
    # Note the spaces at the end of the string to clear previous, longer lines
    status_line = (
        f"\rProgress: {count}/{total} ({progress_percent:.1f}%) | "
        f"Last Case: [f={f:.2f}, uf={uf:.1f}, ua={ua:.1f}, w={w*100:.0f}cm] -> T_max={Tmax:.0f}K | "
        f"Elapsed: {elapsed/60:.1f}min  "
    )
    # Print to the same line without a newline, flushing to ensure it appears immediately
    print(status_line, end="", flush=True)


# ---------------------------------------------------------------------------
# CORE SIMULATION FUNCTION
# ---------------------------------------------------------------------------

def run_case(args):
    """
    Runs a single counterflow flame simulation for a given case.
    
    Args:
        A tuple containing (f, uf, ua, width).
    
    Returns:
        A tuple with results: (f, uf, ua, width, Tmax, stable).
    """
    f, uf, ua, width = args
    
    gas = ct.Solution(MECH)

    # Calculate fuel-side mass flow rate
    gas.TPX = T_FUEL, P_ATM, cracked_fuel_X(f)
    mdot_f = gas.density * uf

    # Calculate air-side mass flow rate
    gas.TPX = T_AIR, P_ATM, AIR_X
    mdot_a = gas.density * ua

    # Create the flame object
    flame = ct.CounterflowDiffusionFlame(gas, width=width)
    flame.P = P_ATM

    # Set fuel and oxidizer inlet properties
    flame.fuel_inlet.mdot = mdot_f
    flame.fuel_inlet.T = T_FUEL
    flame.fuel_inlet.X = cracked_fuel_X(f)
    flame.oxidizer_inlet.mdot = mdot_a
    flame.oxidizer_inlet.T = T_AIR
    flame.oxidizer_inlet.X = AIR_X

    flame.set_initial_guess()

    try:
        # Solve the flame with suppressed output and auto-grid refinement
        flame.solve(loglevel=0, auto=True)
        Tmax = float(np.max(flame.T))
        # Check if the maximum temperature is above the stability threshold
        stable = int(Tmax >= TEMP_THRESHOLD)
    except Exception:
        # If any Cantera error occurs, the flame is considered non-convergent/extinct
        Tmax = np.nan
        stable = 0

    return f, uf, ua, width, Tmax, stable


# ---------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------------------------

def main():
    """
    Main function to orchestrate the simulation, data processing, and plotting.
    """
    # --- SETUP ---
    FIG_DIR.mkdir(exist_ok=True)
    start_time = time.perf_counter()

    # --- SIMULATION BATCH ---
    all_params = list(it.product(crack_fractions, uf_range, ua_range, widths))
    total_cases = len(all_params)
    tasks = [(f, fix_velocity(uf), fix_velocity(ua), w) for f, uf, ua, w in all_params]
    all_results = []

    print(f"Settings configured successfully. Using chemical mechanism: {MECH}")
    print(f"Setting up to run {total_cases} total simulation cases on {CPUS} CPU core(s)...")

    with mp.Pool(CPUS) as pool:
        # imap_unordered is efficient for collecting results as they complete
        for i, res in enumerate(pool.imap_unordered(run_case, tasks), 1):
            all_results.append(res)
            
            # Unpack the result to get case details for progress reporting
            f_res, uf_res, ua_res, w_res, Tmax_res, _ = res
            current_elapsed = time.perf_counter() - start_time
            
            # Call the dynamic progress printer for EVERY case
            print_progress(i, total_cases, f_res, uf_res, ua_res, w_res, Tmax_res, current_elapsed)
    
    print("\n\nAll simulations completed.")

    # --- DATA PROCESSING ---
    df = pd.DataFrame(all_results,
                      columns=["f_crack", "u_fuel", "u_air", "width", "T_max", "stable"])
    df.sort_values(["f_crack", "width", "u_fuel", "u_air"], inplace=True)

    # --- SAVE RESULTS TO EXCEL ---
    print(f"\nSaving data to Excel file: {OUT_XLSX}")
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="all_data", index=False)
        for f_val in crack_fractions:
            for width_val in widths:
                sheet_name = f"f_{f_val:.2f}_w_{int(width_val*100)}"
                subset_df = df[(df.f_crack == f_val) & (df.width == width_val)]
                subset_df.to_excel(xl, sheet_name=sheet_name, index=False)
    print(f"✅ Results successfully saved ➜ {OUT_XLSX.resolve()}")

    # --- GENERATE AND SAVE PLOTS ---
    print("\nGenerating and saving plots...")
    for f_val in crack_fractions:
        for width_val in widths:
            d = df[(df.f_crack == f_val) & (df.width == width_val)].copy()

            # Replace simulation velocities (1e-9) back to 0 for clear plotting
            d["u_fuel_plot"] = d["u_fuel"].apply(lambda x: 0 if x <= 1e-9 else x)
            d["u_air_plot"]  = d["u_air"].apply(lambda x: 0 if x <= 1e-9 else x)

            plt.figure(figsize=(7, 6))

            stable_data = d[d.stable == 1]
            extinct_data = d[d.stable == 0]
            
            plt.scatter(stable_data.u_fuel_plot, stable_data.u_air_plot,
                        marker="o", label="Stable flame", s=30, c='blue', zorder=2)
            plt.scatter(extinct_data.u_fuel_plot, extinct_data.u_air_plot,
                        marker="x", label="Extinct / no-solution", s=30, c='red', zorder=2)
            
            if not stable_data.empty:
                boundary_df = stable_data.groupby("u_fuel_plot")["u_air_plot"].max().reset_index()
                boundary_df.sort_values(by="u_fuel_plot", inplace=True)
                plt.plot(boundary_df.u_fuel_plot, boundary_df.u_air_plot,
                         linestyle='-', color='green', marker='.', linewidth=2,
                         label="Stability Limit", zorder=3)
            
            plt.grid(True, ls=":", zorder=1)
            plt.xlabel("Fuel velocity $u_{fuel}$  [m/s]")
            plt.ylabel("Air velocity $u_{air}$  [m/s]")
            plt.title(f"Extinction Map\nf = {f_val:.2f}, width = {width_val*100:.0f} mm")
            plt.legend()
            plt.tight_layout()
            
            fname = FIG_DIR / f"extinction_map_f_{f_val:.2f}_w_{int(width_val*100)}.png"
            plt.savefig(fname, dpi=300)
            plt.close() # Close plot to free memory
            print(f"  - Figure saved ➜ {fname.resolve()}")
            
    print("\nAll plots have been generated and saved.")

    # --- FINAL TIME REPORT ---
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"\n🕒 Total simulation time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n")


if __name__ == "__main__":
    # The main() function is called only when the script is executed directly.
    # This is a requirement for the multiprocessing module to work correctly
    # on all operating systems (especially Windows).
    main()