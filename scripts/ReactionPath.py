# =========================================================================
#       FINAL SCRIPT WITH HORIZONTAL REACTION PATH DIAGRAM (ALL PATHS)
# -------------------------------------------------------------------------
# This version shows all reaction pathways by setting threshold=0.0 and
# changes the diagram layout from vertical to horizontal by setting
# rankdir="LR".
# =========================================================================

#%%
import time
from pathlib import Path
import os
import re
import subprocess
import cantera as ct
import numpy as np

# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

MECH = "mechanisms/reactionsokafar.yaml"
P_ATM = ct.one_atm
T_FUEL = 300.0
T_AIR = 300.0
SINGLE_CRACK_FRACTION = 0.50
SINGLE_UF = 1.0
SINGLE_UA = 1.0
SINGLE_WIDTH = 0.01
FIG_DIR = Path("results/hasil_NO_ReactionPath")
FIG_DIR.mkdir(exist_ok=True)

#%%
# ---------------------------------------------------------------------------
# COMPOSITION AND UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def cracked_fuel_X(f):
    """Returns the mole fraction string for a partially cracked ammonia fuel."""
    return f"NH3:{1 - f}, H2:{1.5 * f}, N2:{0.5 * f}"

AIR_X = "O2:0.21, N2:0.79"

def fix_velocity(u):
    """Ensures the velocity is a small positive number to avoid mdot=0."""
    return max(u, 1e-9)

#%%
# ---------------------------------------------------------------------------
# SIMULATION AND REACTION PATH ANALYSIS FUNCTION
# ---------------------------------------------------------------------------

def generate_reaction_path(f_crack, uf, ua, width):
    """
    Solves a counterflow flame and generates a reaction path diagram where
    arrow labels are shown as percentages relative to the largest flux.
    """
    # 1. --- SETUP AND SOLVE THE COUNTERFLOW FLAME ---
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

    print(f"Solving flame for: crack_fraction={f_crack * 100:.0f}%, u_fuel={uf} m/s, u_air={ua} m/s...")
    try:
        flame.solve(loglevel=0, auto=True, refine_grid=True)
        Tmax = float(np.max(flame.T))
        print(f"Flame solved successfully. Maximum Temperature = {Tmax:.2f} K")
    except Exception as e:
        print(f"Error solving flame: {e}")
        print("Cannot proceed with analysis.")
        return

    # 2. --- FIND LOCATION OF MAXIMUM NO PRODUCTION RATE ---
    print("\nAnalyzing flame to find the location of maximum NO production...")
    target_species = 'NO'
    working_gas = ct.Solution(MECH)
    try:
        sp_idx = working_gas.species_index(target_species)
    except ValueError:
        print(f"Error: Species '{target_species}' not found in mechanism '{MECH}'.")
        return

    max_rop_val = -np.inf
    max_rop_location_index = -1
    for j in range(len(flame.grid)):
        working_gas.TPY = flame.T[j], flame.P, flame.Y[:, j]
        net_rates = working_gas.net_rates_of_progress
        sp_net_rop = 0.0
        for i in range(working_gas.n_reactions):
            nu_sp = working_gas.product_stoich_coeffs[sp_idx, i] - working_gas.reactant_stoich_coeffs[sp_idx, i]
            if nu_sp != 0:
                sp_net_rop += nu_sp * net_rates[i]
        if sp_net_rop > max_rop_val:
            max_rop_val = sp_net_rop
            max_rop_location_index = j

    if max_rop_location_index == -1:
        print(f"Could not find a production zone for '{target_species}'. Cannot generate diagram.")
        return

    T_at_max_rop = flame.T[max_rop_location_index]
    z_at_max_rop_mm = flame.grid[max_rop_location_index] * 1000
    print(f"Maximum '{target_species}' ROP found at z = {z_at_max_rop_mm:.3f} mm (T = {T_at_max_rop:.1f} K).")

    # 3. --- GENERATE AND SAVE THE REACTION PATH DIAGRAM WITH % LABELS ---
    print(f"\nGenerating reaction path diagram for element 'N'...")
    working_gas.TPY = flame.T[max_rop_location_index], flame.P, flame.Y[:, max_rop_location_index]
    element = 'N'
    diagram = ct.ReactionPathDiagram(working_gas, element)
    
    # ==================== MODIFICATION 1: SHOW ALL PATHS =====================
    # Set threshold to 0.0 to prevent any pathways from being filtered out.
    diagram.threshold = 0.01
    # =========================================================================

    # ================== MODIFICATION 2: HORIZONTAL LAYOUT ====================
    # We change rankdir from "TB" (Top-to-Bottom) back to "LR" (Left-to-Right).
    diagram.dot_options = 'rankdir="LR";'
    # =========================================================================

    diagram.arrow_width = 2.0
    diagram.font = "Helvetica"
    title = (f'Nitrogen Reaction Path at Max. ROP of {target_species}\n'
             f'T={T_at_max_rop:.1f} K, z={z_at_max_rop_mm:.2f} mm, f_crack={f_crack * 100:.0f}%')
    diagram.title = title

    dot_source = diagram.get_dot()

    # This is the corrected regex pattern to find the simple labels.
    pattern = re.compile(r'(label="\s*)([0-9.eE+-]+)"')

    all_fluxes_str = [match[1] for match in pattern.findall(dot_source)]
    
    modified_dot_source = dot_source
    if all_fluxes_str:
        all_fluxes_float = [float(f) for f in all_fluxes_str]
        max_flux = max(all_fluxes_float)

        def replacer(match):
            prefix = match.group(1)
            flux_value = float(match.group(2))
            percentage = (flux_value / max_flux) * 100
            return f'{prefix}{percentage:.2f}%"'

        modified_dot_source = pattern.sub(replacer, dot_source)
    
    final_png_path = FIG_DIR / f"reaction_path_f{f_crack:.2f}_percentage_horizontal.png"
    dot_file_path = final_png_path.with_suffix('.dot')

    try:
        with open(dot_file_path, 'w') as f:
            f.write(modified_dot_source)

        subprocess.run(
            ['dot', '-Tpng', '-o', str(final_png_path), str(dot_file_path)],
            check=True
        )
        print(f"\n🗺️  Horizontal, percentage-based reaction path diagram saved successfully!")
        print(f"   -> {final_png_path}")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print("\n--- [ERROR] ---")
        print("Failed to generate diagram. Ensure Graphviz is installed and in your PATH.")
        print(f"Error details: {e}")
    finally:
        if os.path.exists(dot_file_path):
            os.remove(dot_file_path)

#%%
def main():
    """The main entry point for the script."""
    start_time = time.perf_counter()
    print("--- Counterflow Flame Simulation for Reaction Path Analysis ---")
    print(f"Mechanism: {MECH}")
    print(f"Conditions: Crack Fraction = {SINGLE_CRACK_FRACTION * 100:.0f}%, "
          f"u_fuel = {SINGLE_UF} m/s, u_air = {SINGLE_UA} m/s")
    print("-" * 60)
    generate_reaction_path(SINGLE_CRACK_FRACTION, SINGLE_UF, SINGLE_UA, SINGLE_WIDTH)
    print("-" * 60)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"🕒 Total time: {elapsed:.2f} seconds.")
    print(f"🕒 Waktu Selesai: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

#%%
if __name__ == "__main__":
    main()
# %%