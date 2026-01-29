#%%
import itertools as it
# import multiprocessing as mp # Not needed
from pathlib import Path
import time

import cantera as ct
import numpy as np
# import pandas as pd # Not needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # For formatting ticks
#%%
# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

MECH                  = "mechanisms/reactionsokafar.yaml"
P_ATM                 = ct.one_atm
T_FUEL                = 300.0
T_AIR                 = 300.0
TEMP_THRESHOLD        = 1400.0

# --- Modified for varying velocities ---
CRACK_FRACTION        = 0.75 # Constant Cracking Fraction for all runs
# List of (fuel_velocity, air_velocity) pairs in m/s to simulate
VELOCITY_PAIRS        = [(1.0, 1.0), (3.0, 3.0), (5.0, 5.0)] 
WIDTH                 = 0.01 # Single Domain Width for all runs

# --- Target species for detailed plots ---
TARGET_SPECIES_ROP_LINES = 'NO'
TARGET_SPECIES_ROP_BAR   = 'NO'


FIG_DIR               = Path("results/hasil_NO_velocity_comparison")
FIG_DIR.mkdir(exist_ok=True)
#%%
# ---------------------------------------------------------------------------
# COMPOSITION AND UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def cracked_fuel_X(f):
    """Returns the fuel composition string for a given cracking fraction f."""
    return f"NH3:{1 - f}, H2:{1.5 * f}, N2:{0.5 * f}"

AIR_X = "O2:0.21, N2:0.79"

def fix_velocity(u):
    """Ensures velocity is a small positive number to avoid simulation errors."""
    return max(u, 1e-9)
#%%
# ---------------------------------------------------------------------------
# CORE SIMULATION AND ROP ANALYSIS FUNCTION
# ---------------------------------------------------------------------------

def run_and_analyze_flame(f_crack, uf, ua, width):
    """
    Runs a single counterflow flame simulation and performs ROP analysis.
    Instead of plotting, it returns a dictionary of results for later processing.
    """
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
    except Exception as e:
        print(f"Error solving flame: {e}")
        return None

    nox_species_to_analyze = ['NO']
    grid_points = flame.grid
    num_grid_points = len(grid_points)

    working_gas = ct.Solution(MECH)
    num_reactions = working_gas.n_reactions

    rop_profiles = {sp: np.zeros(num_grid_points) for sp in nox_species_to_analyze}
    max_rop_info = {
        sp: {'max_val': -np.inf, 'min_val': np.inf, 'grid_idx_max': -1, 'grid_idx_min': -1,
             'T_at_max': -1, 'T_at_min':-1, 'top_producing_reactions': [], 'top_consuming_reactions': []}
        for sp in nox_species_to_analyze
    }

    species_indices = {}
    valid_nox_species = []
    for sp in nox_species_to_analyze:
        try:
            species_indices[sp] = working_gas.species_index(sp)
            valid_nox_species.append(sp)
        except ValueError:
            print(f"Warning: Species '{sp}' not found in mechanism '{MECH}'. Skipping for ROP analysis.")
    nox_species_to_analyze = valid_nox_species

    if not nox_species_to_analyze:
        print("No target NOx species found in the mechanism. Aborting ROP analysis.")
        return None

    print("Starting ROP analysis...")
    for j in range(num_grid_points):
        working_gas.TPY = flame.T[j], flame.P, flame.Y[:, j]
        local_net_rates_of_progress = working_gas.net_rates_of_progress

        for sp_name in nox_species_to_analyze:
            sp_idx = species_indices[sp_name]
            sp_net_rop_at_this_point = 0.0
            reaction_contributions_this_point = []

            for i in range(num_reactions):
                nu_sp_reaction_i = (working_gas.product_stoich_coeffs[sp_idx, i] -
                                    working_gas.reactant_stoich_coeffs[sp_idx, i])
                if nu_sp_reaction_i != 0:
                    contribution = nu_sp_reaction_i * local_net_rates_of_progress[i]
                    sp_net_rop_at_this_point += contribution
                    if abs(contribution) > 1e-12:
                        reaction_contributions_this_point.append({
                            'reaction_index': i, 'contribution': contribution,
                            'equation': working_gas.reaction(i).equation
                        })

            rop_profiles[sp_name][j] = sp_net_rop_at_this_point

            if sp_net_rop_at_this_point > max_rop_info[sp_name]['max_val']:
                max_rop_info[sp_name]['max_val'] = sp_net_rop_at_this_point
                max_rop_info[sp_name]['grid_idx_max'] = j
                max_rop_info[sp_name]['T_at_max'] = flame.T[j]
                reaction_contributions_this_point.sort(key=lambda x: x['contribution'], reverse=True)
                max_rop_info[sp_name]['top_producing_reactions'] = [r for r in reaction_contributions_this_point if r['contribution'] > 0][:5]

            if sp_net_rop_at_this_point < max_rop_info[sp_name]['min_val']:
                max_rop_info[sp_name]['min_val'] = sp_net_rop_at_this_point
                max_rop_info[sp_name]['grid_idx_min'] = j
                max_rop_info[sp_name]['T_at_min'] = flame.T[j]
                reaction_contributions_this_point.sort(key=lambda x: x['contribution'])
                max_rop_info[sp_name]['top_consuming_reactions'] = [r for r in reaction_contributions_this_point if r['contribution'] < 0][:5]

    return {
        'f_crack': f_crack, 'uf': uf, 'ua': ua, 'width': width, 'grid': flame.grid,
        'T': flame.T, 'X': flame.X, 'Y': flame.Y, 'gas': flame.gas, 'P': flame.P,
        'rop_profiles': rop_profiles, 'max_rop_info': max_rop_info,
        'nox_species': nox_species_to_analyze, 'Tmax': Tmax,
        'species_indices': species_indices
    }
#%%
# ---------------------------------------------------------------------------
# PLOTTING FUNCTIONS FOR COMPARISON
# ---------------------------------------------------------------------------

def plot_combined_line_plots(all_results):
    """
    Generates combined line plots for Net ROP and Mole Fractions
    with a two-part legend for improved clarity.
    """
    if not all_results:
        return

    linestyles = ['-', '--', ':', '-.']
    plot_colors = ['blue', 'green', 'purple']
    nox_species = all_results[0]['nox_species']
    
    # Create labels for the varied parameter (velocity)
    if all(res['uf'] == res['ua'] for res in all_results):
        velocity_labels = [f'u = {res["uf"]:.2f} m/s' for res in all_results]
    else:
        velocity_labels = [f'$u_f, u_a$ = {res["uf"]:.2f}, {res["ua"]:.2f} m/s' for res in all_results]


    # --- Net ROP Plot ---
    fig_rop, ax_rop1 = plt.subplots(figsize=(10, 7.5))
    ax_rop2 = ax_rop1.twinx()

    for i, result in enumerate(all_results):
        style = linestyles[i % len(linestyles)]
        for j, sp_name in enumerate(nox_species):
            color = plot_colors[j % len(plot_colors)]
            ax_rop1.plot(result['grid'] * 1000, result['rop_profiles'][sp_name], linestyle=style, color=color)
    
    temp_handle, = ax_rop2.plot(all_results[0]['grid'] * 1000, all_results[0]['T'], 'r--', label='Temperature', alpha=0.7)
    
    # Create the two-part legend for the ROP plot
    species_handles = [plt.Line2D([0], [0], color=c, lw=2) for c in plot_colors]
    species_handles.append(temp_handle)
    species_labels = [f'{sp} ROP' for sp in nox_species]
    species_labels.append('Temperature')
    leg1 = ax_rop1.legend(handles=species_handles, labels=species_labels, loc='upper left', title='Species & T')
    ax_rop1.add_artist(leg1)

    linestyle_handles = [plt.Line2D([0], [0], color='black', linestyle=s) for s in linestyles[:len(velocity_labels)]]
    ax_rop1.legend(handles=linestyle_handles, labels=velocity_labels, loc='upper right', title='Velocities')

    ax_rop1.set_xlabel('Position in flame (mm)')
    ax_rop1.set_ylabel('Net Rate of Production (kmol·m⁻³·s⁻¹)')
    ax_rop2.set_ylabel('Temperature (K)')
    ax_rop1.grid(False)
    ax_rop1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'))
    fig_rop.suptitle(f'Comparison of Net NOx ROP Profiles\nCracking Fraction = {all_results[0]["f_crack"]:.2f}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = FIG_DIR / "combined_net_rop_profiles_vs_velocity_NO.png"
    fig_rop.savefig(fig_path, dpi=300)
    plt.close(fig_rop)
    print(f"\n📊 Combined Net ROP plot saved ➜ {fig_path}")

    # --- Mole Fraction Plot ---
    fig_mf, ax_mf1 = plt.subplots(figsize=(10, 7.5))
    ax_mf2 = ax_mf1.twinx()
    for i, result in enumerate(all_results):
        style = linestyles[i % len(linestyles)]
        for j, sp_name in enumerate(nox_species):
            sp_idx = result['gas'].species_index(sp_name)
            color = plot_colors[j % len(plot_colors)]
            ax_mf1.plot(result['grid'] * 1000, result['X'][sp_idx, :], linestyle=style, color=color)
    
    temp_handle_mf, = ax_mf2.plot(all_results[0]['grid'] * 1000, all_results[0]['T'], 'r--', label='Temperature', alpha=0.7)

    # Create the two-part legend for the Mole Fraction plot
    mf_species_handles = [plt.Line2D([0], [0], color=c, lw=2) for c in plot_colors]
    mf_species_handles.append(temp_handle_mf)
    mf_species_labels = [f'$X_{{{sp}}}$' for sp in nox_species]
    mf_species_labels.append('Temperature')
    leg1_mf = ax_mf1.legend(handles=mf_species_handles, labels=mf_species_labels, loc='upper left', title='Species & T')
    ax_mf1.add_artist(leg1_mf)

    mf_linestyle_handles = [plt.Line2D([0], [0], color='black', linestyle=s) for s in linestyles[:len(velocity_labels)]]
    ax_mf1.legend(handles=mf_linestyle_handles, labels=velocity_labels, loc='upper right', title='Velocities')

    ax_mf1.set_xlabel('Position in flame (mm)')
    ax_mf1.set_ylabel('Mole Fraction')
    ax_mf2.set_ylabel('Temperature (K)')
    ax_mf1.set_yscale('log')
    ax_mf1.set_ylim(bottom=1e-12)
    ax_mf1.grid(False)
    fig_mf.suptitle(f'Comparison of NOx Mole Fraction Profiles\nCracking Fraction = {all_results[0]["f_crack"]:.2f}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = FIG_DIR / "combined_mf_profiles_vs_velocity_NO_only.png"
    fig_mf.savefig(fig_path, dpi=300)
    plt.close(fig_mf)
    print(f"📊 Combined Mole Fraction plot saved ➜ {fig_path}")

def plot_grouped_rop_bar_chart(all_results, target_species=TARGET_SPECIES_ROP_BAR):
    """
    Generates a grouped bar chart comparing ROP contributions for a target species.
    The reactions shown are fixed based on the most important ones from the first case.
    """
    if not all_results:
        print("No results to generate a bar chart.")
        return

    # 1. Determine the set of reference reactions from the first successful case.
    ref_result = all_results[0]
    info_ref = ref_result['max_rop_info'].get(target_species)
    if not info_ref:
        print(f"Target species '{target_species}' for bar chart not found in reference result.")
        return

    reference_reactions = []
    ref_prod = info_ref.get('top_producing_reactions', [])
    ref_cons = info_ref.get('top_consuming_reactions', [])
    reference_reactions.extend(ref_prod)
    existing_indices = {r['reaction_index'] for r in ref_prod}
    for r_info in ref_cons:
        if r_info['reaction_index'] not in existing_indices:
            reference_reactions.append(r_info)

    if not reference_reactions:
        print(f"No significant reactions identified for {target_species} in the reference case to create a bar chart.")
        return
    reference_reactions.sort(key=lambda x: x['contribution'], reverse=True)
    reaction_labels = [f"R{r['reaction_index']+1}: {r['equation']}" for r in reference_reactions]
    ref_reaction_indices = [r['reaction_index'] for r in reference_reactions]

    # 2. For each case, calculate the ROP for these *specific* reference reactions.
    all_contributions = []
    working_gas = ct.Solution(MECH)
    sp_idx_target = working_gas.species_index(target_species)

    for result in all_results:
        idx_max_rop = result['max_rop_info'][target_species]['grid_idx_max']
        case_contributions = []
        if idx_max_rop != -1:
            working_gas.TPY = result['T'][idx_max_rop], result['P'], result['Y'][:, idx_max_rop]
            local_net_rates = working_gas.net_rates_of_progress
            for rxn_idx in ref_reaction_indices:
                nu = (working_gas.product_stoich_coeffs[sp_idx_target, rxn_idx] -
                      working_gas.reactant_stoich_coeffs[sp_idx_target, rxn_idx])
                contribution = nu * local_net_rates[rxn_idx]
                case_contributions.append(contribution)
        else:
            case_contributions = [0] * len(ref_reaction_indices)
        all_contributions.append(case_contributions)

    # 3. Plot the grouped bar chart
    plot_data = np.array(all_contributions).T

    fig, ax = plt.subplots(figsize=(10, max(6, len(reaction_labels) * 0.7)))
    n_cases = len(all_results)
    n_reactions = len(reaction_labels)
    bar_width = 0.8 / n_cases
    y = np.arange(n_reactions)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_cases))

    if all(res['uf'] == res['ua'] for res in all_results):
        bar_labels = [f'{res["uf"]:.2f}' for res in all_results]
        legend_title = "Velocity (m/s)"
    else:
        bar_labels = [f'{res["uf"]:.2f}/{res["ua"]:.2f}' for res in all_results]
        legend_title = "u_f/u_a (m/s)"


    for i in range(n_cases):
        offset = (i - (n_cases - 1) / 2) * bar_width
        ax.barh(y + offset, plot_data[:, i], height=bar_width, label=bar_labels[i], color=colors[i])

    # ================================================================= #
    # ===== NEWLY ADDED CODE TO CENTER THE X-AXIS AT ZERO ============= #
    # ================================================================= #
    # Check if there is data to avoid errors
    if plot_data.size > 0:
        # Find the largest absolute value in the entire dataset
        max_abs_val = np.abs(plot_data).max()
        # Set the x-axis limits to be symmetric around zero, with 10% padding
        ax.set_xlim(-max_abs_val * 1.1, max_abs_val * 1.1)
    # ================================================================= #
    # =================== END OF NEWLY ADDED CODE ===================== #
    # ================================================================= #

    ax.set_xlabel('Rate of Production (kmol·m⁻³·s⁻¹)')
    ax.set_ylabel('Reaction')
    ax.set_yticks(y)
    ax.set_yticklabels(reaction_labels)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'))
    
    ref_vel_label = f'$u_f,u_a$={ref_result["uf"]:.2f},{ref_result["ua"]:.2f} m/s'
    ref_z_mm = ref_result['grid'][info_ref['grid_idx_max']] * 1000
    ax.set_title(f'Comparison of ROP Contributions with Varying Velocities \nat cracking fraction {ref_result["f_crack"]:.2f}')
    ax.legend(title=legend_title)
    ax.grid(axis='x', linestyle=':', which='both')
    plt.tight_layout()

    fig_path = FIG_DIR / f"grouped_bar_chart_{target_species}_rop_vs_velocity.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"📊 Grouped ROP bar chart for {target_species} saved ➜ {fig_path}")

#%%
# ---------------------------------------------------------------------------
# MAIN EXECUTION SCRIPT
# ---------------------------------------------------------------------------
def main():
    start_time = time.perf_counter()
    print("--- Counterflow Flame Comparison with ROP Analysis ---")
    print(f"Mechanism: {MECH}")
    print(f"Constant Cracking Fraction: {CRACK_FRACTION}")
    print(f"Velocity Pairs (uf, ua) to run: {VELOCITY_PAIRS} m/s")
    print("-" * 70)

    all_results = []
    for uf, ua in VELOCITY_PAIRS:
        print(f"\nProcessing case: uf = {uf:.2f} m/s, ua = {ua:.2f} m/s")
        result = run_and_analyze_flame(
            f_crack=CRACK_FRACTION,
            uf=uf,
            ua=ua,
            width=WIDTH
        )
        if result:
            all_results.append(result)
        print("-" * 70)

    if all_results:
        print("\nAll simulations complete. Generating combined plots...")
        plot_combined_line_plots(all_results)
        plot_grouped_rop_bar_chart(all_results, target_species=TARGET_SPECIES_ROP_BAR)
    else:
        print("\nNo simulations succeeded. Cannot generate plots.")

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"\n🕒 Total simulation and analysis time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

#%%
if __name__ == "__main__":
    main()
# %%