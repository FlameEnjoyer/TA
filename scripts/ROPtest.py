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

MECH                = "mechanisms/reactionsokafar.yaml"
P_ATM               = ct.one_atm
T_FUEL              = 300.0
T_AIR               = 300.0
TEMP_THRESHOLD      = 1400.0

SINGLE_CRACK_FRACTION = 0.28
SINGLE_UF             = 0.2
SINGLE_UA             = 0.2
SINGLE_WIDTH          = 0.01

FIG_DIR             = Path("results/hasil_NO")
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

    nox_species_to_analyze = ['NO', 'NO2', 'N2O']
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
        return

    print("\nStarting ROP analysis (calculating net ROPs and identifying key reactions at peak locations)...")
    for j in range(num_grid_points):
        working_gas.TPY = flame.T[j], flame.P, flame.Y[:, j]
        local_net_rates_of_progress = working_gas.net_rates_of_progress

        for sp_name in nox_species_to_analyze:
            sp_idx = species_indices[sp_name]
            sp_net_rop_at_this_point = 0.0
            reaction_contributions_this_point = [] 

            for i in range(num_reactions):
                nu_prod = working_gas.product_stoich_coeffs[sp_idx, i]
                nu_react = working_gas.reactant_stoich_coeffs[sp_idx, i]
                nu_sp_reaction_i = nu_prod - nu_react
                
                if nu_sp_reaction_i != 0:
                    contribution = nu_sp_reaction_i * local_net_rates_of_progress[i]
                    sp_net_rop_at_this_point += contribution
                    if abs(contribution) > 1e-12: 
                        reaction_contributions_this_point.append({
                            'reaction_index': i,
                            'contribution': contribution, 
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

    fig_rop_net, ax_rop_net1 = plt.subplots(figsize=(12, 7))
    plot_colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, sp_name in enumerate(nox_species_to_analyze):
        ax_rop_net1.plot(grid_points * 100, rop_profiles[sp_name] * 1e-3, label=f'Net ROP {sp_name}', color=plot_colors[i % len(plot_colors)])
    
    ax_rop_net1.set_xlabel('Position in flame (mm)')
    ax_rop_net1.set_ylabel('Net Rate of Production (mol·cm⁻³·s⁻¹)')
    ax_rop_net1.grid(True, ls=":")
    ax_rop_net1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'))
    
    ax_rop_net2 = ax_rop_net1.twinx()
    ax_rop_net2.plot(grid_points * 100, flame.T, 'r--', label='Temperature (K)', alpha=0.7)
    ax_rop_net2.set_ylabel('Temperature (K)')
    
    fig_rop_net.suptitle(f'Net NOx ROP Profiles\nFuel: Cracked NH3 (f={f_crack}), $u_f$={uf:.1f} m/s, $u_a$={ua:.1f} m/s', fontsize=10)
    lines, labels = ax_rop_net1.get_legend_handles_labels()
    lines2, labels2 = ax_rop_net2.get_legend_handles_labels()
    ax_rop_net2.legend(lines + lines2, labels + labels2, loc='best')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path_net = FIG_DIR / f"net_rop_profiles_f{f_crack:.2f}_uf{uf:.0f}_ua{ua:.0f}_w{int(width*1000)}.png"
    plt.savefig(fig_path_net, dpi=300)
    plt.close(fig_rop_net)
    print(f"\n📊 Net ROP profile plot saved ➜ {fig_path_net}")

    print("\n--- Net Rate of Production (ROP) Summary (kmol/m³·s⁻¹) ---") 
    for sp_name in nox_species_to_analyze:
        info = max_rop_info[sp_name]
        print(f"\nSpecies: {sp_name}")
        if info['grid_idx_max'] != -1:
            z_max_rop_mm = flame.grid[info['grid_idx_max']] * 1000
            print(f"  Max Net Production: {info['max_val']:.3e} kmol/m^3/s") 
            #print(f"    Occurs at z = {z_max_rop_cm:.3f} cm, T = {info['T_at_max']:.1f} K")
            if info['top_producing_reactions']:
                print(f"    Top producing reactions at this point (contrib in kmol/m^3/s):")
                for r_info in info['top_producing_reactions']:
                    print(f"      R{r_info['reaction_index']+1}: {r_info['equation']} (Contrib: {r_info['contribution']:.3e})")
        if info['grid_idx_min'] != -1 and info['min_val'] < -1e-12 :
            z_min_rop_mm = flame.grid[info['grid_idx_min']] * 1000
            print(f"  Max Net Consumption (Min ROP): {info['min_val']:.3e} kmol/m^3/s") 
            #print(f"    Occurs at z = {z_min_rop_cm:.3f} cm, T = {info['T_at_min']:.1f} K")
            if info['top_consuming_reactions']:
                print(f"    Top consuming reactions at this point (contrib in kmol/m^3/s):")
                for r_info in info['top_consuming_reactions']:
                    print(f"      R{r_info['reaction_index']+1}: {r_info['equation']} (Contrib: {r_info['contribution']:.3e})")

    fig_mf, ax_mf1 = plt.subplots(figsize=(12, 7))
    for i, sp_name in enumerate(nox_species_to_analyze):
        sp_flame_idx = flame.gas.species_index(sp_name) 
        ax_mf1.plot(grid_points * 100, flame.X[sp_flame_idx, :], label=f'$X$_({sp_name})', color=plot_colors[i % len(plot_colors)])
    
    ax_mf1.set_xlabel('Position in flame (mm)')
    ax_mf1.set_ylabel('Mole Fraction')
    ax_mf1.set_yscale('log') 
    min_mf_val = 1e-12 
    if nox_species_to_analyze:
        all_min_positive_mf = []
        for sp_name_for_limit in nox_species_to_analyze:
            sp_idx_for_limit = flame.gas.species_index(sp_name_for_limit)
            positive_mfs = flame.X[sp_idx_for_limit, flame.X[sp_idx_for_limit, :] > 0]
            if positive_mfs.size > 0:
                all_min_positive_mf.append(np.min(positive_mfs))
        if all_min_positive_mf:
             min_mf_val = min(1e-9, np.min(all_min_positive_mf) / 10 ) 
    ax_mf1.set_ylim(bottom=min_mf_val) 

    ax_mf1.grid(True, ls=":")
    ax_mf2 = ax_mf1.twinx()
    ax_mf2.plot(grid_points * 100, flame.T, 'r--', label='Temperature (K)', alpha=0.7)
    ax_mf2.set_ylabel('Temperature (K)')
    fig_mf.suptitle(f'NOx Mole Fraction Profiles\nFuel: Cracked NH3 (f={f_crack}), $u_f$={uf:.1f} m/s, $u_a$={ua:.1f} m/s', fontsize=10)
    lines_mf, labels_mf = ax_mf1.get_legend_handles_labels()
    lines2_mf, labels2_mf = ax_mf2.get_legend_handles_labels()
    ax_mf2.legend(lines_mf + lines2_mf, labels_mf + labels2_mf, loc='best')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path_mf = FIG_DIR / f"mf_profiles_f{f_crack:.2f}_uf{uf:.0f}_ua{ua:.0f}_w{int(width*1000)}.png"
    plt.savefig(fig_path_mf, dpi=300)
    plt.close(fig_mf)
    print(f"\n📊 Mole fraction profile plot saved ➜ {fig_path_mf}")

    target_species_for_line_plot = 'NO'
    if target_species_for_line_plot in species_indices:
        print(f"\nGenerating individual reaction ROP line plot for {target_species_for_line_plot}...")
        sp_idx_target = species_indices[target_species_for_line_plot]
        
        producing_rxns_info = max_rop_info[target_species_for_line_plot]['top_producing_reactions']
        consuming_rxns_info = max_rop_info[target_species_for_line_plot]['top_consuming_reactions']
        
        selected_rxns_for_line_plot_info = []
        selected_rxns_for_line_plot_info.extend(producing_rxns_info[:5]) 
        existing_indices = {r['reaction_index'] for r in selected_rxns_for_line_plot_info}
        for r_info in consuming_rxns_info[:5]: 
            if r_info['reaction_index'] not in existing_indices:
                selected_rxns_for_line_plot_info.append(r_info)
                existing_indices.add(r_info['reaction_index'])
        
        if not selected_rxns_for_line_plot_info:
            print(f"No significant reactions identified for {target_species_for_line_plot} to plot.")
        else:
            individual_reaction_rop_profiles = {} 
            for r_info in selected_rxns_for_line_plot_info:
                rxn_idx = r_info['reaction_index']
                profile = np.zeros(num_grid_points)
                for j in range(num_grid_points):
                    working_gas.TPY = flame.T[j], flame.P, flame.Y[:, j]
                    local_net_rates_of_progress_j = working_gas.net_rates_of_progress
                    
                    nu_prod = working_gas.product_stoich_coeffs[sp_idx_target, rxn_idx]
                    nu_react = working_gas.reactant_stoich_coeffs[sp_idx_target, rxn_idx]
                    nu_sp_reaction_i = nu_prod - nu_react
                    
                    contribution = nu_sp_reaction_i * local_net_rates_of_progress_j[rxn_idx]
                    profile[j] = contribution * 1e-3 
                individual_reaction_rop_profiles[rxn_idx] = {'equation': r_info['equation'], 'profile': profile}

            plt.figure(figsize=(12, 7))
            for i, (rxn_idx, data) in enumerate(individual_reaction_rop_profiles.items()):
                eq_label = data['equation']
                if len(eq_label) > 40: 
                    eq_label = eq_label[:37] + "..."
                plt.plot(grid_points * 100, data['profile'], label=f"R{rxn_idx+1}: {eq_label}", color=plot_colors[i % len(plot_colors)])
            
            plt.xlabel('Position in flame (mm)')
            plt.ylabel('Rate of Production (mol·cm⁻³·s⁻¹)')
            
            # Create the phi_global string separately
            phi_string = "N/A" # Default
            try:
                # This is a rough placeholder for global phi.
                # A true global phi needs inlet flow rates and compositions before they mix/react.
                # flame.gas.set_equivalence_ratio() is for setting, not getting a pre-mixed one.
                # We'll leave it as N/A or a very rough estimate if one can be made.
                # For a diffusion flame, a single "phi" is less well-defined than for premixed.
                # One might define it based on overall inlet C/H/O/N balance.
                # For now, let's try a very basic calculation with the 'gas' object (which is the flame object's gas)
                # This might not be representative of a "global phi" for a diffusion flame.
                # If this is problematic, it's best to remove it or calculate it more rigorously based on inlet mdots.
                # phi_global = gas.equivalence_ratio() # 'gas' here is the initial gas object, not flame.gas
                # phi_string = f"{phi_global:.2f}"

                # Attempting to use inlet properties for a more global sense
                # Create temporary gas objects for fuel and air inlets to calculate phi
                temp_fuel_gas = ct.Solution(MECH)
                temp_fuel_gas.TPX = T_FUEL, P_ATM, cracked_fuel_X(f_crack)
                
                temp_air_gas = ct.Solution(MECH)
                temp_air_gas.TPX = T_AIR, P_ATM, AIR_X

                # Calculate phi based on mass flow rates and stoichiometric O/F ratio
                # This requires knowing the complete combustion products for the specific fuel.
                # For NH3 + 0.75 O2 -> 0.5 N2 + 1.5 H2O (for uncracked NH3)
                # For cracked fuel, it's more complex.
                # As a simplification, we'll use Cantera's equivalence_ratio with the inlet streams
                # if they were to be mixed. This still isn't a perfect representation for a diffusion flame.
                # We need to define what elements constitute fuel and what are oxidizer elements
                # fuel_elements = {'N': temp_fuel_gas.n_atoms('NH3', 'N'), 'H': temp_fuel_gas.n_atoms('NH3', 'H')} # Example for NH3
                # This gets complicated quickly for cracked fuel.
                # For simplicity, we might just report inlet conditions or omit phi.
                # Given the complexity and potential for misinterpretation for a diffusion flame,
                # we'll stick to N/A or a very carefully defined one.
                # For now, sticking with N/A as a robust placeholder.
                pass # Keeping phi_string = "N/A" for now.

            except Exception as e_phi:
                # phi_string = "N/A" # Already default
                pass

            plt.title(f'Key Reaction ROP Contributions to {target_species_for_line_plot}\nFuel: Cracked NH3 (f={f_crack}), $u_f$={uf:.1f}, $u_a$={ua:.1f}')
            plt.grid(True, ls=":")
            plt.legend(loc='best', fontsize='small')
            plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e')) 
            
            fig_path_line = FIG_DIR / f"individual_rop_lines_{target_species_for_line_plot}_f{f_crack:.2f}_uf{uf:.0f}_ua{ua:.0f}_w{int(width*1000)}.png"
            plt.savefig(fig_path_line, dpi=300)
            plt.close()
            print(f"📊 Individual reaction ROP line plot for {target_species_for_line_plot} saved ➜ {fig_path_line}")

    target_species_for_bar_chart = 'NO'
    if target_species_for_bar_chart in species_indices:
        print(f"\nGenerating ROP contribution bar chart for {target_species_for_bar_chart} at its max net ROP location...")
        info = max_rop_info[target_species_for_bar_chart]
        
        reactions_for_bar_chart = []
        if info['top_producing_reactions']:
            reactions_for_bar_chart.extend(info['top_producing_reactions'])
        if info['top_consuming_reactions']:
            current_indices = {r['reaction_index'] for r in reactions_for_bar_chart}
            for r_cons in info['top_consuming_reactions']:
                if r_cons['reaction_index'] not in current_indices:
                    reactions_for_bar_chart.append(r_cons)
        
        if not reactions_for_bar_chart:
            print(f"No significant reactions identified for {target_species_for_bar_chart} bar chart.")
        else:
            labels = [f"R{r['reaction_index']+1}: {r['equation']}" for r in reactions_for_bar_chart]
            contributions = [r['contribution'] * 1e-3 for r in reactions_for_bar_chart] 

            plt.figure(figsize=(10, max(6, len(labels) * 0.5))) 
            bars = plt.barh(labels, contributions, color=['limegreen' if c > 0 else 'salmon' for c in contributions]) 
            
            plt.xlabel('Rate of Production (mol·cm⁻³·s⁻¹)')
            plt.ylabel('Reaction')
            z_val_mm = flame.grid[info["grid_idx_max"]]*1000 if info["grid_idx_max"]!=-1 else np.nan
            plt.title(f'Key Reaction Contributions to {target_species_for_bar_chart}\nAt point of max net {target_species_for_bar_chart} ROP')
            plt.grid(axis='x', linestyle=':', alpha=0.7)
            plt.axvline(0, color='black', linewidth=0.8) 
            plt.gca().invert_yaxis() 
            plt.tight_layout()
            plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'))


            fig_path_bar = FIG_DIR / f"rop_bar_chart_{target_species_for_bar_chart}_f{f_crack:.2f}_uf{uf:.1f}_ua{ua:.1f}_w{int(width*1000)}.png"
            plt.savefig(fig_path_bar, dpi=300)
            plt.close()
            print(f"📊 ROP contribution bar chart for {target_species_for_bar_chart} saved ➜ {fig_path_bar}")
#%%
def main():
    start_time = time.perf_counter()

    print("--- Single Case Counterflow Flame Simulation with ROP Analysis ---")
    print(f"Mechanism: {MECH}")
    print(f"Conditions: Crack Fraction = {SINGLE_CRACK_FRACTION}, "
          f"Fuel Velocity (uf) = {SINGLE_UF} m/s, Air Velocity (ua) = {SINGLE_UA} m/s, "
          f"Domain Width = {SINGLE_WIDTH*100} mm")
    print(f"Fuel Inlet Temp: {T_FUEL} K, Air Inlet Temp: {T_AIR} K, Pressure: {P_ATM/101325:.2f} atm")
    print("-" * 70)


    run_single_flame_and_analyze_rop(
        SINGLE_CRACK_FRACTION,
        SINGLE_UF,
        SINGLE_UA,
        SINGLE_WIDTH
    )

    print("-" * 70)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"🕒 Total simulation and analysis time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
#%%
if __name__ == "__main__":
    main()
# %%
