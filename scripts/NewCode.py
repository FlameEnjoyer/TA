import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from collections import defaultdict

# ----------------------------
# User-defined parameters
# ----------------------------
# Using a well-known mechanism validated for ammonia combustion is crucial.
# Ensure 'mechanisms/reactionsokafar.yaml' is in your working directory or provide a full path.
mechanism = 'mechanisms/reactionsokafar.yaml'
# Burner separation distance. 2 cm is a common value in counterflow experiments.
width = 0.02
# Inlet temperature (K)
T0 = 300.0
# Inlet pressure (Pa)
P0 = ct.one_atm
# Nozzle cross-sectional area (m^2) - used to convert volumetric flow to velocity
nozzle_area = 3.14159e-06
# Number of points in the air-flow sweep for each case
n_air_points = 10
# Number of top reactions to show in the pathway analysis plots
N_top_reactions = 10


# Define the cases for different cracking ratios (alpha) and fuel velocities
fuel_cases = {
    # alpha: {'vfuel_LPM': [list_of_velocities], 'max_air_LPM': [corresponding_max_air_flows]}
    0.28: {'vfuel_LPM': [20.0, 23.0, 25.0], 'max_air_LPM': [190.0, 190.0, 170.0]},
    0.5:  {'vfuel_LPM': [20.0, 30.0, 35.0], 'max_air_LPM': [250.0, 250.0, 250.0]},
    0.75: {'vfuel_LPM': [25.0, 40.0, 50.0], 'max_air_LPM': [250.0, 250.0, 250.0]},
}

# ----------------------------
# Simulation function
# ----------------------------
def simulate_case(args):
    """
    Simulates a single counterflow diffusion flame case.
    Returns a dictionary of results or None if the flame is extinguished.
    """
    alpha, fuel_LPM, air_LPM = args

    try:
        gas = ct.Solution(mechanism)
        fuel_comp = {'NH3': 1 - alpha, 'H2': 1.5 * alpha, 'N2': 0.5 * alpha}
        gas.TPX = T0, P0, fuel_comp
        fuel_rho = gas.density
        fuel_velocity = (fuel_LPM * 1e-3 / 60) / nozzle_area

        air_comp = 'O2:0.21, N2:0.79'
        gas.TPX = T0, P0, air_comp
        air_rho = gas.density
        air_velocity = (air_LPM * 1e-3 / 60) / nozzle_area

        flame = ct.CounterflowDiffusionFlame(gas=gas, width=width)
        flame.fuel_inlet.mdot = fuel_rho * fuel_velocity
        flame.fuel_inlet.X = fuel_comp
        flame.fuel_inlet.T = T0
        flame.oxidizer_inlet.mdot = air_rho * air_velocity
        flame.oxidizer_inlet.X = air_comp
        flame.oxidizer_inlet.T = T0

        flame.set_refine_criteria(ratio=3, slope=0.1, curve=0.1)
        flame.solve(loglevel=0, auto=True)

        # --- Data Extraction ---
        strain_rate = np.max(np.abs(np.gradient(flame.u, flame.grid)))
        
        # --- Reaction Pathway Analysis ---
        reaction_analysis = defaultdict(dict)
        species_to_analyze = ['NO', 'NO2', 'N2O']
        species_indices = {s: gas.species_index(s) for s in species_to_analyze}

        for sp in species_to_analyze:
            sp_idx = species_indices[sp]
            for i, rxn in enumerate(gas.reactions()):
                net_stoich = rxn.products.get(sp, 0.0) - rxn.reactants.get(sp, 0.0)
                if abs(net_stoich) > 1e-8:
                    rop_i = flame.rates_of_progress[:, i]
                    integrated_rop_i = np.trapz(rop_i * net_stoich, flame.grid)
                    reaction_analysis[sp][rxn.equation] = integrated_rop_i
        
        return {
            'alpha': alpha,
            'fuel_LPM': fuel_LPM,
            'air_LPM': air_LPM,
            'strain_rate': strain_rate,
            'reaction_analysis': dict(reaction_analysis) # Convert back to regular dict
        }

    except Exception as e:
        return None

# ----------------------------
# Main execution block
# ----------------------------
def main():
    """
    Sets up and runs the simulations, then processes and plots the results.
    """
    tasks = []
    for alpha, case_details in fuel_cases.items():
        for i, v_fuel in enumerate(case_details['vfuel_LPM']):
            max_air = case_details['max_air_LPM'][i]
            for air in np.linspace(1.0, max_air, n_air_points):
                tasks.append((alpha, v_fuel, air))

    print(f"Starting {len(tasks)} simulations...")
    pool = mp.Pool(processes=max(1, mp.cpu_count() - 1))
    results = pool.map(simulate_case, tasks)
    pool.close()
    pool.join()

    successful_results = [r for r in results if r is not None]
    
    if not successful_results:
        print("All simulations failed. Please check mechanism and parameters.")
        return

    print(f"Completed {len(successful_results)} successful simulations.")
    
    df = pd.DataFrame(successful_results)
    output_filename = 'results/ammonia_combustion_results.csv'
    df.to_csv(output_filename, index=False)
    print(f"Simulation data saved to '{output_filename}'.")

    # --- Plotting Results ---
    plot_stability_limit(df)
    plot_reaction_pathways(df)


def plot_reaction_pathways(df):
    """
    Generates reaction pathway analysis plots for NOx species,
    similar to the user-provided example.
    """
    species_to_analyze = ['NO', 'NO2', 'N2O']
    
    # --- Select representative cases for plotting ---
    # For each alpha, we choose the case with the highest fuel flow,
    # and for that, the highest air flow that resulted in a stable flame.
    representative_cases = []
    for alpha in fuel_cases.keys():
        alpha_df = df[df['alpha'] == alpha]
        if alpha_df.empty:
            continue
        # Find the row with the max fuel LPM
        max_fuel_lpm = alpha_df['fuel_LPM'].max()
        max_fuel_df = alpha_df[alpha_df['fuel_LPM'] == max_fuel_lpm]
        # From those, find the row with the max air LPM
        most_strained_case = max_fuel_df.loc[max_fuel_df['air_LPM'].idxmax()]
        representative_cases.append(most_strained_case)
    
    if not representative_cases:
        print("Could not find any representative cases to plot for pathway analysis.")
        return
        
    for sp in species_to_analyze:
        # --- Identify top reactions across all representative cases ---
        all_rxn_abs_rates = defaultdict(float)
        for case in representative_cases:
            for rxn, rate in case['reaction_analysis'][sp].items():
                all_rxn_abs_rates[rxn] += abs(rate)

        # Sort reactions by their total absolute contribution
        sorted_reactions = sorted(all_rxn_abs_rates.items(), key=lambda item: item[1], reverse=True)
        top_reactions = [rxn for rxn, rate in sorted_reactions[:N_top_reactions]]
        
        # --- Prepare data for plotting ---
        plot_data = defaultdict(list)
        for rxn in top_reactions:
            for case in representative_cases:
                # Get the rate for this reaction in this case
                rate = case['reaction_analysis'][sp].get(rxn, 0.0)
                
                # Normalize by total production for that case
                all_rates_for_case = case['reaction_analysis'][sp].values()
                total_production = sum(r for r in all_rates_for_case if r > 0)
                
                if total_production > 0:
                    relative_rate = (rate / total_production) * 100
                else:
                    relative_rate = 0.0 # Avoid division by zero
                plot_data[rxn].append(relative_rate)
                
        # --- Generate the plot for the current species ---
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(top_reactions))
        bar_height = 0.25
        
        alphas = [case['alpha'] for case in representative_cases]
        colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

        for i, alpha in enumerate(alphas):
            rates_for_this_alpha = [plot_data[rxn][i] for rxn in top_reactions]
            ax.barh(y_pos + i * bar_height - bar_height, rates_for_this_alpha, bar_height, 
                    label=f'α = {alpha}', color=colors[i])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_reactions)
        ax.invert_yaxis()  # To have the most important reaction at the top
        ax.set_xlabel('Relative Contribution to Total Production Rate (%)')
        ax.set_title(f'Reaction Pathway Analysis for {sp} Production')
        ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.8) # Add a line at x=0
        ax.legend(title='Cracking Ratio')
        fig.tight_layout()
        plt.show()


def plot_stability_limit(df):
    """Finds the extinction points and plots the stability curve."""
    if df.empty: return
    extinction_points_idx = df.groupby(['alpha', 'fuel_LPM'])['air_LPM'].idxmax()
    stability_df = df.loc[extinction_points_idx]

    plt.figure(figsize=(10, 7))
    markers = ['o', 's', '^', 'D', 'v']
    vfuel_marker_map = {v_fuel: markers[i % len(markers)] 
                        for i, v_fuel in enumerate(sorted(stability_df['fuel_LPM'].unique()))}

    for v_fuel, group in stability_df.groupby('fuel_LPM'):
        sorted_group = group.sort_values('alpha')
        plt.plot(sorted_group['alpha'], sorted_group['strain_rate'],
                 marker=vfuel_marker_map[v_fuel],
                 linestyle='--', markersize=8,
                 label=f'V_fuel = {v_fuel} LPM')

    plt.title('Flame Stability Limit', fontsize=16)
    plt.xlabel('Ammonia Cracking Ratio (α)', fontsize=12)
    plt.ylabel('Extinction Strain Rate (1/s)', fontsize=12)
    plt.legend(title='Fuel Flow Rate')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    mp.freeze_support()
    main()
