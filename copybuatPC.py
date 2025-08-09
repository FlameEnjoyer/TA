import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from itertools import product
import sys

# ==============================================================================
# 1. USER-DEFINED PARAMETERS
# ==============================================================================

# --- Simulation Setup ---
mechanism = 'reactionsokafar.yaml'  # Your reaction mechanism file
T0 = 300.0                          # Inlet temperature (K)
P0 = ct.one_atm                     # Inlet pressure (Pa)

# --- Burner Geometry ---
# Define a range of widths to test for stability
widths_to_test = np.linspace(0.015, 0.04, 6) # Test from 1.5 cm to 4.0 cm

# --- Operating Conditions ---
# A dictionary defining the fuel compositions (alpha) and corresponding
# fuel velocities (v_fuel) and max air flow rates to test.
fuel_cases = {
    # alpha: {'vfuel_cm_s': [v1, v2, ...], 'max_air_LPM': [a1, a2, ...]}
    0.28: {'vfuel_cm_s': [20.0, 23.0, 25.0], 'max_air_LPM': [190.0, 190.0, 170.0]},
    0.5:  {'vfuel_cm_s': [20.0, 30.0, 35.0], 'max_air_LPM': [250.0, 250.0, 250.0]},
    0.75: {'vfuel_cm_s': [25.0, 40.0, 50.0], 'max_air_LPM': [250.0, 250.0, 250.0]},
}
n_air_points = 10                   # Number of air flowrate points to simulate for each case

# --- Plotting Parameters ---
N_top_reactions = 8                 # Number of top NO-producing reactions to show in the bar chart

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def get_simulation_tasks():
    """Generates a list of all simulation condition tuples."""
    tasks = []
    for alpha, case_data in fuel_cases.items():
        for v_fuel_cm_s, max_air in zip(case_data['vfuel_cm_s'], case_data['max_air_LPM']):
            air_flows = np.linspace(max_air / n_air_points, max_air, n_air_points)
            for air_LPM in air_flows:
                if air_LPM > 0: # Avoid zero flow rate for oxidizer
                    tasks.append({
                        'alpha': alpha,
                        'v_fuel_cm_s': v_fuel_cm_s,
                        'air_LPM': air_LPM
                    })
    return tasks

def check_stability(args):
    """
    A lightweight simulation function to check if a flame solves for given
    conditions. Returns True for success, False for failure.
    """
    width, params = args
    gas = ct.Solution(mechanism)

    # Fuel composition
    X_NH3 = 1.0 - params['alpha']
    X_H2 = 1.5 * params['alpha']
    X_N2 = 0.5 * params['alpha']
    gas.TPX = T0, P0, {'NH3': X_NH3, 'H2': X_H2, 'N2': X_N2}

    # Flame setup
    flame = ct.CounterflowDiffusionFlame(gas=gas, width=width)
    flame.fuel_inlet.mdot = (gas.density * params['v_fuel_cm_s'] / 100.0)
    flame.oxidizer_inlet.X = 'O2:0.21, N2:0.79'
    v_air = (params['air_LPM'] * 1e-3 / 60) / (np.pi * (width/2)**2) # Approximate area
    gas.TPX = T0, P0, 'O2:0.21, N2:0.79'
    flame.oxidizer_inlet.mdot = gas.density * v_air

    try:
        # Use a less strict solver for quick stability checks
        flame.set_refine_criteria(ratio=2, slope=0.05, curve=0.05)
        flame.solve(loglevel=0, auto=True)
        # Check for extinction (low temperature)
        if flame.T.max() < T0 + 200:
            return False
        return True
    except ct.CanteraError:
        return False

def run_detailed_simulation(args):
    """
    Runs a full simulation for a given condition and extracts detailed
    ROP data for individual reactions.
    """
    width, params = args
    gas = ct.Solution(mechanism)

    # 1) Fuel composition and Stoichiometry
    X_NH3 = 1.0 - params['alpha']
    X_H2 = 1.5 * params['alpha']
    X_N2 = 0.5 * params['alpha']
    fuel_X = {'NH3': X_NH3, 'H2': X_H2, 'N2': X_N2}
    gas.TPX = T0, P0, fuel_X
    fuel_rho = gas.density
    stoich_AFR = gas.stoich_air_fuel_ratio('O2:0.21, N2:0.79')

    # 2) Configure flame
    flame = ct.CounterflowDiffusionFlame(gas=gas, width=width)
    flame.fuel_inlet.X = fuel_X
    flame.oxidizer_inlet.X = 'O2:0.21, N2:0.79'
    
    mdot_fuel = fuel_rho * params['v_fuel_cm_s'] / 100.0
    flame.fuel_inlet.mdot = mdot_fuel

    gas.TPX = T0, P0, 'O2:0.21, N2:0.79'
    oxidizer_rho = gas.density
    v_air = (params['air_LPM'] * 1e-3 / 60) / (np.pi * (width/2)**2)
    mdot_air = oxidizer_rho * v_air
    flame.oxidizer_inlet.mdot = mdot_air

    # 3) Calculate Equivalence Ratio (Φ)
    actual_AFR = mdot_air / mdot_fuel
    phi = stoich_AFR / actual_AFR

    # 4) Solve
    flame.set_refine_criteria(ratio=3, slope=0.1, curve=0.1)
    flame.solve(loglevel=0, auto=True)

    # 5) Extract metrics
    grid = flame.grid
    
    # Integrate total species ROP
    rop_NO = np.trapz(flame.net_production_rates[:, gas.species_index('NO')], grid)
    rop_NO2 = np.trapz(flame.net_production_rates[:, gas.species_index('NO2')], grid)
    rop_N2O = np.trapz(flame.net_production_rates[:, gas.species_index('N2O')], grid)

    # Integrate ROP for each reaction contributing to NO
    rop_rxn_no = flame.get_net_production_rates('NO') # Shape: (n_points, n_reactions)
    integrated_rop_rxn_no = np.array([np.trapz(rop_rxn_no[:, i], grid) for i in range(gas.n_reactions)])
    
    result = {
        'alpha': params['alpha'],
        'v_fuel_cm_s': params['v_fuel_cm_s'],
        'air_LPM': params['air_LPM'],
        'phi': phi,
        'max_temp': float(flame.T.max()),
        'rop_NO': rop_NO,
        'rop_NO2': rop_NO2,
        'rop_N2O': rop_N2O,
    }
    # Add individual reaction ROPs to the result dictionary
    for i, rop in enumerate(integrated_rop_rxn_no):
        result[f'R{i+1}'] = rop

    return result

# ==============================================================================
# 3. PLOTTING FUNCTIONS
# ==============================================================================

def plot_total_rop(df):
    """Plots the total ROP for major species vs. air flowrate."""
    for sp in ['NO', 'NO2', 'N2O']:
        plt.figure(figsize=(8, 6))
        for alpha, group in df.groupby('alpha'):
            plt.plot(group['air_LPM'], group[f'rop_{sp}'], 'o-', label=f'alpha = {alpha}')
        plt.xlabel('Air Flowrate (LPM)')
        plt.ylabel(f'Integrated Rate of Production of {sp} [kmol/m/s]')
        plt.title(f'Total ROP of {sp} vs. Air Flowrate')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"total_rop_{sp}.png")
        print(f"Saved total ROP plot for {sp} to total_rop_{sp}.png")
        plt.show()

def plot_reaction_rop_analysis(df, gas):
    """
    Creates a bar chart of relative NO production rates, similar to the
    example image.
    """
    # Identify all reaction columns
    rxn_cols = [col for col in df.columns if col.startswith('R')]
    
    # Find the top N contributing reactions across all conditions
    # Sum the absolute ROP for each reaction to gauge its overall importance
    total_abs_rop = df[rxn_cols].abs().sum()
    top_rxn_indices = total_abs_rop.nlargest(N_top_reactions).index
    
    # We will group results by equivalence ratio bins
    phi_bins = pd.cut(df['phi'], bins=[0.6, 0.9, 1.1, 1.4], labels=['Φ ≈ 0.8', 'Φ ≈ 1.0', 'Φ ≈ 1.2'])
    df['phi_bin'] = phi_bins
    
    # Calculate relative ROP for the top reactions within each phi_bin group
    grouped = df.groupby('phi_bin')
    plot_data = {}
    for name, group in grouped:
        if group.empty:
            continue
        # Sum of all positive NO ROPs (total production) for this group
        group_rxn_rops = group[top_rxn_indices]
        total_production = group_rxn_rops[group_rxn_rops > 0].sum(axis=1).mean()
        
        # Mean ROP for each top reaction in this group
        mean_rops = group[top_rxn_indices].mean()
        
        # Relative rate, normalized by total production
        relative_rates = (mean_rops / total_production) * 100
        plot_data[name] = relative_rates
        
    if not plot_data:
        print("\nCould not generate reaction analysis plot: No data fell into specified phi bins.")
        return

    plot_df = pd.DataFrame(plot_data)
    
    # Get reaction equations for labels
    reaction_labels = [gas.reaction(int(idx[1:])-1).equation for idx in plot_df.index]
    
    # Plotting
    ax = plot_df.plot(kind='barh', figsize=(10, 8), width=0.8)
    
    plt.title('Relative Rate of NO Production Analysis')
    plt.xlabel('Relative Contribution to NO Production [%]')
    plt.ylabel('Reaction')
    ax.set_yticklabels(reaction_labels)
    plt.legend(title='Equivalence Ratio')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis() # Show most important reaction at the top
    plt.tight_layout()
    plt.savefig("relative_no_production_analysis.png")
    print("Saved relative NO production analysis to relative_no_production_analysis.png")
    plt.show()


# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    """Main script execution."""
    mp.freeze_support() # For Windows compatibility
    
    # --- Part 1: Find a Stable Width ---
    print("--- Part 1: Searching for a stable burner width ---")
    all_tasks = get_simulation_tasks()
    stable_width = None

    for width in widths_to_test:
        print(f"\nChecking width = {width*100:.2f} cm...")
        is_stable_for_all = True
        
        # Prepare arguments for multiprocessing
        stability_tasks = [(width, task_params) for task_params in all_tasks]
        
        with mp.Pool(mp.cpu_count() - 1) as pool:
            results = pool.map(check_stability, stability_tasks)
        
        if all(results):
            stable_width = width
            print(f"SUCCESS: Width {width*100:.2f} cm is stable for all conditions.")
            break
        else:
            failed_count = len(results) - sum(results)
            print(f"FAILURE: Width {width*100:.2f} cm is unstable for {failed_count} condition(s).")
    
    if stable_width is None:
        print("\nCRITICAL: No stable width found in the specified range. Aborting.")
        sys.exit(1)

    # --- Part 2: Run Detailed Simulations at the Stable Width ---
    print(f"\n--- Part 2: Running detailed simulations at stable width {stable_width*100:.2f} cm ---")
    
    detailed_tasks = [(stable_width, task_params) for task_params in all_tasks]
    
    with mp.Pool(mp.cpu_count() - 1) as pool:
        detailed_results = pool.map(run_detailed_simulation, detailed_tasks)

    df = pd.DataFrame(detailed_results)
    output_filename = 'simulation_results_detailed.csv'
    df.to_csv(output_filename, index=False)
    print(f"\nSimulations complete. Detailed data saved to '{output_filename}'.")

    # --- Part 3: Plot Results ---
    print("\n--- Part 3: Generating plots ---")
    gas = ct.Solution(mechanism) # For getting reaction equations
    
    # Plot 1: Total ROP of key species
    plot_total_rop(df)
    
    # Plot 2: Relative Reaction ROP Analysis
    plot_reaction_rop_analysis(df, gas)

if __name__ == '__main__':
    main()