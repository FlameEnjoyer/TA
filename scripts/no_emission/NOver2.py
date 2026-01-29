#%%
# Imports and setup
import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#%%
# Simulation parameters
# Burner geometry
domain_width = 2e-3  # 2 mm
# Kinetics mechanism file
kinetics_file = 'mechanisms/reactionsokafar_no_emission.yaml'

# Parameter sweep conditions
testing_conditions = [
    {'cracking_frac': 0.28, 'fuel_vel': 5.0, 'air_vel': 3.0},
    # {'cracking_frac': 0.5, 'fuel_vel': 0.5, 'air_vel': 0.5},
    # {'cracking_frac': 1.0, 'fuel_vel': 0.5, 'air_vel': 0.5},
]

# Output directory
outdir = 'results/no_emission'
os.makedirs(outdir, exist_ok=True)

#%%
# Helper functions
def make_composition(crack_frac):
    """
    Compute fuel composition based on ammonia cracking fraction.
    Reaction: 2 NH3 -> 3 H2 + N2
    """
    uncracked = 1.0 - crack_frac
    h2 = 1.5 * crack_frac
    n2 = 0.5 * crack_frac
    return {'NH3': uncracked, 'H2': h2, 'N2': n2}


def build_stoich_matrix(gas):
    """
    Build stoichiometric matrix nu with shape (n_species, n_reactions).
    """
    n_s, n_r = gas.n_species, gas.n_reactions
    nu = np.zeros((n_s, n_r))
    for j, rxn in enumerate(gas.reactions()):
        for sp, coeff in rxn.reactants.items():
            nu[gas.species_index(sp), j] -= coeff
        for sp, coeff in rxn.products.items():
            nu[gas.species_index(sp), j] += coeff
    return nu

#%%
# Load mechanism and initialize
gas = ct.Solution(kinetics_file)
rxn_equations = gas.reaction_equations()
nu = build_stoich_matrix(gas)

#%%
# Main simulation loop
species_to_track = ['NO', 'NO2', 'N2O']

for cond in testing_conditions:
    crack = cond['cracking_frac']
    fuel_vel = cond['fuel_vel']
    air_vel = cond['air_vel']

    # Generate reactant composition
    comp = make_composition(crack)

    # Setup counterflow diffusion flame
    flame = ct.CounterflowDiffusionFlame(gas=gas, width=domain_width)
    flame.reactants = {'left': comp, 'right': 'O2:1, N2:3.76'}
    rho = gas.density
    flame.fuel_inlet.mdot = fuel_vel * rho
    flame.air_inlet.mdot = air_vel * rho
    flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    flame.solve(loglevel=0)

    # Extract grid
    z = flame.grid

    #%%
    # Save profiles and bar-chart data
    for sp in species_to_track:
        idx = gas.species_index(sp)

        # Net ROP and mole fraction profiles
        net_rop = flame.net_production_rates[idx]
        mole_frac = flame.X[idx]
        df_profile = pd.DataFrame({
            'z_m': z,
            'net_ROP': net_rop,
            'mole_frac': mole_frac
        })
        df_profile.to_csv(os.path.join(outdir, f'profile_{sp}_crack{crack}.csv'), index=False)

        # Determine peak location for detailed ROP breakdown
        peak_idx = np.argmax(mole_frac)
        gas.TPX = flame.T[peak_idx], flame.P[peak_idx], flame.Y[:, peak_idx]
        rop_f = gas.forward_rate_of_progress
        rop_r = gas.reverse_rate_of_progress
        net_rop_rxn = rop_f - rop_r
        contrib = nu[idx, :] * net_rop_rxn

        # Top 10 reactions by magnitude
        top_idx = np.argsort(np.abs(contrib))[-10:]
        top_rxns = [rxn_equations[i] for i in top_idx]
        top_rates = contrib[top_idx]
        df_bar = pd.DataFrame({
            'reaction': top_rxns,
            'net_ROP': top_rates
        }).sort_values('net_ROP', ascending=False)
        df_bar.to_csv(os.path.join(outdir, f'bar_{sp}_crack{crack}.csv'), index=False)

    #%%
    # Generate and save plots
    # 1) Bar charts for each species
    for sp in species_to_track:
        df_bar = pd.read_csv(os.path.join(outdir, f'bar_{sp}_crack{crack}.csv'))
        plt.figure()
        df_bar.plot(kind='barh', x='reaction', y='net_ROP', legend=False)
        plt.title(f'Top 10 ROP reactions for {sp} (crack={crack})')
        plt.xlabel('Net ROP (kmol/m3/s)')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'bar_{sp}_crack{crack}.png'))
        plt.close()

    # 2) Net ROP profile comparison
    plt.figure()
    for sp in species_to_track:
        idx = gas.species_index(sp)
        plt.plot(z, flame.net_production_rates[idx], label=f'net ROP {sp}')
    plt.xlabel('Distance (m)')
    plt.ylabel('Net ROP (kmol/m3/s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'net_rop_profiles_crack{crack}.png'))
    plt.close()

    # 3) Mole fraction profiles in ppm
    plt.figure()
    for sp in species_to_track:
        idx = gas.species_index(sp)
        plt.plot(z, flame.X[idx] * 1e6, label=f'{sp} (ppm)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Mole Fraction (ppm)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'mole_frac_profiles_crack{crack}.png'))
    plt.close()

#%%
print('All simulations complete. Results saved in:', outdir)
