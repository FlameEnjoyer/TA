#%%
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
#%%
# Constants
fuel_velocity = 25  # m/s
air_flow_rate_lpm = 200  # LPM
area = 0.000311018  # m²

# Convert air flow rate from LPM to kg/s
air_flow_rate_kg_s = air_flow_rate_lpm * 1e-3 / 60 * 1.225  # kg/s (assuming air density at 300 K and 1 atm)

# Cracking fractions and corresponding fuel compositions
cracking_fractions = [0.28, 0.5, 0.75]
fuel_compositions = [
    'NH3:0.72, H2:0.28',
    'NH3:0.5, H2:0.5',
    'NH3:0.25, H2:0.75'
]

# Air composition (assumed to be air)
oxidizer_composition = 'O2:0.21, N2:0.79'

# Temperature threshold for flame stability
temperature_threshold = 1400  # K
#%%
def simulate_flame(cracking_fraction, fuel_composition):
    # Initialize gas object with the provided mechanism
    gas = ct.Solution('mechanisms/reactionsokafar.yaml')
    gas.TPX = 300.0, ct.one_atm, fuel_composition

    # Define counterflow diffusion flame
    width = 0.02  # Flame width in meters
    flame = ct.CounterflowDiffusionFlame(gas, width=width)

    # Set inlet conditions
    flame.fuel_inlet.mdot = fuel_velocity * area  # kg/m²/s
    flame.fuel_inlet.X = fuel_composition
    flame.fuel_inlet.T = 300.0  # K

    flame.oxidizer_inlet.mdot = air_flow_rate_kg_s / area  # kg/m²/s
    flame.oxidizer_inlet.X = oxidizer_composition
    flame.oxidizer_inlet.T = 300.0  # K

    # Solve the flame structure
    flame.set_refine_criteria(ratio=4, slope=0.2, curve=0.3, prune=0.04)
    flame.solve(loglevel=1, auto=True)

    # Determine flame stability based on maximum temperature
    max_temp = np.max(flame.T)
    stability = "Stable" if max_temp > temperature_threshold else "Extinct"
#%%
    # Plot Reaction Progress Variable (RPV)
    plt.figure(figsize=(10, 6))
    plt.plot(flame.grid, flame.T, label='Temperature (K)')
    plt.xlabel('Position (m)')
    plt.ylabel('Temperature (K)')
    plt.title(f'Flame Profile for Cracking Fraction {cracking_fraction} ({stability})')
    plt.legend()
    plt.grid(True)
    plt.show()
#%%
    # Plot Rate of Progress (ROP)
    fwd, rev = gas.kinetics.rop()
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(fwd)), fwd, label='Forward ROP')
    plt.bar(np.arange(len(rev)), rev, label='Reverse ROP', alpha=0.7)
    plt.xlabel('Reaction Index')
    plt.ylabel('Rate of Progress (kmol/m³/s)')
    plt.title(f'Rate of Progress for Cracking Fraction {cracking_fraction}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return stability
#%%
# Create interactive widget
interact(simulate_flame, 

        cracking_fraction=fixed(cracking_fractions), 
        fuel_composition=fixed(fuel_compositions));

# %%
