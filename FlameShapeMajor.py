#%%
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
#%%
# Parameters
width = 0.01  # Domain width in meters (10 mm)
kinetics_file = 'reactionsokafar.yaml'

# Parameter ranges (example values)
cracking_fractions = [0.28]     # Fraction of NH3 cracked
fuel_velocities = [1.0]             # m/s
air_velocities = [1.0]              # m/s

# Species to plot
species_to_plot = ['NH3', 'H2', 'H2O', 'O2', 'N2']
#%%
for f in cracking_fractions:
    for uf in fuel_velocities:
        for ua in air_velocities:
            # Create fuel-side gas
            gas_fuel = ct.Solution(kinetics_file)
            gas_fuel.TP = 300.0, ct.one_atm
            # 2 NH3 -> N2 + 3 H2
            X_fuel = {'NH3': 1-f, 'N2': f/2, 'H2': 3*f/2}
            gas_fuel.X = X_fuel
            mdot_fuel = uf * gas_fuel.density

            # Create oxidizer-side gas
            gas_ox = ct.Solution(kinetics_file)
            gas_ox.TP = 300.0, ct.one_atm
            gas_ox.X = {'O2': 0.21, 'N2': 0.79}
            mdot_ox = ua * gas_ox.density

            # Set up counterflow diffusion flame
            flame = ct.CounterflowDiffusionFlame(gas=gas_fuel, width=width)
            flame.fuel_inlet.T = gas_fuel.T
            flame.fuel_inlet.X = gas_fuel.X
            flame.fuel_inlet.mdot = mdot_fuel

            flame.oxidizer_inlet.T = gas_ox.T
            flame.oxidizer_inlet.X = gas_ox.X
            flame.oxidizer_inlet.mdot = mdot_ox

            # Refine grid for accuracy
            flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)

            # Solve
            flame.solve(loglevel=1, auto=True)
#%%
# Define per‐species scale factors (tune these to whatever you need)
scale_factors = {
    'NH3': 1.0,
    'H2': 1.0,
    'N2': 1.0,
    'NO': 1.0,     # plot NO × 10
    'H2O': 1.0,
    'O2': 1.0,
    'OH': 1.0,      # plot OH × 1,000
    'NH2': 1.0      # plot NH2 × 1,000
}

# … inside your existing loops, replace the plotting block with:
plt.figure(figsize=(6, 4.5))
# plot species on left y-axis
for sp in species_to_plot:
    idx = flame.gas.species_index(sp)
    factor = scale_factors.get(sp, 1.0)
    label = sp if factor == 1 else f"{sp} × {factor:g}"
    plt.plot(
        flame.grid * 1000,
        flame.X[idx, :] * factor,
        label=label
    )

# grab current axes and make a second y-axis for temperature
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.set_xlabel('Distance from fuel nozzle (mm)')
# plot temperature on right y-axis
ax2.plot(
    flame.grid * 1000,
    flame.T,
    'r--',
    label='Temperature (K)'
)

# labels and title

ax1.set_ylabel('Species mole fraction')
ax2.set_ylabel('Temperature (K)')
plt.title(f'Cracking={f}, Fuel vel={uf} m/s, Air vel={ua} m/s \n Major Species')

# combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', ncol=2)

# manually set the y-axis limits
species_max = 1.0 #eg. max mole fraction
temp_max    = 2100  # e.g. max temperature in K

plt.xlim(0, 10)
ax1.set_ylim(0, species_max)
ax2.set_ylim(0,temp_max)

plt.tight_layout()
plt.show()
# %%
