#%%
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#%%
# Parameters
width = 0.01  # Domain width in meters (10 mm)
kinetics_file = 'reactionsokafar.yaml'

# Parameter ranges (two paired cases)
cracking_fractions = [0.28, 0.50, 0.75]  # Fraction of NH3 cracked for case 1 and case 2
fuel_velocities   = [1.0, 1.0, 1.0]    # m/s for case 1 and case 2
air_velocities    = [1.0, 1.0, 1.0]    # m/s for case 1 and case 2
line_styles       = ['-',  '--', ':']   # one linestyle per case

# Species to plot
species_to_plot = ['H', 'N2O', 'O', 'NO', 'NO2', 'OH', 'NH2']

# Define per-species scale factors (tune these to whatever you need)
scale_factors = {
    'NH3': 1.0,
    'H2':  1.0,
    'N2':  1.0,
    'NO':  1.0,
    'H2O': 1.0,
    'O2':  1.0,
    'OH':  1.0,
    'NH2': 1.0
}

# Fixed y-axis limits
species_max = 0.04   # max mole fraction
temp_max    = 2100   # max temperature in K

# Single figure with twin axes
plt.figure(figsize=(6, 4.5))
ax1 = plt.gca()
ax2 = ax1.twinx()

# Loop over the paired (cracking fraction, fuel vel, air vel) cases
for i, (f, uf, ua) in enumerate(zip(cracking_fractions, fuel_velocities, air_velocities)):
    ls = line_styles[i]

    # Create fuel-side gas
    gas_fuel = ct.Solution(kinetics_file)
    gas_fuel.TP = 300.0, ct.one_atm
    X_fuel = {'NH3': 1-f, 'N2': f/2, 'H2': 3*f/2}
    gas_fuel.X = X_fuel
    mdot_fuel = uf * gas_fuel.density

    # Create oxidizer-side gas
    gas_ox = ct.Solution(kinetics_file)
    gas_ox.TP = 300.0, ct.one_atm
    gas_ox.X = {'O2': 0.21, 'N2': 0.79}
    mdot_ox = ua * gas_ox.density

    # Set up and solve counterflow diffusion flame
    flame = ct.CounterflowDiffusionFlame(gas=gas_fuel, width=width)
    flame.fuel_inlet.T = gas_fuel.T
    flame.fuel_inlet.X = gas_fuel.X
    flame.fuel_inlet.mdot = mdot_fuel
    flame.oxidizer_inlet.T = gas_ox.T
    flame.oxidizer_inlet.X = gas_ox.X
    flame.oxidizer_inlet.mdot = mdot_ox
    flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    flame.solve(loglevel=1, auto=True)

    # Plot species on left y-axis with scaling
    for sp in species_to_plot:
        idx        = flame.gas.species_index(sp)
        factor     = scale_factors.get(sp, 1.0)
        base_label = sp if factor == 1 else f"{sp} × {factor:g}"
        label      = f"{base_label} (f={f})"
        ax1.plot(
            flame.grid * 1000,
            flame.X[idx, :] * factor,
            linestyle=ls,
            label=label
        )

    # Plot temperature on right y-axis, now with a label
    ax2.plot(
        flame.grid * 1000,
        flame.T,
        linestyle=ls,
        color='r',
        label=f"T (f={f})"
    )

# Axis labels and title
plt.xlabel('Distance (mm)')
ax1.set_ylabel('Species mole fraction')
ax2.set_ylabel('Temperature (K)')
plt.title('Comparison of: Varying Cracking Fraction')

# 1) Legend for cracking fractions (linestyles)
frac_handles = [
    Line2D([0], [0], color='k', linestyle=ls, lw=1.5)
    for ls in line_styles
]
frac_labels = [str(f) for f in cracking_fractions]
legend1 = ax1.legend(frac_handles, frac_labels,
                     title='Cracking fraction',
                     loc='upper right')
ax1.add_artist(legend1)

# 2) Legend for species + temperature (colors)
# gather one handle per species from ax1
spec_handles = []
spec_labels = []
for sp in species_to_plot:
    for line in ax1.get_lines():
        if line.get_label().startswith(sp + ' '):
            spec_handles.append(line)
            spec_labels.append(sp)
            break
# now add the temperature handle(s) from ax2
for line in ax2.get_lines():
    if line.get_label().startswith('T '):
        spec_handles.append(line)
        spec_labels.append('Temperature')

legend2 = ax1.legend(spec_handles, spec_labels,
                     title='Species & T',
                     loc='upper left',
                     frameon=False,
                     fontsize='small')

# Apply fixed y-limits
plt.xlim(0, 10)
ax1.set_ylim(0, species_max)
ax2.set_ylim(300, temp_max)

plt.tight_layout()
plt.show()
# %%
