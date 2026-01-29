#%%
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#%%
# Parameters
width = 0.01  # Domain width in meters (10 mm)
kinetics_file = 'mechanisms/reactionsokafar.yaml'

# Single cracking fraction
cracking_fractions = [0.75]      # Fraction of NH3 cracked

# Three velocity cases
fuel_velocities   = [1.0, 3.0, 5.0]    # m/s for case 1, 2, 3
air_velocities    = [1.0, 3.0, 5.0]    # m/s for case 1, 2, 3
line_styles       = ['-',  '--',  ':']    # one linestyle per case

# Species to plot
species_to_plot = ['NO']

# --- MODIFICATION START ---
# Define a color for each species to keep it consistent across different cases
species_colors = {
    'NH3': 'blue',
    'H2': 'green',
    'H2O': 'cyan',
    'O2': 'orange',
    'N2': 'purple',
    'NO': 'salmon',
}
# --- MODIFICATION END ---

# Define per-species scale factors (all unity here)
scale_factors = {sp: 1.0 for sp in species_to_plot}

# Fixed y-axis limits
species_max = 0.002    # max mole fraction
temp_max    = 2100   # max temperature in K

for f in cracking_fractions:
    plt.figure(figsize=(8, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.set_xlabel('Distance from fuel nozzle (mm)')

    # loop over the three velocity cases
    for i, (uf, ua) in enumerate(zip(fuel_velocities, air_velocities)):
        ls = line_styles[i]

        # set up fuel gas
        gas_fuel = ct.Solution(kinetics_file)
        gas_fuel.TP = 300.0, ct.one_atm
        gas_fuel.X = {'NH3': 1-f, 'N2': f/2, 'H2': 3*f/2}
        mdot_fuel = uf * gas_fuel.density

        # set up oxidizer gas
        gas_ox = ct.Solution(kinetics_file)
        gas_ox.TP = 300.0, ct.one_atm
        gas_ox.X = {'O2': 0.21, 'N2': 0.79}
        mdot_ox = ua * gas_ox.density

        # solve the flame
        flame = ct.CounterflowDiffusionFlame(gas=gas_fuel, width=width)
        flame.fuel_inlet.T  = gas_fuel.T
        flame.fuel_inlet.X  = gas_fuel.X
        flame.fuel_inlet.mdot = mdot_fuel
        flame.oxidizer_inlet.T  = gas_ox.T
        flame.oxidizer_inlet.X  = gas_ox.X
        flame.oxidizer_inlet.mdot= mdot_ox
        flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
        flame.solve(loglevel=1, auto=True)

        # plot species
        for sp in species_to_plot:
            idx    = flame.gas.species_index(sp)
            factor = scale_factors[sp]
            label  = sp  # only species name
            # --- MODIFICATION START ---
            # Added a color argument to use the predefined color for each species
            ax1.plot(
                flame.grid*1000,
                flame.X[idx, :]*factor,
                linestyle=ls,
                color=species_colors.get(sp, 'black'), # Use mapped color
                label=label
            )
            # --- MODIFICATION END ---

        # plot temperature
        ax2.plot(
            flame.grid*1000,
            flame.T,
            linestyle=ls,
            color='r',
            label='Temperature'
        )

    # labels & title
    plt.xlabel('Distance from fuel nozzle (mm)')
    ax1.set_ylabel('Species mole fraction')
    ax2.set_ylabel('Temperature (K)')
    plt.title(f'Cracking={f}, Three Velocity Cases')

    # Legend 1: one entry per species + one Temperature
    species_handles = []
    for sp in species_to_plot:
        # pick first matching line for each sp
        for line in ax1.get_lines():
            if line.get_label() == sp:
                species_handles.append(line)
                break
    temp_handle = next(line for line in ax2.get_lines() if line.get_label()=='Temperature')

    handles1 = species_handles + [temp_handle]
    labels1  = species_to_plot   + ['Temperature']
    legend1 = ax1.legend(
        handles1, labels1,
        title='Species & T',
        loc='upper left',
        frameon=False,
        fontsize='small'
    )
    ax1.add_artist(legend1)

    # Legend 2: velocity linestyles
    vel_handles = [Line2D([0],[0], color='k', linestyle=ls) for ls in line_styles]
    vel_labels  = [f"{uf}/{ua} m/s" for uf,ua in zip(fuel_velocities, air_velocities)]
    ax1.legend(
        vel_handles, vel_labels,
        title='Velocity (u_f/u_a)',
        loc='upper right',
        frameon=False,
        fontsize='small'
    )

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, species_max)
    ax2.set_ylim(300, temp_max)
    plt.tight_layout()
    plt.show()
# %%
