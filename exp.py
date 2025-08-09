#%%
# from pathlib import Path|
import cantera as ct
import matplotlib.pyplot as plt
#%%
p = ct.one_atm  # pressure
tin_f = 300.0  # fuel inlet temperature
tin_o = 300.0  # oxidizer inlet temperature
mdot_o = 0.72  # kg/m^2/s
mdot_f = 0.24  # kg/m^2/s

comp_o = 'O2:0.21, N2:0.78, AR:0.01'  # air composition
comp_f = 'C2H6:1'  # fuel composition

width = 0.02  # Distance between inlets is 2 cm

loglevel = 1  # amount of diagnostic output (0 to 5)
#%%
# Create the gas object used to evaluate all thermodynamic, kinetic, and
# transport properties.
gas = ct.Solution('gri30.yaml')
gas.TP = gas.T, p

# Create an object representing the counterflow flame configuration,
# which consists of a fuel inlet on the left, the flow in the middle,
# and the oxidizer inlet on the right.
f = ct.CounterflowDiffusionFlame(gas, width=width)

# Set the state of the two inlets
f.fuel_inlet.mdot = mdot_f
f.fuel_inlet.X = comp_f
f.fuel_inlet.T = tin_f

f.oxidizer_inlet.mdot = mdot_o
f.oxidizer_inlet.X = comp_o
f.oxidizer_inlet.T = tin_o

# Set the boundary emissivities
f.boundary_emissivities = 0.0, 0.0
# Turn radiation off
f.radiation_enabled = False

f.set_refine_criteria(ratio=4, slope=0.2, curve=0.3, prune=0.04)
# %%
f.solve(loglevel, auto=True)
# %%
no_rad = f.to_array()
f.radiation_enabled = True
f.solve(loglevel=1, refine_grid=False)
# %%
fig, ax = plt.subplots()
ax.plot(no_rad.grid, no_rad.T, label='Temperature without radiation')
plt.plot(f.grid, f.T, label='Temperature with radiation')
ax.set_title('Temperature of the flame')
ax.set(ylim=(0,2500), xlim=(0.000, 0.020))
ax.legend()
fig.savefig('./diffusion_flame.pdf')
plt.show()
# %%
