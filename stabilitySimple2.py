# Code for stability

#%% # Stability map for partially cracked ammonia combustion in a counterflow diffusion flame
import cantera as ct
import numpy as np
import matplotlib
from multiprocessing import Pool, cpu_count

#%% # Set up headless backend for off-screen plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#%% # Worker to compute a single (fraction, fuel‐velocity) slice
def _compute_slice(args):
    """
    Compute a 1×n_air_velocities slice of stability & Tmax for
    cracking fraction f at fuel_velocities[j].
    """
    i, j, f, v_fuel, air_velocities, width, T_fuel, T_air, P, kinetics_file, T_threshold = args
    n_va = len(air_velocities)
    stable_row = np.zeros(n_va, dtype=bool)
    Tmax_row   = np.full(n_va, np.nan)

    # mixture compositions
    comp_fuel = {'NH3': 1 - f, 'N2': 0.5 * f, 'H2': 1.5 * f}
    comp_air  = {'O2': 1.0,     'N2': 3.76}

    # pre‐compute densities for mdot
    gas = ct.Solution(kinetics_file)
    gas.TPX = T_fuel, P, comp_fuel
    rho_f = gas.density
    gas.TPX = T_air,  P, comp_air
    rho_o = gas.density

    for k, v_air in enumerate(air_velocities):
        flame = ct.CounterflowDiffusionFlame(gas=gas, width=width)
        flame.fuel_inlet.mdot      = v_fuel * rho_f
        flame.fuel_inlet.X          = comp_fuel
        flame.fuel_inlet.T          = T_fuel
        flame.oxidizer_inlet.mdot   = v_air  * rho_o
        flame.oxidizer_inlet.X       = comp_air
        flame.oxidizer_inlet.T       = T_air
        flame.set_refine_criteria(ratio=2, slope=0.06, curve=0.12)

        try:
            flame.solve(loglevel=0, auto=True)
            T_max = float(np.max(flame.T))
            Tmax_row[k]   = T_max
            stable_row[k] = (T_max >= T_threshold)
        except Exception:
            # on failure, leave as nan / False
            pass

    return i, j, stable_row, Tmax_row

#%% # Parallelized stability‐map runner with finer task granularity
def run_stability_map(crack_fractions, fuel_velocities, air_velocities,
                      width=0.01,  # m
                      T_fuel=300.0, T_air=300.0,
                      P=ct.one_atm,
                      kinetics_file='reactionsokafar.yaml',
                      T_threshold=1400.0,
                      n_procs=None):
    """
    Compute stability and peak-temperature maps for partially cracked ammonia combustion.
    Tasks are now split per (fraction, fuel‐velocity) so that all CPU cores can be utilized.
    """
    n_f  = len(crack_fractions)
    n_vf = len(fuel_velocities)
    n_va = len(air_velocities)

    stable   = np.zeros((n_f, n_vf, n_va), dtype=bool)
    Tmax_map = np.full((n_f, n_vf, n_va), np.nan)

    if n_procs is None:
        n_procs = cpu_count()

    # build one task per (fraction, fuel-speed)
    args = []
    for i, f in enumerate(crack_fractions):
        for j, v_fuel in enumerate(fuel_velocities):
            args.append((i, j, f, v_fuel, air_velocities,
                         width, T_fuel, T_air, P,
                         kinetics_file, T_threshold))

    # dispatch
    with Pool(processes=n_procs) as pool:
        for i, j, stab_row, Tmax_row in pool.map(_compute_slice, args):
            stable[i, j, :]   = stab_row
            Tmax_map[i, j, :] = Tmax_row

    return stable, Tmax_map

#%% # Compute extinction maps for multiple cracking fractions
def main():
    # Simulation parameters
    width           = 0.01  # m (10 mm)
    crack_fractions = [0.75]
    fuel_velocities = np.linspace(1.0, 25.0, 5)   # m/s
    air_velocities  = np.linspace(1.0, 45.0, 5)   # m/s
    n_procs         = 8  # desired number of processes

    # Compute stability map for all fractions
    stable_map, _ = run_stability_map(crack_fractions,
                                      fuel_velocities,
                                      air_velocities,
                                      width=width,
                                      n_procs=n_procs)
    return crack_fractions, fuel_velocities, air_velocities, stable_map, width

#%% # Plot extinction maps for each cracking fraction
if __name__ == '__main__':
    crack_fractions, fuel_velocities, air_velocities, stable_map, width = main()

    for i, f in enumerate(crack_fractions):
        Z = stable_map[i].T  # True=stable, False=unstable
        X, Y = np.meshgrid(fuel_velocities, air_velocities)

        plt.figure(figsize=(8,6))
        plt.scatter(X[Z],  Y[Z],  c='blue', marker='o', label='Stable',   edgecolors='white')
        plt.scatter(X[~Z], Y[~Z], c='red',  marker='x', label='Unstable')

        # Stability limit curve
        y_limit = []
        for j in range(Z.shape[1]):
            stable_air = Y[:, j][Z[:, j]]
            y_limit.append(np.max(stable_air) if stable_air.size > 0 else np.nan)
        plt.plot(fuel_velocities, y_limit, linewidth=2, label='Stability Limit')

        plt.xlabel('Fuel velocity $u_{fuel}$ [m/s]')
        plt.ylabel('Air velocity $u_{air}$ [m/s]')
        plt.title(f'Extinction Map (f={f:.2f}, width={int(width*1000)} mm)')
        plt.legend(loc='upper right')
        plt.grid(False)
        plt.tight_layout()

        filename = f'extinction_map_f{int(f*100)}.png'
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to {filename}")
        plt.close()

# %%
