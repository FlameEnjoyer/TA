#%% # Stability map for partially cracked ammonia combustion in a counterflow diffusion flame
import cantera as ct
import numpy as np
import matplotlib
from multiprocessing import Pool, cpu_count

#%% # Set up headless backend for off-screen plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#%% # Simulation parameters and stability & temperature map functions

def _compute_fraction(args):
    """
    Worker to compute stable and Tmax slices for a single cracking fraction.
    """
    i, f, fuel_velocities, air_velocities, width, T_fuel, T_air, P, kinetics_file, T_threshold = args
    n_vf = len(fuel_velocities)
    n_va = len(air_velocities)
    stable_i = np.zeros((n_vf, n_va), dtype=bool)
    Tmax_i = np.full((n_vf, n_va), np.nan)

    comp_fuel = {'NH3': 1 - f, 'N2': 0.5 * f, 'H2': 1.5 * f}
    comp_air = {'O2': 1.0, 'N2': 3.76}

    for j, v_fuel in enumerate(fuel_velocities):
        for k, v_air in enumerate(air_velocities):
            gas = ct.Solution(kinetics_file)
            # compute densities for mdot
            gas.TPX = T_fuel, P, comp_fuel
            rho_f = gas.density
            gas.TPX = T_air, P, comp_air
            rho_o = gas.density

            flame = ct.CounterflowDiffusionFlame(gas=gas, width=width)
            flame.fuel_inlet.mdot = v_fuel * rho_f
            flame.fuel_inlet.X = comp_fuel
            flame.fuel_inlet.T = T_fuel
            flame.oxidizer_inlet.mdot = v_air * rho_o
            flame.oxidizer_inlet.X = comp_air
            flame.oxidizer_inlet.T = T_air
            flame.set_refine_criteria(ratio=2, slope=0.06, curve=0.12)

            try:
                flame.solve(loglevel=0, auto=True)
                T_max = float(np.max(flame.T))
                Tmax_i[j, k] = T_max
                stable_i[j, k] = (T_max >= T_threshold)
            except Exception:
                pass

    return i, stable_i, Tmax_i


def run_stability_map(crack_fractions, fuel_velocities, air_velocities,
                      width=0.01,  # m
                      T_fuel=300.0, T_air=300.0,
                      P=ct.one_atm,
                      kinetics_file='reactionsokafar.yaml',
                      T_threshold=1400.0,
                      n_procs=None):  # K, stability criterion
    """
    Compute stability and peak-temperature maps for partially cracked ammonia combustion.
    Uses multiprocessing across crack fractions for speed.
    n_procs: number of parallel worker processes (defaults to all CPUs).
    Returns:
      stable[i,j,k]: bool for stable flame (Tmax >= T_threshold)
      Tmax_map[i,j,k]: peak flame temperature in K (or nan if failed)
    """
    n_f = len(crack_fractions)
    n_vf = len(fuel_velocities)
    n_va = len(air_velocities)
    stable = np.zeros((n_f, n_vf, n_va), dtype=bool)
    Tmax_map = np.full((n_f, n_vf, n_va), np.nan)

    # Determine number of processes
    if n_procs is None:
        n_procs = cpu_count()

    # build argument list for each fraction
    args = []
    for i, f in enumerate(crack_fractions):
        args.append((i, f, fuel_velocities, air_velocities,
                     width, T_fuel, T_air, P, kinetics_file, T_threshold))

    # parallel execution
    with Pool(processes=n_procs) as pool:
        for i, stable_i, Tmax_i in pool.map(_compute_fraction, args):
            stable[i, :, :] = stable_i
            Tmax_map[i, :, :] = Tmax_i

    return stable, Tmax_map

#%% # Compute extinction maps for multiple cracking fractions

def main():
    # Simulation parameters
    width = 0.01  # m (10 mm)
    crack_fractions = [0.28, 0.5]
    fuel_velocities = np.linspace(1.0, 20.0, 2)   # m/s
    air_velocities = np.linspace(1.0, 35.0, 2) # m/s
    n_procs = 6  # set desired number of processes

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
        plt.scatter(X[Z], Y[Z], c='blue', marker='o', label='Stable', edgecolors='white')
        plt.scatter(X[~Z], Y[~Z], c='red', marker='x', label='Unstable')

        # Stability limit curve
        y_limit = []
        for j in range(Z.shape[1]):
            stable_air = Y[:, j][Z[:, j]]
            y_limit.append(np.max(stable_air) if stable_air.size > 0 else np.nan)
        plt.plot(fuel_velocities, y_limit, color='green', linewidth=2, label='Stability Limit')

        plt.xlabel('Fuel velocity $u_{fuel}$ [m/s]')
        plt.ylabel('Air velocity $u_{air}$ [m/s]')
        plt.title(f'Extinction Map (f={f:.2f}, width={int(width*1000)} mm)')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        filename = f'extinction_map_f{int(f*100)}.png'
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to {filename}")
        plt.close()

# %%
