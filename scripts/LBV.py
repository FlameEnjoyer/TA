#%% Imports and global settings
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

#%% Function to compute laminar burning velocity for a given cracking fraction α
def compute_laminar_burning_velocity(alpha,
                                     mechanism='mechanisms/reactionsokafar.yaml',
                                     T0=300.0,
                                     P=ct.one_atm):
    """
    Compute the laminar burning velocity (in m/s) of a stoichiometric
    mixture of partially cracked ammonia for a given cracking fraction α.

    Parameters
    ----------
    alpha : float
        Cracking fraction of ammonia (0 ≤ α ≤ 1).
    mechanism : str
        Path to the Cantera mechanism file.
    T0 : float
        Unburned mixture temperature [K].
    P : float
        Unburned mixture pressure [Pa].

    Returns
    -------
    SL : float
        Laminar burning velocity [m/s].
    """
    # 1. Create gas object
    gas = ct.Solution(mechanism)

    # 2. Fuel composition: NH3 partially cracked by α via 2 NH3 → N2 + 3 H2
    #    Uncracked: (1 − α) NH3 ; Cracked: (α/2) N2 + (3α/2) H2
    Xfuel = {
        'NH3': 1.0 - alpha,
        'N2':   alpha / 2.0,
        'H2':  3.0 * alpha / 2.0,
    }
    # 3. Normalize fuel stream
    total_fuel = sum(Xfuel.values())
    for sp in Xfuel:
        Xfuel[sp] /= total_fuel

    # 4. Stoichiometric O2 requirement: NH3 + 3/4 O2 → …
    O2_stoich = 0.75 * (1.0 - alpha)

    # 5. For φ = 1, actual oxidizer: O2 = O2_stoich, N2 = 3.76 × O2
    O2 = O2_stoich
    N2_air = 3.76 * O2

    # 6. Build full mixture
    comp = Xfuel.copy()
    comp['O2'] = O2
    comp['N2'] = comp.get('N2', 0.0) + N2_air

    # 7. Set state
    gas.TPX = T0, P, comp

    # 8. Create and solve freely propagating premixed flame
    width = 0.03  # 3 cm domain
    flame = ct.FreeFlame(gas, width=width)
    flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    flame.solve(loglevel=0, auto=True)

    # 9. Return burning velocity (m/s)
    return flame.velocity[0]

#%% Compute over specified cracking fractions
if __name__ == '__main__':
    alphas = np.array([0.28, 0.50, 0.75])
    SL_m_per_s = [compute_laminar_burning_velocity(a) for a in alphas]
    SL_cm_per_s = np.array(SL_m_per_s) * 100.0

    #%% Plotting
    plt.figure(figsize=(6,4))
    plt.plot(alphas, SL_cm_per_s, 's-', lw=2, ms=8)
    plt.xlabel('Cracking fraction, $\\alpha$', fontsize=12)
    plt.ylabel('Laminar burning velocity, $S_L$ (cm/s)', fontsize=12)
    plt.title('Laminar burning velocity vs.\nPartially cracked NH$_3$', fontsize=14)
    plt.grid(True, ls='--', alpha=0.5)
    plt.xlim(0.2, 0.8)
    plt.ylim(0, np.max(SL_cm_per_s)*1.1)
    plt.tight_layout()
    plt.show()

# %%
