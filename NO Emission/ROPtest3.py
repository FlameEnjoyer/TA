#%%
"""multi_case_counterflow_rop.py

Parametric counter‑flow diffusion‑flame ROP sweep:
- CRACK_FRACTIONS × UFS × UAS
- For each case: solve flame, extract NOx ROP profiles,
  identify top-10 reactions, generate plots, and record results.
- Saves a comprehensive CSV with peak values and reactions (indices, contributions, equations).
"""
#%% ---------------------------------------------------------------------------
# Imports & globals
# ---------------------------------------------------------------------------
import itertools as it
import time
from pathlib import Path

import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Headless backend for off-screen plotting
plt.switch_backend("Agg")

#%% ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
MECH            = "reactionsokafar.yaml"
P_ATM           = ct.one_atm
T_FUEL          = 300.0  # K
T_AIR           = 300.0  # K
TEMP_THRESHOLD  = 1400.0  # K

CRACK_FRACTIONS = [0.28, 0.50, 0.75]
UFS             = [2.0, 5.0, 8.0]
UAS             = [2.0, 5.0, 8.0]

WIDTH           = 0.02  # m

FIG_DIR = Path("figures_multi_case_rop")
FIG_DIR.mkdir(exist_ok=True)

#%% ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------
def cracked_fuel_X(f: float) -> str:
    return f"NH3:{1-f}, H2:{1.5*f}, N2:{0.5*f}"

AIR_X = "O2:0.21, N2:0.79"

def fix_velocity(u: float) -> float:
    return max(u, 1e-9)

#%% ---------------------------------------------------------------------------
# Single-case analysis
# ---------------------------------------------------------------------------
def run_single_case(f_crack, uf, ua):
    # Setup streams
    gas = ct.Solution(MECH)
    gas.TPX = T_FUEL, P_ATM, cracked_fuel_X(f_crack)
    rho_f = gas.density; mdot_f = rho_f * fix_velocity(uf)
    gas.TPX = T_AIR, P_ATM, AIR_X
    rho_a = gas.density; mdot_a = rho_a * fix_velocity(ua)

    flame = ct.CounterflowDiffusionFlame(gas, width=WIDTH)
    flame.P = P_ATM
    flame.fuel_inlet.mdot = mdot_f
    flame.fuel_inlet.T    = T_FUEL
    flame.fuel_inlet.X    = cracked_fuel_X(f_crack)
    flame.oxidizer_inlet.mdot = mdot_a
    flame.oxidizer_inlet.T    = T_AIR
    flame.oxidizer_inlet.X    = AIR_X
    flame.set_initial_guess()

    print(f"Solving f={f_crack}, uf={uf}, ua={ua} ...", flush=True)
    try:
        flame.solve(loglevel=0, auto=True, refine_grid=True)
    except Exception as e:
        print("  ❌ solve failed:", e)
        return None

    Tmax = float(flame.T.max())
    if Tmax < TEMP_THRESHOLD:
        print(f"  ⚠ Tmax {Tmax:.1f} K below threshold.")
    else:
        print(f"  ✔ Tmax = {Tmax:.1f} K")

    # ROP extraction
    species_list = ["NO", "NO2", "N2O"]
    working = ct.Solution(MECH)
    n_rxn = working.n_reactions
    grid = flame.grid; n = len(grid)

    # profiles and peak info
    profiles = {sp: np.zeros(n) for sp in species_list}
    peak = {sp: {"val": -np.inf, "idx": None, "reactions": []} for sp in species_list}

    # species indices
    idx_map = {}
    valid = []
    for sp in species_list:
        try:
            idx_map[sp] = working.species_index(sp)
            valid.append(sp)
        except ValueError:
            print(f"  • {sp} missing, skip.")

    for j in range(n):
        working.TPY = flame.T[j], flame.P, flame.Y[:, j]
        net_rop = working.net_rates_of_progress
        for sp in valid:
            i_sp = idx_map[sp]
            net = 0.0; contribs = []
            for r in range(n_rxn):
                nu = (working.product_stoich_coeffs[i_sp, r] -
                      working.reactant_stoich_coeffs[i_sp, r])
                if nu == 0: continue
                c = nu * net_rop[r]
                net += c
                if abs(c) > 1e-15:
                    contribs.append({"r": r, "c": c,
                                     "eq": working.reaction(r).equation})
            profiles[sp][j] = net
            if net > peak[sp]["val"]:
                peak[sp]["val"] = net
                peak[sp]["idx"] = j
                # sort by absolute contribution and take top 10
                contribs.sort(key=lambda x: abs(x["c"]), reverse=True)
                peak[sp]["reactions"] = contribs[:10]

    # Plot net ROP and temperature
    fig, ax = plt.subplots(figsize=(12,7))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, sp in enumerate(valid):
        ax.plot(grid*100, profiles[sp]*1e-3,
                label=f"{sp}", color=colors[i%len(colors)])
    ax.set_xlabel("z (cm)"); ax.set_ylabel("Net ROP (mol cm^-3 s^-1)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'))
    ax.grid(ls=':')
    ax2 = ax.twinx(); ax2.plot(grid*100, flame.T, 'r--', alpha=0.7); ax2.set_ylabel('T (K)')
    ax.legend(loc='best')
    fig.suptitle(f"Net NOx ROP f={f_crack}, uf={uf}, ua={ua}, Tmax={Tmax:.0f} K")
    fig.savefig(FIG_DIR/f"net_ROP_f{f_crack}_uf{uf}_ua{ua}.png", dpi=300)
    plt.close(fig)

    # Plot mole fractions
    fig, ax = plt.subplots(figsize=(12,7))
    for i, sp in enumerate(valid):
        idx = flame.gas.species_index(sp)
        ax.plot(grid*100, flame.X[idx,:],
                label=sp, color=colors[i%len(colors)])
    ax.set_xlabel("z (cm)"); ax.set_ylabel("X")
    ax.set_yscale('log'); ax.grid(ls=':')
    ax2 = ax.twinx(); ax2.plot(grid*100, flame.T, 'r--', alpha=0.7); ax2.set_ylabel('T (K)')
    ax.legend(loc='best')
    fig.suptitle(f"NOx Mole Fractions f={f_crack}, uf={uf}, ua={ua}")
    fig.savefig(FIG_DIR/f"mf_f{f_crack}_uf{uf}_ua{ua}.png", dpi=300)
    plt.close(fig)

    # Horizontal bar for NO
    if 'NO' in valid:
        rec = peak['NO']['reactions']
        labels = [f"R{r['r']+1}: {r['eq']}" for r in rec]
        vals   = [r['c']*1e-3 for r in rec]
        fig, ax = plt.subplots(figsize=(10, max(6,len(labels)*0.5)))
        ax.barh(labels, vals)
        ax.set_xlabel('Rate (mol cm^-3 s^-1)'); ax.grid(axis='x', ls=':'); ax.invert_yaxis()
        ax.set_title(f"Top-10 NO ROP contributions f={f_crack}, uf={uf}, ua={ua}")
        path_bar = FIG_DIR/f"bar_ROP_NO_f{f_crack}_uf{uf}_ua{ua}.png"
        fig.savefig(path_bar, dpi=300); plt.close(fig)
    else:
        path_bar = None

    # Prepare summary row
    row = {
        'f_crack': f_crack, 'uf': uf, 'ua': ua, 'Tmax_K': Tmax,
        'net_ROP_NO_kmol_m3_s': peak.get('NO',{}).get('val',np.nan),
        'net_ROP_NO2_kmol_m3_s': peak.get('NO2',{}).get('val',np.nan),
        'net_ROP_N2O_kmol_m3_s': peak.get('N2O',{}).get('val',np.nan),
        'bar_plot_path': str(path_bar) if path_bar else ''
    }
    # add reaction details
    for sp in valid:
        recs = peak[sp]['reactions']
        for i in range(1, 11):
            if i <= len(recs):
                rinfo = recs[i-1]
                row[f"{sp}_top{i}_rxn_idx"] = rinfo['r']
                row[f"{sp}_top{i}_contrib_kmol_m3_s"] = rinfo['c']
                row[f"{sp}_top{i}_rxn_eq"] = rinfo['eq']
            else:
                row[f"{sp}_top{i}_rxn_idx"] = ''
                row[f"{sp}_top{i}_contrib_kmol_m3_s"] = ''
                row[f"{sp}_top{i}_rxn_eq"] = ''

    return row

#%% ---------------------------------------------------------------------------
# Main sweep & CSV
# ---------------------------------------------------------------------------
def main():
    start = time.perf_counter()
    summary = []
    for f_crack, uf, ua in it.product(CRACK_FRACTIONS, UFS, UAS):
        res = run_single_case(f_crack, uf, ua)
        if res is not None:
            summary.append(res)
    df = pd.DataFrame(summary)
    out_csv = Path("summary_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved summary CSV ➜ {out_csv}")
    print(f"Elapsed time: {time.perf_counter()-start:.1f} s")

if __name__ == '__main__':
    main()

# %%
