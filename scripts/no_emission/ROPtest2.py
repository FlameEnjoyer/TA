#%%
"""multi_case_counterflow_rop.py

Run counter‑flow diffusion‑flame **ROP analysis** for many cases and
export a single CSV summary.

You can execute the whole file in VS Code or another IDE that recognises
`#%%` cell dividers – each major block below is a runnable cell.
"""
#%% ---------------------------------------------------------------------------
# Imports & global constants
# ---------------------------------------------------------------------------
import itertools as it
from pathlib import Path
import time

import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Ensure interactive back‑ends don’t choke in head‑less mode
plt.switch_backend("Agg")

#%% ---------------------------------------------------------------------------
# User‑editable sweep settings
# ---------------------------------------------------------------------------
MECH = "mechanisms/reactionsokafar.yaml"
P_ATM = ct.one_atm
T_FUEL = 300.0  # K
T_AIR  = 300.0  # K
TEMP_THRESHOLD = 1400.0  # K – warn if flame is colder than this

# Parameter grids – tweak as needed
CRACK_FRACTIONS = [0.28, 0.50, 0.75]           # NH₃ cracking fractions
UFS             = [2.0, 5.0, 10.0]           # fuel velocities, m s⁻¹
UAS             = [2.0, 5.0, 10.0]           # air  velocities, m s⁻¹

SINGLE_WIDTH = 0.02  # m (domain half‑width)

FIG_DIR = Path("results/no_emission/test_NO")
FIG_DIR.mkdir(exist_ok=True)

#%% ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------

def cracked_fuel_X(f: float) -> str:
    """Return Cantera mol‑fraction string for *cracked* NH₃ fuel."""
    return f"NH3:{1 - f}, H2:{1.5 * f}, N2:{0.5 * f}"

AIR_X = "O2:0.21, N2:0.79"

def fix_velocity(u: float) -> float:
    """Prevent zero mass‑flow (CounterFlow wants *some* velocity)."""
    return max(u, 1e-9)

#%% ---------------------------------------------------------------------------
# Single‑case simulation + ROP diagnostics
# ---------------------------------------------------------------------------

def run_single_flame_and_analyze_rop(
    f_crack: float,
    uf: float,
    ua: float,
    width: float,
    summary_rows: list,
):
    """Solve one counter‑flow diffusion flame and extract NOₓ ROP metrics.

    All diagnostic plots follow the naming scheme used in your original
    single‑case script (now prefixed by *multi*).  The function appends a
    one‑row dict of key results to *summary_rows* so the caller can dump a
    CSV afterwards.
    """
    # ------------------------------------------------------------------
    # 1.  Initialise streams & flame object
    # ------------------------------------------------------------------
    gas = ct.Solution(MECH)

    # Fuel stream
    gas.TPX = T_FUEL, P_ATM, cracked_fuel_X(f_crack)
    rho_f = gas.density
    mdot_f = rho_f * fix_velocity(uf)

    # Air stream
    gas.TPX = T_AIR, P_ATM, AIR_X
    rho_a = gas.density
    mdot_a = rho_a * fix_velocity(ua)

    flame = ct.CounterflowDiffusionFlame(gas, width=width)
    flame.P = P_ATM

    flame.fuel_inlet.mdot = mdot_f
    flame.fuel_inlet.T    = T_FUEL
    flame.fuel_inlet.X    = cracked_fuel_X(f_crack)

    flame.oxidizer_inlet.mdot = mdot_a
    flame.oxidizer_inlet.T    = T_AIR
    flame.oxidizer_inlet.X    = AIR_X

    flame.set_initial_guess()

    # ------------------------------------------------------------------
    # 2.  Solve
    # ------------------------------------------------------------------
    print(
        f"\nSolving → f={f_crack:.2f}, u_f={uf:.1f} m/s, u_a={ua:.1f} m/s, width={width*100:.0f} cm …",
        flush=True,
    )
    try:
        flame.solve(loglevel=0, auto=True, refine_grid=True)
    except Exception as err:
        print(f"❌  Flame failed: {err}")
        return  # skip this case

    Tmax = float(np.max(flame.T))
    if Tmax < TEMP_THRESHOLD:
        print(
            f"⚠️  T_max {Tmax:.0f} K below threshold {TEMP_THRESHOLD} K – flame may be weak.",
            flush=True,
        )
    else:
        print(f"✔️  Solved.  T_max = {Tmax:.0f} K", flush=True)

    # ------------------------------------------------------------------
    # 3.  ROP post‑processing (identical to your original code)
    # ------------------------------------------------------------------
    nox_species = ["NO", "NO2", "N2O"]
    grid_points = flame.grid
    num_pts = len(grid_points)

    working_gas = ct.Solution(MECH)
    n_rxns = working_gas.n_reactions

    rop_profiles = {sp: np.zeros(num_pts) for sp in nox_species}
    max_info = {
        sp: {
            "max_val": -np.inf,
            "grid_idx_max": -1,
        }
        for sp in nox_species
    }

    # Species indices (skip if missing from mechanism)
    species_indices = {}
    valid_species = []
    for sp in nox_species:
        try:
            species_indices[sp] = working_gas.species_index(sp)
            valid_species.append(sp)
        except ValueError:
            print(f"    • {sp} not in mechanism – skipped.")
    if not valid_species:
        print("    ✖ No NOx species present – ROP skipped.")
        return

    # —— loop over axial locations ——
    for j in range(num_pts):
        working_gas.TPY = flame.T[j], flame.P, flame.Y[:, j]
        net_rop = working_gas.net_rates_of_progress

        for sp in valid_species:
            sp_idx = species_indices[sp]
            net = 0.0
            for i in range(n_rxns):
                nu = (
                    working_gas.product_stoich_coeffs[sp_idx, i]
                    - working_gas.reactant_stoich_coeffs[sp_idx, i]
                )
                if nu == 0:
                    continue
                net += nu * net_rop[i]
            rop_profiles[sp][j] = net
            if net > max_info[sp]["max_val"]:
                max_info[sp]["max_val"] = net
                max_info[sp]["grid_idx_max"] = j

    # ------------------------------------------------------------------
    # 4.  Generate two quick plots (net ROP & mole‑fraction) – identical
    #     filenames but under results/no_emission/figures/
    # ------------------------------------------------------------------
    plot_colors = [
        "blue", "green", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"
    ]
    # Net ROP + Temperature
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    for i, sp in enumerate(valid_species):
        ax1.plot(grid_points * 100, rop_profiles[sp] * 1e-3, label=f"Net ROP {sp}", color=plot_colors[i % len(plot_colors)])
    ax1.set_xlabel("Position in flame (cm)")
    ax1.set_ylabel("Net Rate (mol cm⁻³ s⁻¹)")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
    ax1.grid(ls=":")

    axT = ax1.twinx()
    axT.plot(grid_points * 100, flame.T, "r--", label="Temperature (K)", alpha=0.7)
    axT.set_ylabel("Temperature (K)")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = axT.get_legend_handles_labels()
    axT.legend(lines + lines2, labels + labels2, loc="best")

    fig1.suptitle(
        f"Net NOx ROP • f={f_crack}, u_f={uf:.1f}, u_a={ua:.1f}, w={width*100:.0f} cm, T_max={Tmax:.0f} K",
        fontsize=10,
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.95])

    fpath1 = FIG_DIR / (
        f"net_rop_f{f_crack:.2f}_uf{uf:.0f}_ua{ua:.0f}_w{int(width*100)}.png"
    )
    fig1.savefig(fpath1, dpi=300)
    plt.close(fig1)

    # Mole‑fraction + Temperature
    fig2, ax_mf = plt.subplots(figsize=(12, 7))
    for i, sp in enumerate(valid_species):
        sp_idx = flame.gas.species_index(sp)
        ax_mf.plot(grid_points * 100, flame.X[sp_idx, :], label=f"X({sp})", color=plot_colors[i % len(plot_colors)])
    ax_mf.set_xlabel("Position in flame (cm)")
    ax_mf.set_ylabel("Mole Fraction (log)")
    ax_mf.set_yscale("log")
    ax_mf.grid(ls=":")

    axT2 = ax_mf.twinx()
    axT2.plot(grid_points * 100, flame.T, "r--", label="Temperature (K)", alpha=0.7)
    axT2.set_ylabel("Temperature (K)")

    lines, labels = ax_mf.get_legend_handles_labels()
    lines2, labels2 = axT2.get_legend_handles_labels()
    axT2.legend(lines + lines2, labels + labels2, loc="best")

    fig2.suptitle(
        f"NOx Mole Fraction • f={f_crack}, u_f={uf:.1f}, u_a={ua:.1f}, T_max={Tmax:.0f} K",
        fontsize=10,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.95])

    fpath2 = FIG_DIR / (
        f"mf_rop_f{f_crack:.2f}_uf{uf:.0f}_ua{ua:.0f}_w{int(width*100)}.png"
    )
    fig2.savefig(fpath2, dpi=300)
    plt.close(fig2)

    # ------------------------------------------------------------------
    # 5.  Append summary line
    # ------------------------------------------------------------------
    row = {
        "crack_fraction": f_crack,
        "u_fuel_m_s": uf,
        "u_air_m_s": ua,
        "width_m": width,
        "T_max_K": Tmax,
    }
    for sp in valid_species:
        row[f"{sp}_max_net_ROP_kmol_m3_s"] = max_info[sp]["max_val"]
    summary_rows.append(row)

#%% ---------------------------------------------------------------------------
# Main parameter sweep & CSV export
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.perf_counter()
    print("=== Multi‑case Counter‑flow Diffusion‑flame ROP Sweep ===\n")

    summary_rows = []
    case_total = len(CRACK_FRACTIONS) * len(UFS) * len(UAS)
    for idx, (f_crack, uf, ua) in enumerate(it.product(CRACK_FRACTIONS, UFS, UAS), 1):
        print(f"Case {idx:02d}/{case_total}")
        run_single_flame_and_analyze_rop(f_crack, uf, ua, SINGLE_WIDTH, summary_rows)

    # — export —
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = Path("results/no_emission/summary_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n📄 Summary CSV ➜ {csv_path.resolve()}")
    else:
        print("No successful cases – nothing to write.")

    dur = time.perf_counter() - start
    print(f"\n🕒 Sweep finished in {dur:.1f} s ({dur/60:.1f} min).")

#%% ---------------------------------------------------------------------------
# Entry‑point guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

#%%
