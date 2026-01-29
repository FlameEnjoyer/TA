import itertools as it
import multiprocessing as mp
from pathlib import Path
import time

import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# USER SETTINGS -------------------------------------------------------------
# ---------------------------------------------------------------------------

MECH               = "mechanisms/reactionsokafar.yaml"      # chemical mechanism file
P_ATM              = ct.one_atm
T_FUEL             = 300.0              # K
T_AIR              = 300.0              # K
TEMP_THRESHOLD     = 1400.0             # extinction threshold (K)
CPUS               = max(mp.cpu_count() - 1, 1)

crack_fractions    = [0.28, 0.50, 0.75]
uf_range           = np.arange(0.0, 30.0 + 1e-9, 1.0)
ua_range           = np.arange(0.0, 18.0 + 1e-9, 1.0)
widths             = [0.01, 0.02, 0.03]  # meters (1 cm, 2 cm, 3 cm)

OUT_XLSX           = Path("data/extinction_data.xlsx")
FIG_DIR            = Path("results/figures")
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# COMPOSITION FUNCTIONS -----------------------------------------------------
# ---------------------------------------------------------------------------

def cracked_fuel_X(f):
    return f"NH3:{1 - f}, H2:{1.5 * f}, N2:{0.5 * f}"

AIR_X = "O2:0.21, N2:0.79"

def fix_velocity(u):
    return max(u, 1e-9)

# ---------------------------------------------------------------------------
# SIMULATION FUNCTION -------------------------------------------------------
# ---------------------------------------------------------------------------

def run_case(args):
    f, uf, ua, width = args
    gas = ct.Solution(MECH)

    gas.TPX = T_FUEL, P_ATM, cracked_fuel_X(f)
    rho_f = gas.density
    mdot_f = rho_f * uf

    gas.TPX = T_AIR, P_ATM, AIR_X
    rho_a = gas.density
    mdot_a = rho_a * ua

    flame = ct.CounterflowDiffusionFlame(gas, width=width)
    flame.P = P_ATM

    flame.fuel_inlet.mdot = mdot_f
    flame.fuel_inlet.T = T_FUEL
    flame.fuel_inlet.X = cracked_fuel_X(f)

    flame.oxidizer_inlet.mdot = mdot_a
    flame.oxidizer_inlet.T = T_AIR
    flame.oxidizer_inlet.X = AIR_X

    flame.set_initial_guess()

    try:
        flame.solve(loglevel=0, auto=True)
        Tmax = float(np.max(flame.T))
        stable = int(Tmax >= TEMP_THRESHOLD)
    except Exception:
        Tmax = np.nan
        stable = 0

    return f, uf, ua, width, Tmax, stable

# ---------------------------------------------------------------------------
# MAIN FUNCTION -------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    start_time = time.perf_counter()

    all_results = []
    total_cases = len(crack_fractions) * len(uf_range) * len(ua_range) * len(widths)
    print(f"Running {total_cases} cases on {CPUS} CPU core(s)…")

    with mp.Pool(CPUS) as pool:
        for f, width in it.product(crack_fractions, widths):
            tasks = [(f, fix_velocity(uf), fix_velocity(ua), width)
                     for uf, ua in it.product(uf_range, ua_range)]
            for res in pool.imap_unordered(run_case, tasks):
                all_results.append(res)

    df = pd.DataFrame(all_results,
                      columns=["f_crack", "u_fuel", "u_air", "width", "T_max", "stable"])
    df.sort_values(["f_crack", "width", "u_fuel", "u_air"], inplace=True)

    # SAVE EXCEL
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="all_data", index=False)
        for f in crack_fractions:
            for width in widths:
                name = f"f_{f:.2f}_w_{int(width*100)}"
                df[(df.f_crack == f) & (df.width == width)].to_excel(
                    xl, sheet_name=name, index=False
                )

    print(f"\n✅ Results saved ➜ {OUT_XLSX.resolve()}")

    # PLOTS
    for f in crack_fractions:
        for width in widths:
            d = df[(df.f_crack == f) & (df.width == width)].copy()

            # Replace small velocities (1e-9) with 0 ONLY for plotting
            d["u_fuel_plot"] = d["u_fuel"].apply(lambda x: 0 if x <= 1e-9 else x)
            d["u_air_plot"]  = d["u_air"].apply(lambda x: 0 if x <= 1e-9 else x)

            plt.figure(figsize=(7, 6))
            plt.scatter(d.u_air_plot[d.stable == 1],
                        d.u_fuel_plot[d.stable == 1],
                        marker="o", label="Stable flame", s=30)
            plt.scatter(d.u_air_plot[d.stable == 0],
                        d.u_fuel_plot[d.stable == 0],
                        marker="x", label="Extinct / no-solution", s=30)
            plt.grid(True, ls=":")
            plt.xlabel("Air velocity $u_{air}$  [m/s]")
            plt.ylabel("Fuel velocity $u_{fuel}$  [m/s]")
            plt.title(f"Extinction map\nf = {f}, width = {width*100:.0f} mm")
            plt.legend()
            plt.tight_layout()
            fname = FIG_DIR / f"extinction_map_f_{f:.2f}_w_{int(width*100)}.png"
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"📊 Figure saved ➜ {fname}")

    # TIME REPORT
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"\n🕒 Total simulation time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

# ---------------------------------------------------------------------------
# EXECUTE -------------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
