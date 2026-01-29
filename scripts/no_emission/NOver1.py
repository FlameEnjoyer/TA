#%% ---------- Imports ----------
import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

#%% ---------- Version-agnostic helpers ----------
def reaction_equation(gas, i):
    """
    Return the equation string for reaction *i* on any Cantera version.
    """
    if hasattr(gas, "reaction_equation"):          # Cantera ≥ 2.5
        return gas.reaction_equation(i)
    elif hasattr(gas, "reaction_equations"):       # 3.x only has list form
        return gas.reaction_equations([int(i)])[0]
    else:                                          # very old 2.4 fallback
        return gas.reaction(i).equation


def net_stoich_coeffs(gas, k):
    """
    Net stoichiometric coefficients ν_k,i = ν_prod − ν_reac (1×n_reactions).
    Works for 2.x (callables) and 3.x (properties).
    """
    try:   # 3.x property form
        ν = gas.product_stoich_coeffs[k, :] - gas.reactant_stoich_coeffs[k, :]
    except TypeError:  # 2.x callable form
        ν = gas.product_stoich_coeffs()[k, :] - gas.reactant_stoich_coeffs()[k, :]
    return ν

#%% ---------- Flame builder ----------
def run_flame(crack_frac, fuel_vel, air_vel,
              mech="mechanisms/reactionsokafar.yaml", width=0.002):
    gas   = ct.Solution(mech)
    flame = ct.CounterflowDiffusionFlame(gas, width=width)
    flame.P, flame.energy_enabled = ct.one_atm, True

    # cracked-NH3 mixture
    f, y_NH3, y_H2, y_N2 = crack_frac, 1-crack_frac, 1.5*crack_frac, 0.5*crack_frac
    comp_f = {"NH3": y_NH3/(y_NH3+y_H2+y_N2),
              "H2":  y_H2 /(y_NH3+y_H2+y_N2),
              "N2":  y_N2 /(y_NH3+y_H2+y_N2)}

    fuel_g = ct.Solution(mech); fuel_g.TPX = 300, ct.one_atm, comp_f
    flame.fuel_inlet.T, flame.fuel_inlet.X, flame.fuel_inlet.mdot = 300, comp_f, fuel_g.density*fuel_vel

    comp_o = {"O2": 0.21, "N2": 0.79}
    ox_g   = ct.Solution(mech); ox_g.TPX = 300, ct.one_atm, comp_o
    flame.oxidizer_inlet.T, flame.oxidizer_inlet.X, flame.oxidizer_inlet.mdot = 300, comp_o, ox_g.density*air_vel

    flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    flame.solve(loglevel=1, auto=True)
    return flame

#%% ---------- Data extraction ----------
def net_rop(flame, S=("NO","NO2","N2O")):
    z, ω, names = flame.grid, flame.net_production_rates, flame.gas.species_names
    return pd.DataFrame({"z (m)": z, **{f"ω̇_{s}": ω[names.index(s), :] for s in S}})

def mole_ppm(flame, S=("NO","NO2","N2O")):
    z, X = flame.grid, flame.X
    return pd.DataFrame({"z (m)": z, **{f"{s} (ppm)": X[flame.gas.species_index(s), :]*1e6 for s in S}})

def top_rxn(flame, sp, N=10):
    gas, k = flame.gas, flame.gas.species_index(sp)
    ν      = net_stoich_coeffs(gas, k)
    rop    = flame.forward_rates_of_progress - flame.reverse_rates_of_progress
    jstar  = int(np.argmax(flame.T))
    Δω     = ν * rop[:, jstar]
    pick   = np.argsort(np.abs(Δω))[-N:][::-1]
    return pd.DataFrame({"idx": pick,
                         "reaction": [reaction_equation(gas,i) for i in pick],
                         f"Δω̇_{sp} (kmol/m3/s)": Δω[pick]})

def rxn_profiles(flame, sp, idx_list):
    gas, k = flame.gas, flame.gas.species_index(sp)
    ν      = net_stoich_coeffs(gas, k)
    rop    = flame.forward_rates_of_progress - flame.reverse_rates_of_progress
    df     = pd.DataFrame({"z (m)": flame.grid})
    for i in idx_list:
        df[reaction_equation(gas,i)] = ν[i] * rop[i, :]
    return df

#%% ---------- Parameter sweep ----------
cracks, uf_list, uo_list = [0.28,0.5,0/75], [3.0,5.0,8.0], [3.0,5.0,8.0]

for f, uf, uo in product(cracks, uf_list, uo_list):
    tag  = f"f{f:.2f}_fuel{uf:.1f}_air{uo:.1f}"
    F    = run_flame(f, uf, uo)

    df_net, df_ppm = net_rop(F), mole_ppm(F)
    df_NO, df_NO2, df_N2O = (top_rxn(F, s) for s in ("NO","NO2","N2O"))
    df_NO_prof = rxn_profiles(F, "NO", df_NO["idx"])

    # save CSVs
    df_net.to_csv(f"{tag}_net_ROP.csv", index=False)
    df_ppm.to_csv(f"{tag}_NOx_ppm.csv", index=False)
    df_NO.to_csv(f"{tag}_top10_NO.csv", index=False)
    df_NO2.to_csv(f"{tag}_top10_NO2.csv", index=False)
    df_N2O.to_csv(f"{tag}_top10_N2O.csv", index=False)
    df_NO_prof.to_csv(f"{tag}_NO_key_rxn_profiles.csv", index=False)

    # bar charts
    for frame, sp in ((df_NO,"NO"), (df_NO2,"NO2"), (df_N2O,"N2O")):
        plt.figure(figsize=(8,4))
        plt.barh(frame["reaction"], frame[f"Δω̇_{sp} (kmol/m3/s)"])
        plt.xlabel("Δω̇ (kmol/m³·s)"); plt.title(f"{sp} Top-10 ROP @ peak — {tag}")
        plt.tight_layout(); plt.savefig(f"{tag}_bar_{sp}.png"); plt.close()

    # net ROP profiles
    plt.figure()
    for sp in ("NO","NO2","N2O"):
        plt.plot(df_net["z (m)"], df_net[f"ω̇_{sp}"], label=sp)
    plt.xlabel("Distance (m)"); plt.ylabel("ω̇ (kmol/m³·s)")
    plt.title(f"Net ROP profiles — {tag}"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{tag}_net_ROP_profiles.png"); plt.close()

    # NO key-reaction profiles
    plt.figure()
    for col in df_NO_prof.columns[1:]:
        plt.plot(df_NO_prof["z (m)"], df_NO_prof[col], label=col)
    plt.xlabel("Distance (m)"); plt.ylabel("Δω̇_NO (kmol/m³·s)")
    plt.title(f"Key NO-reaction ROP profiles — {tag}")
    plt.legend(fontsize="small", ncol=2); plt.tight_layout()
    plt.savefig(f"{tag}_NO_key_rxn_profiles.png"); plt.close()

print("✔ All simulations finished – CSV and PNG files created.")

# %%
