#%%
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
 
#%%
# ─── 1) PARAMETERS ────────────────────────────────────────────────────────────
fuel_velocity       = 5.0           # m/s
air_flow_rate_LPM   = 10.0          # L/min
area                = 0.000311018    # m²
air_flow_rate_m3_s  = air_flow_rate_LPM / 1000.0 / 60.0   # m³/s
air_velocity        = air_flow_rate_m3_s / area          # m/s
threshold_temp      = 1400.0         # K extinction criterion
cracking_fractions  = [0.28]
 
# Storage
stability_results   = {}
max_temps           = {}
temperature_profiles= {}
rop_results         = {}
flames              = {}
 
#%%
# ─── 2) RUN SIMULATIONS FOR MULTIPLE WIDTHS ───────────────────────────────────
widths = [0.05, 0.10, 0.15, 0.20]   # in meters, add or remove as you wish
 
# re‐initialize storage keyed by width → cracking fraction
stability_results = {w: {} for w in widths}
max_temps          = {w: {} for w in widths}
flames             = {w: {} for w in widths}
rop_results        = {w: {} for w in widths}
 
for w in widths:
    for cf in cracking_fractions:
        # 2.1) Fuel mixture
        comp_fuel = {'NH3': 1.0 - cf, 'H2': cf}
 
        # 2.2) Mass fluxes
        gas = ct.Solution('reactionsokafar.yaml')
        gas.TPX = 300.0, ct.one_atm, comp_fuel
        mdot_f = gas.density_mass * fuel_velocity
        gas.TPX = 300.0, ct.one_atm, {'O2':0.21, 'N2':0.79}
        mdot_o = gas.density_mass * air_velocity
 
        # 2.3) Create flame with width = w
        gas.TPX = 300.0, ct.one_atm, comp_fuel
        flame = ct.CounterflowDiffusionFlame(gas=gas, width=w)
        flame.fuel_inlet.mdot      = mdot_f
        flame.fuel_inlet.X         = comp_fuel
        flame.fuel_inlet.T         = 300.0
        flame.oxidizer_inlet.mdot  = mdot_o
        flame.oxidizer_inlet.X     = {'O2':0.21, 'N2':0.79}
        flame.oxidizer_inlet.T     = 300.0
 
        # 2.4) Mesh & initial guess
        flame.set_refine_criteria(ratio=2, slope=0.05, curve=0.1, prune=0.01)
        flame.set_initial_guess()
 
        # 2.5) Solve
        try:
            flame.solve(loglevel=0, auto=True)
        except ct.CanteraError as e:
            print(f"[w={w:.2f} m, CF={cf:.2f}] solve failed: {e}")
            stability_results[w][cf] = 'Solve failed'
            continue
 
        # 2.6) Stability check
        T_max = flame.T.max()
        max_temps[w][cf] = T_max
        if not np.isfinite(T_max):
            print(f"[w={w:.2f}, CF={cf:.2f}] T_max={T_max}, skipping")
            stability_results[w][cf] = 'Invalid'
            continue
 
        status = 'On' if T_max > threshold_temp else 'Extinct'
        stability_results[w][cf] = status
        flames[w][cf] = flame
 
        # 2.7) ROP at peak for “On” only
        if status == 'On':
            idx = np.argmax(flame.T)
            T_peak = flame.T[idx]
            X_peak = flame.X[:, idx]
            comp_peak = dict(zip(flame.gas.species_names, X_peak))
 
            gas_rop = ct.Solution('reactionsokafar.yaml')
            gas_rop.TPX = T_peak, ct.one_atm, comp_peak
            fwd = gas_rop.forward_rates_of_progress
            rev = gas_rop.reverse_rates_of_progress
            rop_results[w][cf] = {'forward': fwd, 'reverse': rev}
 
#%%
# ─── 3) PRINT SUMMARY FOR ALL WIDTHS & CFs ────────────────────────────────────
for w in widths:
    print(f"\n=== Width = {w:.3f} m ===")
    for cf in cracking_fractions:
        stat = stability_results[w].get(cf, 'No data')
        Tm   = max_temps[w].get(cf, np.nan)
        print(f"  CF={cf:.2f} → {stat:<8} (T_max = {Tm:6.1f} K)")
#%%
# ─── 4) FLAME STABILITY BAR PLOT ─────────────────────────────────────────────
plt.figure()
labels = [f"{c:.2f}" for c in cracking_fractions]
vals   = [1 if stability_results.get(c)=='On' else 0 for c in cracking_fractions]
plt.bar(labels, vals)
plt.ylim(-0.2, 1.2)
plt.xlabel('Cracking Fraction')
plt.ylabel('Flame Status (1=On, 0=Extinct)')
plt.title('Flame Stability vs Cracking Fraction')
plt.show()
 
#%%
# ─── 5) ROP BAR PLOTS AT PEAK ────────────────────────────────────────────────
for c in cracking_fractions:
    if stability_results.get(c) != 'On':
        continue
    data = rop_results[c]
    idx  = np.arange(len(data['forward']))
    plt.figure()
    plt.bar(idx, data['forward'], label='Forward ROP')
    plt.bar(idx, data['reverse'], label='Reverse ROP', alpha=0.7)
    plt.xlabel('Reaction Index')
    plt.ylabel('Rate of Progress (kmol/m³·s)')
    plt.title(f'ROP at Flame Peak (CF={c:.2f})')
    plt.legend()
    plt.show()
 
#%%
# ─── 6) INTEGRATED NET-PRODUCTION FOR NO, NO₂, N₂O ───────────────────────────
species_interest = ['NO', 'NO2', 'N2O']
relative_production = {}
 
for c in cracking_fractions:
    if stability_results.get(c) != 'On':
        continue
    flame = flames[c]
    x_grid = flame.grid
 
    gas2 = ct.Solution('reactionsokafar.yaml')
    integrated = []
    for sp in species_interest:
        idx_sp = gas2.species_index(sp)
        rate_sp = np.zeros_like(x_grid)
        for i, _ in enumerate(x_grid):
            comp_dict = {name: flame.X[k, i] for k, name in enumerate(gas2.species_names)}
            gas2.TPX = flame.T[i], ct.one_atm, comp_dict
            rate_sp[i] = gas2.net_production_rates[idx_sp]
        # integrate net production rate over space
        integrated.append(np.trapz(rate_sp, x_grid))
 
    # normalize to get % share
    total = np.sum(np.abs(integrated))
    relative_production[c] = [val/total*100 for val in integrated]
 
# Plot grouped bar chart
plt.figure()
x = np.arange(len(species_interest))
width = 0.2
for j, (c, shares) in enumerate(relative_production.items()):
    plt.bar(x + j*width, shares, width, label=f'CF={c:.2f}')
plt.xticks(x + width*(len(relative_production)-1)/2, species_interest)
plt.ylabel('Relative Integrated Net Production (%)')
plt.title('NO, NO₂, N₂O Production for Stable Flames')
plt.legend()
plt.show()
 
#%%
# ─── 7) NEXT STEPS ───────────────────────────────────────────────────────────
# Add new cells below to:
#   • Plot T(x) using temperature_profiles[c]
#   • Examine full species profiles (flame.Y or flame.X)
#   • Compute emission indices (e.g., total NOx)
#   • Explore heat‐release rates, mixture‐fraction, etc.

#%%
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
 
#%%
# ─── 1) PREP ───────────────────────────────────────────────────────────────────
# identify which cases actually burned
on_fractions = [cf for cf, st in stability_results.items() if st == 'On']
on_fractions.sort()
 
# load mechanism once
mech = ct.Solution('reactionsokafar.yaml')
reaction_eqns = mech.reaction_equations()  # list of strings
 
# build a dict of net‐ROP arrays (absolute) normalized to % of sum for each CF
net_rop_pct = {}
for cf in on_fractions:
    fwd = rop_results[cf]['forward']
    rev = rop_results[cf]['reverse']
    net = fwd - rev
    abs_net = np.abs(net)
    net_rop_pct[cf] = abs_net / abs_net.sum() * 100
 
# pick how many top reactions to display (e.g. top 20 by overall max)
# compute overall maximum across all CFs
max_by_rxn = np.array([max(net_rop_pct[cf][i] for cf in on_fractions) 
                       for i in range(len(reaction_eqns))])
top_idxs = np.argsort(max_by_rxn)[-20:][::-1]  # top 20 reactions
 
#%%
# ─── 2) PLOTTING ──────────────────────────────────────────────────────────────
species_plots = {'NO':0, 'NO2':1, 'N2O':2}  # mapping just to iterate 3 plots
 
for sp, _ in species_plots.items():
    # one figure per species
    fig, ax = plt.subplots(figsize=(8,8))
    # for each CF, plot a horizontal bar line
    n = len(on_fractions)
    offsets = np.linspace(-0.3, 0.3, n)   # offset each bar group
    height = 0.25
    for j, cf in enumerate(on_fractions):
        vals = net_rop_pct[cf][top_idxs]
        y = np.arange(len(top_idxs)) + offsets[j]
        ax.barh(y, vals, height=height, label=f"CF={cf:.2f}")
    # set y‐ticks to reaction labels
    ax.set_yticks(np.arange(len(top_idxs)))
    ax.set_yticklabels([reaction_eqns[i] for i in top_idxs])
    ax.invert_yaxis()  # highest at top
    ax.set_xlabel('% Net‐ROP contribution')
    ax.set_title(f'Reaction Sensitivity (% ROP) for {sp}')
    ax.legend(loc='best', frameon=False)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
# %%
