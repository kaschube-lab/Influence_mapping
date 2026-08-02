"""
supfig_plausible_sigma_ratio.py
================================
Supplementary figure: biologically plausible regime in the W_EE × W_EI
parameter space for five σ_I/σ_E ratios (0.75, 1.0, 1.25, 1.5, 1.75).

Plausibility criteria (same four as compute_plausible() / get_plausible_bool()
in helpers.py):

  1. stability  < 1     – network is stable
  2. avg_infl   < 0     – net suppressive influence within 1.6 σ_I
  3. local_infl < 0     – locally suppressive (within avg_range / 4)
  4. max_supress > 1    – the most-suppressed neuron is not at the stim site

Per-panel colour code (matching fig4draft_dec.ipynb panel 'e')
---------------------
  OrRd colour  – biologically plausible region (boolean cast to float)
  white        – unstable (masked with contourf, colours='white')
  black line   – stability boundary (contour at max_eig = 1)
  black vline  – w_EE = 1 (boundary of ISN regime)
  text labels  – example LELI and Cross-dominant parameter points

Model construction follows sigma_ratio_param_scan.py (direct weight-matrix
build, no default_params) for speed; all other parameters match
compute_plausible() in helpers.py.

Output
------
supfig_plausible_sigma_ratio.pdf / .png  saved to savepath
"""

import os, sys, math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

# ── path setup ────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
savepath = '/Users/fionakong/Downloads/kaschube-lab/Influence_mapping/rebuttal_nc/sigma_ratio'
sys.path.insert(0, os.path.join(HERE, ".."))

from Model.ring import generate_model, recurrent_connections, ff_connections

# ── plot style (Nature style, matching fig4draft_dec.ipynb) ───────────────────
# old STYLE had font.size=9; updated to Nature 7 pt to match the notebook
STYLE = {
    "font.size":         7,
    "axes.titlesize":    7,
    "axes.labelsize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
    "lines.linewidth":   1.0,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "figure.dpi":        300,
}
mpl.rcParams.update(STYLE)

# ── scan parameters (match compute_plausible in helpers.py) ───────────────────
WEE     = np.arange(0.1, 2.7, 0.1)   # 26 values of w_EE
WEI     = np.arange(0.1, 3.2, 0.1)   # 31 values of w_EI = w_IE
WII     = 2.5
SIGMA_E = 10
N       = 100
LOC     = 20   # stimulated neuron index (same as compute_plausible)

SIGMA_RATIOS = [0.75, 1.0, 1.25, 1.5, 1.75]

# ── example parameter points to mark on every panel ───────────────────────────
# "tag" is the short text label drawn at the point (matching panel 'e' style
# from fig4draft_dec.ipynb which uses single letters in a white rounded box)
EXAMPLE_PARAMS = {
    "LELI":           {"wee": 2.49, "wei": 2.39, "tag": "LELI"},
    "Cross dominant": {"wee": 1.1,  "wei": 2.89, "tag": "CD"},
}


# ══════════════════════════════════════════════════════════════════════════════
#  Model builder
# ══════════════════════════════════════════════════════════════════════════════

def _make_model(wee, wei, sigma_ratio):
    """
    Build a LinearModel for the given (wee, wei, sigma_ratio).
    sigma_ie is fixed at SIGMA_E; sigma_ii = sigma_ei = SIGMA_E * sigma_ratio.
    """
    rec_params = {
        "N"       : N,
        "npop"    : 2,
        "sigma"   : SIGMA_E,
        "r"       : None,
        "wee"     : wee,
        "wie"     : wei,
        "wei"     : wei,
        "wii"     : WII,
        "sigma_ie": SIGMA_E,
        "sigma_ii": SIGMA_E * sigma_ratio,
        "sigma_ei": SIGMA_E * sigma_ratio,
    }
    ff_params = {"N": N, "npop": 2, "sigma": SIGMA_E}
    W    = recurrent_connections(N, rtype="MH", params=rec_params)
    W_ff = ff_connections(N, fftype="uniform", params=ff_params)
    return generate_model(
        {
            "N"    : N,
            "W"    : torch.tensor(W,    dtype=torch.float32),
            "W_ff" : torch.tensor(W_ff, dtype=torch.float32),
            "npop" : 2,
            "sigma": SIGMA_E,
        },
        linear=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Plausibility scan
# ══════════════════════════════════════════════════════════════════════════════

def run_plausible_scan(sigma_ratio):
    """
    Sweep WEE × WEI and evaluate the four plausibility criteria for a given
    sigma_ratio.  Returns four (len(WEI), len(WEE)) arrays:

        stability   – max real eigenvalue
        avg_infl    – mean E-E influence within 1.6 σ_I  (NaN if unstable)
        local_infl  – mean E-E influence within avg_range/4  (NaN if unstable)
        max_supress – distance of the most-suppressed neuron  (NaN if unstable)

    Averaging ranges match compute_plausible() in helpers.py:
        avg_range   = int(1.6 * SIGMA_E * sigma_ratio)
        local_range = ceil(avg_range / 4)
    """
    avg_range   = int(1.6 * SIGMA_E * sigma_ratio)
    local_range = math.ceil(avg_range / 4)

    stability   = np.full((len(WEI), len(WEE)), np.nan)
    avg_infl    = np.full((len(WEI), len(WEE)), np.nan)
    local_infl  = np.full((len(WEI), len(WEE)), np.nan)
    max_supress = np.full((len(WEI), len(WEE)), np.nan)

    total = len(WEI) * len(WEE)
    done  = 0

    for i, wei in enumerate(WEI):
        for j, wee in enumerate(WEE):
            model   = _make_model(wee, wei, sigma_ratio)
            max_eig = model.get_max_eigenvalue()
            stability[i, j] = max_eig

            if max_eig < 1.0:
                dist, infl_e, _ = model.get_influence_distance(loc=LOC, pop="E")
                dist   = dist.cpu().detach().numpy()
                infl_e = infl_e.cpu().detach().numpy()

                mask_avg   = dist < avg_range
                mask_local = dist < local_range

                avg_infl[i, j]    = infl_e[mask_avg].mean()   if mask_avg.any()   else np.nan
                local_infl[i, j]  = infl_e[mask_local].mean() if mask_local.any() else np.nan
                max_supress[i, j] = dist[np.argmin(infl_e)]

            done += 1
            if done % 50 == 0 or done == total:
                print(f"    {done}/{total}", end="\r", flush=True)

    print()
    return stability, avg_infl, local_infl, max_supress


# ── run all ratios ─────────────────────────────────────────────────────────────
print("=" * 55)
print("Computing plausibility scans …")
print("=" * 55)

all_results = {}
for ratio in SIGMA_RATIOS:
    print(f"\n  sigma_I/sigma_E = {ratio:.2f}")
    all_results[ratio] = run_plausible_scan(ratio)

print("\nAll scans done.  Building figure …")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure  (style matches fig4draft_dec.ipynb panel 'e')
# ══════════════════════════════════════════════════════════════════════════════

# With set_aspect('equal'), each panel is taller than wide (x-range 2.5,
# y-range 3.0).  constrained_layout sizes the figure automatically.
# old: FIG_W, FIG_H = 8.27, 3.2 with green/gray coloring; no equal aspect
fig, axes = plt.subplots(
    1, len(SIGMA_RATIOS),
    figsize=(8.27, 4.0),
    constrained_layout=True,
)

X, Y = np.meshgrid(WEE, WEI)

for ax, ratio in zip(axes, SIGMA_RATIOS):
    stability, avg_infl, local_infl, max_supress = all_results[ratio]

    # ── plausibility boolean masks ────────────────────────────────────────────
    bool_stable  = np.where(np.isnan(stability),   False, stability   < 1)
    bool_avg     = np.where(np.isnan(avg_infl),    False, avg_infl    < 0)
    bool_local   = np.where(np.isnan(local_infl),  False, local_infl  < 0)
    bool_supress = np.where(np.isnan(max_supress), False, max_supress > 1)
    plausible    = bool_stable & bool_avg & bool_local & bool_supress

    # ── plausible region: OrRd colourmap on boolean data (matches panel 'e') ─
    ax.pcolormesh(X, Y, plausible.astype(float), shading="auto", cmap="OrRd")

    # ── mask unstable region with white contourf (matches panel 'e') ─────────
    stab_clean = np.nan_to_num(stability, nan=0.0)
    inst_mask  = np.ma.masked_where(stab_clean <= 1, stab_clean)
    if inst_mask.count() > 0:
        ax.contourf(X, Y, inst_mask,
                    levels=[1, stab_clean.max()], colors="white")

    # ── stability boundary ────────────────────────────────────────────────────
    ax.contour(X, Y, stab_clean, levels=[1],
               colors="black", linewidths=1)

    # ── vertical line at w_EE = 1 (ISN boundary, matches panel 'e') ─────────
    ax.axvline(x=1, color="black", linestyle="-", linewidth=0.8)

    # ── example parameter text labels (matches panel 'e' style) ──────────────
    # for label, ep in EXAMPLE_PARAMS.items():
    #     ax.text(
    #         ep["wee"], ep["wei"], ep["tag"],
    #         fontsize=6, color="black", ha="center", va="center",
    #         bbox=dict(facecolor="white", edgecolor="none",
    #                   boxstyle="round,pad=0.1"),
    #         zorder=5,
    #     )

    # ── axes formatting (matching panel 'e') ──────────────────────────────────
    ax.set_title(rf"$\sigma_I/\sigma_E = {ratio:.2f}$", pad=2)
    ax.set_xlabel(r"$w_{EE}$", labelpad=0)
    if ax is axes[0]:
        ax.set_ylabel(r"$\sqrt{w_{EI}w_{IE}}$", labelpad=-2)
    else:
        ax.set_ylabel("")
    ax.set_xticks([0.1, 1, 2.5], ["$0$", "$1$", "$2.5$"])
    ax.set_yticks([0.1, 2.5],    ["$0$", "$2.5$"])
    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", pad=1)
    ax.set_aspect("equal")   # same physical length = same data range on both axes

# ── save ──────────────────────────────────────────────────────────────────────
pdf_path = os.path.join(savepath, "supfig_plausible_sigma_ratio.pdf")
png_path = os.path.join(savepath, "supfig_plausible_sigma_ratio.png")

fig.savefig(pdf_path, dpi=300)
fig.savefig(png_path, dpi=150)
plt.close(fig)

print(f"\nSaved → {pdf_path}")
print(f"Saved → {png_path}")
print("Done.")
