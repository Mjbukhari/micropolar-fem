#!/usr/bin/env python3
"""
Publication-quality plots for the micropolar FEM paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------------------------------------------------
# Data from your runs (copy-paste the numbers you already have)
# ----------------------------------------------------------------------

# --- Mesh refinement study at nu_r = 1.0 ---
h = np.array([0.125, 0.0625, 0.03125, 0.015625, 0.0078125])

err_u_L2 = np.array([1.057063e-02, 1.332706e-03, 1.672246e-04, 2.092752e-05, 2.616773e-06])
err_u_H1 = np.array([6.166315e-01, 1.587293e-01, 3.999869e-02, 1.002020e-02, 2.506354e-03])
err_p_L2 = np.array([5.539407e-02, 4.713905e-03, 5.443660e-04, 1.051820e-04, 2.524794e-05])
err_w_L2 = np.array([1.336859e-03, 1.072818e-04, 1.008249e-05, 1.125067e-06, 1.360125e-07])
err_w_H1 = np.array([3.607216e-02, 8.589054e-03, 2.120275e-03, 5.283591e-04, 1.319823e-04])

# --- Parameter study (errors at fixed h = 1/64) ---
nu_r = np.array([0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
err_u_vs_nur = np.array([2.092571e-05, 2.092571e-05, 2.092576e-05,
                         2.092752e-05, 2.095901e-05, 2.174408e-05, 5.390474e-05])
err_w_vs_nur = np.array([1.075348e-06, 1.075432e-06, 1.076619e-06,
                         1.125067e-06, 2.794557e-06, 2.022300e-05, 1.906317e-04])

# --- Observed rates vs nu_r (from your table) ---
rates_u_L2 = np.array([2.993, 2.993, 2.993, 2.994, 3.018, 3.355, 3.797])
rates_u_H1 = np.array([1.989, 1.989, 1.989, 1.989, 1.989, 1.991, 2.139])
rates_p_L2 = np.array([2.634, 2.641, 2.701, 3.114, 3.575, 3.676, 3.714])
rates_w_L2 = np.array([2.999, 3.000, 3.018, 3.411, 3.937, 3.940, 3.818])
rates_w_H1 = np.array([1.997, 1.997, 1.997, 2.018, 2.647, 2.936, 2.746])

# ----------------------------------------------------------------------
# Global style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

outdir = Path("figures")
outdir.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Figure 1 – Convergence rates (log-log)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5.5))

ax.loglog(h, err_u_L2, "o-", label=r"$\|u-u_h\|_{L^2}$")
ax.loglog(h, err_u_H1, "s-", label=r"$|u-u_h|_{H^1}$")
ax.loglog(h, err_p_L2, "^-", label=r"$\|p-p_h\|_{L^2}$")
ax.loglog(h, err_w_L2, "D-", label=r"$\|w-w_h\|_{L^2}$")
ax.loglog(h, err_w_H1, "v-", label=r"$|w-w_h|_{H^1}$")

# Reference slopes
href = np.array([0.125, 0.0078125])
ax.loglog(href, 3e-2 * (href/href[0])**3, "k--", lw=1.2, label=r"$O(h^3)$")
ax.loglog(href, 8e-1 * (href/href[0])**2, "k:",  lw=1.2, label=r"$O(h^2)$")

ax.set_xlabel(r"mesh size $h$")
ax.set_ylabel("error")
ax.set_title(r"Convergence at $\nu_r = 1$ (P2/P1/P2)")
ax.legend(loc="lower right", frameon=True)
ax.grid(True, which="both", ls=":", alpha=0.6)

fig.savefig(outdir / "convergence_nu_r_1.pdf")
fig.savefig(outdir / "convergence_nu_r_1.png")
plt.close()

# ----------------------------------------------------------------------
# Figure 2 – Error growth with nu_r (fixed mesh)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

# skip nu_r = 0 for the log-x plot
mask = nu_r > 0
ax.loglog(nu_r[mask], err_u_vs_nur[mask], "o-", label=r"$\|u-u_h\|_{L^2}$")
ax.loglog(nu_r[mask], err_w_vs_nur[mask], "s-", label=r"$\|w-w_h\|_{L^2}$")

# reference line O(nu_r)
ref = 2e-6 * nu_r[mask]
ax.loglog(nu_r[mask], ref, "k--", lw=1.2, label=r"$O(\nu_r)$")

ax.set_xlabel(r"vortex viscosity $\nu_r$")
ax.set_ylabel("error (fixed $h=1/64$)")
ax.set_title("Error growth with vortex viscosity")
ax.legend()
ax.grid(True, which="both", ls=":", alpha=0.6)

fig.savefig(outdir / "error_vs_nur.pdf")
fig.savefig(outdir / "error_vs_nur.png")
plt.close()

# ----------------------------------------------------------------------
# Figure 3 – Observed rates vs nu_r
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

ax.semilogx(nu_r, rates_u_L2, "o-", label=r"$u$ $L^2$ (expected 3)")
ax.semilogx(nu_r, rates_u_H1, "s-", label=r"$u$ $H^1$ (expected 2)")
ax.semilogx(nu_r, rates_p_L2, "^-", label=r"$p$ $L^2$ (expected 2)")
ax.semilogx(nu_r, rates_w_L2, "D-", label=r"$w$ $L^2$ (expected 3)")
ax.semilogx(nu_r, rates_w_H1, "v-", label=r"$w$ $H^1$ (expected 2)")

ax.axhline(3, color="k", ls="--", lw=0.8, alpha=0.5)
ax.axhline(2, color="k", ls=":",  lw=0.8, alpha=0.5)

ax.set_xlabel(r"vortex viscosity $\nu_r$")
ax.set_ylabel("observed rate")
ax.set_title("Convergence rates versus $\\nu_r$")
ax.legend(loc="upper left", ncol=2, fontsize=10)
ax.set_ylim(1.5, 4.2)
ax.grid(True, which="both", ls=":", alpha=0.6)

fig.savefig(outdir / "rates_vs_nur.pdf")
fig.savefig(outdir / "rates_vs_nur.png")
plt.close()

print("Figures saved in ./figures/")
print("  - convergence_nu_r_1.pdf / .png")
print("  - error_vs_nur.pdf / .png")
print("  - rates_vs_nur.pdf / .png")