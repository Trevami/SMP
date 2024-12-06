#!/usr/bin/env python3

import numpy as np
import scipy.linalg


def lj_potential(r_ij: np.ndarray) -> float:
    r_norm = np.linalg.norm(r_ij)
    return 4 * ((1 / r_norm) ** 12 - (1 / r_norm) ** 6)

    
def lj_force(r_ij: np.ndarray) -> np.ndarray:
    r_norm = np.linalg.norm(r_ij)
    return -24 * r_ij * (r_norm ** 6 - 2) / r_norm ** 14


def lj_pot_rep(r_ij: np.ndarray) -> float:
    r_norm = np.linalg.norm(r_ij)
    # value corresponds to r=2**(1/6) at minimum of LJ
    if r_norm <= 1.122462048309373:
        # schift by -epsilon (+1)
        return 4 * ((1 / r_norm) ** 12 - (1 / r_norm) ** 6) + 1
    else:
        return 0.0


def lj_force_rep(r_ij: np.ndarray) -> np.ndarray:
    r_norm = np.linalg.norm(r_ij)
    # value corresponds to r=2**(1/6) at minimum of LJ
    if r_norm <= 1.122462048309373:
        # schift by -epsilon (+1)
        return -24 * r_ij * (r_norm ** 6 - 2) / r_norm ** 14
    else:
        return np.array([0.0, 0.0])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    num_vals = 1000
    d_vec = np.array([np.linspace(0.85, 2.5, num_vals), np.zeros(num_vals)])

    lj_pot_energies = np.array([lj_potential(r_ij) for r_ij in d_vec.T])
    lj_forces = np.array([lj_force(r_ij) for r_ij in d_vec.T]).T

    lj_pot_energies_rep = np.array([lj_pot_rep(r_ij) for r_ij in d_vec.T])
    lj_forces_rep = np.array([lj_force_rep(r_ij) for r_ij in d_vec.T]).T

    plot_path = Path(__file__).resolve().parent.parent/"plots"
    plt.axhline(0, color="gray", linewidth=1,
                linestyle="--")  # Horizontal line at y=0
    plt.plot(d_vec[0], lj_pot_energies, label=r"LJ potential")
    plt.plot(d_vec[0], lj_forces[0], label=r"LJ force")
    plt.xlabel(r"$r\;[\sigma]$")
    plt.ylabel(r"$V_\text{LJ}(r),\;F_\text{LJ}(r)\;[\epsilon]$")
    plt.xlim(0.85,2.5)
    plt.ylim(-2.5, 2.5)
    plt.legend()
    plt.savefig(plot_path/"ex_3_2_plot_1.png")
    plt.clf()

    plt.axhline(0, color="gray", linewidth=1,
                linestyle="--")  # Horizontal line at y=0
    plt.plot(d_vec[0], lj_pot_energies, label=r"LJ potential")
    plt.plot(d_vec[0], lj_forces[0], label=r"LJ force")
    plt.plot(d_vec[0], lj_pot_energies_rep, label=r"LJ potential, fully repulsive")
    plt.plot(d_vec[0], lj_forces_rep[0], label=r"LJ force, fully repulsive")
    plt.xlabel(r"$r\;[\sigma]$")
    plt.ylabel(r"$V_\text{LJ}(r),\;F_\text{LJ}(r)\;[\epsilon]$")
    plt.xlim(0.85,2.5)
    plt.ylim(-2.5, 2.5)
    plt.legend()
    plt.savefig(plot_path/"ex_3_2_plot_2.png")
    plt.show()