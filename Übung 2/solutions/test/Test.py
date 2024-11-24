#!/usr/bin/env python3

import numpy as np
import scipy.linalg


def lj_potential(r_ij: np.ndarray) -> float:
    r_norm = np.linalg.norm(r_ij)
    return 4 * ((1 / r_norm) ** 12 - (1 / r_norm) ** 6)

    
def lj_force(r_ij: np.ndarray) -> np.ndarray:
    r_norm = np.linalg.norm(r_ij)
    return -24 * r_ij * (r_norm ** 6 - 2) / r_norm ** 14


def lj_pot_rep(r_ij: np.ndarray, r_cut:float) -> float:
    r_norm = np.linalg.norm(r_ij)
    if r_norm < r_cut:
        return 4 * (1 / r_norm) ** 12 - 4 * (1 / r_cut) ** 12
    else: return 0.0


def lj_force_rep(r_ij: np.ndarray, r_cut:float) -> np.ndarray:
    r_norm = np.linalg.norm(r_ij)
    if r_norm < r_cut:
        return 48 * r_ij / r_norm ** 14
    else: return np.array([0.0, 0.0])


def lj_pot_rep2(r_ij: np.ndarray) -> float:
    r_norm = np.linalg.norm(r_ij)
    # value corresponds to r=2**(1/6) at minimum of LJ
    if r_norm <= 1.122462048309373:
        # schift by -epsilon (+1)
        return 4 * ((1 / r_norm) ** 12 - (1 / r_norm) ** 6) + 1
    else:
        return 0.0


def lj_force_rep2(r_ij: np.ndarray) -> np.ndarray:
    r_norm = np.linalg.norm(r_ij)
    # value corresponds to r=2**(1/6) at minimum of LJ
    if r_norm <= 1.122462048309373:
        # schift by -epsilon (+1)
        return -24 * r_ij * (r_norm ** 6 - 2) / r_norm ** 14
    else:
        return np.array([0.0, 0.0])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    num_vals = 1000
    d_vec = np.array([np.linspace(0.85, 2.5, num_vals), np.zeros(num_vals)])

    lj_pot_energies = np.array([lj_potential(r_ij) for r_ij in d_vec.T])
    lj_forces = np.array([lj_force(r_ij) for r_ij in d_vec.T]).T

    lj_pot_energies_rep = np.array([lj_pot_rep(r_ij, 1.5) for r_ij in d_vec.T])
    lj_forces_rep = np.array([lj_force_rep(r_ij, 1.5) for r_ij in d_vec.T]).T

    lj_pot_energies_rep2 = np.array([lj_pot_rep2(r_ij) for r_ij in d_vec.T])
    lj_forces_rep2 = np.array([lj_force_rep2(r_ij) for r_ij in d_vec.T]).T

    plt.axhline(0, color="gray", linewidth=1,
                linestyle="--")  # Horizontal line at y=0
    # plt.plot(d_vec[0], lj_pot_energies)
    # plt.plot(d_vec[0], lj_forces[0])
    # plt.plot(d_vec[0], lj_pot_energies_rep)
    # plt.plot(d_vec[0], lj_forces_rep[0])
    plt.plot(d_vec[0], lj_pot_energies_rep2)
    plt.plot(d_vec[0], lj_forces_rep2[0])
    plt.ylim(-2.5, 2.5)
    plt.xlim(0.8,1.5)
    plt.show()