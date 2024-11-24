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
    if np.linalg.norm(r_ij) <= 1.122462048309373:
        return -24 * r_ij * (r_norm ** 6 - 2) / r_norm ** 14
    else:
        return 0.0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    num_vals = 1000
    d_vec = np.array([np.linspace(0.85, 2.5, num_vals), np.zeros(num_vals)])

    # parameter signature="(n)->()" drops a dimension
    lj_pot_vec_func = np.vectorize(lj_potential, signature="(n)->()")
    lj_force_vec_func = np.vectorize(lj_force)

    lj_pot_energies = lj_pot_vec_func(d_vec.T)
    lj_forces = lj_force_vec_func(d_vec)

    plt.axhline(0, color="gray", linewidth=1,
                linestyle="--")  # Horizontal line at y=0
    plt.plot(d_vec[0], lj_pot_energies)
    plt.plot(d_vec[0], lj_forces[0])
    plt.ylim(-2.5, 1)
    plt.xlim(0.5,2.5)
    plt.show()
