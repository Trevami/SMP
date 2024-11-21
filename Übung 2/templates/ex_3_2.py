#!/usr/bin/env python3

import numpy as np
import scipy.linalg


def lj_potential(r_ij: np.ndarray) -> float:
    print(r_ij)
    return 4 * ((1 / np.linalg.norm(r_ij)) ** 12 - (1 / np.linalg.norm(r_ij)) ** 6)


def lj_force(r_ij: np.ndarray) -> np.ndarray:
    r_norm = np.linalg.norm(r_ij)
    x_comp = -4 * (12 * r_ij[0] / r_norm**14 - 6 * r_ij[0] / r_norm**8)
    y_comp = -4 * (12 * r_ij[1] / r_norm**14 - 6 * r_ij[1] / r_norm**8)
    return np.array([x_comp, y_comp])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    d_vec = np.array([np.linspace(0.85, 2.5, 1), np.zeros(1)])
    print(d_vec)
    # d_vec = d_vec.transpose()
    # print(np.shape(d_vec))
    # print(d_vec[0:10])
    vfunc_lj = np.vectorize(lj_potential)
    # vfunc_force = np.vectorize(lj_force)
    lj_vec = vfunc_lj(d_vec)
    # lj_force = vfunc_force(d_vec)
    # plt.plot(lj_vec)
