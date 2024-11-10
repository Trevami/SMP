#!/usr/bin/env python3

import numpy as np
import scipy.constants
import matplotlib.pyplot as plt

from pathlib import Path


def force(x_ij, m_i, m_j, g):
    return -g * m_i * m_j * x_ij / (np.linalg.norm(x_ij) ** 3)


def forces(x, masses, g):
    dim = np.shape(x)
    force_mat = np.zeros((dim[1], dim[0], dim[0]), dtype="float")
    for i in range(dim[0]):
        for j in range(dim[0]):
            if j < i:
                force_xy = force(x[i] - x[j], masses[i], masses[j], g)
                # Diagonal of matrix is 0
                # -> planet exhibits no force on itself
                # Lower diagonal half:
                force_mat[0, i, j] = force_xy[0]  # Force x-components
                force_mat[1, i, j] = force_xy[1]  # Force y-components
                # Upper diagonal half (antisymmetric to upper):
                force_mat[0, j, i] = -force_xy[0]  # Force x-components
                force_mat[1, j, i] = -force_xy[1]  # Force y-components
    return force_mat


def step_euler(x, v, dt, masses, g, forces):
    for i in range(np.shape(x)[0]):
        x[i] = x[i] + v[i] * dt
        force_tot = np.array(
            [np.sum(forces[0, i, :]), np.sum(forces[1, i, :])])
        v[i] = v[i] + force_tot / masses[i] * dt
    return x, v


def run(x, v, dt, masses, g):
    x_new = x.copy()
    v_new = v.copy()
    trajectory = [x_new.copy()]
    for time_step in range(int(1 / dt)):
        force_mat = forces(x_new, masses, g)
        x_new, v_new = step_euler(x_new, v_new, dt, masses, g, force_mat)
        trajectory.append(x_new.copy())
    return np.array(trajectory)


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / \
        "files/solar_system.npz"
    data = np.load(data_path)
    names = data["names"]

    x_init = data["x_init"]
    v_init = data["v_init"]
    m = data["m"]
    g = data["g"]
    # Transpose to seperate x and v vectors
    x = np.array(x_init).transpose()
    v = np.array(v_init).transpose()

    # Calculate trajectories of planets
    trajectories_4dig = run(x, v, 10e-4, m, g).transpose()
    trajectories_5dig = run(x, v, 10e-5, m, g).transpose()

    plot_path = Path(__file__).resolve().parent.parent/"plots"
    # Trajectories plot dt=10e-4
    for i in range(np.shape(trajectories_4dig)[1]):
        plt.plot(
            trajectories_4dig[0, i, :],
            trajectories_4dig[1, i, :],
            "-",
            label=names[i].decode("UTF-8"),
        )
    plt.xlabel(r"$x$ in au")
    plt.ylabel(r"$y$ in au")
    plt.legend(loc="lower right")
    plt.savefig(plot_path/"Exc1_Plot_3_1_part1.png")
    plt.clf()

    # Trajectories plot dt=10e-5
    for i in range(np.shape(trajectories_5dig)[1]):
        plt.plot(
            trajectories_5dig[0, i, :],
            trajectories_5dig[1, i, :],
            "-",
            label=names[i].decode("UTF-8"),
        )
    plt.xlabel(r"$x$ in au")
    plt.ylabel(r"$y$ in au")
    plt.legend(loc="lower right")
    plt.savefig(plot_path/"Exc1_Plot_3_1_part2.png")
