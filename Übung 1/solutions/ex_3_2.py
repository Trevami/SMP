#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ex_3_1


def step_symplectic_euler(x, v, dt, mass, g, forces):

    for i in range(np.shape(x)[0]):
        v[i] = (
            v[i]
            + np.array([np.sum(forces[0, i, :]), np.sum(forces[1, i, :])])
            / mass[i]
            * dt
        )
        x[i] = x[i] + v[i] * dt

    return x, v


def step_velocity_verlet(x, v, dt, mass, g, force_old):
    acc_first = np.zeros(np.shape(x))
    acc_second = np.zeros(np.shape(x))

    for i in range(np.shape(x)[0]):
        acc_first[i] = (
            np.array([np.sum(force_old[0, i, :]), np.sum(force_old[1, i, :])]) / mass[i]
        )
        x[i] = x[i] + v[i] * dt + acc_first[i] / 2 * dt**2

    force_updated = ex_3_1.forces(x, mass, g)
    for i in range(np.shape(x)[0]):
        acc_second[i] = (
            np.array([np.sum(force_updated[0, i, :]), np.sum(force_updated[1, i, :])])
            / mass[i]
        )
        v[i] = v[i] + (acc_first[i] + acc_second[i]) * dt / 2
    return x, v


def run(x, v, dt, masses, g):
    trajectory = [x.copy()]
    for step in range(int(1 / (dt))):
        F_mat = ex_3_1.forces(x, masses, g)
        x, v = step_velocity_verlet(x, v, dt, masses, g, F_mat)
        trajectory.append(x.copy())
    return np.array(trajectory)


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "files" / "solar_system.npz"
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
    trajectories = run(x, v, 10e-4, m, g).transpose()

    # Trajectories plot
    for i in range(np.shape(trajectories)[1]):
        plt.plot(
            trajectories[0, i, :],
            trajectories[1, i, :],
            "-",
            label=names[i].decode("UTF-8"),
        )
    plt.xlabel(r"$x$ in au")
    plt.ylabel(r"$y$ in au")
    plt.legend(loc="lower right")
    plt.show()
