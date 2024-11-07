#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ex_3_1


def step_symplectic_euler(x, v, dt, masses, g, forces):
    # Changed pos. of x and v calculations compared to previous exercises
    for i in range(np.shape(x)[0]):
        force_tot = np.array(
            [np.sum(forces[0, i, :]), np.sum(forces[1, i, :])])
        v[i] = v[i] + force_tot / masses[i] * dt
        x[i] = x[i] + v[i] * dt
    return x, v


def step_velocity_verlet(x, v, dt, masses, g, forces_old):
    # First step of the velocity verlet algorithm:
    # Calculates new pos. with accelerations (accs) form given forces
    accs_first = np.zeros(np.shape(x))
    for i in range(np.shape(x)[0]):
        force_tot = np.array(
            [np.sum(forces_old[0, i, :]), np.sum(forces_old[1, i, :])])
        accs_first[i] = (force_tot / masses[i])
        x[i] = x[i] + v[i] * dt + accs_first[i] / 2 * dt**2

    # Second step of the velocity verlet alogrihtm:
    # Calculates new forces from updatet postions of step 1
    # then uses old acc (step 1) and new accs form updated forces
    # to calculate the velocities
    forces_updated = ex_3_1.forces(x, masses, g)
    accs_second = np.zeros(np.shape(x))
    for i in range(np.shape(x)[0]):
        force_tot = np.array(
            [np.sum(forces_updated[0, i, :]), np.sum(forces_updated[1, i, :])])
        accs_second[i] = (force_tot / masses[i])
        v[i] = v[i] + (accs_first[i] + accs_second[i]) * dt / 2
    return x, v


def run(x, v, dt, duration, masses, g, step_method):
    # Duration and time step is specified in years
    trajectory = [x.copy()]
    for timestep in range(int(duration / (dt))):
        force_mat = ex_3_1.forces(x, masses, g)
        x, v = step_method(x, v, dt, masses, g, force_mat)
        trajectory.append(x.copy())
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
    trajectories = run(x, v, 10e-4, 1, m, g, step_velocity_verlet).transpose()

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
