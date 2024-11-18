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
    return x, v, forces_updated


def run(x, v, dt, duration, masses, g, step_method):
    # Duration and time step are specified in years
    x_new = x.copy()
    v_new = v.copy()
    trajectory = [x_new.copy()]
    if step_method == step_velocity_verlet:
        force_mat = ex_3_1.forces(x_new, masses, g)
        for timestep in range(int(duration / (dt))):
            x_new, v_new, force_mat = step_method(x_new, v_new, dt, masses, g, force_mat)
            trajectory.append(x_new.copy())
    else:
        for timestep in range(int(duration / (dt))):
            force_mat = ex_3_1.forces(x_new, masses, g)
            x_new, v_new = step_method(x_new, v_new, dt, masses, g, force_mat)
            trajectory.append(x_new.copy())
    return np.array(trajectory)


def calc_moon_to_earth(trajectories):
    # Calculates the trajectory of the moon compared to earth
    trat_moon_to_earth = []
    for timestep in range(np.shape(trajectories)[2]):
        trat_moon_to_earth.append(
            trajectories[:, 2, timestep] - trajectories[:, 1, timestep])
    return np.array(trat_moon_to_earth)


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

    plot_path = Path(__file__).resolve().parent.parent/"plots"
    # Trajectory of the moon compared to earth for symplectic Euler algorithm
    trajectory_moon_earth = calc_moon_to_earth(
        run(x.copy(), v.copy(), 1e-2, 20, m, g, step_symplectic_euler).transpose())
    plt.plot(
        trajectory_moon_earth[:, 0],
        trajectory_moon_earth[:, 1],
        "-",
        label=r"moon compared to earth"
    )
    plt.xlabel(r"$x$ in au")
    plt.ylabel(r"$y$ in au")
    plt.legend(loc="lower right")
    plt.savefig(plot_path/"Exc1_Plot_3_2_part1.png")
    plt.clf()

    # Trajectory of the moon compared to earth for velocity Verlet algorithm
    trajectory_moon_earth = calc_moon_to_earth(
        run(x, v, 1e-2, 1, m, g, step_velocity_verlet).transpose())
    plt.plot(
        trajectory_moon_earth[:, 0],
        trajectory_moon_earth[:, 1],
        "-",
        label=r"moon compared to earth"
    )
    plt.xlabel(r"$x$ in au")
    plt.ylabel(r"$y$ in au")
    plt.legend(loc="lower right")
    plt.savefig(plot_path/"Exc1_Plot_3_2_part2.png")
