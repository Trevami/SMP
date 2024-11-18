#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import ex_3_1


def step_symplectic_euler(x, v, dt, masses, g, forces):
    # Changed pos. of x and v calculations compared to previous exercises
    for i in range(np.shape(x)[0]):
        force_tot = np.array([np.sum(forces[0, i, :]), np.sum(forces[1, i, :])])
        v[i] = v[i] + force_tot / masses[i] * dt
        x[i] = x[i] + v[i] * dt
    return x, v


def step_velocity_verlet(x, v, dt, masses, g, forces_old):
    # First step of the velocity verlet algorithm:
    # Calculates new pos. with accelerations (accs) form given forces
    accs_first = np.zeros(np.shape(x))
    for i in range(np.shape(x)[0]):
        force_tot = np.array([np.sum(forces_old[0, i, :]), np.sum(forces_old[1, i, :])])
        accs_first[i] = force_tot / masses[i]
        x[i] = x[i] + v[i] * dt + accs_first[i] / 2 * dt**2

    # Second step of the velocity verlet alogrihtm:
    # Calculates new forces from updatet postions of step 1
    # then uses old acc (step 1) and new accs form updated forces
    # to calculate the velocities
    forces_updated = ex_3_1.forces(x, masses, g)
    accs_second = np.zeros(np.shape(x))
    for i in range(np.shape(x)[0]):
        force_tot = np.array(
            [np.sum(forces_updated[0, i, :]), np.sum(forces_updated[1, i, :])]
        )
        accs_second[i] = force_tot / masses[i]
        v[i] = v[i] + (accs_first[i] + accs_second[i]) * dt / 2
    return x, v


def run(x, v, dt, duration, masses, g, step_method):
    # Duration and time step are specified in years
    x_new = x.copy()
    v_new = v.copy()
    trajectory = [x_new.copy()]
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
            trajectories[:, 2, timestep] - trajectories[:, 1, timestep]
        )
    return np.array(trat_moon_to_earth)


if __name__ == "__main__":
    x = np.array(x_init).transpose()
    v = np.array(v_init).transpose()

    # Calculate trajectories of planets and the distance beween earth and sun
    # Calculate trajectories of planets and the distance beween earth and moon
    timestep = 0.01
    duration = 20

    traject_euler = ex_3_2.run(
        x.copy(), v.copy(), timestep, duration, m, g, ex_3_1.step_euler).transpose()
    dists_e_s_euler = get_dists_earth_sun(traject_euler)
    dists_e_s_euler = get_dists_earth_moon(traject_euler)

    traject_sympl_euler = ex_3_2.run(
        x.copy(), v.copy(), timestep, duration, m, g, ex_3_2.step_symplectic_euler).transpose()
    dists_e_s_symp_euler = get_dists_earth_sun(traject_sympl_euler)
    dists_e_s_symp_euler = get_dists_earth_moon(traject_sympl_euler)

    traject_vv = ex_3_2.run(
        x.copy(), v.copy(), timestep, duration, m, g, ex_3_2.step_velocity_verlet).transpose()
    dists_e_s_vv = get_dists_earth_sun(traject_vv)
    dists_e_s_vv = get_dists_earth_moon(traject_vv)

    # Plot the distance of earth and sun for the 3 integrators
    plot_path = Path(__file__).resolve().parent.parent/"plots"
    # Plot the distance of earth and moon for the 3 integrators
    timeline = np.linspace(0, duration, int(duration/timestep)+1)
    plt.plot(timeline, dists_e_s_euler, '-',
             label="Euler")
    plt.plot(timeline, dists_e_s_symp_euler, '-',
             label="Simplectic Euler")
    plt.plot(timeline, dists_e_s_vv, '-',
             label="Velocity Verlet")
    plt.xlabel(r"$t$ in jears")
    plt.ylabel(r"$\Delta r$ in au")
    plt.xlabel(r"$t$ in years")
    plt.ylabel(r"$\Delta r_{earh/moon}$ in au")
    plt.legend(loc="upper left")
    plt.show()
    plt.savefig(plot_path/"Exc1_Plot_3_3_part1.png")
