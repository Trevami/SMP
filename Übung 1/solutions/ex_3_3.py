#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import ex_3_1
import ex_3_2


def get_dists_earth_sun(trajectories):
    # Calculates the distances of erth and sun from trajectories
    dists_earth_sun = []
    for timestep in range(np.shape(trajectories)[2]):
        dists_earth_sun.append(np.linalg.norm(
            trajectories[:, 1, timestep] - trajectories[:, 0, timestep]))
    return dists_earth_sun


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

    # Calculate trajectories of planets and the distance beween earth and sun
    timestep = 0.01
    duration = 20

    traject_euler = ex_3_2.run(
        x.copy(), v.copy(), timestep, duration, m, g, ex_3_1.step_euler).transpose()
    dists_e_s_euler = get_dists_earth_sun(traject_euler)

    traject_sympl_euler = ex_3_2.run(
        x.copy(), v.copy(), timestep, duration, m, g, ex_3_2.step_symplectic_euler).transpose()
    dists_e_s_symp_euler = get_dists_earth_sun(traject_sympl_euler)

    traject_vv = ex_3_2.run(
        x.copy(), v.copy(), timestep, duration, m, g, ex_3_2.step_velocity_verlet).transpose()
    dists_e_s_vv = get_dists_earth_sun(traject_vv)

    # Plot the distance of earth and sun for the 3 integrators
    timeline = np.linspace(0, duration, int(duration/timestep)+1)
    plt.plot(timeline, dists_e_s_euler, '-',
             label="Euler")
    plt.plot(timeline, dists_e_s_symp_euler, '-',
             label="Simplectic Euler")
    plt.plot(timeline, dists_e_s_vv, '-',
             label="Velocity Verlet")
    plt.xlabel(r"$t$ in jears")
    plt.ylabel(r"$\Delta r$ in au")
    plt.legend(loc="upper left")
    plt.show()
