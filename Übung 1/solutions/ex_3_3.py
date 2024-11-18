#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import ex_3_1
import ex_3_2


def get_dists_earth_moon(trajectories):
    # Calculates the distances of erth and moon from trajectories
    dists_earth_moon = []
    for timestep in range(np.shape(trajectories)[2]):
        dists_earth_moon.append(np.linalg.norm(
            trajectories[:, 2, timestep] - trajectories[:, 1, timestep]))
    return dists_earth_moon


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

    # Calculate trajectories of planets and the distance beween earth and moon
    timestep = 1e-2
    duration = 20

    traject_euler = ex_3_2.run(
        x, v, timestep, duration, m, g, ex_3_1.step_euler).transpose()
    dists_e_s_euler = get_dists_earth_moon(traject_euler)

    traject_sympl_euler = ex_3_2.run(
        x, v, timestep, duration, m, g, ex_3_2.step_symplectic_euler).transpose()
    dists_e_s_symp_euler = get_dists_earth_moon(traject_sympl_euler)

    traject_vv = ex_3_2.run(
        x, v, timestep, duration, m, g, ex_3_2.step_velocity_verlet).transpose()
    dists_e_s_vv = get_dists_earth_moon(traject_vv)

    plot_path = Path(__file__).resolve().parent.parent/"plots"

    # Plot the distance of earth and moon for the 3 integrators
    timeline = np.linspace(0, duration, int(duration/timestep)+1)
    plt.plot(timeline, dists_e_s_euler, '-',
             label="Euler")
    plt.plot(timeline, dists_e_s_symp_euler, '-',
             label="Symplectic Euler")
    plt.plot(timeline, dists_e_s_vv, '-',
             label="Velocity Verlet")
    plt.xlabel(r"$t$ in years")
    plt.ylabel(r"$\Delta r_{earh/moon}$ in au")
    plt.legend(loc="upper left")
    plt.savefig(plot_path/"Exc1_Plot_3_3_part1.png")
    plt.clf()

    # Trajectories plot for symplectic euler
    traject_sympl_euler = ex_3_2.run(
    x, v, timestep, 17.5, m, g, ex_3_2.step_symplectic_euler).transpose()
    for i in range(np.shape(traject_sympl_euler)[1]):
        plt.plot(
            traject_sympl_euler[0, i, :],
            traject_sympl_euler[1, i, :],
            "-",
            label=names[i].decode("UTF-8"),
        )
    # Plot point for earth
    plt.scatter(
        traject_sympl_euler[0, 1, -1],
        traject_sympl_euler[1, 1, -1],
        label=names[1].decode("UTF-8"),
    )
    # Plot point for moon
    plt.scatter(
        traject_sympl_euler[0, 2, -1],
        traject_sympl_euler[1, 2, -1],
        label=names[2].decode("UTF-8"),
    )
    plt.xlabel(r"$x$ in au")
    plt.ylabel(r"$y$ in au")
    plt.legend(loc="lower right")
    plt.savefig(plot_path/"Exc1_Plot_3_3_part2.png")
