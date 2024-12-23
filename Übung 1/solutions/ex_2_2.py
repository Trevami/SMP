#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ex_2_1

from pathlib import Path


def force(mass, gravity, v, gamma, v_0):
    return np.array([0, -mass * gravity]) - gamma * (v - v_0)


def run(x, v, dt, mass, gravity, gamma, v_0):
    x_new = x.copy()
    v_new = v.copy()
    trajectory = [x_new.copy()]
    for timestep in range(int(1e4)):
        force_xy = force(mass, gravity, v_new, gamma, v_0)
        x_new, v_new = ex_2_1.step_euler(
            x_new, v_new, dt, mass, gravity, force_xy)
        if x_new[1] >= 0:
            trajectory.append(x_new.copy())
        else:
            break
    return np.array(trajectory)


if __name__ == "__main__":
    mass = 2.0
    gravity = 9.81
    x = np.array([0, 0])
    v = np.array([60, 60])
    dt = 0.1

    trajectory_2_1 = ex_2_1.run(x, v, dt, mass, gravity)
    trajectory_vw_0 = run(x, v, dt, mass, gravity, 0.1, np.array([0, 0]))
    trajectory_vw = run(x, v, dt, mass, gravity, 0.1, np.array([-30, 0]))

    plot_path = Path(__file__).resolve().parent.parent/"plots"
    # Trajectory comparison plot
    plt.plot(trajectory_2_1[:, 0], trajectory_2_1[:, 1], '-',
             label="without friction")
    plt.plot(trajectory_vw_0[:, 0], trajectory_vw_0[:, 1], '-',
             label=r"$v_w=0\;\frac{\text{m}}{\text{s}}$")
    plt.plot(trajectory_vw[:, 0], trajectory_vw[:, 1], '-',
             label=r"$v_w=-30\;\frac{\text{m}}{\text{s}}$")
    plt.xlabel(r"$x$ in m")
    plt.ylabel(r"$y$ in m")
    plt.legend(loc="upper right")
    plt.savefig(plot_path/"Exc1_Plot_2_2_part1.png")
    plt.clf()

    # Trajectory with different wind speeds in -x direction plot
    for vw in np.linspace(0, -195, num=8):
        trajectory_vw = run(x, v, dt, mass, gravity, 0.1, np.array([vw, 0]))
        plt.plot(trajectory_vw[:, 0], trajectory_vw_0[:, 1], '-',
                 label=r"$v_w= %.2f \;\frac{\text{m}}{\text{s}}$" % vw)
    plt.xlabel(r"$x$ in m")
    plt.ylabel(r"$y$ in m")
    plt.legend(loc="upper right")
    plt.savefig(plot_path/"Exc1_Plot_2_2_part2.png")
