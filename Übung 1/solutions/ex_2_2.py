#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ex_2_1

def force(mass, gravity, v, gamma, v_0):
    return np.array([0, -mass * gravity]) - gamma * (v - v_0)

def run(x, v, dt, mass, gravity, gamma, v_0):
    trajectory = [x.copy()]
    for step in range(int(10e4)):
        f = force(mass, gravity, v, gamma, v_0)
        x, v = ex_2_1.step_euler(x, v, dt, mass, gravity, f)
        if x[1] >= 0:
            trajectory.append(x.copy())
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

    # Trajectory comparison plot
    plt.plot(trajectory_2_1[:,0], trajectory_2_1[:,1], '-',
             label="without friction")
    plt.plot(trajectory_vw_0[:,0], trajectory_vw_0[:,1], '-',
             label=r"$v_w=0\;\frac{\text{m}}{\text{s}}$")
    plt.plot(trajectory_vw[:,0], trajectory_vw[:,1], '-',
             label=r"$v_w=-30\;\frac{\text{m}}{\text{s}}$")
    plt.xlabel(r"$x$ in m")
    plt.ylabel(r"$y$ in m")
    plt.legend(loc="upper right")
    plt.show()

    # Trajectory wiht different wind speeds in -x direction plot
    for vw in np.linspace(0, -195, num=8):
        trajectory_vw = run(x, v, dt, mass, gravity, 0.1, np.array([vw, 0]))
        plt.plot(trajectory_vw[:,0], trajectory_vw_0[:,1], '-',
                 label=r"$v_w= %.2f \;\frac{\text{m}}{\text{s}}$" % vw)
    plt.xlabel(r"$x$ in m")
    plt.ylabel(r"$y$ in m")
    plt.legend(loc="upper right")
    plt.show()