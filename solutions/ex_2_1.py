#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def force(mass, gravity):
    return np.array([0, -mass * gravity])

def step_euler(x, v, dt, mass, gravity, f):
    x = x + v * dt
    v = v + f / mass * dt
    return x, v

def run(x, v, dt, mass, gravity):
    trajectory = [x.copy()]
    for step in range(int(10e4)):
        x, v = step_euler(x, v, dt, mass, gravity, force(mass, gravity))
        if x[1] >= 0:
            trajectory.append(x.copy())
        else:
            break
    return np.array(trajectory)

if __name__ == "__main__":
    mass = 20.0
    gravity = 9.81
    x = np.array([0, 0])
    v = np.array([60, 60])
    dt = 0.1

    trajectory = run(x, v, dt, mass, gravity)

    plt.plot(trajectory[:,0], trajectory[:,1], '-')
    plt.xlabel(r"$x$ in m")
    plt.ylabel(r"$y$ in m")
    plt.show()