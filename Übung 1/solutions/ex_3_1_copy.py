#!/usr/bin/env python3

import numpy as np
import scipy.constants
import matplotlib.pyplot as plt

from pathlib import Path

def force(r_ij, m_i, m_j, g):
    return -g * m_i * m_j * r_ij / np.linalg.norm(r_ij)**3

def step_euler(x, v, dt, mass, g, forces):
    for i in range(np.shape(x)[0]):
        x = x + v * dt
        force_x = np.sum(force[i][0])
        v = v + force_x / mass * dt
    return x, v

def forces(x, masses, g):
    dim = np.shape(x)[0]
    F_x = np.zeros((dim, dim),dtype="float")
    F_y = np.zeros((dim, dim),dtype="float")
    for i in range(np.shape(F_x)[0]):
        for j in range(np.shape(F_x)[1]):
            if i == j:
                pass
            elif j < i:
                force_xy = force(x[i] - x[j], masses[i], masses[j], g)
                F_x[i][j] = force_xy[0]
                F_x[j][i] = -force_xy[0]
                F_y[i][j] = force_xy[1]
                F_y[j][i] = -force_xy[1]
    return F_x, F_y


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "files" / "solar_system.npz"
    data = np.load(data_path)
    names = data ["names"]
    x_init = data ["x_init"]
    v_init = data ["v_init"]
    m = data ["m"]
    g = data ["g"]

    x = np.array(x_init).transpose()
    # print(x)
    print(forces(x, m, g))
