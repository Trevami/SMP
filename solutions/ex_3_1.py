#!/usr/bin/env python3

import numpy as np
import scipy.constants
import matplotlib.pyplot as plt

from pathlib import Path

def force(x_ij, m_i, m_j, g):
    return -g * m_i * m_j * x_ij / np.linalg.norm(x_ij)**3

def forces(x, masses, g):
    dim = np.shape(x)
    F_mat = np.zeros((dim[0], dim[0], dim[1]), dtype="float")
    for i in range(dim[0]):
        for j in range(dim[0]):
            if j < i:
                force_xy = force(x[i] - x[j], masses[i], masses[j], g)
                # Diagonal of matrix is 0 -> planet exhibits no force on itself
                # Lower diagonal half:
                F_mat[i, j, 0] = force_xy[0] # Force x-components
                F_mat[i, j, 1] = force_xy[1] # Force y-components
                # Upper diagonal half (anitsymmetric to upper):
                F_mat[j, i, 0] = -force_xy[0] # Force x-components
                F_mat[j, i, 1] = -force_xy[1] # Force y-components
            else:
                pass
    return F_mat

def step_euler(x, v, dt, masses, g, forces):
    for i in range(np.shape(x)[0]):
        x[i] = x[i] + v[i] * dt
        v[i] = v[i] + np.sum(forces[i,:,:]) / masses[i] * dt
    return x, v

def run(x, v, dt, masses, g):
    trajectory = [x.copy()]
    for step in range(1, int(1/dt)):
        F_mat = forces(x, masses, g)
        x, v = step_euler(x, v, dt, masses, g, F_mat)
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

    # print(np.round(forces(x, m, g)[:,:,1], 3))

    # Calculate trajectories of planets
    trajectories = run(x, v, 10e-4, m, g).transpose()

    # print(v)

    # x1, v1 = step_euler(x, v, 10e-4, m, g, forces(x, m, g))

    # x_new_expected = np.array([[0.00000000e+00, 1.00000000e+00, 1.00256267e+00, 1.52410000e+00, 7.23000000e-01, 5.20300000e+00],
    #                         [0.00000000e+00, 6.28318531e-04, 6.49535585e-04, 5.23250598e-04, 7.40223744e-04, 2.75644293e-04]]).transpose()

    # print(x1 - x_new_expected)

    # Trajectories plot
    for i in range(np.shape(trajectories)[1]):
        plt.plot(trajectories[0,i,:], trajectories[1,i,:], "-",
                 label=names[i].decode('UTF-8'))
    plt.xlabel(r"$x$ in au")
    plt.ylabel(r"$y$ in au")
    plt.legend(loc="lower right")
    plt.show()


