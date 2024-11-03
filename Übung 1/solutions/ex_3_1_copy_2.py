#!/usr/bin/env python3

import numpy as np
import scipy.constants
import matplotlib.pyplot as plt

from pathlib import Path

# Scheint richtig zu sein, Taschenrechner liefert das selbe Ergebnis
def force(x_ij, m_i, m_j, g):
    return -g * m_i * m_j * x_ij / np.linalg.norm(x_ij)**3

# Ist equivalent zu einer Matrix, die vollstäding zu berechent wurde
# Überüft für euler_step(300)
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
    return F_mat

def step_euler(x, v, dt, masses, g, forces):
    for i in range(np.shape(x)[0]):
        x[i] = x[i] + v[i] * dt
        # Irgendwas stimmt mit den Geschwindigkeiten nicht, Kräfte passen 1:1 zu test_ex_3_1
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

    for i in range(1):
        x, v = step_euler(x, v, 1e-4, m, g, forces(x, m, g))

    # Test:
    x_new_expected = np.array([[0.00000000e+00, 1.00000000e+00, 1.00256267e+00, 1.52410000e+00, 7.23000000e-01, 5.20300000e+00],
                            [0.00000000e+00, 6.28318531e-04, 6.49535585e-04, 5.23250598e-04, 7.40223744e-04, 2.75644293e-04]]).transpose()
    v_new_expected = np.array([[1.69985657e-07, -3.91953720e-03, -5.72474203e-03, -1.69684599e-03, -7.54100118e-03, -1.45620182e-04],
                            [0.00000000e+00, 6.28318531e+00, 6.49535585e+00, 5.23250598e+00, 7.40223744e+00, 2.75644293e+00]]).transpose()
    print(v - v_new_expected)
    # print(x - x_new_expected)

    # Calculate trajectories of planets
    # trajectories = run(x, v, 10e-4, m, g).transpose()

    # Trajectories plot
    # for i in range(np.shape(trajectories)[1]):
    #     plt.plot(trajectories[0,i,:], trajectories[1,i,:], "-",
    #              label=names[i].decode('UTF-8'))
    # plt.xlabel(r"$x$ in au")
    # plt.ylabel(r"$y$ in au")
    # plt.legend(loc="lower right")
    # plt.show()


