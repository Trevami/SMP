#!/usr/bin/env python3

import numpy as np

import ex_3_2


def lj_force(r_ij, r_cut):
    if np.linalg.norm(r_ij) < r_cut:
        return ex_3_2.lj_force(r_ij)
    else: return np.zeros_like(r_ij)


def forces(x: np.ndarray, r_cut: float, box=(15, 15)) -> np.ndarray:
    """Compute and return the forces acting onto the particles,
    depending on the positions x."""
    N = x.shape[1]
    f = np.zeros_like(x)
    pbc_box = np.array(box).T
    for i in range(1, N):
        for j in range(i):
            # distance vector
            r_ij = minimum_image_vector(x[:, j], x[:, i], pbc_box)
            # apply cutoff for LJ-potential forces:
            f_ij = lj_force(r_ij, r_cut)
            f[:, i] -= f_ij
            f[:, j] += f_ij
    return f


def lj_potential(r_ij, r_cut, shift):
    if np.linalg.norm(r_ij) < r_cut:
        E_pot = ex_3_2.lj_potential(r_ij)
        if shift == None:
            # create dummy vector for LJ-potential shifting
            dummy_vec = np.array([0, r_cut])
            # calculate and shift the LJ-potential to be continous at cutoff
            E_pot -= ex_3_2.lj_potential(dummy_vec)
        elif shift:
            E_pot += shift
    return E_pot


def total_energy(x: np.ndarray, v: np.ndarray, r_cut: float, shift=None, box=(15, 15)) -> np.ndarray:
    """Compute and return the total energy of the system with the
    particles at positions x and velocities v."""
    N = x.shape[1]
    E_pot = 0.0
    E_kin = 0.0
    pbc_box = np.array(box).T
    # sum up potential energies
    for i in range(1, N):
        for j in range(i):
            # distance vector
            r_ij = minimum_image_vector(x[:, j], x[:, i], pbc_box)
            # apply cutoff for LJ-potential:
            E_pot += lj_potential(r_ij, r_cut, shift)

    # sum up kinetic energy
    for i in range(N):
        E_kin += 0.5 * np.dot(v[:, i], v[:, i])
    return E_pot + E_kin


def minimum_image_vector(xj, xi, pbc_box):
    r_ij = xj - xi
    # apply PCB:
    # For each k, adjust the distance r_ij,k​ so that it
    # is the shortest distance between particles i and j
    r_ij -= pbc_box * np.round(r_ij / pbc_box)
    return r_ij


def step_vv(x: np.ndarray, v: np.ndarray, f: np.ndarray, dt: float, r_cut: float, box=(15, 15)):
    # update positions
    x += v * dt + 0.5 * f * dt * dt
    # half update of the velocity
    v += 0.5 * f * dt

    # compute new forces
    f = forces(x, r_cut, box)
    # we assume that all particles have a mass of unity

    # second half update of the velocity
    v += 0.5 * f * dt

    return x, v, f


def apply_bounce_back(x: np.ndarray, v: np.ndarray, box_cent_pos=(0, 0), box_l=15):
    # invert x-component if particle leaves boundaries of the box in x:
    x_idx_left_out = np.where(x[0, :] <= box_cent_pos[0] - box_l / 2)
    x_idx_right_out = np.where(x[0, :] >= box_cent_pos[0] + box_l / 2)
    v[0, x_idx_left_out] = -v[0, x_idx_left_out]
    v[0, x_idx_right_out] = -v[0, x_idx_right_out]
    # invert y-component if particle leaves boundaries of the box in y:
    y_idx_left_out = np.where(x[1, :] <= box_cent_pos[1] - box_l / 2)
    y_idx_right_out = np.where(x[1, :] >= box_cent_pos[1] + box_l / 2)
    v[1, y_idx_left_out] = -v[1, y_idx_left_out]
    v[1, y_idx_right_out] = -v[1, y_idx_right_out]
    return v


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    DT = 0.01
    T_MAX = 20.0
    N_TIME_STEPS = int(T_MAX / DT)
    R_CUT = 10
    PBC_BOX = (10, 10)
    WRAP_COORDS = False

    # running variables
    time = 0.0

    # particle positions
    x = np.zeros((2, 2))
    x[:, 0] = [3.9, 3.0]
    x[:, 1] = [6.1, 5.0]

    # particle velocities
    v = np.zeros((2, 2))
    v[:, 0] = [-2.0, -2.0]
    v[:, 1] = [2.0, 2.0]

    f = forces(x, R_CUT, PBC_BOX)

    N_PART = x.shape[1]

    positions = np.full((N_TIME_STEPS, 2, N_PART), np.nan)
    energies = np.full((N_TIME_STEPS), np.nan)

    # main loop
    with open('ljbillards.vtf', 'w') as vtffile:
        # write the structure of the system into the file:
        # N particles ("atoms") with a radius of 0.5
        vtffile.write(f'atom 0:{N_PART - 1} radius 0.5\n')
        for i in range(N_TIME_STEPS):
            x, v, f = step_vv(x, v, f, DT, R_CUT, PBC_BOX)
            # v = apply_bounce_back(x, v, (7.4, 0), box_l=15)
            time += DT

            # warp x coordinates inside box
            if WRAP_COORDS:
                x = np.array([r % np.array(PBC_BOX) for r in x.T]).T

            positions[i, :2] = x
            energies[i] = total_energy(x, v, R_CUT, None, PBC_BOX)

            # write out that a new timestep starts
            vtffile.write('timestep\n')
            # write out the coordinates of the particles
            for p in x.T:
                vtffile.write(f"{p[0]} {p[1]} 0.\n")

    traj = np.array(positions)

    plot_path = Path(__file__).resolve().parent.parent/"plots"
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for i in range(N_PART):
        ax1.plot(positions[:, 0, i], positions[:, 1, i], label='{}'.format(i))
    ax1.set_title('Trajectory')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x position')
    ax1.set_ylabel('y position')
    ax1.legend()

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Total energy")
    ax2.plot(energies)
    ax2.set_title('Total energy')
    plt.savefig(plot_path/"ex_3_4_plot_1.png")
    plt.show()
