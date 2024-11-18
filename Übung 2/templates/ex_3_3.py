#!/usr/bin/env python3

import numpy as np

import ex_3_2


def forces(x: np.ndarray) -> np.ndarray:
    """Compute and return the forces acting onto the particles,
    depending on the positions x."""
    N = x.shape[1]
    f = np.zeros_like(x)
    for i in range(1, N):
        for j in range(i):
            # distance vector
            r_ij = x[:, j] - x[:, i]
            f_ij = ex_3_2.lj_force(r_ij)
            f[:, i] -= f_ij
            f[:, j] += f_ij
    return f


def total_energy(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute and return the total energy of the system with the
    particles at positions x and velocities v."""
    N = x.shape[1]
    E_pot = 0.0
    E_kin = 0.0
    # sum up potential energies
    for i in range(1, N):
        for j in range(i):
            # distance vector
            r_ij = x[:, j] - x[:, i]
            E_pot += ex_3_2.lj_potential(r_ij)
    # sum up kinetic energy
    for i in range(N):
        E_kin += 0.5 * np.dot(v[:, i], v[:, i])
    return E_pot + E_kin


def step_vv(x: np.ndarray, v: np.ndarray, f: np.ndarray, dt: float):
    # update positions
    x += v * dt + 0.5 * f * dt * dt
    # half update of the velocity
    v += 0.5 * f * dt

    # compute new forces
    f = forces(x)
    # we assume that all particles have a mass of unity

    # second half update of the velocity
    v += 0.5 * f * dt

    return x, v, f


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    DT = 0.01
    T_MAX = 20.0
    N_TIME_STEPS = int(T_MAX / DT)

    # running variables
    time = 0.0

    # particle positions
    x = np.zeros((2, 5))
    x[:, 0] = [0.0, 0.0]
    x[:, 1] = [5.0, 0.3]
    x[:, 2] = [8.0, 1.8]
    x[:, 3] = [10.9, 0.3]
    x[:, 4] = [12.0, 7.0]

    # particle velocities
    v = np.zeros((2, 5))
    v[:, 0] = [2.0, 0.0]
    v[:, 1] = [0.0, 0.0]
    v[:, 2] = [0.0, 0.0]
    v[:, 3] = [0.0, 0.0]
    v[:, 4] = [0.0, 0.0]

    f = forces(x)

    N_PART = x.shape[1]

    positions = np.full((N_TIME_STEPS, 2, N_PART), np.nan)
    energies = np.full((N_TIME_STEPS), np.nan)


    # main loop
    with open('ljbillards.vtf', 'w') as vtffile:
        # write the structure of the system into the file:
        # N particles ("atoms") with a radius of 0.5
        vtffile.write(f'atom 0:{N_PART - 1} radius 0.5\n')
        for i in range(N_TIME_STEPS):
            x, v, f = step_vv(x, v, f, DT)
            time += DT

            positions[i, :2] = x
            energies[i] = total_energy(x, v)

            # write out that a new timestep starts
            vtffile.write('timestep\n')
            # write out the coordinates of the particles
            for p in x.T:
                vtffile.write(f"{p[0]} {p[1]} 0.\n")

    traj = np.array(positions)

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
    plt.show()
