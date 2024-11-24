#!/usr/bin/env python3

import typing

import numpy as np
import scipy.linalg

import ex_3_2
import ex_3_4


def forces(x: np.ndarray, r_cut: float, box: np.ndarray, verlet_list: np.ndarray) -> np.ndarray:
    """Compute and return the forces acting onto the particles,
    depending on the positions x."""
    N = x.shape[1]
    f = np.zeros_like(x)
    for pair in verlet_list:
            # distance vector
            r_ij = ex_3_4.minimum_image_vector(x[:, pair[0]], x[:, pair[1]], box) 
            f_ij = ex_3_4.lj_force(r_ij, r_cut)
            f[:, pair[0]] -= f_ij
            f[:, pair[1]] += f_ij
    return f


def total_energy(x: np.ndarray, v: np.ndarray, r_cut: float, shift: float, box: np.ndarray, verlet_list: np.ndarray) -> float:
    """Compute and return the total energy of the system with the
    particles at positions x and velocities v."""
    N = x.shape[1]
    E_pot = 0.0
    E_kin = 0.0
    # sum up potential energies
    for pair in verlet_list:
        # distance vector
        r_ij = ex_3_4.minimum_image_vector(x[:, pair[0]], x[:, pair[1]], box)
        E_pot += ex_3_4.lj_potential(r_ij, r_cut, shift)
    # sum up kinetic energy
    for i in range(N):
        E_kin += 0.5 * np.dot(v[:, i], v[:, i])
    return E_pot + E_kin

def get_verlet_list(x: np.ndarray, r_cut: float, skin: float, box: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Create a list of interaction partners.

    """
    N = x.shape[1]
    verlet_list = []

    # TODO: YOUR IMPLEMENTATION OF VERLET LISTS GOES HERE...

    return np.copy(x), np.array(verlet_list)

def step_vv(x: np.ndarray, v: np.ndarray, f: np.ndarray, dt: float, r_cut: float, skin: float, box: np.ndarray, x0: np.ndarray, verlet_list: np.ndarray):
    # update positions
    x += v * dt + 0.5 * f * dt * dt
    # check for maximum distance a particle moved
    max_dx = np.max(np.linalg.norm(x - x0, axis=0))
    if max_dx > 0.5 * skin:
        x0, verlet_list = get_verlet_list(x, r_cut, skin, box)

    # half update of the velocity
    v += 0.5 * f * dt

    # compute new forces
    f = forces(x, r_cut, box, verlet_list)
    # we assume that all particles have a mass of unity

    # second half update of the velocity
    v += 0.5 * f * dt

    return x, v, f, x0, verlet_list

if __name__ == "__main__":
    import argparse
    import itertools
    import tqdm
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'N_per_side',
        type=int,
        nargs=1,
        help='Number of particles per lattice side.')
    parser.add_argument(
        'skin',
        type=float,
        nargs=1,
        help='Skin for Verlet lists.')
    args = parser.parse_args()

    DT = 0.01
    T_MAX = 40.0
    N_TIME_STEPS = int(T_MAX / DT)

    R_CUT = 2.5
    SHIFT = 0.016316891136

    SKIN = args.skin[0]
    nargs=1,

    DIM = 2
    DENSITY = 0.7
    N_PER_SIDE = args.N_per_side[0]
    N_PART = N_PER_SIDE**DIM
    VOLUME = N_PART / DENSITY
    BOX = np.ones(DIM) * VOLUME**(1. / DIM)

    # particle positions
    x = np.array(list(itertools.product(np.linspace(0, BOX[0], N_PER_SIDE, endpoint=False),
                                        np.linspace(0, BOX[1], N_PER_SIDE, endpoint=False)))).T

    # random particle velocities
    v = 2.0 * np.random.random((DIM, N_PART)) - 1.0

    x0, verlet_list = get_verlet_list(x, R_CUT, SKIN, BOX)

    f = forces(x, R_CUT, BOX, verlet_list)

    positions = np.zeros((N_TIME_STEPS, DIM, N_PART))
    energies = np.zeros(N_TIME_STEPS)

    for i in tqdm.tqdm(range(N_TIME_STEPS)):
        x, v, f, x0, verlet_list = step_vv(x, v, f, DT, R_CUT, SKIN, BOX, x0, verlet_list)

        positions[i] = x
        energies[i] = total_energy(x, v, R_CUT, SHIFT, BOX, verlet_list)

    plt.plot(energies)
    plt.show()
