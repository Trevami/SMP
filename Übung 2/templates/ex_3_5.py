#!/usr/bin/env python3

import itertools

import numpy as np
import time

import ex_3_4

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'N_per_side',
        type=int,
        help='Number of particles per lattice side.')
    args = parser.parse_args()

    DT = 0.01
    T_MAX = 1.0
    N_TIME_STEPS = int(T_MAX / DT)

    R_CUT = 2.5
    SHIFT = 0.016316891136

    DENSITY = 0.7
    N_PER_SIDE = args.N_per_side
    N_PART = N_PER_SIDE**2
    VOLUME = N_PART / DENSITY
    BOX = np.ones(2) * VOLUME**(1. / 2.)

    # particle positions

    # SET UP THE PARTICLE POSITIONS ON A LATTICE HERE
    # x = ...

    # random particle velocities
    v = 2.0 * np.random.random((2, N_PART)) - 1.0

    f = ex_3_4.forces(x, R_CUT, BOX)

    positions = np.full((N_TIME_STEPS, 2, N_PART), np.nan)
    energies = np.full((N_TIME_STEPS), np.nan)

    start_time = time.time()

    for i in range(N_TIME_STEPS):
        x, v, f = ex_3_4.step_vv(x, v, f, DT, R_CUT, BOX)

        positions[i] = x
        energies[i] = ex_3_4.total_energy(x, v, R_CUT, SHIFT, BOX)

    end_time = time.time()
    print(f"{N_PART}\t{end_time - start_time}")
