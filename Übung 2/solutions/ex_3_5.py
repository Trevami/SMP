#!/usr/bin/env python3

import itertools

import numpy as np
import time

import ex_3_4

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import csv
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'N_per_side',
        type=int,
        nargs='?',
        default=10,
        help='Number of particles per lattice side (default: 10).')
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
    x_pos = np.linspace(0, BOX[0], N_PER_SIDE, endpoint=False)
    y_pos = np.linspace(0, BOX[1], N_PER_SIDE, endpoint=False)
    grid_x, grid_y = np.meshgrid(x_pos, y_pos)

    # stack grid points into a (N_PART, 2) array
    x = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).T

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
    fieldnames = ["N_PER_SIDE", "N_PART", "CALCTIME"]
    with open("ex_3_5-results.csv", "a") as csv_file:
        file_is_empty = os.stat("ex_3_5-results.csv").st_size == 0
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if file_is_empty:
            writer.writeheader()
        writer.writerow({"N_PER_SIDE":N_PER_SIDE,"N_PART":N_PART, "CALCTIME": end_time - start_time})
    print(f"{N_PART}\t{end_time - start_time}")
