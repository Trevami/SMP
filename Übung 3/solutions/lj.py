#!/usr/bin/env python3

# introduce classes to the students
class Simulation:
    def __init__(self, dt, x, v, box, r_cut, shift):
        self.dt = dt
        self.x = x.copy()
        self.v = v.copy()
        self.box = box.copy()
        self.r_cut = r_cut
        self.shift = shift

        self.n_dims = self.x.shape[0]
        self.n = self.x.shape[1]  # number of particles
        self.f = np.zeros_like(x)

        # both r_ij_matrix and f_ij_matrix are computed in self.forces()
        self.r_ij_matrix = np.zeros((self.n, self.n, self.n_dims))
        self.f_ij_matrix = np.zeros((self.n, self.n, self.n_dims))
        # computed in e_pot_ij_matrix
        self.e_pot_ij_matrix = np.zeros((self.n, self.n))

    def distances(self):
        self.r_ij_matrix = np.repeat([self.x.transpose()], self.n, axis=0)
        self.r_ij_matrix -= np.transpose(self.r_ij_matrix, axes=[1, 0, 2])
        # minimum image convention
        image_offsets = self.r_ij_matrix.copy()
        for nth_box_component, box_component in enumerate(self.box):
            image_offsets[:, :, nth_box_component] = \
                np.rint(
                    image_offsets[:, :, nth_box_component] / box_component) * box_component
        self.r_ij_matrix -= image_offsets

    def energies(self):
        r = np.linalg.norm(self.r_ij_matrix, axis=2)
        with np.errstate(all='ignore'):
            # e_pot_ij_matrix is a N x N matrix with interaction energies
            # i is the row index and j is the column index
            self.e_pot_ij_matrix = np.where((r != 0.0) & (r < self.r_cut),
                                            4.0 * (np.power(r, -12.) - np.power(r, -6.)) + self.shift, 0.0)

    def forces(self):
        # first update the distance vector matrix, obeying minimum image convention
        self.distances()
        self.f_ij_matrix = self.r_ij_matrix.copy()
        r = np.linalg.norm(self.r_ij_matrix, axis=2)
        with np.errstate(all='ignore'):
            fac = np.where((r != 0.0) & (r < self.r_cut),
                           4.0 * (12.0 * np.power(r, -13.) - 6.0 * np.power(r, -7.)), 0.0)
        for dim in range(self.n_dims):
            with np.errstate(invalid='ignore'):
                self.f_ij_matrix[:, :, dim] *= np.where(r != 0.0, fac / r, 0.0)
        self.f = np.sum(self.f_ij_matrix, axis=0).transpose()

    def cap_force(self, cap_value):
        # Initializes forces if not done allready
        if not np.any(self.f):
            self.forces()
        if cap_value:
            N = self.f.shape[1]
            for i in range(N):
                force_norm = np.linalg.norm(self.f[:, i])
                if force_norm > cap_value:
                    scale_factor = cap_value / force_norm
                    self.f[:, i] *= scale_factor

    def energy(self):
        """Compute and return the energy components of the system."""
        # compute energy matrix
        self.energies()

        # compute interaction energy from self.e_pot_ij_matrix
        N = self.x.shape[1]
        E_pot = 0.0
        for i in range(1, N):
            for j in range(i):
                E_pot += self.e_pot_ij_matrix[i, j]

        # calculate kinetic energy from the velocities self.v
        E_kin = self.kin_energy()

        # return both energy components
        return E_pot, E_kin

    def kin_energy(self):
        N = self.x.shape[1]
        E_kin = 0.0
        for i in range(N):
            E_kin += 0.5 * np.dot(self.v[:, i], self.v[:, i])
        return E_kin

    def temperature(self):
        # temerature in multiples of k_B
        E_kin = self.kin_energy()
        N = self.x.shape[1]
        return E_kin / N

    def thermostat(self, temperature, new_temperature):
        if new_temperature:
            self.v *= np.sqrt(new_temperature / temperature)

    def pressure(self):
        E_kin = self.kin_energy()
        N = self.x.shape[1]
        E_force = 0.0
        for i in range(1, N):
            for j in range(i):
                E_force += np.dot(self.f_ij_matrix[i, j],
                                  self.r_ij_matrix[i, j])
        # pressure calculation
        return 1/(2 * self.box[0] * self.box[1]) * (E_kin + E_force)

    def rdf(self, num_bins, density, distance_range):
        N = self.x.shape[1]
        distances = []  # list for histogram bining
        
        # get list of distances
        for i in range(1, N):
            for j in range(i):
                r_ij = self.r_ij_matrix[i, j]
                distances.append(np.linalg.norm(r_ij))
        distances = np.array(distances)
        
        # calculate concentric shell amplitudes as well as the radii r_in and r_out
        bin_edges = np.linspace(distance_range[0], distance_range[1], num_bins + 1)
        bins, _ = np.histogram(distances, bins=bin_edges)
        
        # calculate the r vlaues as bin midpoints
        r_values = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # 2D -> areas of concentric shells pi(r_out^2 - r_in^2)
        shell_areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)

        # output of normalized rdf
        # factor 2 added as the distance list only accounts for half of the distances
        # -> r_ij distance matrix is symmetric
        return r_values, 2 * bins / (density * shell_areas * N)

    def propagate(self):
        # update positions
        self.x += self.v * self.dt + 0.5 * self.f * self.dt * self.dt

        # half update of the velocity
        self.v += 0.5 * self.f * self.dt

        # compute new forces
        self.forces()
        # we assume that all particles have a mass of unity

        # second half update of the velocity
        self.v += 0.5 * self.f * self.dt


def write_checkpoint(state, path, overwrite=False):
    if os.path.exists(path) and not overwrite:
        raise RuntimeError("Checkpoint file already exists")
    with open(path, 'wb') as fp:
        pickle.dump(state, fp)


if __name__ == "__main__":
    import argparse
    import pickle
    import itertools
    import logging

    import os.path
    from pathlib import Path
    import matplotlib.pyplot as plt

    import numpy as np
    import scipy.spatial  # todo: probably remove in template
    import tqdm

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'N_per_side',
        type=int,
        help='Number of particles per lattice side.')
    parser.add_argument(
        '--cpt',
        type=str,
        help='Path to checkpoint.')
    parser.add_argument(
        '--tstps',
        type=int,
        help='Path to checkpoint.')
    parser.add_argument(
        '--tst',
        type=float,
        help='Thermostat temperature.')
    parser.add_argument(
        '--fc',
        type=float,
        help='Force cap for warmup.')
    args = parser.parse_args()

    # np.random.seed(2)

    DT = 0.01
    T_MAX = 10.0
    # T_MAX = 0.1
    N_TIME_STEPS = args.tstps if args.tstps else int(T_MAX / DT)

    R_CUT = 2.5
    SHIFT = 0.016316891136

    DIM = 2
    DENSITY = 0.316
    N_PER_SIDE = args.N_per_side
    N_PART = N_PER_SIDE**DIM
    VOLUME = N_PART / DENSITY
    BOX = np.ones(DIM) * VOLUME**(1. / DIM)

    THERMOSTAT_TEMP = args.tst if args.tst else 0.0
    FORCE_CAP = args.fc if args.fc else None
    RANDOM_PARTICLE_POS = True
    PLOT_INIT_PARTICLE_POS = True

    RDF_BIN_NUM = 100
    RDF_RANGE = (0.8, 5.0)

    SAMPLING_STRIDE = 3

    if not args.cpt or not os.path.exists(args.cpt):
        logging.info("Starting from scratch.")
        # particle positions
        if RANDOM_PARTICLE_POS:
            x = np.random.uniform(
                low=[0, 0], high=[BOX[0], BOX[1]], size=(N_PER_SIDE**2, 2)).T
        else:
            x = np.array(list(itertools.product(np.linspace(
                0, BOX[0], N_PER_SIDE, endpoint=False), np.linspace(0, BOX[1], N_PER_SIDE, endpoint=False)))).T
        # save positions plot
        if PLOT_INIT_PARTICLE_POS:
            plot_path = Path(__file__).resolve().parent.parent/'plots'
            fig = plt.scatter(x[0, :], x[1, :])
            plt.title("Initial Particle Positions")
            plt.xlabel(r"$x$-postions in a. u.")
            plt.ylabel(r"$y$-postions in a. u.")
            plt.tight_layout()
            plt.savefig(plot_path/f"{args.cpt}_init_pos.svg")

        # random particle velocities
        v = 0.5*(2.0 * np.random.random((DIM, N_PART)) - 1.0)

        positions = []
        energies = []
        pressures = []
        temperatures = []
        rdfs = []
    elif args.cpt and os.path.exists(args.cpt):
        logging.info("Reading state from checkpoint.")
        with open(args.cpt, 'rb') as fp:
            data = pickle.load(fp)
            # load from file for checkpoint
            x = data['last_positions']
            v = data['last_velocities']

    # create simulation onject
    sim = Simulation(DT, x, v, BOX, R_CUT, SHIFT)

    # load data from picle file
    # if checkpoint is used, also the forces have to be reloaded!
    if args.cpt and os.path.exists(args.cpt):
        # load from file for checkpoint to assign to simulation object
        sim.f = data['last_forces']
        # load from file for appending observables data
        positions = data['positions']
        energies = data['energies']
        pressures = data['pressures']
        temperatures = data['temperatures']
        rdfs = data['rdfs']

    # Perform warmup cycles
    warmup_performed = False
    if FORCE_CAP:
        force_cap_val = FORCE_CAP
        for i in range(int(1e3)):
            sim.propagate()
            sim.cap_force(force_cap_val)
            if np.max(np.linalg.norm(sim.f, axis=0)) > force_cap_val:
                warmup_performed = True
                curr_temperature = sim.temperature()
                sim.thermostat(curr_temperature, THERMOSTAT_TEMP)

                print("Warmup cycle {:<4}  E: {:<.2E}  P: {:<.2E}  T: {:<.2E}".format(
                    i, sim.pressure(), np.sum(sim.energy()), curr_temperature), end=" | ")
                print("Max f norm: {:<.2E}  Force cap: {:<.2E}".format(
                    np.max(np.linalg.norm(sim.f, axis=0)), force_cap_val))
                force_cap_val *= 1.1
            else:
                break

    # Perform simulation
    for i in tqdm.tqdm(range(N_TIME_STEPS)):
        sim.propagate()

        if i % SAMPLING_STRIDE == 0:
            positions.append(sim.x.copy())

            pressures.append(sim.pressure())
            energies.append(sim.energy())

            curr_temperature = sim.temperature()
            temperatures.append(curr_temperature)
            sim.thermostat(curr_temperature, THERMOSTAT_TEMP)

            rdfs.append(sim.rdf(RDF_BIN_NUM, DENSITY, RDF_RANGE))

    # Save data to plickle file
    if args.cpt:
        state = {
            'warmup': warmup_performed,
            'delta_time': DT,
            'last_positions': sim.x,
            'last_velocities': sim.v,
            'last_forces': sim.f,
            'positions': positions,
            'energies': energies,
            'pressures': pressures,
            'temperatures': temperatures,
            'rdfs': rdfs
        }
        write_checkpoint(state, args.cpt, overwrite=True)
