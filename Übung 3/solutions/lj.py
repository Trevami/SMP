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
        self.n = self.x.shape[1] # number of particles
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
                np.rint(image_offsets[:, :, nth_box_component] / box_component) * box_component
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
        # Temerature in multiples of k_B
        E_kin = self.kin_energy()
        N = self.x.shape[1]
        return 2/3 * E_kin / N

    def pressure(self):
        E_kin = self.kin_energy()
        N = self.x.shape[1]
        E_force = 0.0
        for i in range(1, N):
            for j in range(i):
                E_force += np.dot(self.f_ij_matrix[i, j], self.r_ij_matrix[i, j])
        return 1/(2 * self.box[0] * self.box[1]) * (E_kin + E_force) # pressure calculation

    def rdf(self):
        #TODO
        pass

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
    args = parser.parse_args()

    np.random.seed(2)

    DT = 0.01
    T_MAX = 10.0
    # T_MAX = 0.1
    N_TIME_STEPS = int(T_MAX / DT)

    R_CUT = 2.5
    SHIFT = 0.016316891136

    DIM = 2
    DENSITY = 0.316
    N_PER_SIDE = args.N_per_side
    N_PART = N_PER_SIDE**DIM
    VOLUME = N_PART / DENSITY
    BOX = np.ones(DIM) * VOLUME**(1. / DIM)

    SAMPLING_STRIDE = 3

    if not args.cpt or not os.path.exists(args.cpt):
        logging.info("Starting from scratch.")
        # particle positions
        x = np.array(list(itertools.product(np.linspace(0, BOX[0], N_PER_SIDE, endpoint=False),
                                            np.linspace(0, BOX[1], N_PER_SIDE, endpoint=False)))).T

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

    # Create simulation onject
    sim = Simulation(DT, x, v, BOX, R_CUT, SHIFT)

    # Load data from picle file
    # If checkpoint is used, also the forces have to be reloaded!
    if args.cpt and os.path.exists(args.cpt):
        # load from file for checkpoint
        sim.x = data['last_positions']
        sim.v = data['last_velocities']
        sim.f = data['last_forces']
        # load from file for appending observables data
        positions = data['positions']
        energies = data['energies']
        pressures = data['pressures']
        temperatures = data['temperatures']
        rdfs = data['rdfs']

    # Perform simulation
    for i in tqdm.tqdm(range(N_TIME_STEPS)):
        sim.propagate()

        if i % SAMPLING_STRIDE == 0:
            positions.append(sim.x.copy())
            pressures.append(sim.pressure())
            energies.append(np.sum(sim.energy()))
            temperatures.append(sim.temperature())
            rdfs.append(sim.rdf())

    # Save data to plikcle file
    if args.cpt:
        state = {
            'delta_time' : DT,
            'last_positions' : sim.x,
            'last_velocities' : sim.v,
            'last_forces' : sim.f,
            'positions' : positions,
            'energies': energies,
            'pressures' : pressures,
            'temperatures' : temperatures,
            'rdfs' : rdfs
        }
        write_checkpoint(state, args.cpt, overwrite=True)
