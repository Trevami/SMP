import numpy as np
import pprint

x = np.array([
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1]
])

n = x.shape[1]  # number of particles
r_cut = 1.5
shift = 0.016316891136

r_ij_matrix = np.repeat([x.transpose()], n, axis=0)
r_ij_matrix -= np.transpose(r_ij_matrix, axes=[1, 0, 2])

e_pot_ij_matrix = np.zeros((n, n))

r = np.linalg.norm(r_ij_matrix, axis=2)
with np.errstate(all='ignore'):
    e_pot_ij_matrix = np.where((r != 0.0) & (r < r_cut),
                                    4.0 * (np.power(r, -12.) - np.power(r, -6.)) + shift, 0.0)

pprint.pprint(e_pot_ij_matrix)