import numpy as np
import pprint

x = np.array([
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1]
])

N = x.shape[1]  # number of particles
r_cut = 2.5
shift = 0.016316891136

r_ij_matrix = np.repeat([x.transpose()], N, axis=0)
r_ij_matrix -= np.transpose(r_ij_matrix, axes=[1, 0, 2])

e_pot_ij_matrix = np.zeros((N, N))

r = np.linalg.norm(r_ij_matrix, axis=2)
with np.errstate(all='ignore'):
    e_pot_ij_matrix = np.where((r != 0.0) & (r < r_cut),
                                    4.0 * (np.power(r, -12.) - np.power(r, -6.)) + shift, 0.0)

E_pot = 0.0
for i in range(1, N):
    for j in range(i):
        E_pot += e_pot_ij_matrix[i, j]
# print(E_pot)

DIM = 2
DENSITY = 0.316
N_PER_SIDE = 5
N_PART = N_PER_SIDE**DIM
VOLUME = N_PART / DENSITY
BOX = np.ones(DIM) * VOLUME**(1. / DIM)
print(BOX)