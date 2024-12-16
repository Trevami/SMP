import argparse
import gzip
import pickle

import numpy as np

# SYSTEM CONSTANTS
# timestep
DT = 0.01
# length of run
TIME_MAX = 2000.0
# desired temperature
T = 0.3
# total number of particles
N = 50
# friction coefficient
GAMMA_LANGEVIN = 0.8
# number of steps to do before the next measurement
MEASUREMENT_STRIDE = 50


parser = argparse.ArgumentParser()
parser.add_argument('id', type=int, help='Simulation id')
args = parser.parse_args()


def compute_temperature(v):
    T = 2*compute_energy(v) / (3*N)
    return T

def compute_energy(v):
    return (v * v).sum() / 2.

def step_vv(x, v, f, dt):
    
    # update positions
    x += v * dt + 0.5 * f * dt * dt

    # half update of the velocity
    v += 0.5 * f * dt

    # for this excercise no forces from other particles
    f = np.zeros_like(x)

    # second half update of the velocity
    v += 0.5 * f * dt

    return x, v, f


def step_vv_langevin(x, v, f, dt, gamma):
    sigma = np.sqrt(2.0 * T * gamma / dt)

    x += v * dt * (1 - dt*gamma/2.0) + 0.5 * f * dt * dt

    # half update of the velocity

    v = (v * (1 - dt*gamma/2.0) + dt*f/2.0)/(1+dt*gamma/2.0)

    # for this excercise no forces from other particles
    f = uniform(sigma, v.shape)

    # second half update of the velocity
    v += 0.5 * f * dt / (1+dt*gamma/2.0)

    return x, v, f

def uniform(sigma, s):
    return np.random.uniform(-sigma*np.sqrt(3), sigma*np.sqrt(3), size=s)


# SET UP SYSTEM OR LOAD IT
print("Starting simulation...")
t = 0.0
step = 0

# random particle positions
x = np.random.random((N, 3))
v = np.zeros((N, 3))

# variables to cumulate data
ts = []
Es = []
Tms = []
vels = []
abs_vels = []
traj = []


# main loop
f = np.zeros_like(x)

import copy

print(f"Simulating until tmax={TIME_MAX}...")

while t < TIME_MAX:
    x, v, f = step_vv_langevin(x, v, f, DT, GAMMA_LANGEVIN)

    t += DT
    step += 1



    if step % MEASUREMENT_STRIDE == 0:
        E = compute_energy(v)
        Tm = compute_temperature(v)
        vels.append(copy.copy(v))
        traj.append(copy.copy(x))

        # print(f"t={t}, E={E}, T_m={Tm}")
        abs_vels.append(np.linalg.norm(v, axis=1))
        ts.append(t)
        Es.append(E)
        Tms.append(Tm)



import matplotlib.pyplot as plt

plt.plot(ts, Tms)
plt.xlabel('t')
plt.ylabel('T') 
plt.savefig(f'{args.id}_T.png')
plt.show()

from scipy.stats import maxwell
import scipy

Tmean = np.mean(Tms)  # average temperature
abs_vels = np.array(abs_vels).flatten()

plt.hist(abs_vels, bins=50, density=True, color='g')

kb = scipy.constants.Boltzmann

def maxwell(x, T):
    return 4*np.pi* (1/(2*np.pi*T))**(3/2) * x**2 * np.exp(-1*x**2/(2*T))

plt.plot(np.linspace(0, 3, 100), maxwell(np.linspace(0, 3, 100), Tmean), color='r')

# Plot the Maxwell-Boltzmann distribution on the same graph
plt.title('Distribution of Avg Particle Velocity and Maxwell-Boltzmann Distribution')
plt.xlabel('Velocity')
plt.ylabel('Probability Density')
plt.legend(['Maxwell-Boltzmann', 'Avg Particle Velocity'])
plt.show()

# at the end of the simulation, write out the final state
datafilename = f'{args.id}.dat.gz'
print(f"Writing simulation data to {datafilename}.")
vels = np.array(vels)
traj = np.array(traj)

datafile = gzip.open(datafilename, 'wb')
data_dict = {
    'N': N,
    'T': T,
    'GAMMA_LANGEVIN': GAMMA_LANGEVIN,
    'x': x,
    'v': v,
    'ts': ts,
    'Es': Es,
    'Tms': Tms,
    'vels': vels,
    'traj': traj
}
pickle.dump(data_dict, datafile)
datafile.close()

print("Finished simulation.")
