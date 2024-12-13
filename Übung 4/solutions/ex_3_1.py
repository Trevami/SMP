import numpy as np
from tqdm import tqdm
import pickle

import matplotlib.pyplot as plt

def step_vv(x, v, f, dt):
    """
    Perform a single step of the velocity Verlet algorithm
    x: position
    v: velocity
    f: force
    dt: time step
    """

    # Update position
    x += v*dt + 0.5*f*dt**2

    # Half update of velocity
    v += 0.5*f*dt

    # Update force
    f = np.zeros_like(f)

    # Second half update of velocity
    v += 0.5*f*dt

    return x, v, f


def step_vv_langevin(x, v, f, dt, gamma, T):
    """Perform a single tep of the Langevin dynamics algorithm

    Args:
        x (np.ndarray): position
        v (np.ndarray): velocity
        f (np.ndarray): forces
        dt (float): time step
        gamma (float): firction coefficient
        T (float): temperature

    Returns:
        tuple: updated postion, updated velocitiy, updated force
    """
    

    # Generate random numbers with scaled normal destribution
    mean, std = 0.0, np.sqrt(2 * T * gamma / dt)
    w = np.zeros_like(f)
    for i in range(np.shape(f)[0]):
        w[i, :] = np.random.normal(mean, std, 3)

    # calculate the total force
    f += w # scale force by random norm. dist. number

    # tranform velocity for Langevin thermostat (LT)
    v *= (1 - dt * gamma / 2)

    # Update position
    x += v * dt + 0.5 * f * dt**2

    # Half update of velocity
    b = (1 + dt * gamma / 2) # LT velocity update denominator
    v /= b 
    v += 0.5 * f * dt / b

    # Update force
    f = np.zeros_like(f)

    # Second half update of velocity
    v += 0.5 * f * dt / b

    return x, v, f
    

def initialize_system(n_particles):
    """
    Initialize the system
    n_particles: number of particles
    """

    x = np.zeros((n_particles, 3))
    v = np.zeros((n_particles, 3))
    f = np.zeros((n_particles, 3))

    return x, v, f

def compute_instaneous_temperature(v) -> float:
    """Calculate instaneous temperature

    Args:
        v (np.ndarray): velocity

    Returns:
        float: temperature
    """
    N = v.shape[1]

    # calculate E_kin
    E_kin = 0.0
    for i in range(N):
        E_kin += 0.5 * np.dot(v[:, i], v[:, i])

    # return temperature
    return 2 * E_kin / (3 * N)

if __name__ == "__main__":
    # System parameters
    N_PARTICLES = 100
    T_INT = 200.0
    DT = 0.01

    GAMMA = 1.0
    T = 1.0

    # Lists for observables
    temperature = []
    particle_speeds = []
    particle_velocities = []
    particle_positions = []

    # Initialize the system
    x, v, f = initialize_system(N_PARTICLES)

    # Perform time integration
    for i in tqdm(range(int(T_INT/DT))):
        x, v, f = step_vv_langevin(x, v, f, DT, GAMMA, T)
        temperature.append(compute_instaneous_temperature(v))
        particle_velocities.append(v.copy())
        particle_speeds.append(np.linalg.norm(v, axis=1))
        particle_positions.append(x.copy())

    # Write the observables to file
    data = {
        "temperature": np.array(temperature),
        "particle_speeds": np.array(particle_speeds),
        "particle_velocities": np.array(particle_velocities),
        "particle_positions": np.array(particle_positions),
        "DT": DT,
        "T": T,
    }
    pickle.dump(data, open("data.pkl", "wb"))


    plt.plot(np.linspace(0, T_INT*10, np.shape(temperature)[0]), temperature)
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.show()
