import numpy as np

def lj_force(positions, epsilon, sigma):
    """
    Compute Lennard-Jones forces for a system of particles in 2D.

    Parameters:
        positions (np.ndarray): 2 x N array of particle positions.
        epsilon (float): Depth of the potential well.
        sigma (float): Distance at which potential is zero.

    Returns:
        np.ndarray: 2 x N array of force vectors acting on each particle.
    """
    num_particles = positions.shape[1]
    forces = np.zeros_like(positions)  # Initialize forces array (2 x N)

    for i in range(num_particles):
        for j in range(i + 1, num_particles):  # Only compute upper triangle (no self-interaction)
            # Displacement vector
            rij = positions[:, j] - positions[:, i]
            # Distance
            r = np.linalg.norm(rij)
            if r > 0:  # Avoid division by zero
                # Lennard-Jones force magnitude
                f_magnitude = 24 * epsilon * (2 * (sigma / r)**12 - (sigma / r)**6) / r**2
                # Force vector
                f_vector = f_magnitude * rij / r
                # Apply equal and opposite forces
                forces[:, i] -= f_vector
                forces[:, j] += f_vector

    return forces

# Example usage
N = 5  # Number of particles
positions = np.random.rand(2, N)  # Random 2D positions
epsilon = 1.0  # Example epsilon
sigma = 1.0  # Example sigma

forces = lj_force(positions, epsilon, sigma)
print("Forces on particles:\n", forces)
