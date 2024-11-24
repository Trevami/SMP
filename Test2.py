import numpy as np

def apply_bounce_back(x:np.ndarray, v:np.ndarray, box_l=15):
    # invert x component if particle leaves bouniding box:
    v[0, np.where(x[0, :] >= box_l)] = -v[0, np.where(x[0, :] >= box_l)]
    v[0, np.where(x[0, :] <= 0)] = -v[0, np.where(x[0, :] <= 0)]
    # invert y component if particle leaves bouniding box:
    v[1, np.where(x[1, :] >= box_l)] = -v[1, np.where(x[1, :] >= box_l)]
    v[1, np.where(x[1, :] <= 0)] = -v[1, np.where(x[1, :] <= 0)]

if __name__ == "__main__":
    x = np.array([
        [1.0, 1.0],
        [5.0, -1.0],
        [-1.0, 5.0],
        [-1.0, -1.0],
        [20.0, 5.0],
        [5.0, 20.0],
        [20.0, 20.0]
    ]).T
    v = np.array([
        [1.0, 1.0],
        [2.0, -1.0],
        [-1.0, 2.0],
        [-1.0, -1.0],
        [2.0, 1.0],
        [1.0, 2.0],
        [2.0, 2.0]
    ]).T