import numpy as np


def normalized_gaussian_neighborhood(p, step_size):
    """
    normalized the gaussian neighborhood as a unit vector
    ---
    input
    - p: 1D array of values in [0,1] representing the current point in the search space
    - step_size: float, the size of the step to take
    ---
    output
    - Neighbor: 1D array of values in [0,1] representing the new point in the search space
    """
    
    z = np.random.normal(0.0, 1.0, size=p.shape)
    z_norm = np.linalg.norm(z)
    if z_norm == 0:
        return p.copy()
    p_new = p + step_size * (z / z_norm)
    return reflection_bound(p_new)

def reflection_bound(p):
    """
    Reflects values in p to ensure they are within the [0,1] bounds.
    ---
    input
    - p: 1D array of values that may be outside the [0,1] range
    ---
    output
    - p_reflected: 1D array of values where any value < 0 is reflected to -value, 
                    and any value > 1 is reflected to 2 - value, ensuring 
                    all values are within [0,1].
    """
    
    return 1 - np.abs(1 - np.mod(p, 2))