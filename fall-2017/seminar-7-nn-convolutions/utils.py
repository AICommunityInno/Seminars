import numpy as np

def gaussian(shape, mean_x, mean_y, sigma_x, sigma_y):
    a = np.fromfunction(lambda y, x: -((x-mean_x) ** 2 / (2*sigma_x**2) + \
                                          (y-mean_y) ** 2 / (2*sigma_y**2)), shape)
    return np.exp(a)