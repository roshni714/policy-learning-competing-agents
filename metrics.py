import numpy as np


def mse(emp_gradients, exp_gradients):
    emp_gradients = np.array(emp_gradients)
    exp_gradients = np.array(exp_gradients)

    return np.sqrt(np.sum((emp_gradients - exp_gradients) ** 2)).item()
