import numpy as np
import pandas as pd
from scipy.optimize import minimize

def evaluate_deviation_with_bestfit(points_meas: list, points_ref: list, dofs: list) -> list:
    """ Computes the deviation from two lists of points by the means of a least-square minimization (best-fit). """

    # init array with parameters that will be manipulated throughout the iterations
    params = np.zeros(len(dofs))

    # applying the best-fit to find the transformation resultant
    deviation = minimize(fun=calculate_euclidian_distance, x0=params, args=(points_meas, points_ref, dofs),\
                            method='SLSQP', options={'ftol': 1e-10, 'disp': False})['x']

    # needs signal invertion
    deviation = [dof*(-1) for dof in deviation]

    return deviation

def calculate_euclidian_distance(params: list, *args):
    """ Function passed to the minimizing method. It calculates the euclidian
        distance between the two lists of points and return it, which in turn 
        will be minimize over the iterations. """

    x0 = args[0]
    x_ref = args[1]
    dofs = args[2].copy()
    dofs_backup = args[2]

    x0 = np.array(x0)
    x_ref = np.array(x_ref)

    (Tx, Ty, Tz, Rx, Ry, Rz) = (0, 0, 0, 0, 0, 0)
    # here we assume that the parameters are ordered
    for param in params:
        if 'Tx' in dofs:
            Tx = param
            dofs.pop(dofs.index('Tx'))
        elif 'Ty' in dofs:
            Ty = param
            dofs.pop(dofs.index('Ty'))
        elif 'Tz' in dofs:
            Tz = param
            dofs.pop(dofs.index('Tz'))
        elif 'Rx' in dofs:
            Rx = param
            dofs.pop(dofs.index('Rx'))
        elif 'Ry' in dofs:
            Ry = param
            dofs.pop(dofs.index('Ry'))
        elif 'Rz' in dofs:
            Rz = param
            dofs.pop(dofs.index('Rz'))

    diff = []
    for i in range(np.shape(x0)[0]):

        rot_z = np.array([[np.cos(Rz*10**-3), -np.sin(Rz*10**-3), 0],
                            [np.sin(Rz*10**-3), np.cos(Rz*10**-3), 0],
                            [0, 0, 1]])
        rot_y = np.array([[np.cos(Ry*10**-3), 0, np.sin(Ry*10**-3)],
                            [0, 1, 0],
                            [-np.sin(Ry*10**-3), 0, np.cos(Ry*10**-3)]])
        rot_x = np.array([[1, 0, 0],
                          [0, np.cos(Rx*10**-3), -np.sin(Rx*10**-3)],
                          [0, np.sin(Rx*10**-3), np.cos(Rx*10**-3)]])
        ROT = rot_z @ rot_y @ rot_x
        xr = np.dot(ROT, x0[i])

        xt = xr + np.array([Tx, Ty, Tz])

        if 'Tx' in dofs_backup:
            diff.append(((x_ref[i, 0]-xt[0])**2).sum())
        if 'Ty' in dofs_backup:
            diff.append(((x_ref[i, 1]-xt[1])**2).sum())
        if 'Tz' in dofs_backup:
            diff.append(((x_ref[i, 2]-xt[2])**2).sum())

    return np.sqrt(np.sum(diff))

