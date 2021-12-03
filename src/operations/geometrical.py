import numpy as np
import pandas as pd
from scipy.optimize import minimize

def evaluateDeviation(ptsMeas, ptsRef, dofs):
    # inicializando array com parâmetros a serem manipulados durante as iterações da minimização
    params = np.zeros(len(dofs))

    # aplicando a operação de minimização para achar os parâmetros de transformação
    deviation = minimize(fun=calculateEuclidianDistance, x0=params, args=(ptsMeas, ptsRef, dofs),\
                            method='SLSQP', options={'ftol': 1e-10, 'disp': False})['x']

    # invertendo o sinal do resultado p/ adequação
    deviation = [dof*(-1) for dof in deviation]

    return deviation

def calculateEuclidianDistance(params, *args):
    x0 = args[0]
    x_ref = args[1]
    dofs = args[2].copy()
    dofs_backup = args[2]

    x0 = np.array(x0)
    x_ref = np.array(x_ref)

    (Tx, Ty, Tz, Rx, Ry, Rz) = (0, 0, 0, 0, 0, 0)
    # ** assume-se que os parametros estão ordenados
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

    # inicializando variável para cálculo do(s) valor a ser minimizado
    diff = []

    for i in range(np.shape(x0)[0]):

        rot_z = np.array([[np.cos(Rz*10**-3), -np.sin(Rz*10**-3), 0],
                            [np.sin(Rz*10**-3), np.cos(Rz*10**-3), 0], [0, 0, 1]])
        rot_y = np.array([[np.cos(Ry*10**-3), 0, np.sin(Ry*10**-3)],
                            [0, 1, 0], [-np.sin(Ry*10**-3), 0, np.cos(Ry*10**-3)]])
        rot_x = np.array([[1, 0, 0], [0, np.cos(
            Rx*10**-3), -np.sin(Rx*10**-3)], [0, np.sin(Rx*10**-3), np.cos(Rx*10**-3)]])
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

def calculateMagnetsRelativeDeviations(deviationDF):
    relDev = deviationDF.copy()
    relDevList = []

    k = 0

    while k < len(relDev.iloc[:,0]):
        currGirder = relDev[k].split('-')[0] + '-' + relDev[k].split('-')[1]

        try:
            nextGirder = relDev[k+1].split('-')[0] + '-' + relDev[k+1].split('-')[1]
        except IndexError:
            nextGirder = None
        
        if(currGirder == nextGirder):
            k += 2
            continue
        elif (nextGirder):
            diff = [(relDev.at[k+1, 'Tx'] - relDev.at[k, 'Tx']), (relDev.at[k+1, 'Ty'] - relDev.at[k, 'Ty'])]
            deviation = pd.DataFrame([diff], columns=['Magnet', 'Tx', 'Ty'])
            relDevList.append(deviation)
        
        k += 1

    deviations = pd.concat(relDevList, ignore_index=True)
    deviations = deviations.set_index('Magnet')

    # retorna o df com os termos no tipo numérico certo
    return deviations.astype('float32')



