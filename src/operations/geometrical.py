import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from geom_entitites.point import Point
from geom_entitites.transformation import Transformation

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

def transformToLocalFrame(objectDict, frameDict, accelerator):
    magnetsNotComputed = ""
    for objectName in objectDict:
        magnetDict = objectDict[objectName]
            
        for magnetName in magnetDict:
            magnet = magnetDict[magnetName]
            targetFrame = magnetName + "-NOMINAL"
            try:
                magnet.transformFrame(frameDict, targetFrame)
            except KeyError:
                magnetsNotComputed += magnet.name + ','

    # at least one frame was not computed
    if (magnetsNotComputed != ""):
        #print("[Transformação p/ frames locais] Imãs não computados: " + magnetsNotComputed[:-1])
        pass

def debug_transformToLocalFrame(objectDict, frameDict, accelerator):
    magnetsNotComputed = ""
    for objectName in objectDict:
        magnetDict = objectDict[objectName]
            
        for magnetName in magnetDict:
            magnet = magnetDict[magnetName]

            separetedName = magnetName.split('-')
            sectorId = separetedName[0]
            girderId = separetedName[1]

            if (girderId != 'B05' and girderId != 'B09' and girderId != 'B07'):
                targetFrame = f"{sectorId}-{girderId}-QUAD01-NOMINAL"
            else:
                targetFrame = magnetName + "-NOMINAL"

            try:
                magnet.transformFrame(frameDict, targetFrame)
            except KeyError:
                magnetsNotComputed += magnet.name + ','

    # at least one frame was not computed
    if (magnetsNotComputed != ""):
        print("[Transformação p/ frames locais] Imãs não computados: " +
                magnetsNotComputed[:-1])
        pass

def transformToMachineLocal(girderDict, frameDict):
    for girderName in girderDict:
        magnetDict = girderDict[girderName]
        for magnetName in magnetDict:
            magnet = magnetDict[magnetName]
            magnet.transformFrame(frameDict, "machine-local")

def debug_translatePoints(girderDict, frameDict):
    output = ""

    for girderName in girderDict:
        magnetDict = girderDict[girderName]
        for magnetName in magnetDict:
            magnet = magnetDict[magnetName]
            pointDict = magnet.pointDict

            
            for pointName in pointDict:
                point = pointDict[pointName]

                cp_point = Point.copyFromPoint(point)

                sectorID = pointName.split('-')[0]
                girderID = pointName.split('-')[1]
                magnetID = pointName.split('-')[2]

                # print(f"{cp_point.name}: [{cp_point.x}, {cp_point.y}, {cp_point.z}]")

                # vertical translating
                if ('QUAD' in magnetID):
                    cp_point.y -= 0.0125
                elif ('SEXT' in magnetID):
                    cp_point.y += 0.017 + 0.0125
                elif ('B1' in magnetID):
                    cp_point.y += 0.0125

                # horizontal translating
                if ('QUAD' in magnetID):
                    if ('C02' in cp_point.name or 'C03' in cp_point.name):
                        cp_point.x -= 0.0125
                    elif ('C01' in cp_point.name or 'C04' in cp_point.name):
                        cp_point.x += 0.0125
                elif ('SEXT' in magnetID):
                    if ('C02' in cp_point.name or 'C03' in cp_point.name):
                        cp_point.x += 0.017 + 0.0125
                    elif ('C01' in cp_point.name or 'C04' in cp_point.name):
                        cp_point.x += 0.0125
                elif ('B1' in magnetID):
                    cp_point.x += 0.0125

                # applying specific shifts to B1's
                shiftsB1 = {"S01-B03-B1": -0.21939260387942738,"S01-B11-B1": 0.5027928637375751,"S02-B03-B1": -0.6566497181853421,"S02-B11-B1": 0.3949965569748386,"S03-B03-B1": -0.20433956473073067,"S03-B11-B1": 0.43980701894961527,"S04-B03-B1": -0.24083142212426623,"S04-B11-B1": 0.044734592439588994,"S05-B03-B1": -0.5419523768496219,"S05-B11-B1": 0.18519311704547903,"S06-B03-B1": 0.06556785208046989,"S06-B11-B1": 0.2463624895503429,"S07-B03-B1": -0.11493942111696498,"S07-B11-B1": 0.1979572509557599,"S08-B03-B1": -0.19108205778576348,"S08-B11-B1": 0.10247298117068482,"S09-B03-B1": -0.12550137421514052,"S09-B11-B1": 0.06038905678307316,"S10-B03-B1": 0.08284427370889347,"S10-B11-B1": 0.4413268321516668,"S11-B03-B1": -0.08184888494565712,"S11-B11-B1": 0.08674365614044177,"S12-B03-B1": -0.3405172535192946,"S12-B11-B1": -0.2162778490154338,"S13-B03-B1": -0.20894238262729203,"S13-B11-B1": 0.007992350452042274,"S14-B03-B1": -0.44218076120701255,"S14-B11-B1": 0.19238108862685266,"S15-B03-B1": -0.14013324602614574,"S15-B11-B1": 0.16677316354694938,"S16-B03-B1": -0.8252640711741677,"S16-B11-B1": -0.056585429443245516,"S17-B03-B1": -0.542567297776479,"S17-B11-B1": 0.1909879411927733,"S18-B03-B1": -0.1966650964553054,"S18-B11-B1": 0.15873723593284694,"S19-B03-B1": -0.4565826348706068,"S19-B11-B1": 0.2918019854017899,"S20-B03-B1": -0.4598210056558685,"S20-B11-B1": 0.5146069215769487} 

                if ('B1' in magnetID):
                    # transforming from current frame to its own B1 frame
                    initialFrame = f"{sectorID}-{girderID}-QUAD01-NOMINAL"
                    targetFrame = f"{sectorID}-{girderID}-B1-NOMINAL"
                    transformation = Transformation.evaluateTransformation(frameDict, initialFrame, targetFrame)
                    cp_point.transform(transformation, targetFrame)

                    # applying shift
                    magnetName = f"{sectorID}-{girderID}-B1"
                    cp_point.x -= shiftsB1[magnetName]

                # transforming back to machine-local
                initialFrame = f"{sectorID}-{girderID}-{magnetID}-NOMINAL"
                if (girderID != 'B05' and girderID != 'B09' and girderID != 'B07'):
                    if (magnetID != 'B1'):
                        initialFrame = f"{sectorID}-{girderID}-QUAD01-NOMINAL"

                transformation = Transformation.evaluateTransformation(frameDict, initialFrame, 'machine-local')
                cp_point.transform(transformation, 'machine-local')

                # print(f"{cp_point.name}: [{cp_point.x}, {cp_point.y}, {cp_point.z}]")
                
                # acrescentando a tag de nomes dos imãs que definem a longitudinal
                if ('B01-QUAD01' in cp_point.name):
                    cp_point.name += '-LONG'
                elif ('B02' in cp_point.name):
                    if (sectorID == 'S01' or sectorID == 'S05' or sectorID == 'S09' or sectorID == 'S13' or sectorID == 'S17'):
                        if('QUAD01' in cp_point.name):
                            cp_point.name += '-LONG'
                    else:
                        if (len(pointName.split('-')) == 3):
                            if ('QUAD01' in cp_point.name):
                                cp_point.name = f"{sectorID}-{girderID}-QUAD02-LONG"
                            elif ('QUAD02' in cp_point.name):
                                cp_point.name = f"{sectorID}-{girderID}-QUAD01"
                        else:
                            if('QUAD02' in cp_point.name):
                                cp_point.name += '-LONG'

                elif ('B04-QUAD02' in cp_point.name):
                    cp_point.name += '-LONG'
                elif ('B06-QUAD02' in cp_point.name):
                    cp_point.name += '-LONG'
                elif ('B08-QUAD01' in cp_point.name):
                    cp_point.name += '-LONG'
                elif ('B10-QUAD01' in cp_point.name):
                    cp_point.name += '-LONG'
                elif ('B03' in girderID or 'B11' in girderID):
                    if ('B1' in magnetID):
                        cp_point.name += '-LONG'

                newPoint = f"{cp_point.name}, {cp_point.x}, {cp_point.y}, {cp_point.z}\n"
                output += newPoint

    # writing the modified list of points in output file 
    with open('../data/output/anel-novos_pontos.txt', 'w') as f:
        f.write(output)


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

def calculateMagnetsDeviations2(frameDict):
    header = ['Magnet', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
    deviationList = []

    for frameName in frameDict:
        frame = frameDict[frameName]

        if ('NOMINAL' in frame.name or frame.name == 'machine-local'):
            continue

        magnetName = ""
        splitedFrameName = frame.name.split('-')
        for i in range(len(splitedFrameName) - 1):
            magnetName += (splitedFrameName[i]+'-')

        frameFrom = magnetName + 'NOMINAL'
        frameTo = frame.name

        try:
            transformation = Transformation.evaluateTransformation(
                frameDict, frameFrom, frameTo)
        except KeyError:
            print("[Cálculo dos desvios] Desvio do imã " +
                    magnetName[:-1] + "não calculado.")
            continue

        dev = Transformation.individualDeviationsFromEquations(
            transformation)

        dev.insert(0, magnetName[:-1])

        deviation = pd.DataFrame([dev], columns=header)
        deviationList.append(deviation)

    deviations = pd.concat(deviationList, ignore_index=True)
    deviations = deviations.set_index(header[0])

    # retorna o df com os termos no tipo numérico certo
    return deviations.astype('float32')

def calculateMagnetsDistances(frameDict, dataType):
        header = ['Reference', 'Distance (mm)']

        if (dataType == 'measured'):
            ignoreTag = 'NOMINAL'
            firstIndex = 0
            lastIndexShift = 3
        else:
            ignoreTag = 'MEASURED'
            firstIndex = 1
            lastIndexShift = 2

        distancesList = []
        frameDictKeys = list(frameDict.keys())
        for (index, frameName) in enumerate(frameDict):
            frame = frameDict[frameName]

            if (ignoreTag in frame.name or 'machine-local' in frame.name):
                continue

            if (index == firstIndex):
                firstMagnetsFrame = frame

            if (index == len(frameDict) - lastIndexShift):
                # first frame of the dictionary
                magnet2frame = firstMagnetsFrame
            else:
                # next measured frame
                magnet2frame = frameDict[frameDictKeys[index+2]]

            magnet1frame = frame

            magnet1coord = np.array([magnet1frame.transformation.Tx,
                                     magnet1frame.transformation.Ty, magnet1frame.transformation.Tz])
            magnet2coord = np.array([magnet2frame.transformation.Tx,
                                     magnet2frame.transformation.Ty, magnet2frame.transformation.Tz])

            vector = magnet2coord - magnet1coord
            distance = math.sqrt(vector[0]**2 + vector[1]**2)

            splitedName1 = magnet1frame.name.split('-')
            splitedName2 = magnet2frame.name.split('-')

            magnet1frameName = splitedName1[0] + '-' + \
                splitedName1[1] + '- ' + splitedName1[2]
            magnet2frameName = splitedName2[0] + '-' + \
                splitedName2[1] + '- ' + splitedName2[2]

            distanceReference = magnet1frameName + ' x ' + magnet2frameName

            distanceDF = pd.DataFrame(
                [[distanceReference, distance]], columns=header)
            distancesList.append(distanceDF)

        distances = pd.concat(distancesList, ignore_index=True)
        distances = distances.set_index(header[0])

        # retorna o df com os termos no tipo numérico certo
        return distances.astype('float32')