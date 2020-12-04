import numpy as np
import pandas as pd
import json
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy.linalg import inv


class Point(object):
    def __init__(self, name, x, y, z, frame):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.frameName = frame

    def transform(self, transfMatrix, newFrameName):
        point = np.array([[self.x], [self.y], [self.z], [1]])

        transformedPoint = transfMatrix @ point
        self.x = transformedPoint[0, 0]
        self.y = transformedPoint[1, 0]
        self.z = transformedPoint[2, 0]

        self.frameName = newFrameName


class Frame(object):
    def __init__(self, name, homogeneosTransf):
        self.name = name
        self.transformation = homogeneosTransf


class Transformation(object):
    def __init__(self, frameFrom, frameTo, Tx, Ty, Tz, Rx, Ry, Rz):
        self.frameFrom = frameFrom
        self.frameTo = frameTo
        self.Tx = Tx
        self.Ty = Ty
        self.Tz = Tz
        self.Rx = Rx
        self.Ry = Ry
        self.Rz = Rz
        self.createTransfMatrix()

    def createTransfMatrix(self):

        self.transfMatrix = np.zeros(shape=(4, 4))

        rot_z = np.array([[np.cos(self.Rz*10**-3), -np.sin(self.Rz*10**-3), 0],
                          [np.sin(self.Rz*10**-3), np.cos(self.Rz*10**-3), 0], [0, 0, 1]])
        rot_y = np.array([[np.cos(self.Ry*10**-3), 0, np.sin(self.Ry*10**-3)],
                          [0, 1, 0], [-np.sin(self.Ry*10**-3), 0, np.cos(self.Ry*10**-3)]])
        rot_x = np.array([[1, 0, 0], [0, np.cos(
            self.Rx*10**-3), -np.sin(self.Rx*10**-3)], [0, np.sin(self.Rx*10**-3), np.cos(self.Rx*10**-3)]])

        rotationMatrix = rot_z @ rot_y @ rot_x

        self.transfMatrix[:3, :3] = rotationMatrix
        self.transfMatrix[3, 3] = 1
        self.transfMatrix[:3, 3] = [self.Tx, self.Ty, self.Tz]

        # self.transfMatrix = inv(self.transfMatrix)

    @staticmethod
    def evaluateTransformation(frameDict, initialFrame, targetFrame):
        transformationList = []
        transformationList.insert(
            0, frameDict[initialFrame].transformation.transfMatrix)

        try:
            transformationList.insert(
                0, inv(frameDict[targetFrame].transformation.transfMatrix))
        except KeyError:
            # target frame not found
            return (None, True)

        transformation = transformationList[0] @ transformationList[1]

        return (transformation, False)


class Magnet(object):
    def __init__(self, name, type):
        self.name = name
        self.pointDict = {}
        self.type = type
        self.shift = 0

    def transformFrame(self, frameDict, targetFrame):

        # extracting the name of the initial frame
        for point in self.pointDict:
            initialFrame = self.pointDict[point].frameName
            break

        (transformation, error) = Transformation.evaluateTransformation(
            frameDict, initialFrame, targetFrame)

        if (not error):
            # iterate over points and transform then
            for point in self.pointDict:
                self.pointDict[point].transform(transformation, targetFrame)
        else:
            print(
                "Target frame not computed. Not applying transformation on points of magnet " + self.name)
            return

    def addPoint(self, point):
        self.pointDict[point.name] = point


class Girder(object):
    def __init__(self, name, girderType):
        self.name = name
        self.type = girderType
        self.magnetDict = {}


class DataUtils():
    @staticmethod
    def generateFrameDict(lookupTableDF):
        frameDict = {}

        # creating base frame
        frameDict['machine-local'] = Frame('machine-local', Transformation(
            'machine-local', 'machine-local', 0, 0, 0, 0, 0, 0))

        # iterate over dataframe
        for frameName, dof in lookupTableDF.iterrows():
            newFrameName = frameName + "-NOMINAL"
            transformation = Transformation(
                'machine-local', newFrameName, dof['Tx'], dof['Ty'], dof['Tz'], dof['Rx'], dof['Ry'], dof['Rz'])

            frame = Frame(newFrameName, transformation)
            frameDict[newFrameName] = frame

        return frameDict

    @staticmethod
    def reorderDict(girderDict):
        for girderName in girderDict:
            girder = girderDict[girderName]
            if (girder.type == 'B1' and girderName[-3:] == 'B03'):
                magnetDict = girderDict[girderName].magnetDict

                orderedList = []
                newMagnetDict = {}

                for magnetName in magnetDict:
                    magnet = magnetDict[magnetName]
                    orderedList.insert(0, (magnetName, magnet))

                for item in orderedList:
                    newMagnetDict[item[0]] = item[1]

                girderDict[girderName].magnetDict = newMagnetDict

    @staticmethod
    def createObjectsStructure(pointsDF, isMeasured=False, shiftsB1=None):

        girderDict = {}

        # iterate over dataframe
        for pointName, coordinate in pointsDF.iterrows():
            splitedPointName = pointName.split('-')

            girderName = splitedPointName[0] + '-' + splitedPointName[1]

            girderType = DataUtils.checkGirderType(girderName)

            # default magnet's name (most cases)
            magnetName = girderName + '-' + splitedPointName[2]

            # checking exceptions
            if (girderType == "B1"):
                try:
                    if (splitedPointName[3][:2] == "B1"):
                        magnetName = girderName + "-DIP_B1"
                except IndexError:
                    if (splitedPointName[2] == "CENTER"):
                        magnetName = girderName + "-DIP_B1"

            elif (girderType == "B2"):
                magnetName = girderName + "-DIP_B2"
            elif (girderType == "BC"):
                magnetName = girderName + "-DIP_BC"

            # instantiating a Point object
            point = Point(
                pointName, coordinate['x'], coordinate['y'], coordinate['z'], 'machine-local')

            # finding Girder object or instantiating a new one
            if (girderName in girderDict):
                girder = girderDict[girderName]
            else:
                girderType = DataUtils.checkGirderType(girderName)
                girder = Girder(girderName, girderType)
                girderDict[girderName] = girder

            # finding Magnet object or instantiating a new one
            if (magnetName in girder.magnetDict):
                # reference to the Magnet object
                magnet = girder.magnetDict[magnetName]
                # append point into its point list
                magnet.addPoint(point)
            else:
                magnetType = DataUtils.checkMagnetType(magnetName)

                # instantiate new Magnet object
                magnet = Magnet(magnetName, magnetType)
                magnet.addPoint(point)
                if (isMeasured and magnetType == 'dipole-B1'):
                    magnet.shift = shiftsB1[magnetName]

                # including on data structure
                girderDict[girderName].magnetDict[magnetName] = magnet

        DataUtils.reorderDict(girderDict)

        return girderDict

    @staticmethod
    def readExcel(fileDir, dataType, sheetName=0):

        # create header
        if (dataType == 'lookuptable'):
            header = ['frame', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        else:
            header = ['Magnet', 'x', 'y', 'z']

        # reading file
        df = pd.read_excel(fileDir, sheet_name=sheetName,
                           header=None, names=header)

        # checking if it has a bult-in header and droping it
        if (type(df.iloc[0, 1]) is str):
            df = df.drop([0])

        # sorting and indexing df
        df = df.sort_values(by=header[0])
        df.reset_index(drop=True, inplace=True)
        df = df.set_index(header[0])

        return df

    @staticmethod
    def writeToExcel(fileDir, data):
        data.to_excel(fileDir)

    @staticmethod
    def plotDevitationData(deviations):
        # pegando uma cópia do df original
        plotData = deviations.copy()

        plot_args = {'y_list': ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz'], 'title_list': [
            'Transversal', 'Longitudinal', 'Vertical', 'Pitch', 'Roll', 'Yaw'], 'fig_title': 'Global Alignment Profile - Storage Ring'}

        xAxisTitle = 'Magnet'

        colum1_name = 'Tx'
        df_colums_dict = {
            'Tx': 'd_transv (mm)', 'Ty': 'd_long (mm)', 'Tz': 'd_vert (mm)', 'Rx': 'd_pitch (mrad)', 'Ry': 'd_roll (mrad)', 'Rz': 'd_yaw (mrad)'}

        xAxisTicks = []
        for i in range(0, plotData.iloc[:, 0].size):
            sector = int(i/17) + 1
            xAxisTicks.append(str(sector))

        """ configurando df """
        new_index = np.linspace(
            0, plotData[colum1_name].size-1, plotData[colum1_name].size)

        plotData['Index'] = new_index
        plotData.insert(0, xAxisTitle, plotData.index)
        plotData.set_index('Index', drop=True, inplace=True)
        plotData.reset_index(drop=True, inplace=True)
        plotData.rename(columns=df_colums_dict, inplace=True)

        # invertendo sinal das translações transversais
        plotData["d_transv (mm)"] = -plotData["d_transv (mm)"]

        """ configurando parâmetros do plot """
        # definindo o numero de linhas e colunas do layout do plot
        grid_subplot = [3, 2]
        # definindo se as absicissas serão compartilhadas ou não
        share_xAxis = 'col'
        # criando subplots com os parâmetros gerados acima
        fig, axs = plt.subplots(
            grid_subplot[0], grid_subplot[1], figsize=(18, 9), sharex=share_xAxis)

        plt.subplots_adjust(hspace=0.3)

        tickpos = np.linspace(
            0, plotData.iloc[:, 0].size, int(plotData.iloc[:, 0].size/17))

        tickLabels = []
        [tickLabels.append(str(i)) for i in range(0, len(tickpos))]

        """salvando args de entrada"""
        plot_colors = ['red', 'limegreen', 'blue', 'purple', 'green', 'black']
        # a lista de títulos é direta
        plot_titles = plot_args['title_list']

        # titulo da figura
        fig.suptitle(plot_args['fig_title'], fontsize=16)

        # para a lista de colunas do df a serem plotadas, deve-se mapear a lista 'y_list' de entrada
        # em relação ao dict 'df_colums_dict' para estar em conformidade com os novos nomes das colunas
        y_list = []
        for y in plot_args['y_list']:
            if y in df_colums_dict:
                y_list.append(df_colums_dict[y])

        for i in range(len(y_list)):
            plotData.plot.scatter(
                xAxisTitle, y_list[i], c=plot_colors[i], ax=axs[i % 3][int(i/3)], title=plot_titles[i])

            axs[i % 3][int(i/3)].tick_params(axis='x', which='major', direction='in', bottom=True, top=True, labelrotation=45,
                                             labelsize=5)
            axs[i % 3][int(i/3)].set_xticks(tickpos)
            # axs[i % 3][int(i/3)].set_xticklabels(tickLabels)
            axs[i % 3][int(i/3)].xaxis.labelpad = 10
            axs[i % 3][int(i/3)].grid(b=True, axis='both',
                                      which='major', linestyle='--', alpha=0.5)

        # mostrando plots
        # plt.minorticks_off()

        # acrescentando intervalo entre plots
        plt.draw()
        plt.pause(0.001)

        # condicional para definir se a figura irá congelar o andamento do app
        # usado no ultimo plot, pois senão as janelas fecharão assim que o app acabar de executar
        plt.ioff()

        # evocando a tela com a figura
        plt.show()

    @staticmethod
    def transformToLocalFrame(girderDict, frameDict):
        for girderName in girderDict:
            magnetDict = girderDict[girderName].magnetDict
            for magnetName in magnetDict:
                magnet = magnetDict[magnetName]
                targetFrame = magnetName + "-NOMINAL"
                magnet.transformFrame(frameDict, targetFrame)

    @staticmethod
    def transformToMachineLocal(girderDict, frameDict):
        for girderName in girderDict:
            magnetDict = girderDict[girderName].magnetDict
            for magnetName in magnetDict:
                magnet = magnetDict[magnetName]
                magnet.transformFrame(frameDict, "machine-local")

    @staticmethod
    def checkGirderType(girderName):
        if (girderName[4:] == 'B03' or girderName[4:] == 'B11'):
            girderType = 'B1'
        elif (girderName[4:] == 'B05' or girderName[4:] == 'B09'):
            girderType = 'B2'
        elif (girderName[4:] == 'B07'):
            girderType = 'BC'
        else:
            girderType = 'multi'
        return girderType

    @staticmethod
    def printDictData(girderDict, mode='console'):
        if (mode == 'console'):
            for girder in girderDict:
                print(girder)
                magnetDict = girderDict[girder].magnetDict
                for magnet in magnetDict:
                    pointDict = magnetDict[magnet].pointDict
                    shift = magnetDict[magnet].shift
                    print('\t', magnet)
                    print('\t\tshift: ', shift)
                    print('\t\tpointlist: ')
                    for point in pointDict:
                        pt = pointDict[point]
                        print('\t\t\t', pt.name, pt.x, pt.y, pt.z)
        else:
            output = ""
            for girder in girderDict:
                output += girder + '\n'
                magnetDict = girderDict[girder].magnetDict
                for magnet in magnetDict:
                    pointDict = magnetDict[magnet].pointDict
                    shift = magnetDict[magnet].shift
                    output += '\t'+magnet+'\n'
                    output += '\t\tshift: '+str(shift)+'\n'
                    output += '\t\t\tpointlist:\n'
                    for point in pointDict:
                        pt = pointDict[point]
                        output += '\t\t\t\t' + pt.name + ' ' + \
                            str(pt.x) + ' ' + str(pt.y) + \
                            ' ' + str(pt.z) + '\n'

            with open('../data/output/dataDict.txt', 'w') as f:
                f.write(output)

    @staticmethod
    def checkMagnetType(magnetName):
        magnet = magnetName.split('-')[2]
        if (magnet[:4] == 'QUAD'):
            magnetType = 'quadrupole'
        elif (magnet == 'DIP_B1'):
            magnetType = 'dipole-B1'
        elif (magnet == 'DIP_B2'):
            magnetType = 'dipole-B2'
        elif (magnet == 'DIP_BC'):
            magnetType = 'dipole-BC'
        else:
            magnetType = 'sextupole'

        return magnetType


class Analysis():
    @staticmethod
    def calculateLongitudinalDistances(points):

        distancesList = []
        header = ['Reference', 'Distance']
        for i in range(points.iloc[:, 0].size):
            point1 = points.iloc[i, :]
            point1Name = points.index[i]

            if (i+1 >= points.iloc[:, 0].size):
                point2 = firstPoint
                point2Name = firstPointName
            else:
                point2 = points.iloc[i+1, :]
                point2Name = points.index[i+1]

            vector = point2 - point1

            distance = math.sqrt(
                vector[0]**2 + vector[1]**2)

            distance = round(distance, 4)

            distName = point1Name + ' x ' + point2Name

            if (i == 0):
                firstPoint = point1
                firstPointName = point1Name

            distanceDF = pd.DataFrame([[distName, distance]], columns=header)
            distancesList.append(distanceDF)

        distances = pd.concat(distancesList, ignore_index=True)
        distances = distances.set_index(header[0])

        # retorna o df com os termos no tipo numérico certo
        return distances.astype('float32')

    @staticmethod
    def calculateLongitudinalPoint(magnet):
        xList, yList, zList = [], [], []

        # coletando os pontos do imã, diferenciando graus de liberdade no caso do B2
        for pointName in magnet.pointDict:
            point = magnet.pointDict[pointName]
            if (magnet.type == 'dipole-B2'):
                # tratamento especial, considerando pontos de nivel e vert/transv
                if (point.name[-4:-2] == 'LV'):
                    zList.append(point.z)
                else:
                    xList.append(point.x)
                    yList.append(point.y)
            else:
                xList.append(point.x)
                yList.append(point.y)
                zList.append(point.z)

        coordinates = [None, None, None]

        if (magnet.type == 'quadrupole'):
            coordinates = [
                np.mean(xList), np.mean(yList), np.mean(zList)]
        else:
            if (magnet.type != 'sextupole'):
                midpoint = [np.mean(xList), np.mean(yList), np.mean(zList)]
                dipoleType = magnet.type[-2:]

                shiftZ = -228.5
                if (dipoleType == 'B1'):
                    dist = 62.522039008
                    # applying magnet's specific shift to compensate manufacturing variation
                    dist += magnet.shift
                elif (dipoleType == 'B2'):
                    dist = 56.0719937133
                else:
                    dist = - 61.7029799998
                    shiftZ = -351.5

                Rz = magnet.deviation['Rz'] * 10**-3

                shiftX = math.cos(Rz) * dist
                shiftY = - math.sin(Rz) * dist

                # aplicando shift e assim definindo o ponto
                coordinates = [midpoint[0] + shiftX,
                               midpoint[1] + shiftY, midpoint[2] + shiftZ]

        point = Point(
            magnet.name, coordinates[0], coordinates[1], coordinates[2], magnet.name)

        return point

    @staticmethod
    def generateLongitudinalDistances(girderDict):
        pointDFList = []

        header = ['point-name', 'x', 'y', 'z']

        for girderName in girderDict:
            magnetDict = girderDict[girderName].magnetDict
            for magnetName in magnetDict:
                magnet = magnetDict[magnetName]

                point = Analysis.calculateLongitudinalPoint(magnet)

                tm = magnet.transfMatrix
                tm = {'Tx': -tm['Tx'], 'Ty': -tm['Ty'], 'Tz': -tm['Tz'],
                      'Rx': -tm['Rx'], 'Ry': -tm['Ry'], 'Rz': -tm['Rz']}

                try:
                    point.transform(tm, 'machine-local')

                    row = pd.DataFrame(
                        [[point.name, point.x, point.y, point.z]], columns=header)
                    pointDFList.append(row)
                except TypeError:
                    # caso do sextupolo
                    pass

        pointDF = pd.concat(pointDFList, ignore_index=True)
        pointDF = pointDF.set_index(header[0])

        return Analysis.calculateLongitudinalDistances(pointDF)

    @staticmethod
    def evaluateTransformation(params, *args):

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

    @staticmethod
    def calculateDeviation(ptsMeas, ptsRef, dofs):

        # inicializando array com parâmetros a serem manipulados durante as iterações da minimização
        params = np.zeros(len(dofs))

        # aplicando a operação de minimização para achar os parâmetros de transformação
        deviation = minimize(fun=Analysis.evaluateTransformation, x0=params, args=(ptsMeas, ptsRef, dofs),
                             method='SLSQP', options={'ftol': 1e-06, 'disp': False})['x']

        # invertendo o sinal do resultado p/ adequação
        deviation = [dof*(-1) for dof in deviation]

        return deviation

    @staticmethod
    def appendPoints(pointDict, girderDictNom, magnet, girderName):
        pointsMeasured = [[], []]
        pointsNominal = [[], []]

        for pointName in pointDict:
            pointMeas = pointDict[pointName]
            pointNom = girderDictNom[girderName].magnetDict[magnet.name].pointDict[pointName]

            pointName = pointMeas.name
            pointMeasCoord = [pointMeas.x, pointMeas.y, pointMeas.z]
            pointNomCoord = [pointNom.x, pointNom.y, pointNom.z]

            # tratamento  bestfits que exijam composição de graus de liberdade
            if (magnet.type == 'dipole-B2'):
                # diferenciando conjunto de pontos para comporem diferentes graus de liberdade
                if (pointName[-4:-2] == 'LV'):
                    # DoFs: [Tz, Rx, Ry]
                    pointsMeasured[0].append(pointMeasCoord)
                    pointsNominal[0].append(pointNomCoord)
                else:
                    # DoFs: [Tx, Ty, Rz]
                    pointsMeasured[1].append(pointMeasCoord)
                    pointsNominal[1].append(pointNomCoord)
            else:

                # adicionando pontos às suas respectivas listas
                pointsMeasured[0].append(pointMeasCoord)
                pointsNominal[0].append(pointNomCoord)

        return (pointsMeasured, pointsNominal)

    @staticmethod
    def calculateLocalDeviationsByTypeOfMagnet(magnet, girderName, pointDict, frameDict, pointList):
        pointsMeasured = pointList[0]
        pointsNominal = pointList[1]

        if (magnet.type == 'dipole-B1'):
            # changuing points' frame
            frameFrom = magnet.name+'-NOMINAL'
            frameTo = girderName + '-QUAD01-MEASURED'
            (transformation, error) = Transformation.evaluateTransformation(
                frameDict, frameFrom, frameTo)

            if (not error):
                for pointName in pointDict:
                    point = pointDict[pointName]
                    point.transform(
                        transformation.transfMatrix, frameTo)

            # calculating mid point position
            midPoint = np.mean(pointsMeasured, axis=0)

            # defining shift parameters
            dist = 62.522039008
            # applying magnet's specific shift to compensate manufacturing variation
            dist += magnet.shift
            dRz = -24.044474 * 10**-3  # rad

            # calculating shift in both planar directions
            shiftX = dist * math.sin(dRz)
            shiftY = -dist * math.cos(dRz)

            # calculating shifted point position, which defines the origin of the frame dipole-measured
            shiftedPointPosition = [
                midPoint[0] + shiftX, midPoint[1] + shiftY, midPoint[2]]

            # defining a transformation between dipole-measured and quadrupole-measured
            frameFrom = girderName + '-QUAD01-MEASURED'
            frameTo = magnet.name + '-MEASURED'
            localTransformation = Transformation(
                frameFrom, frameTo, 0, shiftedPointPosition[1], 0, 0, 0, dRz)
        else:
            if (magnet.type == 'dipole-B2'):
                # calculating transformation for the 1st group of points associated
                # with specific degrees of freedom defin
                dofs = ['Tz', 'Rx', 'Ry']
                deviationPartial1 = Analysis.calculateDeviation(
                    pointsMeasured[0], pointsNominal[0], dofs)

                # 2nd group of points
                dofs = ['Tx', 'Ty', 'Rz']
                deviationPartial2 = Analysis.calculateDeviation(
                    pointsMeasured[1], pointsNominal[1], dofs)

                # agregating parcial dofs
                deviation = np.array(
                    [deviationPartial2[0], deviationPartial2[1], deviationPartial1[0], deviationPartial1[1], deviationPartial1[2], deviationPartial2[2]])
            else:
                # no dof separation
                dofs = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
                deviation = Analysis.calculateDeviation(
                    pointsMeasured[0], pointsNominal[0], dofs)

            # creating new Frames and inserting in the frame dictionary
            frameFrom = magnet.name + '-NOMINAL'  # hard-coded, but auto would be better
            frameTo = magnet.name + '-MEASURED'

            localTransformation = Transformation(
                frameFrom, frameTo, deviation[0], deviation[1], deviation[2], deviation[3], deviation[4], deviation[5])

        return localTransformation

    @staticmethod
    def generateMeasuredFrames(typeOfMagnets, girderDictMeas, girderDictNom, frameDict):

        for girderName in girderDictMeas:
            magnetDict = girderDictMeas[girderName].magnetDict
            for magnetName in magnetDict:
                magnet = magnetDict[magnetName]
                pointDict = magnet.pointDict

                if (magnet.type == typeOfMagnets):
                    # creating lists for measured and nominal points
                    pointList = Analysis.appendPoints(
                        pointDict, girderDictNom, magnet, girderName)

                    # calculating the transformation from the nominal points to the measured ones
                    localTransformation = Analysis.calculateLocalDeviationsByTypeOfMagnet(
                        magnet, girderName, pointDict, frameDict, pointList)

                    # referencing the transformation from the frame of nominal magnet to the machine-local
                    baseTransformation = frameDict[localTransformation.frameFrom].transformation

                    # calculating the homogeneous matrix of the transformation from the frame of the measured magnet to the machine-local
                    transfMatrix = baseTransformation.transfMatrix @ localTransformation.transfMatrix

                    # updating the homogeneous matrix and frameFrom of the already created Transformation
                    localTransformation.transfMatrix = transfMatrix
                    localTransformation.frameFrom = 'machine-local'

                    # creating a new Frame with the transformation from the measured magnet to machine-local
                    measMagnetFrame = Frame(
                        localTransformation.frameTo, localTransformation)

                    # adding it to the frame dictionary
                    frameDict[measMagnetFrame.name] = measMagnetFrame

    @staticmethod
    def calculateGirderToGirderDeviation(girderDictMeas, girderDictNom, frameDict):
        pass
        # header = ['Magnet', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']

        # deviationList = []

        # for girderName in girderDictMeas:
        #     girderType = DataUtils.checkGirderType(girderName)
        #     magnetDict = girderDictMeas[girderName].magnetDict
        #     for magnetName in magnetDict:
        #         magnet = magnetDict[magnetName]

        #         magnetDeviation = pd.DataFrame([[magnetName, deviation[0], deviation[1], deviation[2],
        #                                             deviation[3], deviation[4], deviation[5]]], columns=header)

        #         deviationList.append(magnetDeviation)

        # deviation = pd.concat(deviationList, ignore_index=True)

        # deviation = deviation.set_index(header[0])

        # # retorna o df com os termos no tipo numérico certo
        # return deviation.astype('float32')


class App():
    def __init__(self):
        self.lookupTable = None
        self.measured = None
        self.nominals = None
        self.girderDictMeas = None
        self.girderDictNom = None

        self.shiftsB1 = None

        self.runInDevMode()

    def readNominalsFile(self, filePath):
        nominals = DataUtils.readExcel(filePath, "points")
        self.nominals = nominals

    def readMeasuredFile(self, filePath):
        measured = DataUtils.readExcel(filePath, "points")
        self.measured = measured

    def readLookuptableFile(self, filePath):
        lookupTable = DataUtils.readExcel(filePath, "lookuptable")
        self.lookupTable = lookupTable

    def loadEnv(self):
        with open("config.json", "r") as file:
            config = json.load(file)

        self.readNominalsFile(config['ptsNomFileName'])
        self.readMeasuredFile(config['ptsMeasFileName'])
        self.readLookuptableFile(config['lookupFileName'])

        self.shiftsB1 = config['shiftsB1']

    def generateMeasuredFrames(self):
        Analysis.generateMeasuredFrames(
            'quadrupole', self.girderDictMeas, self.girderDictNom, self.frameDict)
        Analysis.generateMeasuredFrames(
            'dipole-B1', self.girderDictMeas, self.girderDictNom, self.frameDict)
        Analysis.generateMeasuredFrames(
            'dipole-B2', self.girderDictMeas, self.girderDictNom, self.frameDict)
        Analysis.generateMeasuredFrames(
            'dipole-BC', self.girderDictMeas, self.girderDictNom, self.frameDict)

    def runInDevMode(self):
        self.loadEnv()

        self.frameDict = DataUtils.generateFrameDict(self.lookupTable)

        self.girderDictNom = DataUtils.createObjectsStructure(self.nominals)

        self.girderDictMeas = DataUtils.createObjectsStructure(
            self.measured, isMeasured=True, shiftsB1=self.shiftsB1)

        DataUtils.transformToLocalFrame(self.girderDictNom, self.frameDict)
        DataUtils.transformToLocalFrame(self.girderDictMeas, self.frameDict)

        self.generateMeasuredFrames()

        for frameName in self.frameDict:
            frame = self.frameDict[frameName]
            print(frame.name)


if __name__ == "__main__":
    App()
