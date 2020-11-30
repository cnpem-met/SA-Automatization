import numpy as np
import pandas as pd
import json
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class Point(object):
    def __init__(self, name, x, y, z, frame):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.frame = frame

    def transform(self, transfMatrix, newFrameName, method):
        if (method == 'direct'):
            self.translate(transfMatrix['Tx'],
                           transfMatrix['Ty'], transfMatrix['Tz'])
            self.rotate(transfMatrix['Rx'],
                        transfMatrix['Ry'], transfMatrix['Rz'])
        else:
            self.rotate(transfMatrix['Rx'],
                        transfMatrix['Ry'], transfMatrix['Rz'])
            self.translate(transfMatrix['Tx'],
                           transfMatrix['Ty'], transfMatrix['Tz'])
        self.frame = newFrameName

    def rotate(self, Rx, Ry, Rz):
        rot_z = np.array([[np.cos(Rz*10**-3), -np.sin(Rz*10**-3), 0],
                          [np.sin(Rz*10**-3), np.cos(Rz*10**-3), 0], [0, 0, 1]])
        rot_y = np.array([[np.cos(Ry*10**-3), 0, np.sin(Ry*10**-3)],
                          [0, 1, 0], [-np.sin(Ry*10**-3), 0, np.cos(Ry*10**-3)]])
        rot_x = np.array([[1, 0, 0], [0, np.cos(
            Rx*10**-3), -np.sin(Rx*10**-3)], [0, np.sin(Rx*10**-3), np.cos(Rx*10**-3)]])

        rotationMatrix = rot_z @ rot_y @ rot_x

        point = np.array([[self.x], [self.y], [self.z]])

        rotatedPoint = np.transpose(rotationMatrix) @ point

        self.x = rotatedPoint[0, 0]
        self.y = rotatedPoint[1, 0]
        self.z = rotatedPoint[2, 0]

    def translate(self, Tx, Ty, Tz):
        self.x -= Tx
        self.y -= Ty
        self.z -= Tz


class Magnet(object):
    def __init__(self, name, type, transfMatrix):
        self.name = name
        self.pointList = {}
        self.transfMatrix = transfMatrix
        self.deviation = {'Tx': 0, 'Ty': 0, 'Tz': 0, 'Rx': 0, 'Ry': 0, 'Rz': 0}
        self.type = type
        self.shift = 0

    def transformFrame(self, frameTo="magnet"):

        # default values for the transformation matrix and the targed frame
        tm = self.transfMatrix
        frame = self.name
        method = 'direct'

        # inverted matrix for transformation to frame 'machine local'
        if (frameTo == "machine-local"):
            tm = {'Tx': -tm.Tx, 'Ty': -tm.Ty, 'Tz': -tm.Tz,
                  'Rx': -tm.Rx, 'Ry': -tm.Ry, 'Rz': -tm.Rz}
            frame = frameTo
            method = 'indirect'

        # iterate over points and transform then
        for point in self.pointList:
            self.pointList[point].transform(tm, frame, method)

    def addPoint(self, point):
        self.pointList[point.name] = point


class Girder(object):
    def __init__(self, name, girderType):
        self.name = name
        self.type = girderType
        self.magnetList = {}


class TransfMatrix(object):
    def __init__(self, Tx, Ty, Tz, Rx, Ry, Rz):
        self.Tx = Tx
        self.Ty = Ty
        self.Tz = Tz
        self.Rx = Rx
        self.Ry = Ry
        self.Rz = Rz


class DataUtils():
    @staticmethod
    def createLookupDict(lookupTableDF):
        lookupTable = {}

        # iterate over dataframe
        for frameName, dof in lookupTableDF.iterrows():
            lookupTable[frameName] = {'Tx': dof['Tx'], 'Ty': dof['Ty'],
                                      'Tz': dof['Tz'], 'Rx': dof['Rx'], 'Ry': dof['Ry'], 'Rz': dof['Rz']}

        return lookupTable

    @staticmethod
    def reorderList(girderDict):
        for girderName in girderDict:
            girder = girderDict[girderName]
            if (girder.type == 'B1' and girderName[-3:] == 'B03'):
                magnetList = girderDict[girderName].magnetList

                orderedList = []
                newMagnetDict = {}

                for magnetName in magnetList:
                    magnet = magnetList[magnetName]
                    orderedList.insert(0, (magnetName, magnet))

                for item in orderedList:
                    newMagnetDict[item[0]] = item[1]

                girderDict[girderName].magnetList = newMagnetDict

    @staticmethod
    def createObjectsStructure(lookupTable, pointsDF):

        girderList = {}

        # iterate over dataframe
        for pointName, coordinate in pointsDF.iterrows():
            splitedPointName = pointName.split('-')

            girderName = splitedPointName[0] + '-' + splitedPointName[1]

            girderType = GenericUtils.checkGirderType(girderName)

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
            if (girderName in girderList):
                girder = girderList[girderName]
            else:
                girderType = GenericUtils.checkGirderType(girderName)
                girder = Girder(girderName, girderType)
                girderList[girderName] = girder

            # finding Magnet object or instantiating a new one
            if (magnetName in girder.magnetList):
                # reference to the Magnet object
                magnet = girder.magnetList[magnetName]
                # append point into its point list
                magnet.addPoint(point)
            else:
                # TIRAR TRATATIVA (DEIXANDO SÓ 'magnetName') QUANDO LOOKUPTABLE ATUALIZAR
                try:
                    transfMatrix = lookupTable[magnetName]
                except KeyError:
                    transfMatrix = lookupTable[girderName]

                magnetType = GenericUtils.checkMagnetType(magnetName)

                # instantiate new Magnet object
                magnet = Magnet(magnetName, magnetType, transfMatrix)
                magnet.addPoint(point)

                # including on data structure
                girderList[girderName].magnetList[magnetName] = magnet

        DataUtils.reorderList(girderList)

        return girderList

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
    def calculateDeviation(ptsMeas, ptsRef, dofs):

        # inicializando array com parâmetros a serem manipulados durante as iterações da minimização
        params = np.zeros(len(dofs))

        # aplicando a operação de minimização para achar os parâmetros de transformação
        deviation = minimize(fun=DataUtils.evaluateTransformation, x0=params, args=(ptsMeas, ptsRef, dofs),
                             method='SLSQP', options={'ftol': 1e-06, 'disp': False})['x']

        # invertendo o sinal do resultado p/ adequação
        deviation = [dof*(-1) for dof in deviation]

        return deviation

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
    def calculateGirderToGirderDeviation(shiftsB1, girderDictMeas, girderDictNom):

        header = ['Magnet', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']

        deviationList = []

        for girderName in girderDictMeas:
            girderType = GenericUtils.checkGirderType(girderName)

            magnetList = girderDictMeas[girderName].magnetList
            for magnetName in magnetList:
                magnet = magnetList[magnetName]
                pointList = magnet.pointList

                pointsMeasured = [[], []]
                pointsNominal = [[], []]

                for pointName in pointList:
                    pointMeas = pointList[pointName]
                    pointNom = girderDictNom[girderName].magnetList[magnetName].pointList[pointName]

                    pointName = pointMeas.name
                    pointMeasCoord = [pointMeas.x, pointMeas.y, pointMeas.z]
                    pointNomCoord = [pointNom.x, pointNom.y, pointNom.z]

                    # tratamento  bestfits que exijam composição de graus de liberdade
                    if (girderType == 'B2'):
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

                if (girderType == 'B2'):
                    # calculating transformation for the 1st group of points associated
                    # with specific degrees of freedom defin
                    dofs = ['Tz', 'Rx', 'Ry']
                    transfPartial1 = DataUtils.calculateDeviation(
                        pointsMeasured[0], pointsNominal[0], dofs)

                    # 2nd group of points
                    dofs = ['Tx', 'Ty', 'Rz']
                    transfPartial2 = DataUtils.calculateDeviation(
                        pointsMeasured[1], pointsNominal[1], dofs)

                    # agregating parcial dofs
                    transformation = np.array(
                        [transfPartial2[0], transfPartial2[1], transfPartial1[0], transfPartial1[1], transfPartial1[2], transfPartial2[2]])

                else:
                    # no dof separation
                    dofs = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
                    transformation = DataUtils.calculateDeviation(
                        pointsMeasured[0], pointsNominal[0], dofs)

                magnetDeviation = pd.DataFrame([[magnet, transformation[0], transformation[1], transformation[2],
                                                 transformation[3], transformation[4], transformation[5]]], columns=header)

                deviationList.append(magnetDeviation)

                # saving deviation into magnet's properties for later usage
                magnet.deviation = {'Tx': transformation[0], 'Ty': transformation[1], 'Tz': transformation[2],
                                    'Rx': transformation[3], 'Ry': transformation[4], 'Rz': transformation[5]}

                if (magnet.type == 'dipole-B1'):
                    magnet.shift = shiftsB1[magnetName]

        deviation = pd.concat(deviationList, ignore_index=True)

        deviation = deviation.set_index(header[0])

        # retorna o df com os termos no tipo numérico certo
        return deviation.astype('float32')

    @staticmethod
    def transformToLocalFrame(girderDict):
        for girderName in girderDict:
            magnetList = girderDict[girderName].magnetList
            for magnetName in magnetList:
                magnet = magnetList[magnetName]
                magnet.transformFrame()


class GenericUtils():
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
    def printDictData(girderDict):
        for girder in girderDict:
            print(girder)
            magnetList = girderDict[girder].magnetList
            for magnet in magnetList:
                pointList = magnetList[magnet].pointList
                transfMatrix = magnetList[magnet].transfMatrix
                shift = magnetList[magnet].shift
                print('\t\t', magnet)
                print('\t\t\tshift: ', shift)
                print('\t\t\ttransfMatrix: ', transfMatrix)
                print('\t\t\tpointlist: ')
                for point in pointList:
                    pt = pointList[point]
                    print('\t\t\t\t', pt.name, pt.x, pt.y, pt.z)

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


class FacModel():
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
        for pointName in magnet.pointList:
            point = magnet.pointList[pointName]
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
            magnetList = girderDict[girderName].magnetList
            for magnetName in magnetList:
                magnet = magnetList[magnetName]

                point = FacModel.calculateLongitudinalPoint(magnet)

                tm = magnet.transfMatrix
                tm = {'Tx': -tm['Tx'], 'Ty': -tm['Ty'], 'Tz': -tm['Tz'],
                      'Rx': -tm['Rx'], 'Ry': -tm['Ry'], 'Rz': -tm['Rz']}

                try:
                    point.transform(tm, 'machine-local', 'indirect')

                    row = pd.DataFrame(
                        [[point.name, point.x, point.y, point.z]], columns=header)
                    pointDFList.append(row)
                except TypeError:
                    # caso do sextupolo
                    pass

        pointDF = pd.concat(pointDFList, ignore_index=True)
        pointDF = pointDF.set_index(header[0])

        return FacModel.calculateLongitudinalDistances(pointDF)


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

    def runInDevMode(self):
        self.loadEnv()

        lookupDict = DataUtils.createLookupDict(self.lookupTable)

        self.girderDictNom = DataUtils.createObjectsStructure(
            lookupDict, self.nominals)

        self.girderDictMeas = DataUtils.createObjectsStructure(
            lookupDict, self.measured)

        DataUtils.transformToLocalFrame(self.girderDictMeas)
        DataUtils.transformToLocalFrame(self.girderDictNom)

        girdersDeviation = DataUtils.calculateGirderToGirderDeviation(
            self.shiftsB1, self.girderDictMeas, self.girderDictNom)

        GenericUtils.printDictData(self.girderDictMeas)

        # DataUtils.plotDevitationData(girdersDeviation)

        # distancesNom = FacModel.generateLongitudinalDistances(
        #     self.girderDictNom)

        # distancesMeas = FacModel.generateLongitudinalDistances(
        #     self.girderDictMeas)

        # print(distancesNom.head(40))
        # print(distancesMeas.head(40))


if __name__ == "__main__":
    App()
