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

    @staticmethod
    def changeFrame(frameFrom, frameTo, pointDict, frameDict):
        transformation = Transformation.evaluateTransformation(
            frameDict, frameFrom, frameTo)

        # checking if the transformation's calculation went ok
        # if (not isTransformOk):
        #     return True

        # iterating over magnet's points and transforming then to the new frame
        for pointName in pointDict:
            point = pointDict[pointName]
            point.transform(
                transformation, frameTo)


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

    @staticmethod
    def evaluateTransformation(frameDict, initialFrame, targetFrame):
        transformationList = []
        transformationList.insert(
            0, frameDict[initialFrame].transformation.transfMatrix)

        transformationList.insert(
            0, inv(frameDict[targetFrame].transformation.transfMatrix))

        transformation = transformationList[0] @ transformationList[1]

        return transformation

    @staticmethod
    def individualDeviationsFromHomogMatrix(transfMatrix):
        p1 = Point('p1', 1, 0, 0, 'base')
        p2 = Point('p2', 0, 1, 0, 'base')
        p3 = Point('p3', 0, 0, 1, 'base')
        p1t = Point('p1', 1, 0, 0, 'base')
        p2t = Point('p2', 0, 1, 0, 'base')
        p3t = Point('p3', 0, 0, 1, 'base')

        p1t.transform(transfMatrix, 'transformed')
        p2t.transform(transfMatrix, 'transformed')
        p3t.transform(transfMatrix, 'transformed')

        p = [[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z], [p3.x, p3.y, p3.z]]
        pt = [[p1t.x, p1t.y, p1t.z], [p2t.x, p2t.y, p2t.z], [p3t.x, p3t.y, p3t.z]]

        dofs = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        deviation = DataUtils.evaluateDeviation(
            pt, p, dofs)

        return deviation

    @staticmethod
    def individualDeviationsFromEquations(transfMatrix):
        # Ry
        ry = math.asin(-transfMatrix[2, 0])
        # Rz
        rz = math.asin(transfMatrix[1, 0]/math.cos(ry))
        # Rx
        rx = math.asin(transfMatrix[2, 1]/math.cos(ry))
        # Tx, Ty, Tz
        (tx, ty, tz) = (transfMatrix[0, 3],
                        transfMatrix[1, 3], transfMatrix[2, 3])

        return [tx, ty, tz, rx * 10**3, ry * 10**3, rz * 10**3]


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

        transformation = Transformation.evaluateTransformation(
            frameDict, initialFrame, targetFrame)

        # iterate over points and transform then
        for point in self.pointDict:
            self.pointDict[point].transform(transformation, targetFrame)

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
    def readCSV(fileDir, dataType):
        # create header
        if (dataType == 'lookuptable'):
            header = ['frame', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        else:
            header = ['Magnet', 'x', 'y', 'z']

        # reading csv file with automatic detection of its delimeter
        data = pd.read_csv(fileDir, sep=None, engine='python', names=header)

        # checking if it has a bult-in header and droping it
        if (type(data.iloc[0, 1]) is str):
            data = data.drop([0])

        # sorting and indexing df
        data = data.sort_values(by=header[0])
        data.reset_index(drop=True, inplace=True)
        data = data.set_index(header[0])

        return data

    @staticmethod
    def writeToExcel(fileDir, data):
        data.to_excel(fileDir)

    @staticmethod
    def plotDevitationData(deviations):  # precisa adaptação pra booster
        # pegando uma cópia do df original
        plotData = deviations.copy()

        plot_args = {'y_list': ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz'], 'title_list': [
            'Transversal', 'Longitudinal', 'Vertical', 'Pitch', 'Roll', 'Yaw'], 'fig_title': 'Global Alignment Profile - Booster'}

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
    def transformToLocalFrame(objectDict, frameDict, accelerator):
        magnetsNotComputed = ""
        for objectName in objectDict:
            if (accelerator == 'SR'):
                magnetDict = objectDict[objectName].magnetDict
            else:
                magnetDict = objectDict[objectName]
            for magnetName in magnetDict:
                magnet = magnetDict[magnetName]
                targetFrame = magnetName + "-NOMINAL"

                # error = magnet.transformFrame(frameDict, targetFrame)
                # if(error):
                #     magnetsNotComputed += magnet.name + ','
                try:
                    magnet.transformFrame(frameDict, targetFrame)
                except KeyError:
                    magnetsNotComputed += magnet.name + ','

        # at least one frame was not computed
        if (magnetsNotComputed != ""):
            print("[Transformação p/ frames locais] Imãs não computados: " +
                  magnetsNotComputed[:-1])

    @staticmethod
    def transformToMachineLocal(girderDict, frameDict):
        for girderName in girderDict:
            magnetDict = girderDict[girderName].magnetDict
            for magnetName in magnetDict:
                magnet = magnetDict[magnetName]
                magnet.transformFrame(frameDict, "machine-local")

    @staticmethod
    def printDictData(objectDict, accelerator, mode='console'):
        output = ""
        for objectName in objectDict:
            output += objectName + '\n'
            if (accelerator == 'SR'):
                magnetDict = objectDict[objectName].magnetDict
            elif (accelerator == 'booster'):
                magnetDict = objectDict[objectName]
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

        if (mode == 'console'):
            print(output)
        else:
            with open('../data/output/dataDict.txt', 'w') as f:
                f.write(output)

    @staticmethod
    def printFrameDict(frameDict):
        for frameName in frameDict:
            frame = frameDict[frameName]
            print(frame.name)
            print('\t(Tx, Ty, Tz, Rx, Ry, Rz): ('+str(frame.transformation.Tx)+', '+str(frame.transformation.Ty)+', ' +
                  str(frame.transformation.Tz)+', '+str(frame.transformation.Rx)+', '+str(frame.transformation.Ry)+', '+str(frame.transformation.Rz)+')')

    @staticmethod
    def calculateMagnetsDeviations(frameDict):
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

    @staticmethod
    # FALTA ADAPTAÇÃO PARA O BOOSTER
    def sortFrameDictByBeamTrajectory(frameDict, accelerator="SR"):
        # sorting in alphabetical order
        keySortedList = sorted(frameDict)
        sortedFrameDict = {}
        for key in keySortedList:
            sortedFrameDict[key] = frameDict[key]

        if(accelerator == 'SR'):
            finalFrameDict = {}
            b1FrameList = []
            # correcting the case of S0xB03
            for frameName in sortedFrameDict:
                frame = sortedFrameDict[frameName]

                if ('B03-B1' in frame.name):
                    b1FrameList.append(frame)
                    continue

                finalFrameDict[frame.name] = frame

                if ('B03-QUAD01-NOMINAL' in frame.name):
                    for b1Frame in b1FrameList:
                        finalFrameDict[b1Frame.name] = b1Frame
                    b1FrameList = []
        else:
            # incluir exceções do booster aqui
            finalFrameDict = sortedFrameDict

        return finalFrameDict

    @staticmethod
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

    @staticmethod
    def evaluateDeviation(ptsMeas, ptsRef, dofs):

        # inicializando array com parâmetros a serem manipulados durante as iterações da minimização
        params = np.zeros(len(dofs))

        # aplicando a operação de minimização para achar os parâmetros de transformação
        deviation = minimize(fun=DataUtils.calculateEuclidianDistance, x0=params, args=(ptsMeas, ptsRef, dofs),
                             method='SLSQP', options={'ftol': 1e-10, 'disp': False})['x']

        # invertendo o sinal do resultado p/ adequação
        deviation = [dof*(-1) for dof in deviation]

        return deviation

    @staticmethod
    def generateMeasuredFrames(typeOfMagnets, objectDictMeas, objectDictNom, frameDict, accelerator):

        for objectName in objectDictMeas:

            if (accelerator == 'SR'):
                magnetDict = objectDictMeas[objectName].magnetDict
            elif (accelerator == 'booster'):
                magnetDict = objectDictMeas[objectName]

            for magnetName in magnetDict:
                magnet = magnetDict[magnetName]
                pointDict = magnet.pointDict

                if (magnet.type == typeOfMagnets):
                    if (accelerator == 'SR'):
                        # creating lists for measured and nominal points
                        try:
                            pointList = SR.appendPoints(
                                pointDict, objectDictNom, magnet, objectName)
                        except KeyError:
                            print('[Geração dos frames medidos] Falha no imã ' + magnet.name +
                                  ': o nome de 1 ou mais pontos nominais não correspondem aos medidos.')
                            continue

                        # calculating the transformation from the nominal points to the measured ones
                        try:
                            localTransformation = SR.calculateLocalDeviationsByTypeOfMagnet(
                                magnet, objectName, pointDict, frameDict, pointList)
                        except KeyError:
                            print('[Geração dos frames medidos] Falha no imã ' + magnet.name +
                                  ': frame medido do quadrupolo adjacente ainda não foi calculado.')
                            continue
                    else:
                        try:
                            pointList = Booster.appendPoints(
                                pointDict, objectDictNom, magnet, objectName)
                        except KeyError:
                            print('[Geração dos frames medidos] Falha no imã ' + magnet.name +
                                  ': o nome de 1 ou mais pontos nominais não correspondem aos medidos.')
                            continue

                        localTransformation = Booster.calculateLocalDeviationsByTypeOfMagnet(
                            magnet, objectName, pointDict, frameDict, pointList)

                    try:
                        # referencing the transformation from the frame of nominal magnet to the machine-local
                        baseTransformation = frameDict[localTransformation.frameFrom].transformation
                    except KeyError:
                        print('[Geração dos frames medidos] Falha no imã ' +
                              magnet.name + ': frame nominal não foi computado; checar tabela com transformações.')
                        continue

                    # calculating the homogeneous matrix of the transformation from the frame of the measured magnet to the machine-local
                    transfMatrix = baseTransformation.transfMatrix @ localTransformation.transfMatrix

                    # updating the homogeneous matrix and frameFrom of the already created Transformation
                    localTransformation.transfMatrix = transfMatrix
                    localTransformation.frameFrom = 'machine-local'

                    localTransformation.Tx = transfMatrix[0, 3]
                    localTransformation.Ty = transfMatrix[1, 3]
                    localTransformation.Tz = transfMatrix[2, 3]
                    localTransformation.Rx = baseTransformation.Rx + localTransformation.Rx
                    localTransformation.Ry = baseTransformation.Ry + localTransformation.Ry
                    localTransformation.Rz = baseTransformation.Rz + localTransformation.Rz

                    # creating a new Frame with the transformation from the measured magnet to machine-local
                    measMagnetFrame = Frame(
                        localTransformation.frameTo, localTransformation)

                    # adding it to the frame dictionary
                    frameDict[measMagnetFrame.name] = measMagnetFrame

    @staticmethod
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


class Booster():
    @staticmethod
    def appendPoints(pointDict, wallDictNom, magnet, wall):
        pointsMeasured = [[], []]
        pointsNominal = [[], []]

        for pointName in pointDict:
            pointMeas = pointDict[pointName]
            pointNom = wallDictNom[wall][magnet.name].pointDict[pointName]

            # pointName = pointMeas.name
            pointMeasCoord = [pointMeas.x, pointMeas.y, pointMeas.z]
            pointNomCoord = [pointNom.x, pointNom.y, pointNom.z]

            # adicionando pontos às suas respectivas listas
            pointsMeasured[0].append(pointMeasCoord)
            pointsNominal[0].append(pointNomCoord)

        return (pointsMeasured, pointsNominal)

    @staticmethod
    def calculateLocalDeviationsByTypeOfMagnet(magnet, girderName, pointDict, frameDict, pointList):
        pointsMeasured = pointList[0]
        pointsNominal = pointList[1]

        # no dof separation
        dofs = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        deviation = DataUtils.evaluateDeviation(
            pointsMeasured[0], pointsNominal[0], dofs)

        # creating new Frames and inserting in the frame dictionary
        # [--- hard-coded, but auto would be better ---]
        frameFrom = magnet.name + '-NOMINAL'
        frameTo = magnet.name + '-MEASURED'

        localTransformation = Transformation(
            frameFrom, frameTo, deviation[0], deviation[1], deviation[2], deviation[3], deviation[4], deviation[5])

        return localTransformation

    @staticmethod
    def generateCredentials(pointName):
        details = pointName.split('-')

        wallID = details[0]
        magnetName = details[0] + '-' + details[1]
        magnetRef = details[1]

        if ('QUAD' in magnetRef):
            magnetType = 'quadrupole'
        elif ('SEXT' in magnetRef):
            magnetType = 'sextupole'
        elif ('DIP' in magnetRef):
            magnetType = 'dipole-booster'
        else:
            magnetType = 'other'

        credentials = {}
        credentials["wall"] = wallID
        credentials["magnetType"] = magnetType
        credentials["pointName"] = pointName
        credentials["magnetName"] = magnetName

        return credentials

    @staticmethod
    def createObjectsStructure(pointsDF):

        wallDict = {}

        # iterate over dataframe
        for pointName, coordinate in pointsDF.iterrows():

            credentials = Booster.generateCredentials(pointName)

            # instantiating a Point object
            point = Point(
                credentials['pointName'], coordinate['x'], coordinate['y'], coordinate['z'], 'machine-local')

            # finding Girder object or instantiating a new one
            if (not credentials['wall'] in wallDict):
                wallDict[credentials['wall']] = {}

            magnetDict = wallDict[credentials['wall']]

            # finding Magnet object or instantiating a new one
            if (credentials['magnetName'] in magnetDict):
                # reference to the Magnet object
                magnet = magnetDict[credentials['magnetName']]
                # append point into its point list
                magnet.addPoint(point)
            else:
                # instantiate new Magnet object
                magnet = Magnet(
                    credentials['magnetName'], credentials['magnetType'])
                magnet.addPoint(point)

                # including on data structure
                wallDict[credentials['wall']
                         ][credentials['magnetName']] = magnet

        # SR.reorderDict(girderDict)

        return wallDict


class SR():
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
    def checkMagnetType(magnetName):
        magnet = magnetName.split('-')[2]
        if (magnet[:4] == 'QUAD'):
            magnetType = 'quadrupole'
        elif (magnet == 'B1'):
            magnetType = 'dipole-B1'
        elif (magnet == 'B2'):
            magnetType = 'dipole-B2'
        elif (magnet == 'BC'):
            magnetType = 'dipole-BC'
        else:
            magnetType = 'sextupole'

        return magnetType

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
            # changuing points' frame: from the current activated frame ("dipole-nominal") to the "quadrupole-measured"
            frameFrom = magnet.name+'-NOMINAL'
            frameTo = girderName + '-QUAD01-MEASURED'
            Frame.changeFrame(frameFrom, frameTo, pointDict, frameDict)

            # listing the points' coordinates in the new frame
            points = []
            for pointName in pointDict:
                point = pointDict[pointName]
                points.append([point.x, point.y, point.z])

            # changuing back point's frame
            frameFrom = girderName + '-QUAD01-MEASURED'
            frameTo = magnet.name+'-NOMINAL'
            Frame.changeFrame(frameFrom, frameTo, pointDict, frameDict)

            # calculating mid point position
            midPoint = np.mean(points, axis=0)

            # defining shift parameters
            dist = 62.522039008

            # applying magnet's specific shift to compensate manufacturing variation
            dist -= magnet.shift

            dRz = 24.044474 * 10**-3  # rad
            # differentiating the rotation's signal according to the type of assembly (girder B03 or B11)
            if (girderName[-3:] == 'B03'):
                dRz = - 24.044474 * 10**-3  # rad

            # calculating shift in both planar directions
            shiftX = dist * math.cos(dRz)
            shiftY = dist * math.sin(dRz)
            shiftZ = -228.5

            # calculating shifted point position, which defines the origin of the frame "dipole-measured"
            shiftedPointPosition = [
                midPoint[0] + shiftX, midPoint[1] + shiftY, midPoint[2] + shiftZ]

            # defining a transformation between "dipole-measured" and "quadrupole-measured"
            frameFrom = girderName + '-QUAD01-MEASURED'
            frameTo = magnet.name + '-MEASURED'
            localTransformation = Transformation(
                frameFrom, frameTo, 0, shiftedPointPosition[1], 0, 0, 0, dRz * 10**3)  # mrad

        else:
            if (magnet.type == 'dipole-B2'):
                # calculating transformation for the 1st group of points associated
                # with specific degrees of freedom
                dofs = ['Tz', 'Rx', 'Ry']
                deviationPartial1 = DataUtils.evaluateDeviation(
                    pointsMeasured[0], pointsNominal[0], dofs)

                # 2nd group of points
                dofs = ['Tx', 'Ty', 'Rz']
                deviationPartial2 = DataUtils.evaluateDeviation(
                    pointsMeasured[1], pointsNominal[1], dofs)

                # agregating parcial dofs
                deviation = np.array(
                    [deviationPartial2[0], deviationPartial2[1], deviationPartial1[0], deviationPartial1[1], deviationPartial1[2], deviationPartial2[2]])
            else:
                # no dof separation
                dofs = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
                deviation = DataUtils.evaluateDeviation(
                    pointsMeasured[0], pointsNominal[0], dofs)

            # creating new Frames and inserting in the frame dictionary
            # [--- hard-coded, but auto would be better ---]
            frameFrom = magnet.name + '-NOMINAL'
            frameTo = magnet.name + '-MEASURED'

            localTransformation = Transformation(
                frameFrom, frameTo, deviation[0], deviation[1], deviation[2], deviation[3], deviation[4], deviation[5])

        return localTransformation

    @staticmethod
    def createObjectsStructure(pointsDF, isMeasured=False, shiftsB1=None):

        girderDict = {}

        # iterate over dataframe
        for pointName, coordinate in pointsDF.iterrows():
            splitedPointName = pointName.split('-')

            girderName = splitedPointName[0] + '-' + splitedPointName[1]

            girderType = SR.checkGirderType(girderName)

            # default magnet's name (most cases)
            magnetName = girderName + '-' + splitedPointName[2]

            # checking exceptions
            if (girderType == "B1"):
                try:
                    if (splitedPointName[3][:2] == "B1"):
                        magnetName = girderName + "-B1"
                except IndexError:
                    if (splitedPointName[2] == "CENTER"):
                        magnetName = girderName + "-B1"

            elif (girderType == "B2"):
                magnetName = girderName + "-B2"
            elif (girderType == "BC"):
                magnetName = girderName + "-BC"

            # instantiating a Point object
            point = Point(
                pointName, coordinate['x'], coordinate['y'], coordinate['z'], 'machine-local')

            # finding Girder object or instantiating a new one
            if (girderName in girderDict):
                girder = girderDict[girderName]
            else:
                girderType = SR.checkGirderType(girderName)
                girder = Girder(girderName, girderType)
                girderDict[girderName] = girder

            # finding Magnet object or instantiating a new one
            if (magnetName in girder.magnetDict):
                # reference to the Magnet object
                magnet = girder.magnetDict[magnetName]
                # append point into its point list
                magnet.addPoint(point)
            else:
                magnetType = SR.checkMagnetType(magnetName)

                # instantiate new Magnet object
                magnet = Magnet(magnetName, magnetType)
                magnet.addPoint(point)
                if (isMeasured and magnetType == 'dipole-B1'):
                    magnet.shift = shiftsB1[magnetName]

                # including on data structure
                girderDict[girderName].magnetDict[magnetName] = magnet

        SR.reorderDict(girderDict)

        return girderDict


class App():
    def __init__(self):
        # SR variables
        self.SR_lookupTable = None
        self.SR_measured = None
        self.SR_nominals = None
        self.SR_girderDictMeas = None
        self.SR_girderDictNom = None
        self.SR_frameDict = None
        self.shiftsB1 = None

        # Booster variables
        self.booster_lookupTable = None
        self.booster_measured = None
        self.booster_nominals = None
        self.booster_wallDictMeas = None
        self.booster_wallDictNom = None
        self.booster_frameDict = None

        self.runInDevMode()

    def readNominalsFile(self, filePath, accelerator):

        if (accelerator == 'SR'):
            nominals = DataUtils.readExcel(filePath, "points")
            self.SR_nominals = nominals
        else:
            nominals = DataUtils.readCSV(filePath, "points")
            self.booster_nominals = nominals

    def readMeasuredFile(self, filePath, accelerator):
        if (accelerator == 'SR'):
            measured = DataUtils.readExcel(filePath, "points")
            self.SR_measured = measured
        else:
            measured = DataUtils.readCSV(filePath, "points")
            self.booster_measured = measured

    def readLookuptableFile(self, filePath, accelerator):
        if (accelerator == 'SR'):
            lookupTable = DataUtils.readExcel(filePath, "lookuptable")
            self.SR_lookupTable = lookupTable
        else:
            lookupTable = DataUtils.readCSV(filePath, "lookuptable")
            self.booster_lookupTable = lookupTable

    def loadEnv(self):
        with open("config.json", "r") as file:
            config = json.load(file)

        # SR files
        self.readNominalsFile(config['SR_ptsNomFileName'], 'SR')
        self.readMeasuredFile(config['SR_ptsMeasFileName'], 'SR')
        self.readLookuptableFile(config['SR_lookupFileName'], 'SR')
        self.shiftsB1 = config['shiftsB1']

        # Booster files
        self.readNominalsFile(config['booster_ptsNomFileName'], 'booster')
        self.readMeasuredFile(config['booster_ptsMeasFileName'], 'booster')
        self.readLookuptableFile(config['booster_lookupFileName'], 'booster')

    def generateMeasuredFrames(self, accelerator):
        if (accelerator == 'SR'):
            DataUtils.generateMeasuredFrames(
                'quadrupole', self.SR_girderDictMeas, self.SR_girderDictNom, self.SR_frameDict, 'SR')
            DataUtils.generateMeasuredFrames(
                'dipole-B1', self.SR_girderDictMeas, self.SR_girderDictNom, self.SR_frameDict, 'SR')
            DataUtils.generateMeasuredFrames(
                'dipole-B2', self.SR_girderDictMeas, self.SR_girderDictNom, self.SR_frameDict, 'SR')
            DataUtils.generateMeasuredFrames(
                'dipole-BC', self.SR_girderDictMeas, self.SR_girderDictNom, self.SR_frameDict, 'SR')
            return

        if (accelerator == 'booster'):
            DataUtils.generateMeasuredFrames(
                'quadrupole', self.booster_girderDictMeas, self.booster_girderDictNom, self.booster_frameDict, 'booster')
            DataUtils.generateMeasuredFrames(
                'dipole-booster', self.booster_girderDictMeas, self.booster_girderDictNom, self.booster_frameDict, 'booster')
            return

    def runInDevMode(self):
        self.loadEnv()

        self.booster_frameDict = DataUtils.generateFrameDict(
            self.booster_lookupTable)

        self.booster_girderDictNom = Booster.createObjectsStructure(
            self.booster_nominals)
        self.booster_girderDictMeas = Booster.createObjectsStructure(
            self.booster_measured)

        DataUtils.transformToLocalFrame(
            self.booster_girderDictNom, self.booster_frameDict, 'booster')
        DataUtils.transformToLocalFrame(
            self.booster_girderDictMeas, self.booster_frameDict, 'booster')

        self.generateMeasuredFrames('booster')

        self.booster_frameDict = DataUtils.sortFrameDictByBeamTrajectory(
            self.booster_frameDict, 'booster')

        self.booster_magnetsDeviations = DataUtils.calculateMagnetsDeviations(
            self.booster_frameDict)

        DataUtils.plotDevitationData(self.booster_magnetsDeviations)

        # self.frameDict = DataUtils.generateFrameDict(self.SR_lookupTable)

        # self.SR_girderDictNom = SR.createObjectsStructure(self.SR_nominals)

        # self.SR_girderDictMeas = SR.createObjectsStructure(
        #     self.SR_measured, isMeasured=True, shiftsB1=self.shiftsB1)

        # DataUtils.transformToLocalFrame(self.SR_girderDictNom, self.frameDict)
        # DataUtils.transformToLocalFrame(self.SR_girderDictMeas, self.frameDict)

        # self.generateMeasuredFrames()

        # self.frameDict = DataUtils.sortFrameDictByBeamTrajectory(
        #     self.frameDict)

        # self.magnetsDeviations = DataUtils.calculateMagnetsDeviations(
        #     self.frameDict)

        # self.magnetsDistancesMeas = DataUtils.calculateMagnetsDistances(
        #     self.frameDict, 'measured')

        # self.magnetsDistancesNom = DataUtils.calculateMagnetsDistances(
        #     self.frameDict, 'nominal')

        # DataUtils.writeToExcel(
        #     "../data/output/distances-measured.xlsx", self.magnetsDistancesMeas)
        # DataUtils.writeToExcel(
        #     "../data/output/distances-nominal.xlsx", self.magnetsDistancesNom)

        # DataUtils.plotDevitationData(self.magnetsDeviations)


if __name__ == "__main__":
    App()
