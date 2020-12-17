from entities import (Point, Frame, Transformation)
from accelerators import (SR, Booster)
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


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
    def plotDevitationData(deviations, accelerator):  # precisa adaptação pra booster
        # pegando uma cópia do df original
        plotData = deviations.copy()

        if (accelerator == 'SR'):
            plotTitle = 'Global Alignment Profile - Storage Ring'
        elif (accelerator == 'booster'):
            plotTitle = 'Global Alignment Profile - Booster'

        plot_args = {'y_list': ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz'], 'title_list': [
            'Transversal', 'Longitudinal', 'Vertical', 'Pitch', 'Roll', 'Yaw'], 'fig_title': plotTitle}

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
            # print("[Transformação p/ frames locais] Imãs não computados: " +
            #       magnetsNotComputed[:-1])
            pass

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
    def generateMeasuredFrames(typeOfMagnets, objectDictMeas, objectDictNom, frameDict, accelerator, ui):

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
                            ui.logMessage('Falha no imã ' + magnet.name +': o nome de 1 ou mais pontos nominais não correspondem aos medidos.', severity="danger")
                            # print('Falha no imã ' + magnet.name +
                            #       ': o nome de 1 ou mais pontos nominais não correspondem aos medidos.')
                            continue

                        # calculating the transformation from the nominal points to the measured ones
                        try:
                            localTransformation = SR.calculateLocalDeviationsByTypeOfMagnet(
                                magnet, objectName, pointDict, frameDict, pointList)
                        except KeyError:
                            ui.logMessage('Falha no imã ' + magnet.name +': frame medido do quadrupolo adjacente ainda não foi calculado.', severity='danger')
                            # print('Falha no imã ' + magnet.name +
                            #       ': frame medido do quadrupolo adjacente ainda não foi calculado.')
                            continue
                    else:
                        try:
                            pointList = Booster.appendPoints(
                                pointDict, objectDictNom, magnet, objectName)
                        except KeyError:
                            ui.logMessage('Falha no imã ' + magnet.name +': o nome de 1 ou mais pontos nominais não correspondem aos medidos.', severity='danger')
                            # print('Falha no imã ' + magnet.name +
                            #       ': o nome de 1 ou mais pontos nominais não correspondem aos medidos.')
                            continue

                        localTransformation = Booster.calculateLocalDeviationsByTypeOfMagnet(
                            magnet, pointList)

                    try:
                        # referencing the transformation from the frame of nominal magnet to the machine-local
                        baseTransformation = frameDict[localTransformation.frameFrom].transformation
                    except KeyError:
                        ui.logMessage('Falha no imã ' + magnet.name + ': frame nominal não foi computado; checar tabela com transformações.', severity='danger')
                        # print('Falha no imã ' +
                        #       magnet.name + ': frame nominal não foi computado; checar tabela com transformações.')
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
