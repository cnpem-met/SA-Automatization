from pointGroup import PointGroup
from girderGroup import GirderGroup
import pandas as pd
import numpy as np
import math


class FacModel (object):
    def __init__(self, girdersMeasured, girdersNominal):
        self.girdersMeas = girdersMeasured
        self.girdersNom = girdersNominal
        self.longitudinalPoints = None
        self.magnetsDevMatrix = None

    @staticmethod
    def generateMagnetsDictionary(girdersDictionary):
        magnetPointsDict = {}

        # iterando sobre pontos medidos e aplicando bestfit
        for currentGirder in girdersDictionary['girder']:
            # extraindo índice do berço atual *
            # * se todos os berços estiverem presentes em ambos os arquivos de entrada, os índices devem coincidir
            girderMeasIndex = girdersDictionary['girder'].index(currentGirder)

            # identificando tipo do berço
            girderType = GirderGroup.checkGirderType(currentGirder)

            # itera sobre os pontos do berço
            for pointName in girdersDictionary['x_list'][girderMeasIndex]['pts_name']:
                # salva o indice referente ao nome do ponto
                ptMeasIndex = girdersDictionary['x_list'][girderMeasIndex]['pts_name'].index(
                    pointName)

                # extrai as coordenadas do referido ponto
                ptMeas = [girdersDictionary['x_list'][girderMeasIndex]['pts_val'][ptMeasIndex], girdersDictionary['y_list']
                          [girderMeasIndex]['pts_val'][ptMeasIndex], girdersDictionary['z_list'][girderMeasIndex]['pts_val'][ptMeasIndex]]

                # cria arrays de pontos de cada imã
                if (girderType == 'multi'):
                    magnet = pointName[-10:-4]

                elif (girderType == 'B1'):
                    # diferenciando conjunto de pontos para comporem diferentes graus de liberdade
                    if(pointName[-2:] == 'B1' or pointName[-4:] == 'B1MR'):
                        magnet = "DIP(B1)"
                    else:
                        magnet = pointName[-10:-4]

                elif (girderType == 'B2'):
                    magnet = "DIP(B2)"
                else:
                    magnet = "DIP(BC)"

                if (not currentGirder in magnetPointsDict):
                    magnetPointsDict[currentGirder] = {}

                if (not magnet in magnetPointsDict[currentGirder]):
                    magnetPointsDict[currentGirder][magnet] = []

                magnetPointsDict[currentGirder][magnet].append(
                    (pointName, ptMeas))

        return magnetPointsDict

    @staticmethod
    def checkMagnetType(magnetName):

        if (magnetName[:4] == 'QUAD'):
            magnetType = 'quadrupole'
        elif (magnetName == 'DIP(B1)'):
            magnetType = 'dipole-B1'
        elif (magnetName == 'DIP(B2)'):
            magnetType = 'dipole-B2'
        elif (magnetName == 'DIP(BC)'):
            magnetType = 'dipole-BC'
        else:
            return 'not a valid magnet'

        return magnetType

    def calculateLongitudinalPoint(self, typeOfPoints, girder, magnet, magnetType, girderDev=None):
        xList, yList, zList = [], [], []

        # coletando os pontos do imã, diferenciando graus de liberdade no caso do B2
        for point in magnet:
            ptName = point[0]
            coordinates = point[1]

            if (magnetType == 'quadrupole' or magnetType == 'dipole-B1' or magnetType == 'dipole-BC'):
                xList.append(coordinates[0])
                yList.append(coordinates[1])
                zList.append(coordinates[2])
            elif (magnetType == 'dipole-B2'):
                # tratamento especial, considerando pontos de nivel e vert/transv
                if (ptName[:2] == 'LV'):
                    zList.append(coordinates[2])
                else:
                    xList.append(coordinates[0])
                    yList.append(coordinates[1])

        # calculando o ponto de interesse para o calculo das distancias longitudinais
        # segundo criterios para cada tipo de imã

        if (magnetType == 'quadrupole'):
            longitudinalPoint = [
                np.mean(xList), np.mean(yList), np.mean(zList)]
        else:  # caso do dipolo
            midpoint = [np.mean(xList), np.mean(yList), np.mean(zList)]
            dipoleType = magnetType[-2:]

            if (dipoleType == 'B1'):
                # geometria do B1
                dist = 62.522039008
                theta = 88.6223509126 * math.pi/180  # em radianos

                # desvio angular do quadrupolo do mesmo berço
                Rz = 0
                if (typeOfPoints == 'measured'):
                    Rz = girderDev['Rz'][girder] * 10**-3  # em rad

                # definindo pra qual sentido shiftar em Y
                directionYFactor = -1
                girderNum = girder[-3:]
                if(girderNum == 'B11'):
                    directionYFactor = 1

                # definindo direção do shift
                shiftDir = theta - Rz

                # calculando shift
                shiftX = math.sin(shiftDir) * dist
                shiftY = directionYFactor * math.cos(shiftDir) * dist
                shiftZ = -228.5

            else:
                if (dipoleType == 'B2'):
                    # geometria do B1
                    dist = 56.0719937133
                    shiftZ = -228.5
                else:  # caso do BC
                    dist = - 61.7029799998
                    shiftZ = -351.5

                Rz = 0
                if (typeOfPoints == 'measured'):
                    Rz = girderDev['Rz'][girder] * 10**-3

                shiftX = math.cos(Rz) * dist
                shiftY = - math.sin(Rz) * dist

            # aplicando shift e assim definindo o ponto
            longitudinalPoint = [midpoint[0] + shiftX,
                                 midpoint[1] + shiftY, midpoint[2] + shiftZ]

        return longitudinalPoint

    def calculateLongitudinalDistances(self, points):

        distancesList = np.array([])
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

            distancesList = np.append(distancesList, [distName, distance])

            if (i == 0):
                firstPoint = point1
                firstPointName = point1Name

        distancesList = distancesList.reshape(int(len(distancesList)/2), 2)

        return distancesList

    def generateLongitudinalDistances(self, typeOfPoints):

        if (typeOfPoints == 'measured'):
            ptList = self.girdersMeas.pointGroup.ptList
            girdersDeviation = GirderGroup.evalDiff_bestFit(
                self.girdersMeas.pointGroup.ptList, self.girdersNom.pointGroup.ptList)
        else:
            ptList = self.girdersNom.pointGroup.ptList

        # gerando estruturas com pontos mapeados por berço
        pointsDict = GirderGroup.generateGirderDictionary(ptList)
        magnetsDict = FacModel.generateMagnetsDictionary(pointsDict)

        longidutinalPointList = []

        for girder in magnetsDict:
            for magnet in magnetsDict[girder]:
                magnetType = FacModel.checkMagnetType(magnet)

                if (typeOfPoints == 'measured'):
                    point = self.calculateLongitudinalPoint(
                        typeOfPoints, girder, magnetsDict[girder][magnet], magnetType, girderDev=girdersDeviation)
                else:
                    point = self.calculateLongitudinalPoint(typeOfPoints, girder,
                                                            magnetsDict[girder][magnet], magnetType)

                longidutinalPointList.append((girder+'-'+magnet, point))

        coordinates = np.array([])
        names = np.array([])

        for point in longidutinalPointList:
            girder = point[0][:7]
            coordinates = np.append(
                coordinates, [float(point[1][0]), float(point[1][1]), float(point[1][2])])
            names = np.append(names, [girder, point[0]])

        # mudando a forma de vetor para matriz
        coordinates = coordinates.reshape(int(len(coordinates)/3), 3)
        names = names.reshape(int(len(names)/2), 2)

        # criando dataframe primeiro apenas com as coordenadas dos pontos
        longidutinalPoints = pd.DataFrame(
            coordinates, columns=['x', 'y', 'z'])

        # após a criação, inserção de colunas com referências em formato de string (nome do berço e do imã)
        longidutinalPoints.insert(0, 'Girder', names[:, 0])
        longidutinalPoints.insert(
            len(longidutinalPoints.columns), 'Magnet', names[:, 1])

        # compatibilizando estrutura do df com o pedido dentro do método transformToLocalFrame
        longidutinalPoints.reset_index(drop=True, inplace=True)
        longidutinalPoints = longidutinalPoints.set_index('Girder')

        # instanciando novo objeto do tipo PointGroup para ter acesso a função de conversão de frame
        longPointGroup = PointGroup(longidutinalPoints, self.girdersMeas.type,
                                    lookuptable=self.girdersMeas.pointGroup.lookuptable, frame='local')

        longPointGroup.ptList.dropna(inplace=True)

        # convertendo para o machine-local
        longPointGroup.transformToLocalFrame(convers_type='indirect')

        # voltando ao formato desejado, ou seja, sem a informação dos berços e com os imãs como índice
        longPointGroup.ptList.reset_index(drop=True, inplace=True)
        longPointGroup.ptList = longPointGroup.ptList.set_index('Magnet')

        # finalmente, calculando as distâncias entre os centros dos imãs
        longDistancesList = self.calculateLongitudinalDistances(
            longPointGroup.ptList)

        reference = longDistancesList[:, 0]
        distance = longDistancesList[:, 1].astype('float32')

        longDistancesDf = pd.DataFrame(reference, columns=[
                                       'Reference'])
        longDistancesDf.insert(
            1, 'Distance (mm)', distance)
        longDistancesDf.reset_index(drop=True, inplace=True)
        longDistancesDf = longDistancesDf.set_index('Reference')

        longDistancesDf.astype({'Distance (mm)': 'float32'})

        return longDistancesDf
