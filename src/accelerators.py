from entities import *
import numpy as np
import math


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
    def calculateLocalDeviationsByTypeOfMagnet(magnet, pointList):
        pointsMeasured = pointList[0]
        pointsNominal = pointList[1]

        # no dof separation
        dofs = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        deviation = DataUtils.evaluateDeviation(
            pointsMeasured[0], pointsNominal[0], dofs)

        frameFrom = magnet.name + '-NOMINAL'
        frameTo = magnet.name + '-MEASURED'

        localTransformation = Transformation(
            frameFrom, frameTo, deviation[0], deviation[1], deviation[2], deviation[3], deviation[4], deviation[5])

        return localTransformation

    @staticmethod
    def generateCredentials(pointName):
        credentials = {}
        credentials["isBooster"] = True
        credentials["isValidPoint"] = True

        details = pointName.split('-')
        prefix = details[0]

        if (prefix[0] != 'P'):
            credentials["isBooster"] = False
            return credentials

        if (len(details) != 3):
            credentials["isValidPoint"] = False
            return credentials
        
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

        credentials["wall"] = prefix
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

            if (not credentials['isBooster'] or not credentials['isValidPoint']):
                continue

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


class LTB():
    @staticmethod
    def appendPoints(pointDict, transportDictNom, magnet, transport):
        pointsMeasured = [[], []]
        pointsNominal = [[], []]

        for pointName in pointDict:
            pointMeas = pointDict[pointName]
            pointNom = transportDictNom[transport][magnet.name].pointDict[pointName]

            pointMeasCoord = [pointMeas.x, pointMeas.y, pointMeas.z]
            pointNomCoord = [pointNom.x, pointNom.y, pointNom.z]

            # adicionando pontos às suas respectivas listas
            pointsMeasured[0].append(pointMeasCoord)
            pointsNominal[0].append(pointNomCoord)

        return (pointsMeasured, pointsNominal)

    @staticmethod
    def calculateLocalDeviationsByTypeOfMagnet(magnet, pointList):
        pointsMeasured = pointList[0]
        pointsNominal = pointList[1]

        # no dof separation
        dofs = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        deviation = DataUtils.evaluateDeviation(
            pointsMeasured[0], pointsNominal[0], dofs)

        frameFrom = magnet.name + '-NOMINAL'
        frameTo = magnet.name + '-MEASURED'

        localTransformation = Transformation(
            frameFrom, frameTo, deviation[0], deviation[1], deviation[2], deviation[3], deviation[4], deviation[5])

        return localTransformation

    @staticmethod
    def generateCredentials(pointName):
        credentials = {}
        credentials["isLTB"] = True
        credentials["isValidPoint"] = True

        details = pointName.split('-')
        prefix = details[0]

        if (prefix[0] != 'P'):
            credentials["isLTB"] = False
            return credentials

        if (len(details) != 3):
            credentials["isValidPoint"] = False
            return credentials

        magnetName = details[0] + '-' + details[1]
        magnetRef = details[1]

        if (magnetRef == 'QD' or magnetRef == 'QF'):
            magnetType = 'quadrupole'
        elif (magnetRef[0] == 'B'):
            magnetType = 'dipole-LTB'
        elif ('CV' in magnetRef or 'CF' in magnetRef):
            magnetType = 'broker'
        else:
            magnetType = 'other'

        credentials["prefix"] = prefix
        credentials["magnetType"] = magnetType
        credentials["pointName"] = pointName
        credentials["magnetName"] = magnetName

        return credentials

    @staticmethod
    def createObjectsStructure(pointsDF):

        transportDict = {}

        # iterate over dataframe
        for pointName, coordinate in pointsDF.iterrows():

            credentials = Booster.generateCredentials(pointName)

            if (not credentials['isLTB'] or not credentials['isValidPoint']):
                continue

            # instantiating a Point object
            point = Point(
                credentials['pointName'], coordinate['x'], coordinate['y'], coordinate['z'], 'machine-local')

            # finding Girder object or instantiating a new one
            if (not credentials['prefix'] in transportDict):
                transportDict[credentials['prefix']] = {}

            magnetDict = transportDict[credentials['prefix']]

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
                transportDict[credentials['prefix']
                         ][credentials['magnetName']] = magnet

        # SR.reorderDict(girderDict)

        return transportDict


from dataUtils import DataUtils