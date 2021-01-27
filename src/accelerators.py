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
                magnet = Magnet(credentials['magnetName'], credentials['magnetType'])
                magnet.addPoint(point)

                # including on data structure
                wallDict[credentials['wall']][credentials['magnetName']] = magnet

        # SR.reorderDict(girderDict)

        return wallDict


class SR():
    @staticmethod
    def reorderDict(girderDict):
        for girderName in girderDict:
            if ('B03' in girderName):
                magnetDict = girderDict[girderName]

                orderedList = []
                newMagnetDict = {}

                for magnetName in magnetDict:
                    magnet = magnetDict[magnetName]
                    orderedList.insert(0, (magnetName, magnet))

                for item in orderedList:
                    newMagnetDict[item[0]] = item[1]

                girderDict[girderName] = newMagnetDict

    @staticmethod
    def appendPoints(pointDict, girderDict, magnet, girderName):
        pointsMeasured = [[], []]
        pointsNominal = [[], []]

        for pointName in pointDict:
            pointMeas = pointDict[pointName]
            pointNom = girderDict[girderName][magnet.name].pointDict[pointName]

            pointName = pointMeas.name
            pointMeasCoord = [pointMeas.x, pointMeas.y, pointMeas.z]
            pointNomCoord = [pointNom.x, pointNom.y, pointNom.z]

            # tratamento  bestfits que exijam composição de graus de liberdade
            if (magnet.type == 'dipole-B2'):
                # diferenciando conjunto de pontos para comporem diferentes graus de liberdade
                if ('LVL' in pointName):
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
            # dist = 62.522039008
            dist = 62.4384

            # applying magnet's specific shift to compensate manufacturing variation
            dist -= magnet.shift

            dRy = 24.044474 * 10**-3  # rad
            # differentiating the rotation's signal according to the type of assembly (girder B03 or B11)
            if (girderName[-3:] == 'B03'):
                dRy = - 24.044474 * 10**-3  # rad

            # calculating shift in both planar directions
            shiftTransv = dist * math.cos(dRy)
            shiftLong = dist * math.sin(dRy)
            shiftVert = -228.5

            # calculating shifted point position, which defines the origin of the frame "dipole-measured"
            shiftedPointPosition = [
                midPoint[0] - shiftTransv, midPoint[1] + shiftVert, midPoint[2] + shiftLong]

            # defining a transformation between "dipole-measured" and "quadrupole-measured"
            frameFrom = girderName + '-QUAD01-MEASURED'
            frameTo = magnet.name + '-MEASURED'
            localTransformation = Transformation(
                frameFrom, frameTo, 0, 0, shiftedPointPosition[2], 0, dRy * 10**3, 0)  # mrad

        else:
            if (magnet.type == 'dipole-B2'):
                # calculating transformation for the 1st group of points associated
                # with specific degrees of freedom
                dofs = ['Ty', 'Rx', 'Rz']
                deviationPartial1 = DataUtils.evaluateDeviation(
                    pointsMeasured[0], pointsNominal[0], dofs)
                
                # # TENTATIVA DE FAZER COM QUE GRUPO CORRESPONDENTE MEXA TAMBÉM
                # transfPoints = []
                # for point in pointsNominal[1]:
                #     transfPt = Point('n/c', point[0], point[1], point[2], 'n/c')
                #     # transfPt = Point.copyFromPoint(point)
                #     transform = Transformation('n/c', 'n/c', 0, deviationPartial1[0], 0, \
                #                                             deviationPartial1[1], 0, deviationPartial1[2])
                #     transfPt.transform(transform.transfMatrix, 'n/c')
                #     transfPoints.append([transfPt.x, transfPt.y, transfPt.z])

                # print(transfPoints)
                # print(pointsNominal[1])
                # print('---------')

                # 2nd group of points
                dofs = ['Tx', 'Tz', 'Ry']
                deviationPartial2 = DataUtils.evaluateDeviation(
                    pointsMeasured[1], pointsNominal[1], dofs)

                # agregating parcial dofs
                deviation = np.array(
                    [deviationPartial2[0], deviationPartial1[0], deviationPartial2[1], deviationPartial1[1], deviationPartial1[2], deviationPartial2[2]])
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
    def generateCredentials(pointName):
        credentials = {}
        credentials["isSR"] = True
        credentials["isValidPoint"] = True

        details = pointName.split('-')
        sectorID = details[0]
        girderID = details[1]
        girder = sectorID + '-' + girderID

        if (sectorID[0] != 'S' or girderID[0] != 'B'):
            credentials["isSR"] = False
            return credentials

        if (len(details) < 4):
            credentials["isValidPoint"] = False
            return credentials

        magnetRef = details[2]

        if (details[len(details)-1] == 'LONG'):
            magnetName = sectorID + '-' + girderID + '-' + magnetRef + '-' + details[len(details)-1]
        else:
            magnetName = sectorID + '-' + girderID + '-' + magnetRef

        if (magnetRef[:4] == 'QUAD'):
            magnetType = 'quadrupole'
        elif (magnetRef[:4] == 'SEXT'):
            magnetType = 'sextupole'
        elif (magnetRef == 'B1'):
            magnetType = 'dipole-B1'
        elif (magnetRef == 'B2'):
            magnetType = 'dipole-B2'
        elif (magnetRef == 'BC'):
            magnetType = 'dipole-BC'
        else:
            magnetType = 'other'

        credentials["girder"] = girder
        credentials["magnetType"] = magnetType
        credentials["pointName"] = pointName
        credentials["magnetName"] = magnetName

        return credentials
    
    @staticmethod
    def createObjectsStructure(pointsDF, isMeasured=False, shiftsB1=None):

        girderDict = {}

        # iterate over dataframe
        for pointName, coordinate in pointsDF.iterrows():
            
            credentials = SR.generateCredentials(pointName)

            if (not credentials['isSR'] or not credentials['isValidPoint']):
                continue

            # instantiating a Point object
            point = Point(
                credentials['pointName'], coordinate['x'], coordinate['y'], coordinate['z'], 'machine-local')

            # finding Girder object or instantiating a new one
            if (not credentials['girder'] in girderDict):
                girderDict[credentials['girder']] = {}

            magnetDict = girderDict[credentials['girder']]

            # finding Magnet object or instantiating a new one
            if (credentials['magnetName'] in magnetDict):
                # reference to the Magnet object
                magnet = magnetDict[credentials['magnetName']]
                # append point into its point list
                magnet.addPoint(point)
                if (shiftsB1):
                    magnet.shift = shiftsB1['magnetName']
            else:
                # instantiate new Magnet object
                magnet = Magnet(credentials['magnetName'], credentials['magnetType'])
                magnet.addPoint(point)
                if (shiftsB1):
                    magnet.shift = shiftsB1['magnetName']

                # including on data structure
                girderDict[credentials['girder']][credentials['magnetName']] = magnet

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

            credentials = LTB.generateCredentials(pointName)

            if (not credentials['isLTB'] or not credentials['isValidPoint']):
                continue

            # instantiating a Point object
            point = Point(credentials['pointName'], coordinate['x'], coordinate['y'], coordinate['z'], 'machine-local')

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
                magnet = Magnet(credentials['magnetName'], credentials['magnetType'])
                magnet.addPoint(point)

                # including on data structure
                transportDict[credentials['prefix']][credentials['magnetName']] = magnet

        # SR.reorderDict(girderDict)

        return transportDict

class BTS():
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
        credentials["isBTS"] = True
        credentials["isValidPoint"] = True

        details = pointName.split('-')
        prefix = details[0]

        if (prefix[0] != 'BTS'):
            credentials["isBTS"] = False
            return credentials

        if (len(details) != 3):
            credentials["isValidPoint"] = False
            return credentials

        magnetName = details[0] + '-' + details[1]
        magnetRef = details[1]

        if ('MP' in magnetRef):
            magnetType = 'quadrupole'
        elif ('DIP' in magnetRef):
            magnetType = 'dipole-BTS'
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

            credentials = BTS.generateCredentials(pointName)

            if (not credentials['isBTS'] or not credentials['isValidPoint']):
                continue

            # instantiating a Point object
            point = Point(credentials['pointName'], coordinate['x'], coordinate['y'], coordinate['z'], 'machine-local')

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
                magnet = Magnet(credentials['magnetName'], credentials['magnetType'])
                magnet.addPoint(point)

                # including on data structure
                transportDict[credentials['prefix']][credentials['magnetName']] = magnet

        # SR.reorderDict(girderDict)

        return transportDict



from dataUtils import DataUtils