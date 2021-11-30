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
            pointNom = wallDictNom[wall][magnet.name].pointDict[pointName] # PODE IR PRA FORA DA FUNÇÃO, E DE QUEBRA ELA FICA COM MENOS ARGUMENTOS

            # pointName = pointMeas.name
            pointMeasCoord = [pointMeas.x, pointMeas.y, pointMeas.z]
            pointNomCoord = [pointNom.x, pointNom.y, pointNom.z]

            # adicionando pontos às suas respectivas listas
            pointsMeasured[0].append(pointMeasCoord)
            pointsNominal[0].append(pointNomCoord)

        return (pointsMeasured, pointsNominal)

    @staticmethod
    def calculateLocalDeviations(magnet, pointList):
        (pointsMeasured, pointsNominal) = pointList
        # pointsNominal = pointList[1]

        # no dof separation
        dofs = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        deviation = DataUtils.evaluateDeviation(pointsMeasured[0], pointsNominal[0], dofs)

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
                print(pointName)
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
            Frame.changeFrame(frameTo, frameFrom, pointDict, frameDict)

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
                dRy = - dRy

            # calculating shift in both planar directions
            shiftTransv = dist * math.cos(dRy)
            shiftLong = dist * math.sin(dRy)
            shiftVert = -228.5

            # calculating shifted point position, which defines the origin of the frame "dipole-measured"
            shiftedPointPosition = [midPoint[0] - shiftTransv, midPoint[1] + shiftVert, midPoint[2] + shiftLong]

            # defining a transformation between "dipole-measured" and "quadrupole-measured"
            frameFrom = girderName + '-QUAD01-MEASURED'
            frameTo = magnet.name + '-MEASURED'
            localTransformation = Transformation(frameFrom, frameTo, 0, 0, shiftedPointPosition[2], 0, dRy * 10**3, 0)  # mrad

        else:
            if (magnet.type == 'dipole-B2'):
                # calculating transformation for the 1st group of points associated
                # with specific degrees of freedom
                dofs = ['Ty', 'Rx', 'Rz']
                # print(pointsMeasured, pointsNominal)
                deviationPartial1 = DataUtils.evaluateDeviation(pointsMeasured[0], pointsNominal[0], dofs)
            

                # Forcing the "corresponding group"
                transfPoints = []
                for point in pointsNominal[1]:
                    transfPt = Point('n/c', point[0], point[1], point[2], 'n/c')
                    # transfPt = Point.copyFromPoint(point)
                    transform = Transformation('n/c', 'n/c', 0, deviationPartial1[0], 0, deviationPartial1[1], 0, deviationPartial1[2])
                    transfPt.transform(transform.transfMatrix, 'n/c')
                    transfPoints.append([transfPt.x, transfPt.y, transfPt.z])

                # 2nd group of points
                dofs = ['Tx', 'Tz', 'Ry']
                deviationPartial2 = DataUtils.evaluateDeviation(pointsMeasured[1], transfPoints, dofs)

                # agregating parcial dofs
                deviation = np.array(
                    [deviationPartial2[0], deviationPartial1[0], deviationPartial2[1], deviationPartial1[1], deviationPartial2[2], deviationPartial1[2]])
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

        # filtering points that aren't fiducials (especially on B1)
        if ('C01' not in pointName and 'C02' not in pointName and 'C03' not in pointName and 'C04' not in pointName and 'LVL' not in pointName):
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
    def createObjectsStructure(pointsDF, shiftsB1=None, isMeasured=False):

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

            # print(shiftsB1)

            # finding Magnet object or instantiating a new one
            if (credentials['magnetName'] in magnetDict):
                # reference to the Magnet object
                magnet = magnetDict[credentials['magnetName']]
                # append point into its point list
                magnet.addPoint(point)
                if (shiftsB1 and credentials['magnetType'] == 'dipole-B1'):
                    magnet.shift = shiftsB1[credentials['magnetName']]
            else:
                # instantiate new Magnet object
                magnet = Magnet(credentials['magnetName'], credentials['magnetType'])
                magnet.addPoint(point)
                if (shiftsB1 and credentials['magnetType'] == 'dipole-B1'):
                    magnet.shift = shiftsB1[credentials['magnetName']]

                # including on data structure
                girderDict[credentials['girder']][credentials['magnetName']] = magnet

        SR.reorderDict(girderDict)

        return girderDict

    @staticmethod
    def calcB1PerpendicularTranslations(deviations, objectDictMeas, objectDictNom, frameDict):
        for objectName in objectDictMeas:
            magnetDict = objectDictMeas[objectName]
            for magnetName in magnetDict:
                magnet = magnetDict[magnetName]

                if (magnet.type != 'dipole-B1'):
                    continue
                
                # extracting measured points and changuing to nominal frame
                measPoints = []
                pointDict = magnet.pointDict
                for pointName in pointDict:
                    point = pointDict[pointName]
                    # copying for making manipulations
                    cp_point = Point.copyFromPoint(point)
                    # changuing frame to the nominal one
                    targetFrame = magnetName+'-NOMINAL'
                    cp_point.changeFrame(frameDict, targetFrame)
                    measPoints.append(cp_point.coords)
                
                # making calculations to determine the deviation from nominal
                nomPtDict = objectDictNom[objectName][magnetName].pointDict
                nomPoints = [nomPtDict[pt].coords for pt in nomPtDict]
                measMean = np.mean(measPoints, axis=0)
                nomMean = np.mean(nomPoints, axis=0)
                dev = measMean - nomMean
                deviations.loc[magnetName, ['Tx', 'Ty']] = dev[:2]
        return deviations

    @staticmethod
    def addTranslationsFromMagnetMeasurements(deviations):
        for magnet, dof in deviations.iterrows():
            translations = SR.getUncertaintyAndTranslationsPerMagnet(magnet, 'trans')
            deviations.at[magnet, 'Tx'] = dof['Tx'] + translations[0]
            deviations.at[magnet, 'Ty'] = dof['Ty'] + translations[1]
            deviations.at[magnet, 'Tz'] = dof['Tz'] + translations[2]
            deviations.at[magnet, 'Rz'] = dof['Rz'] + translations[3]
        return deviations

    @staticmethod
    def getUncertaintyAndTranslationsPerMagnet(magnetName, uncOrTrans, DOF_num=0):
        sectorA = ['S01', 'S05', 'S09', 'S13', 'S17']
        sectorB = ['S02', 'S04', 'S06', 'S08', 'S10', 'S12', 'S14', 'S16', 'S18', 'S20']
        sectorP = ['S03', 'S07', 'S11', 'S15', 'S19']

        mns = magnetName.split('-')
        sector, girder, magnet = mns[0], mns[1], mns[2]

        trans = np.array([1,2,3,4])

        # girder influenced by the sector type
        if (girder == 'B02' or girder == 'B03' or girder == 'B11' or girder == 'B01'):
            # high-beta sectors (type A)
            if (sector in sectorA):
                if (girder == 'B02' or girder == 'B01'):
                    # quad type: QFA
                    if ('QUAD01' in magnet):
                        unc = [0.0183, 0.0435, 0.023, 0.104]
                elif (girder == 'B03' or girder == 'B11'):
                    # quad type: QDA
                    if ('QUAD01' in magnet):
                        unc = [0.0189, 0.0434, 0.023, 0.104]
                    # dipole
                    elif ('B1' in magnet):
                        unc = [0.0269, 0.0474, 0.023, 0.03]
            # low-beta sectors (type B)
            elif (sector in sectorB):
                if (girder == 'B01'):
                    # quad type: QFB
                    if ('QUAD01' in magnet):
                        unc = [0.0194, 0.0434, 0.023, 0.058]
                    # quad type: QDB2
                    elif ('QUAD02' in magnet):
                        unc = [0.0195, 0.0433, 0.023, 0.104]
                elif (girder == 'B02'):
                    # quad type: QFB
                    if ('QUAD02' in magnet):
                        unc = [0.0194, 0.0434, 0.023, 0.058]
                    # quad type: QDB2
                    elif ('QUAD01' in magnet):
                        unc = [0.0195, 0.0433, 0.023, 0.104]
                elif (girder == 'B03' or girder == 'B11'):
                    # quad type: QDB1
                    if ('QUAD01' in magnet):
                        unc = [0.019, 0.0435, 0.023, 0.104]
                    # dipole
                    elif ('B1' in magnet):
                        unc = [0.0269, 0.0474, 0.023, 0.03]
            # low-beta sectors (type P)
            elif (sector in sectorP):
                if (girder == 'B01'):
                    # quad type: QFP
                    if ('QUAD01' in magnet):
                        unc = [0.0188, 0.0433, 0.023, 0.104]
                    # quad type: QDP2
                    elif ('QUAD02' in magnet):
                        unc = [0.0189, 0.0435, 0.023, 0.104]
                elif (girder == 'B02'):
                    # quad type: QFP
                    if ('QUAD02' in magnet):
                        unc = [0.0188, 0.0433, 0.023, 0.104]
                    # quad type: QDP2
                    elif ('QUAD01' in magnet):
                        unc = [0.0189, 0.0435, 0.023, 0.104]
                elif (girder == 'B03' or girder == 'B11'):
                    # quad type: QDP1
                    if ('QUAD01' in magnet):
                        unc = [0.0189, 0.0433, 0.023, 0.104]
                    # dipole
                    elif ('B1' in magnet):
                        unc = [0.0269, 0.0474, 0.023, 0.03]

        elif (girder == 'B04'):
            # quad type: Q1
            if ('QUAD01' in magnet):
                unc = [0.0193, 0.0434, 0.023, 0.104]
            # quad type: Q2
            elif ('QUAD02' in magnet):
                unc = [0.0188, 0.0435, 0.023, 0.104]
        elif (girder == 'B05'):
            # dipole
            if ('B2' in magnet):
                unc = [0.0269, 0.0474, 0.023, 0.03]
        elif (girder == 'B06'):
            # quad type: Q3
            if ('QUAD01' in magnet):
                unc = [0.019, 0.0436, 0.023, 0.104]
            # quad type: Q4
            elif ('QUAD02' in magnet):
                unc = [0.0189, 0.0434, 0.023, 0.104]
        elif (girder == 'B07'):
            # dipole
            if ('BC' in magnet):
                unc = [0.0269, 0.0474, 0.023, 0.03]
        elif (girder == 'B08'):
            # quad type: Q4
            if ('QUAD01' in magnet):
                unc = [0.0189, 0.0434, 0.023, 0.104]
            # quad type: Q3
            elif ('QUAD02' in magnet):
                unc = [0.019, 0.0436, 0.023, 0.104]
        elif (girder == 'B09'):
            # dipole
            if ('B2' in magnet):
                unc = [0.0269, 0.0474, 0.023, 0.03]
        elif (girder == 'B10'):
            # quad type: Q2
            if ('QUAD01' in magnet):
                unc = [0.0188, 0.0435, 0.023, 0.104]
            # quad type: Q1
            elif ('QUAD02' in magnet):
                unc = [0.0193, 0.0434, 0.023, 0.104]
        
        if (uncOrTrans == 'unc'):
            output = unc[DOF_num]
        else:
            output = trans

        return output

    @staticmethod
    def checkMagnetFamily(magnet):
        sectorA = ['S01', 'S05', 'S09', 'S13', 'S17']
        sectorB = ['S02', 'S04', 'S06', 'S08', 'S10', 'S12', 'S14', 'S16', 'S18', 'S20']
        sectorP = ['S03', 'S07', 'S11', 'S15', 'S19']

        mns = magnet.split('-')
        sector, girder, magnet = mns[0], mns[1], mns[2]

        # girder influenced by the sector type
        if (girder == 'B02' or girder == 'B03' or girder == 'B11' or girder == 'B01'):
            # high-beta sectors (type A)
            if (sector in sectorA):
                if (girder == 'B02' or girder == 'B01'):
                    # quad type: QFA
                    if ('QUAD01' in magnet):
                        family = 'QFA'
                elif (girder == 'B03' or girder == 'B11'):
                    # quad type: QDA
                    if ('QUAD01' in magnet):
                        family = 'QDA'
                    # dipole
                    elif ('B1' in magnet):
                        family = 'dipole-B1'
            # low-beta sectors (type B)
            elif (sector in sectorB):
                if (girder == 'B01'):
                    # quad type: QFB
                    if ('QUAD01' in magnet):
                        family = 'QFB'
                    # quad type: QDB2
                    elif ('QUAD02' in magnet):
                        family = 'QDB2'
                elif (girder == 'B02'):
                    # quad type: QFB
                    if ('QUAD02' in magnet):
                        family = 'QFB'
                    # quad type: QDB2
                    elif ('QUAD01' in magnet):
                        family = 'QDB2'
                elif (girder == 'B03' or girder == 'B11'):
                    # quad type: QDB1
                    if ('QUAD01' in magnet):
                        family = 'QDB1'
                    # dipole
                    elif ('B1' in magnet):
                        family = 'dipole-B1'
            # low-beta sectors (type P)
            elif (sector in sectorP):
                if (girder == 'B01'):
                    # quad type: QFP
                    if ('QUAD01' in magnet):
                        family = 'QFP'
                    # quad type: QDP2
                    elif ('QUAD02' in magnet):
                        family = 'QDP2'
                elif (girder == 'B02'):
                    # quad type: QFP
                    if ('QUAD02' in magnet):
                        family = 'QFP'
                    # quad type: QDP2
                    elif ('QUAD01' in magnet):
                        family = 'QDP2'
                elif (girder == 'B03' or girder == 'B11'):
                    # quad type: QDP1
                    if ('QUAD01' in magnet):
                        family = 'QDP1'
                    # dipole
                    elif ('B1' in magnet):
                        family = 'dipole-B1'

        elif (girder == 'B04'):
            # quad type: Q1
            if ('QUAD01' in magnet):
                family = 'Q1'
            # quad type: Q2
            elif ('QUAD02' in magnet):
                family = 'Q2'
        elif (girder == 'B05'):
            # dipole
            if ('B2' in magnet):
                family = 'dipole-B2'
        elif (girder == 'B06'):
            # quad type: Q3
            if ('QUAD01' in magnet):
                family = 'Q3'
            # quad type: Q4
            elif ('QUAD02' in magnet):
                family = 'Q4'
        elif (girder == 'B07'):
            # dipole
            if ('BC' in magnet):
                family = 'dipole-BC'
        elif (girder == 'B08'):
            # quad type: Q4
            if ('QUAD01' in magnet):
                family = 'Q4'
            # quad type: Q3
            elif ('QUAD02' in magnet):
                family = 'Q3'
        elif (girder == 'B09'):
            # dipole
            if ('B2' in magnet):
                family = 'dipole-B2'
        elif (girder == 'B10'):
            # quad type: Q2
            if ('QUAD01' in magnet):
                family = 'Q2'
            # quad type: Q1
            elif ('QUAD02' in magnet):
                family = 'Q1'

        return family

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
    def calculateLocalDeviations(magnet, pointList):
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

        if ('LTB' not in prefix):
            credentials["isLTB"] = False
            return credentials

        if (len(details) != 3):
            credentials["isValidPoint"] = False
            return credentials

        magnetName = details[0] + '-' + details[1]
        magnetRef = details[1]

        if ('QD' in magnetRef or 'QF' in magnetRef):
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
    def calculateLocalDeviations(magnet, pointList):
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

        if ('BTS' not in prefix):
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

class FE():
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
    def calculateLocalDeviations(magnet, pointList):
        pointsMeasured = pointList[0]
        pointsNominal = pointList[1]

        # no dof separation
        dofs = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        deviation = DataUtils.evaluateDeviation(pointsMeasured[0], pointsNominal[0], dofs)

        frameFrom = magnet.name + '-NOMINAL'
        frameTo = magnet.name + '-MEASURED'

        localTransformation = Transformation(frameFrom, frameTo, deviation[0], deviation[1], deviation[2], deviation[3], deviation[4], deviation[5])

        return localTransformation

    @staticmethod
    def generateCredentials(pointName):
        credentials = {}
        credentials["isFE"] = True
        credentials["isValidPoint"] = True

        details = pointName.split('-')
        prefixes = ['MOG', 'EMA', 'CAR', 'CAT', 'MAN', 'IPE']
        prefix = details[0]

        if (prefix not in prefixes):
            credentials["isFE"] = False
            return credentials

        if (len(details) != 3):
            credentials["isValidPoint"] = False
            return credentials

        magnetName = details[0] + '-' + details[1]

        # dummy magnet type, because this information isn't important for FE analysis
        magnetType = 'generic'

        credentials["prefix"] = prefix
        credentials["magnetType"] = magnetType
        credentials["pointName"] = pointName
        credentials["magnetName"] = magnetName

        return credentials

    @staticmethod
    def createObjectsStructure(pointsDF):

        FEDict = {}

        # iterate over dataframe
        for pointName, coordinate in pointsDF.iterrows():

            credentials = FE.generateCredentials(pointName)

            if (not credentials['isFE'] or not credentials['isValidPoint']):
                continue

            # instantiating a Point object
            point = Point(credentials['pointName'], coordinate['x'], coordinate['y'], coordinate['z'], 'machine-local')

            # finding Girder object or instantiating a new one
            if (not credentials['prefix'] in FEDict):
                FEDict[credentials['prefix']] = {}

            magnetDict = FEDict[credentials['prefix']]

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
                FEDict[credentials['prefix']][credentials['magnetName']] = magnet

        # SR.reorderDict(girderDict)

        return FEDict

from dataUtils import DataUtils