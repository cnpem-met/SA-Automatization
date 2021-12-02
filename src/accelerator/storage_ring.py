from geom_entitites.magnet import Magnet
from geom_entitites.point import Point
from geom_entitites.transformation import Transformation
from geom_entitites.frame import Frame

from operations.geometrical import evaluateDeviation

import numpy as np
import math

class SR():
    def __init__(self):
        self.lookupTable = None
        self.frames = {'measured': {}, 'nominal': {}}
        self.points = {'measured': None, 'nominal': None}
        self.girders = {'measured': {}, 'nominal': {}}
        self.shiftsB1 = None

    def generateFrameDict(self):
        # creating base frame
        self.frames['nominal']['machine-local'] = Frame('machine-local', Transformation('machine-local', 'machine-local', 0, 0, 0, 0, 0, 0))

        # iterate over dataframe
        for frameName, dof in self.lookupTable.iterrows():
            # newFrameName = frameName + "-NOMINAL"
            transformation = Transformation('machine-local', frameName, dof['Tx'], dof['Ty'], dof['Tz'], dof['Rx'], dof['Ry'], dof['Rz'])

            frame = Frame(frameName, transformation)
            self.frames['nominal'][frameName] = frame

    def createObjectsStructure(self, type_of_points):
        # iterate over dataframe
        for pointName, coordinate in self.points[type_of_points].iterrows():
            
            credentials = SR.generateCredentials(pointName)

            if (not credentials['isSR'] or not credentials['isValidPoint']):
                continue

            # instantiating a Point object
            point = Point(credentials['pointName'], coordinate['x'], coordinate['y'], coordinate['z'], 'machine-local')

            # finding Girder object or instantiating a new one
            if (not credentials['girder'] in self.girders[type_of_points]):
                self.girders[type_of_points][credentials['girder']] = {}

            magnetDict = self.girders[type_of_points][credentials['girder']]

            # finding Magnet object or instantiating a new one
            if (credentials['magnetName'] in magnetDict):
                # reference to the Magnet object
                magnet = magnetDict[credentials['magnetName']]
                # append point into its point list
                magnet.addPoint(point)
                if (type_of_points == 'measured' and credentials['magnetType'] == 'dipole-B1'):
                    magnet.shift = self.shiftsB1[credentials['magnetName']]
            else:
                # instantiate new Magnet object
                magnet = Magnet(credentials['magnetName'], credentials['magnetType'])
                magnet.addPoint(point)
                if (type_of_points == 'measured' and credentials['magnetType'] == 'dipole-B1'):
                    magnet.shift = self.shiftsB1[credentials['magnetName']]

                # including on data structure
                self.girders[type_of_points][credentials['girder']][credentials['magnetName']] = magnet

        self.reorderDict(type_of_points)

    def transformToLocalFrame(self, type_of_points):
        magnetsNotComputed = ""
        for girder in self.girders[type_of_points]:
            magnetDict = self.girders[type_of_points][girder]
                
            for magnetName in magnetDict:
                magnet = magnetDict[magnetName]
                # targetFrame = magnetName + "-NOMINAL"
                try:
                    magnet.transformFrame(self.frames['nominal'], self.frames['nominal'][magnetName])
                except KeyError:
                    magnetsNotComputed += magnet.name + ','

        # at least one frame was not computed
        if (magnetsNotComputed != ""):
            print("[Transformação p/ frames locais] Imãs não computados: " + magnetsNotComputed[:-1])
            pass

    def generateMeasuredFrames(self, ui):
        typeOfMagnets = ['quadrupole', 'dipole-B1', 'dipole-B2', 'dipole-BC']

        for girder in self.girders['measured']:
            magnetDict = self.girders['measured'][girder]
            for magnetName in magnetDict:
                magnet = magnetDict[magnetName]
                pointDict = magnet.pointDict

                if (magnet.type not in typeOfMagnets):
                    continue

                # creating lists for measured and nominal points
                try:
                    pointList = SR.appendPoints(pointDict, self.girders['nominal'], magnet, girder)
                except KeyError:
                    ui.logMessage('Falha no imã ' + magnet.name +': o nome de 1 ou mais pontos nominais não correspondem aos medidos.', severity="danger")
                    continue
                
                # calculating the transformation from the nominal points to the measured ones
                try:
                    localTransf, baseTransf = self.calculateTransformationsByTypeOfMagnet(magnet, girder, pointDict, pointList)
                except KeyError:
                    ui.logMessage('Falha no imã ' + magnet.name +': frame medido do quadrupolo adjacente ainda não foi calculado.', severity='danger')
                    continue

                # calculating the homogeneous matrix of the transformation from the frame of the measured magnet to the machine-local
                transfMatrix = baseTransf.transfMatrix @ localTransf.transfMatrix

                # updating the homogeneous matrix and frameFrom of the already created Transformation
                # tf = transfMatrix
                # transformation = Transformation('machine-local', localTransformation.frameTo, tf[0], tf[1], tf[2], tf[3], tf[4], tf[5])
                transformation = localTransf
                transformation.transfMatrix = transfMatrix
                transformation.frameFrom = 'machine-local'

                transformation.Tx = transfMatrix[0, 3]
                transformation.Ty = transfMatrix[1, 3]
                transformation.Tz = transfMatrix[2, 3]
                transformation.Rx = baseTransf.Rx + localTransf.Rx
                transformation.Ry = baseTransf.Ry + localTransf.Ry
                transformation.Rz = baseTransf.Rz + localTransf.Rz

                # creating a new Frame with the transformation from the measured magnet to machine-local
                measMagnetFrame = Frame(transformation.frameTo, transformation)

                # adding it to the frame dictionary
                self.frames['measured'][measMagnetFrame.name] = measMagnetFrame

    def sortFrameDictByBeamTrajectory(self, type_of_points):
        # sorting in alphabetical order
        keySortedList = sorted(self.frames[type_of_points])
        sortedFrameDict = {}
        for key in keySortedList:
            sortedFrameDict[key] = self.frames[type_of_points][key]

        finalFrameDict = {}
        b1FrameList = []
        # correcting the case of S0xB03
        for frameName in sortedFrameDict:
            frame = sortedFrameDict[frameName]

            if ('B03-B1' in frame.name):
                b1FrameList.append(frame)
                continue

            finalFrameDict[frame.name] = frame

            if ('B03-QUAD01' in frame.name):
                for b1Frame in b1FrameList:
                    finalFrameDict[b1Frame.name] = b1Frame
                b1FrameList = []

            self.frames[type_of_points] = finalFrameDict

    def reorderDict(self, type_of_points):
        for girderName in self.girders[type_of_points]:
            if ('B03' in girderName or 'B11' in girderName):
                magnetDict = self.girders[type_of_points][girderName]

                orderedList = []
                newMagnetDict = {}

                for magnetName in magnetDict:
                    magnet = magnetDict[magnetName]
                    orderedList.insert(0, (magnetName, magnet))

                for item in orderedList:
                    newMagnetDict[item[0]] = item[1]

                self.girders[type_of_points][girderName] = newMagnetDict

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

    def calculateTransformationsByTypeOfMagnet(self, magnet, girderName, pointDict, pointList):
        pointsMeasured = pointList[0]
        pointsNominal = pointList[1]

        if (magnet.type == 'dipole-B1'):
            # changuing points' frame: from the current activated frame ("dipole-nominal") to the "quadrupole-measured"
            frameFrom = self.frames['nominal'][f'{magnet.name}']
            frameTo = self.frames['measured'][f'{girderName}-QUAD01']

            Frame.changeFrame(frameFrom, frameTo, pointDict)

            # listing the points' coordinates in the new frame
            points = []
            for pointName in pointDict:
                point = pointDict[pointName]
                points.append([point.x, point.y, point.z])

            # changuing back point's frame
            Frame.changeFrame(frameTo, frameFrom, pointDict)

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
            frameFrom = girderName + '-QUAD01'
            frameTo = magnet.name
            localTransformation = Transformation(frameFrom, frameTo, 0, 0, shiftedPointPosition[2], 0, dRy * 10**3, 0)  # mrad
            baseTransformation = self.frames['measured'][frameFrom].transformation

        else:
            if (magnet.type == 'dipole-B2'):
                # calculating transformation for the 1st group of points associated
                # with specific degrees of freedom
                dofs = ['Ty', 'Rx', 'Rz']
                # print(pointsMeasured, pointsNominal)
                deviationPartial1 = evaluateDeviation(pointsMeasured[0], pointsNominal[0], dofs)
            

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
                deviationPartial2 = evaluateDeviation(pointsMeasured[1], transfPoints, dofs)

                # agregating parcial dofs
                deviation = np.array(
                    [deviationPartial2[0], deviationPartial1[0], deviationPartial2[1], deviationPartial1[1], deviationPartial2[2], deviationPartial1[2]])
            else:
                # no dof separation
                dofs = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
                deviation = evaluateDeviation(pointsMeasured[0], pointsNominal[0], dofs)

            # creating new Frames and inserting in the frame dictionary
            # [--- hard-coded, but auto would be better ---]
            frameFrom = magnet.name
            frameTo = magnet.name

            localTransformation = Transformation(frameFrom, frameTo, deviation[0], deviation[1], deviation[2], deviation[3], deviation[4], deviation[5])
            baseTransformation = self.frames['nominal'][frameFrom].transformation


        return localTransformation, baseTransformation

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
                    # targetFrame = magnetName+'-NOMINAL'
                    targetFrame = magnetName
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