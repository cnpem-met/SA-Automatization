from geom_entitites.magnet import Magnet
from geom_entitites.point import Point
from geom_entitites.transformation import Transformation
from geom_entitites.frame import Frame

from operations.geometrical import evaluateDeviation
from accelerator import Accelerator

import numpy as np
import math

class SR(Accelerator):
    def __init__(self, name):
        super().__init__(name)

    def populate_magnets_with_points(self, type_of_points):
        # iterate over dataframe
        for pointName, coordinate in self.points[type_of_points].iterrows():
            
            credentials = self.get_point_info(pointName)

            if (not credentials['isCurrentAcceleratorPoint'] or not credentials['isValidPoint']):
                continue

            # instantiating a Point object
            point = Point(credentials['pointName'], coordinate['x'], coordinate['y'], coordinate['z'], Frame.machine_local())

            # finding Magnet object or instantiating a new one
            if (credentials['magnetName'] in self.magnets[type_of_points]):
                # reference to the Magnet object
                magnet = self.magnets[type_of_points][credentials['magnetName']]
                # append point into its point list
                magnet.addPoint(point)
                # if B1 and measured, insert offset into magnets properties
                if (type_of_points == 'measured' and credentials['magnetType'] == 'dipole-B1'):
                    magnet.shift = self.shiftsB1[credentials['magnetName']]
            else:
                # instantiate new Magnet object
                magnet = Magnet(credentials['magnetName'], credentials['magnetType'], credentials['location'])
                magnet.addPoint(point)
                if (type_of_points == 'measured' and credentials['magnetType'] == 'dipole-B1'):
                    magnet.shift = self.shiftsB1[credentials['magnetName']]
                # # including on data structure
                self.magnets[type_of_points][credentials['magnetName']] = magnet
            
        self.reorderMagnets(type_of_points)

    def sort_frames_by_beam_trajectory(self, type_of_points):
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

    def reorderMagnets(self, type_of_points):
        # sorting list of magnet names
        magnet_names_sorted = sorted(self.magnets[type_of_points].keys())

        # exchanguing S0x-B03-B1 with S0x-B03-QUAD01 (same applies to B11)
        for i, magnet_name in enumerate(magnet_names_sorted):
            if 'B03-QUAD01' in magnet_name or 'B11-QUAD01' in magnet_name:
                b1 = f'{magnet_name.split("-")[0]}-{magnet_name.split("-")[1]}-B1-LONG'
                b1_idx = magnet_names_sorted.index(b1)
                magnet_names_sorted[b1_idx] = magnet_names_sorted[i]
                magnet_names_sorted[i] = b1
                
        # reordering magnet's dictionary based on the new order
        self.magnets[type_of_points] = {k: self.magnets[type_of_points][k] for k in magnet_names_sorted}

    def get_points_to_bestfit(self, magnet_meas, magnet_nom):

        if magnet_meas.type != 'dipole-B2':
            points_measured = magnet_meas.get_ordered_points_coords()
            points_nominal = magnet_nom.get_ordered_points_coords()
        else:
            points_measured = {'TyRxRz': [], 'TxTzRy': []}
            points_nominal = {'TyRxRz': [], 'TxTzRy': []}

            pointDict = magnet_meas.pointDict

            for pointName in pointDict:
                pointMeas = pointDict[pointName]
                pointNom = magnet_nom.pointDict[pointName]

                # diferenciando conjunto de pontos para comporem diferentes graus de liberdade
                if ('LVL' in pointName):
                    # DoFs: ['Ty', 'Rx', 'Rz']
                    points_measured['TyRxRz'].append(pointMeas.coord_list)
                    points_nominal['TyRxRz'].append(pointNom.coord_list)
                else:
                    # DoFs: ['Tx', 'Tz', 'Ry']
                    points_measured['TxTzRy'].append(pointMeas.coord_list)
                    points_nominal['TxTzRy'].append(pointNom.coord_list)

        return {'measured': points_measured, 'nominal': points_nominal}

    def calculate_nominal_to_measured_transformation(self, magnet, pointList) -> Transformation:
        if (magnet.type == 'dipole-B1'):
            pointDict = magnet.pointDict
            girderName = f'{magnet.name.split("-")[0]}-{magnet.name.split("-")[1]}'

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
                deviationPartial1 = evaluateDeviation(pointList['measured']['TyRxRz'], pointList['nominal']['TyRxRz'], dofs)
            
                # Forcing the "corresponding group"
                transfPoints = []
                for point in pointList['nominal']['TxTzRy']:
                    transfPt = Point('n/c', point[0], point[1], point[2])
                    # transfPt = Point.copyFromPoint(point)
                    transform = Transformation('n/c', 'n/c', 0, deviationPartial1[0], 0, deviationPartial1[1], 0, deviationPartial1[2])
                    transfPt.transform(transform.transfMatrix, 'n/c')
                    transfPoints.append([transfPt.x, transfPt.y, transfPt.z])

                # 2nd group of points
                dofs = ['Tx', 'Tz', 'Ry']
                deviationPartial2 = evaluateDeviation(pointList['measured']['TxTzRy'], transfPoints, dofs)

                # agregating parcial dofs
                deviation = np.array(
                    [deviationPartial2[0], deviationPartial1[0], deviationPartial2[1], deviationPartial1[1], deviationPartial2[2], deviationPartial1[2]])
            else:
                # no dof separation
                dofs = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
                deviation = evaluateDeviation(pointList['measured'], pointList['nominal'], dofs)

            localTransformation = Transformation(magnet.name, magnet.name, *deviation)
            baseTransformation = self.frames['nominal'][magnet.name].transformation

        transformation = Transformation.compose_transformations(baseTransformation, localTransformation, 'machine-local', magnet.name)
        return transformation

    def get_point_info(self, pointName):
        credentials = {}
        credentials["isCurrentAcceleratorPoint"] = True
        credentials["isValidPoint"] = True

        details = pointName.split('-')
        sectorID = details[0]
        girderID = details[1]
        girder = sectorID + '-' + girderID

        if (sectorID[0] != 'S' or girderID[0] != 'B'):
            credentials["isCurrentAcceleratorPoint"] = False
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

        credentials["location"] = girder
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