from abc import ABC, abstractmethod
import math
import numpy as np

import pandas as pd
from geom_entitites.magnet import Magnet
from geom_entitites.point import Point

from geom_entitites.transformation import Transformation
from geom_entitites.frame import Frame
from operations.geometrical import evaluateDeviation

class Accelerator(ABC):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
        self.lookupTable = None
        self.frames = {'measured': {}, 'nominal': {}}
        self.points = {'measured': None, 'nominal': None}
        # self.girders = {'measured': {}, 'nominal': {}}
        self.magnets = {'measured': {}, 'nominal': {}}
        self.deviations = None
    
    def generateFrameDict(self):
        # creating base frame
        self.frames['nominal']['machine-local'] = Frame.machine_local()

        # iterate over dataframe
        for frameName, dof in self.lookupTable.iterrows():
            transformation = Transformation('machine-local', frameName, dof['Tx'], dof['Ty'], dof['Tz'], dof['Rx'], dof['Ry'], dof['Rz'])

            frame = Frame(frameName, transformation)
            self.frames['nominal'][frameName] = frame

    def transformToLocalFrame(self, type_of_points):
        magnetsNotComputed = ""  
        for magnetName in self.magnets[type_of_points]:
            magnet = self.magnets[type_of_points][magnetName]
            try:
                magnet.transformFrame(self.frames['nominal'][magnetName])
            except KeyError:
                magnetsNotComputed += magnet.name + ','

        # at least one frame was not computed
        if (magnetsNotComputed != ""):
            # print("[Transformação p/ frames locais] Imãs não computados: " + magnetsNotComputed[:-1])
            pass
    
    def generateMeasuredFrames(self, ui, magnet_types_to_ignore: list = []):
        for magnetName in self.magnets['measured']:
            magnet_meas = self.magnets['measured'][magnetName]
            magnet_nom = self.magnets['nominal'][magnetName]

            if magnet_meas.type in magnet_types_to_ignore:
                continue

            # checking if points are in both measured and nominal magnet objects
            if not self.check_points_coexistance(magnet_meas, magnet_nom):
                ui.logMessage('Falha no imã ' + magnet_meas.name +': o nome de 1 ou mais pontos nominais não correspondem aos medidos.', severity="danger")
                continue
            
            # get list of measured and nominal point coordinates to further compute transformation by the means of a bestfit
            points = self.get_points_to_bestfit(magnet_meas, magnet_nom)

            # calculating the transformation from the nominal points to the measured ones
            try:
                transformation = self.calculateNominalToMeasuredTransformation(magnet_meas, points)
            except KeyError:
                ui.logMessage('Falha no imã ' + magnet_meas.name +': frame medido do quadrupolo adjacente ainda não foi calculado.', severity='danger')
                continue

            # creating a new Frame with the transformation from the measured magnet to machine-local
            measMagnetFrame = Frame(magnet_meas.name, transformation)

            # adding it to the frame dictionary
            self.frames['measured'][measMagnetFrame.name] = measMagnetFrame

    def calculateMagnetsDeviations(self) -> bool:
        header = ['Magnet', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        deviationList = []

        # at least one measured and nominal frame has to exist
        if self.frames['measured'] == {} or self.frames['nominal'] == {}:
            return False

        for frameName in self.frames['nominal']:
            frame = self.frames['nominal'][frameName]

            if (frame.name == 'machine-local'):
                continue

            try:
                frameFrom = self.frames['measured'][frame.name]
                frameTo = self.frames['nominal'][frame.name]
            except KeyError:
                print(f'error in magnet {frame.name}')
                continue

            try:
                transformation = Transformation.evaluateTransformation(frameFrom, frameTo)
                dev = Transformation.individualDeviationsFromEquations(transformation)
            except KeyError:
                dev = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]
                pass

            dev.insert(0, frame.name)

            deviation = pd.DataFrame([dev], columns=header)
            deviationList.append(deviation)

        deviations = pd.concat(deviationList, ignore_index=True)
        deviations = deviations.set_index(header[0])

        # retorna o df com os termos no tipo numérico certo
        self.deviations = deviations.astype('float32')

        # return positivite flag to allow plotting
        return True

    def calculateMagnetsLongitudinalDistances(self, type_of_points):
        header = ['Reference', 'Distance (mm)']

        distancesList = []
        frameDictKeys = list(self.frames[type_of_points].keys())

        for (index, frameName) in enumerate(self.frames[type_of_points]):
            frame = self.frames[type_of_points][frameName]

            if ('machine-local' in frame.name):
                continue

            if (index == 0):
                firstMagnetsFrame = frame

            if (index == len(self.frames[type_of_points]) - 1):
                # first frame of the dictionary
                magnet2frame = firstMagnetsFrame
            else:
                # next measured frame
                magnet2frame = self.frames[type_of_points][frameDictKeys[index+1]]

            magnet1frame = frame

            magnet1coord = np.array([magnet1frame.transformation.Tx, magnet1frame.transformation.Ty, magnet1frame.transformation.Tz])
            magnet2coord = np.array([magnet2frame.transformation.Tx, magnet2frame.transformation.Ty, magnet2frame.transformation.Tz])

            vector = magnet2coord - magnet1coord
            distance = math.sqrt(vector[0]**2 + vector[1]**2)

            splitedName1 = magnet1frame.name.split('-')
            splitedName2 = magnet2frame.name.split('-')

            magnet1frameName = splitedName1[0] + '-' + splitedName1[1] + '- ' + splitedName1[2]
            magnet2frameName = splitedName2[0] + '-' + splitedName2[1] + '- ' + splitedName2[2]

            distanceReference = magnet1frameName + ' x ' + magnet2frameName

            distanceDF = pd.DataFrame([[distanceReference, distance]], columns=header)
            distancesList.append(distanceDF)

        distances = pd.concat(distancesList, ignore_index=True)
        distances = distances.set_index(header[0])

        # retorna o df com os termos no tipo numérico certo
        return distances.astype('float32')

    def check_points_coexistance(self, magnet_meas, magnet_nom):
        for point_name in magnet_meas.pointDict:
            if not point_name in magnet_nom.pointDict:
                return False
        return True

    @abstractmethod
    def get_points_to_bestfit(self, magnet_meas, magnet_nom):

        points_measured = magnet_meas.get_ordered_points_coords()
        points_nominal = magnet_nom.get_ordered_points_coords()

        return {'measured': points_measured, 'nominal': points_nominal}

    @abstractmethod
    def calculateNominalToMeasuredTransformation(self, magnet, pointList) -> Transformation:
        # (pointsMeasured, pointsNominal) = pointList

        # no dof separation
        dofs = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        deviation = evaluateDeviation(pointList['measured'], pointList['nominal'], dofs)

        localTransformation = Transformation(magnet.name, magnet.name, *deviation)
        baseTransformation = self.frames['nominal'][magnet.name].transformation
        
        transformation = Transformation.compose_transformations(baseTransformation, localTransformation, 'machine-local', magnet.name)

        return transformation

    @abstractmethod
    def sortFrameDictByBeamTrajectory(self, type_of_points):
        # sorting in alphabetical order
        keySortedList = sorted(self.frames[type_of_points])
        sortedFrameDict = {}
        for key in keySortedList:
            sortedFrameDict[key] = self.frames[type_of_points][key]
        
        self.frames[type_of_points] = sortedFrameDict

    @abstractmethod
    def createObjectsStructure(self, type_of_points):
        # iterate over dataframe
        for pointName, coordinate in self.points[type_of_points].iterrows():
            
            credentials = self.generateCredentials(pointName)

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
            else:
                # instantiate new Magnet object
                magnet = Magnet(credentials['magnetName'], credentials['magnetType'], credentials['location'])
                magnet.addPoint(point)
                # # including on data structure
                self.magnets[type_of_points][credentials['magnetName']] = magnet
            
            
            


    @abstractmethod
    def generateCredentials(self, pointName):
        pass