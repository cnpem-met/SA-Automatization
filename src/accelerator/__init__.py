from abc import ABC, abstractmethod

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
        self.girders = {'measured': {}, 'nominal': {}}
        self.deviations = None
    
    def generateFrameDict(self):
        # creating base frame
        self.frames['nominal']['machine-local'] = Frame('machine-local', Transformation('machine-local', 'machine-local', 0, 0, 0, 0, 0, 0))

        # iterate over dataframe
        for frameName, dof in self.lookupTable.iterrows():
            transformation = Transformation('machine-local', frameName, dof['Tx'], dof['Ty'], dof['Tz'], dof['Rx'], dof['Ry'], dof['Rz'])

            frame = Frame(frameName, transformation)
            self.frames['nominal'][frameName] = frame

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
    
    def generateMeasuredFrames(self, ui, magnet_types_to_ignore: list = []):
        for girder in self.girders['measured']:
            magnetDict = self.girders['measured'][girder]
            for magnetName in magnetDict:
                magnet_meas = magnetDict[magnetName]
                pointDict = magnet_meas.pointDict

                if magnet_meas.type in magnet_types_to_ignore:
                    continue

                # creating lists for measured and nominal points
                try:
                    magnet_nom = self.girders['nominal'][girder][magnetName]
                    pointList = self.appendPoints(magnet_meas, magnet_nom)
                except KeyError:
                    ui.logMessage('Falha no imã ' + magnet_meas.name +': o nome de 1 ou mais pontos nominais não correspondem aos medidos.', severity="danger")
                    continue
                
                # calculating the transformation from the nominal points to the measured ones
                try:
                    localTransf, baseTransf = self.calculateNominalToMeasuredTransformation(magnet_meas, pointDict, pointList)
                except KeyError:
                    ui.logMessage('Falha no imã ' + magnet_meas.name +': frame medido do quadrupolo adjacente ainda não foi calculado.', severity='danger')
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

    @abstractmethod
    def appendPoints(self, magnet_meas, magnet_nom):
        pointsMeasured = [[], []]
        pointsNominal = [[], []]

        pointDict = magnet_meas.pointDict

        for pointName in pointDict:
            pointMeas = pointDict[pointName]
            pointNom = magnet_nom.pointDict[pointName]

            pointName = pointMeas.name
            pointMeasCoord = [pointMeas.x, pointMeas.y, pointMeas.z]
            pointNomCoord = [pointNom.x, pointNom.y, pointNom.z]

            # adicionando pontos às suas respectivas listas
            pointsMeasured[0].append(pointMeasCoord)
            pointsNominal[0].append(pointNomCoord)

        return (pointsMeasured, pointsNominal)

    @abstractmethod
    def calculateNominalToMeasuredTransformation(self, magnet, pointDict, pointList):
        (pointsMeasured, pointsNominal) = pointList

        # no dof separation
        dofs = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        deviation = evaluateDeviation(pointsMeasured[0], pointsNominal[0], dofs)

        frameFrom = magnet.name
        frameTo = magnet.name

        localTransformation = Transformation(frameFrom, frameTo, deviation[0], deviation[1],
                                             deviation[2], deviation[3], deviation[4], deviation[5])
        baseTransformation = self.frames['nominal'][frameFrom].transformation
        
        return localTransformation, baseTransformation

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
            point = Point(credentials['pointName'], coordinate['x'], coordinate['y'], coordinate['z'], 'machine-local')

            # finding Girder object or instantiating a new one
            if (not credentials['location'] in self.girders[type_of_points]):
                self.girders[type_of_points][credentials['location']] = {}

            magnetDict = self.girders[type_of_points][credentials['location']]

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
                self.girders[type_of_points][credentials['location']][credentials['magnetName']] = magnet

    @abstractmethod
    def generateCredentials(self, pointName):
        pass