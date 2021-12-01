from geom_entitites.magnet import Magnet
from geom_entitites.point import Point
from geom_entitites.transformation import Transformation

from operations.geometrical import evaluateDeviation

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
        deviation = evaluateDeviation(pointsMeasured[0], pointsNominal[0], dofs)

        frameFrom = magnet.name + '-NOMINAL'
        frameTo = magnet.name + '-MEASURED'

        localTransformation = Transformation(frameFrom, frameTo, deviation[0], deviation[1], deviation[2], deviation[3], deviation[4], deviation[5])

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
