from geom_entitites.transformation import Transformation
from geom_entitites.point import Point
from geom_entitites.magnet import Magnet

from operations.geometrical import evaluateDeviation

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
        deviation = evaluateDeviation(pointsMeasured[0], pointsNominal[0], dofs)

        frameFrom = magnet.name + '-NOMINAL'
        frameTo = magnet.name + '-MEASURED'

        localTransformation = Transformation(frameFrom, frameTo, deviation[0], deviation[1],
                                             deviation[2], deviation[3], deviation[4], deviation[5])

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
            point = Point(credentials['pointName'], coordinate['x'], coordinate['y'], coordinate['z'], 'machine-local')

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
