from accelerator import Accelerator

class BTS(Accelerator):
    def __init__(self, name) -> None:
        super().__init__(name)

    def appendPoints(self, magnet_meas, magnet_nom):
        return super().appendPoints(magnet_meas, magnet_nom)

    def calculateNominalToMeasuredTransformation(self, magnet, pointDict, pointList):
        return super().calculateNominalToMeasuredTransformation(magnet, pointDict, pointList)

    def createObjectsStructure(self, type_of_points):
        super().createObjectsStructure(type_of_points)

    def sortFrameDictByBeamTrajectory(self, type_of_points):
        super().sortFrameDictByBeamTrajectory(type_of_points)

    def generateCredentials(self, pointName):
        credentials = {}
        credentials["isCurrentAcceleratorPoint"] = True
        credentials["isValidPoint"] = True

        details = pointName.split('-')
        prefix = details[0]

        if ('BTS' not in prefix):
            credentials["isCurrentAcceleratorPoint"] = False
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

        credentials["location"] = prefix
        credentials["magnetType"] = magnetType
        credentials["pointName"] = pointName
        credentials["magnetName"] = magnetName

        return credentials