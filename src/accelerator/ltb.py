from accelerator import Accelerator

class LTB(Accelerator):
    def __init__(self, name) -> None:
        super().__init__(name)

    def get_points_to_bestfit(self, magnet_meas, magnet_nom):
        return super().get_points_to_bestfit(magnet_meas, magnet_nom)

    def calculateNominalToMeasuredTransformation(self, magnet, pointList):
        return super().calculateNominalToMeasuredTransformation(magnet, pointList)

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

        if ('LTB' not in prefix):
            credentials["isCurrentAcceleratorPoint"] = False
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

        credentials["location"] = prefix
        credentials["magnetType"] = magnetType
        credentials["pointName"] = pointName
        credentials["magnetName"] = magnetName

        return credentials