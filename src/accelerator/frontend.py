from accelerator import Accelerator

class FE(Accelerator):
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
        prefixes = ['MOG', 'EMA', 'CAR', 'CAT', 'MAN', 'IPE']
        prefix = details[0]

        if (prefix not in prefixes):
            credentials["isCurrentAcceleratorPoint"] = False
            return credentials

        if (len(details) != 3):
            credentials["isValidPoint"] = False
            return credentials

        magnetName = details[0] + '-' + details[1]

        # dummy magnet type, because this information isn't important for FE analysis
        magnetType = 'generic'

        credentials["location"] = prefix
        credentials["magnetType"] = magnetType
        credentials["pointName"] = pointName
        credentials["magnetName"] = magnetName

        return credentials