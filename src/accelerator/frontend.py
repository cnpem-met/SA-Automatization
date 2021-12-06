from accelerator import Accelerator

class FE(Accelerator):
    def __init__(self, name) -> None:
        super().__init__(name)

    def get_points_to_bestfit(self, magnet_meas, magnet_nom):
        return super().get_points_to_bestfit(magnet_meas, magnet_nom)

    def calculate_nominal_to_measured_transformation(self, magnet, pointList):
        return super().calculate_nominal_to_measured_transformation(magnet, pointList)

    def populate_magnets_with_points(self, type_of_points):
        super().populate_magnets_with_points(type_of_points)

    def sort_frames_by_beam_trajectory(self, type_of_points):
        super().sort_frames_by_beam_trajectory(type_of_points)

    def get_point_info(self, pointName):
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