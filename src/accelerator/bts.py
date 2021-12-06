from accelerator import Accelerator

class BTS(Accelerator):
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