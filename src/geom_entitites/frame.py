from geom_entitites.transformation import Transformation


class Frame:
    """ Class containing implementation of a frame (coordinate system) abstraction. """

    def __init__(self, name: str, homogeneos_transf: Transformation) -> None:
        self.name = name
        self.transformation = homogeneos_transf

    def __str__(self) -> str:
        return f'Frame[{self.name}, {self.transformation}]'

    @staticmethod
    def change_frame(frame_from, frame_to, points: dict) -> None:
        """ Changes the frame from a set of given points """

        transformation = Transformation.evaluateTransformation(frame_from, frame_to)

        # iterating over point and transforming then to the new frame
        for point in points.values():
            point.transform(transformation, frame_to.name)

    @staticmethod
    def machine_local():
        """ Returns a instance of a machine-local frame, which is simply 
            a frame on the zero of the global coordinate system. """

        return Frame('machine-local', Transformation('machine-local', 'machine-local', 0, 0, 0, 0, 0, 0))