from geom_entitites.transformation import Transformation


class Frame(object):
    def __init__(self, name, homogeneosTransf):
        self.name = name
        self.transformation = homogeneosTransf

    def __str__(self) -> str:
        return f'Frame[{self.name}, {self.transformation}]'

    @staticmethod
    def changeFrame(frameFrom, frameTo, pointDict):
        transformation = Transformation.evaluateTransformation(frameFrom, frameTo)

        # iterating over magnet's points and transforming then to the new frame
        for pointName in pointDict:
            point = pointDict[pointName]
            point.transform(transformation, frameTo.name)

    @staticmethod
    def machine_local():
        return Frame('machine-local', Transformation('machine-local', 'machine-local', 0, 0, 0, 0, 0, 0))