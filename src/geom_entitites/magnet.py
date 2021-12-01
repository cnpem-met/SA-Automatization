from geom_entitites.transformation import Transformation


class Magnet(object):
    def __init__(self, name, type):
        self.name = name
        self.pointDict = {}
        self.type = type
        self.shift = 0

    def transformFrame(self, frameDict, targetFrame):

        # extracting the name of the initial frame
        for point in self.pointDict:
            initialFrame = self.pointDict[point].frameName
            break

        initialFrame = frameDict[initialFrame]

        transformation = Transformation.evaluateTransformation(initialFrame, targetFrame)

        # iterate over points and transform then
        for point in self.pointDict:
            self.pointDict[point].transform(transformation, targetFrame)

    def addPoint(self, point):
        self.pointDict[point.name] = point

