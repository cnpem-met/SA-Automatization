import numpy as np

from geom_entitites.transformation import Transformation


class Point(object):
    def __init__(self, name, x, y, z, frame):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.coords = [self.x, self.y, self.z]
        self.frameName = frame

    def __str__(self):
        return f'Point [{self.name}: ({self.x}, {self.y}, {self.z})]'

    @classmethod
    def copyFromPoint(cls, point):
        name = point.name
        x = point.x
        y = point.y
        z = point.z
        frameName = point.frameName
        
        return cls(name, x, y, z, frameName)

    def transform(self, transfMatrix, newFrameName):
        point = np.array([[self.x], [self.y], [self.z], [1]])

        transformedPoint = transfMatrix @ point
        self.x = transformedPoint[0, 0]
        self.y = transformedPoint[1, 0]
        self.z = transformedPoint[2, 0]

        self.coords = [self.x, self.y, self.z]

        self.frameName = newFrameName

    def changeFrame(self, frameDict, targetFrame):

        # extracting the name of the initial frame
        initialFrame = self.frameName

        # evaluating transformation
        transformation = Transformation.evaluateTransformation(frameDict, initialFrame, targetFrame)

        # transform itself
        self.transform(transformation, targetFrame)

