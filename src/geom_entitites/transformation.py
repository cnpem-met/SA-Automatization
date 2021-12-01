import numpy as np
import math
from numpy.linalg import inv

class Transformation(object):
    def __init__(self, frameFrom, frameTo, Tx, Ty, Tz, Rx, Ry, Rz):
        self.frameFrom = frameFrom
        self.frameTo = frameTo
        self.Tx = Tx
        self.Ty = Ty
        self.Tz = Tz
        self.Rx = Rx
        self.Ry = Ry
        self.Rz = Rz
        self.createTransfMatrix()

    def createTransfMatrix(self):

        self.transfMatrix = np.zeros(shape=(4, 4))

        rot_z = np.array([[np.cos(self.Rz*10**-3), -np.sin(self.Rz*10**-3), 0],
                          [np.sin(self.Rz*10**-3), np.cos(self.Rz*10**-3), 0], [0, 0, 1]])
        rot_y = np.array([[np.cos(self.Ry*10**-3), 0, np.sin(self.Ry*10**-3)],
                          [0, 1, 0], [-np.sin(self.Ry*10**-3), 0, np.cos(self.Ry*10**-3)]])
        rot_x = np.array([[1, 0, 0], [0, np.cos(
            self.Rx*10**-3), -np.sin(self.Rx*10**-3)], [0, np.sin(self.Rx*10**-3), np.cos(self.Rx*10**-3)]])

        rotationMatrix = rot_z @ rot_y @ rot_x

        self.transfMatrix[:3, :3] = rotationMatrix
        self.transfMatrix[3, 3] = 1
        self.transfMatrix[:3, 3] = [self.Tx, self.Ty, self.Tz]

    @staticmethod
    def evaluateTransformation(frameDict, initialFrame, targetFrame):
        transformationList = []
        transformationList.insert(
            0, frameDict[initialFrame].transformation.transfMatrix)

        transformationList.insert(
            0, inv(frameDict[targetFrame].transformation.transfMatrix))

        transformation = transformationList[0] @ transformationList[1]

        return transformation

    @staticmethod
    def individualDeviationsFromEquations(transfMatrix):
        # Ry
        ry = math.asin(-transfMatrix[2, 0])
        # Rz
        rz = math.asin(transfMatrix[1, 0]/math.cos(ry))
        # Rx
        rx = math.asin(transfMatrix[2, 1]/math.cos(ry))
        # Tx, Ty, Tz
        (tx, ty, tz) = (transfMatrix[0, 3],
                        transfMatrix[1, 3], transfMatrix[2, 3])

        return [tx, ty, tz, rx * 10**3, ry * 10**3, rz * 10**3]

