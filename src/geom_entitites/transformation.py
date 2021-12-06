from typing import List
import numpy as np
import math
from numpy.linalg import inv

class Transformation:
    """ Class containing implementation of a Homogeneous Transformation abstraction. """

    def __init__(self, frame_from, frame_to, Tx: float, Ty: float, Tz: float, Rx: float, Ry: float, Rz: float):
        self.frameFrom = frame_from
        self.frameTo = frame_to
        self.Tx = Tx
        self.Ty = Ty
        self.Tz = Tz
        self.Rx = Rx
        self.Ry = Ry
        self.Rz = Rz
        self.create_transf_matrix()

    def __str__(self) -> str:
        return f'[{self.Tx}, {self.Ty}, {self.Tz}, {self.Rx}, {self.Ry}, {self.Rz}]'

    def create_transf_matrix(self):
        """ Generates a homogeneos transformation matrix to be used in calculations. """

        self.transf_matrix = np.zeros(shape=(4, 4))

        rot_z = np.array([[np.cos(self.Rz*10**-3), -np.sin(self.Rz*10**-3), 0],
                          [np.sin(self.Rz*10**-3), np.cos(self.Rz*10**-3), 0], [0, 0, 1]])
        rot_y = np.array([[np.cos(self.Ry*10**-3), 0, np.sin(self.Ry*10**-3)],
                          [0, 1, 0], [-np.sin(self.Ry*10**-3), 0, np.cos(self.Ry*10**-3)]])
        rot_x = np.array([[1, 0, 0], [0, np.cos(
            self.Rx*10**-3), -np.sin(self.Rx*10**-3)], [0, np.sin(self.Rx*10**-3), np.cos(self.Rx*10**-3)]])

        rotation_matrix = rot_z @ rot_y @ rot_x

        self.transf_matrix[:3, :3] = rotation_matrix
        self.transf_matrix[3, 3] = 1
        self.transf_matrix[:3, 3] = [self.Tx, self.Ty, self.Tz]

    @staticmethod
    def evaluateTransformation(initial_frame, target_frame):
        """ Computes the transformation between 2 given frames. """

        transf1 = inv(target_frame.transformation.transf_matrix)
        transf2 = initial_frame.transformation.transf_matrix

        transformation = transf1 @ transf2

        return transformation

    @staticmethod
    def individualDeviationsFromEquations(transf_matrix: List[list]) -> list:
        """ Computes de deviations from a given transformation matrix. """

        # rotations
        ry = math.asin(-transf_matrix[2, 0])
        rz = math.asin(transf_matrix[1, 0]/math.cos(ry))
        rx = math.asin(transf_matrix[2, 1]/math.cos(ry))
        
        # translations
        (tx, ty, tz) = (transf_matrix[0, 3], transf_matrix[1, 3], transf_matrix[2, 3])

        return [tx, ty, tz, rx * 10**3, ry * 10**3, rz * 10**3]

    @staticmethod
    def compose_transformations(base_transf, local_transf, frame_from: str, frame_to: str):
        """ Computes the resultant transformation given by two subsequent transformations. """
        
        transf_matrix = base_transf.transf_matrix @ local_transf.transf_matrix
        
        transf = Transformation(frame_from, frame_to, 0, 0, 0, 0, 0, 0)

        transf.Tx = transf_matrix[0, 3]
        transf.Ty = transf_matrix[1, 3]
        transf.Tz = transf_matrix[2, 3]
        transf.Rx = base_transf.Rx + local_transf.Rx
        transf.Ry = base_transf.Ry + local_transf.Ry
        transf.Rz = base_transf.Rz + local_transf.Rz

        transf.transf_matrix = transf_matrix

        return transf
