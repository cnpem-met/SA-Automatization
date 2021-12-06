from typing import List
import numpy as np
from geom_entitites.frame import Frame

from geom_entitites.transformation import Transformation


class Point:
    """ Class containing implementation of a point abstraction. """

    def __init__(self, name, x, y, z, frame: Frame = None):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.frame = frame

    def __str__(self):
        return f'Point [{self.name}: ({self.x}, {self.y}, {self.z})]'

    @property
    def coord_list(self):
        return [self.x, self.y, self.z]

    @classmethod
    def copy_from_point(cls, point):
        """ Auxiliary function to copy a point for manipulation. """

        name = point.name
        x = point.x
        y = point.y
        z = point.z
        frameName = point.frameName
        
        return cls(name, x, y, z, frameName)

    def transform(self, transf_matrix: List[list], new_frame: Frame) -> None:
        """ Applies a transformation to itself, normally related to change of base/frame operations. """

        point = np.array([[self.x], [self.y], [self.z], [1]])

        transformed_point = transf_matrix @ point
        self.x = transformed_point[0, 0]
        self.y = transformed_point[1, 0]
        self.z = transformed_point[2, 0]

        self.frame = new_frame

    def changeFrame(self, target_frame: Frame):
        """ Changes the current frame of the point. """

        # evaluating transformation
        transformation = Transformation.evaluateTransformation(self.frame, target_frame)

        # transform itself
        self.transform(transformation, target_frame)

