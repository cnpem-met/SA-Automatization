from typing import List
from geom_entitites.transformation import Transformation


class Magnet:
    """ Class containing implementation for the a Magnet abstraction, which
        will hold meta data and the list of points that belongs to it. """

    def __init__(self, name, type, parent_ref):
        self.name = name
        self.points = {}
        self.type = type
        self.parent_ref = parent_ref
        self.shift = 0

    def change_frame(self, target_frame):
        """ Changes the frame of its points. """

        # iterate over points and transform then
        for point in self.points.values():
            point.changeFrame(target_frame)

    def add_point(self, point):
        """ Adds points to its dict of points. """

        self.points[point.name] = point

    def get_ordered_points_coords(self) -> List[list]:
        """ Provides a list of coordinates of its points in a sorted order. """

        # sorting list of point names
        point_names_sorted = sorted(self.points.keys())
        
        # generating the sorted version of the dict of points
        sorted_point_dict = {k: self.points[k] for k in point_names_sorted}

        # returning the coordinate list of each point of the sorted dict
        return [sorted_point_dict[point].coord_list for point in sorted_point_dict]
