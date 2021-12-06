from geom_entitites.transformation import Transformation


class Magnet(object):
    def __init__(self, name, type, parent_ref):
        self.name = name
        self.points = {}
        self.type = type
        self.parent_ref = parent_ref
        self.shift = 0

    def transformFrame(self, targetFrame):
        # iterate over points and transform then
        for point in self.points:
            self.points[point].changeFrame(targetFrame)

    def addPoint(self, point):
        self.points[point.name] = point

    def get_ordered_points_coords(self) -> list:
        # sorting list of magnet names
        point_names_sorted = sorted(self.points.keys())
                
        sortedPointDict = {k: self.points[k] for k in point_names_sorted}

        return [sortedPointDict[point].coord_list for point in sortedPointDict]
