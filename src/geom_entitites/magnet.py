from geom_entitites.transformation import Transformation


class Magnet(object):
    def __init__(self, name, type, parent_ref):
        self.name = name
        self.pointDict = {}
        self.type = type
        self.parent_ref = parent_ref
        self.shift = 0

    def transformFrame(self, targetFrame):
        # iterate over points and transform then
        for point in self.pointDict:
            self.pointDict[point].changeFrame(targetFrame)

    def addPoint(self, point):
        self.pointDict[point.name] = point

    def get_ordered_points_coords(self) -> list:
        # sorting list of magnet names
        point_names_sorted = sorted(self.pointDict.keys())
                
        sortedPointDict = {k: self.pointDict[k] for k in point_names_sorted}

        return [sortedPointDict[point].coord_list for point in sortedPointDict]
