from abc import ABC, abstractmethod
from typing import Dict

import math
import numpy as np
import pandas as pd

import config
from geom_entitites.magnet import Magnet
from geom_entitites.point import Point

from geom_entitites.transformation import Transformation
from geom_entitites.frame import Frame
from operations.geometrical import evaluateDeviation

from ui.ui import Ui


class Accelerator(ABC):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.ml_to_nominal_frame_mapping = None
        self.points = {'measured': None, 'nominal': None}
        self.magnets = {'measured': {}, 'nominal': {}}
        self.frames = {'measured': {}, 'nominal': {}}
        self.deviations = None

    def create_nominal_frames(self) -> None:
        """ Creates magnets' nominals frames based on the ML to local frames lookup table (inputted by the user)  """

        # creating base ML frame
        self.frames['nominal']['machine-local'] = Frame.machine_local()

        # for each magnet create a nominal frame based on the ML to local frame mapping data
        for frame_name, dof in self.ml_to_nominal_frame_mapping.iterrows():
            transformation = Transformation('machine-local', frame_name, dof['Tx'], dof['Ty'], dof['Tz'], dof['Rx'], dof['Ry'], dof['Rz'])

            frame = Frame(frame_name, transformation)
            self.frames['nominal'][frame_name] = frame

    def change_to_local_frame(self, type_of_points: str) -> None:
        """ Changue all point's frame from Machine Local(ML) to local/magnet (nominal) ones.
        
        This function expects that all the points are referenced in ML at this stage. """

        magnets_not_affected = ""
        for magnet in self.magnets[type_of_points].values():
            try:
                magnet.transformFrame(self.frames['nominal'][magnet.name])
            except KeyError:
                magnets_not_affected += magnet.name + ','

        # at least one frame was not computed
        if (magnets_not_affected != ""):
            if config.DEBUG_MODE:
                print(f'[Transformação p/ frames locais] Imãs não computados: {magnets_not_affected[:-1]}')

    def calculate_measured_frames(self, ui: Ui, magnet_types_to_ignore: list = []) -> None:
        """ Calculate measured/real frame position of each magnet, i.e. the relation
            between machine-local (ML) frame and the magnet one recreated with measured data.
            
            The default mechanism of evaluation is a simple nominal-to-measured transformation, but
            it can be more specific to the type of magnet and accelerator (e.g. B1 of Storage Ring).
            
            This function expects that all the points were previously transformed to local nominal frame. """

        # iterate over each measured magnet data (only what was measured needs to be evaluated)
        for magnet_meas in self.magnets['measured'].values():
            magnet_nom = self.magnets['nominal'][magnet_meas.name]

            # ignoring some type of magnet (e.g. sextupoles)
            if magnet_meas.type in magnet_types_to_ignore:
                continue

            # checking if points are in both measured and nominal magnet objects
            if not self.check_points_coexistance(magnet_meas, magnet_nom):
                ui.logMessage(f'Falha no imã {magnet_meas.name}: o nome de 1 ou mais pontos nominais não correspondem aos medidos.', severity="danger")
                continue

            # get list of measured and nominal point coordinates to further compute transformation by the means of a bestfit
            points = self.get_points_to_bestfit(magnet_meas, magnet_nom)

            # calculating the transformation from the nominal points to the measured ones
            try:
                transformation = self.calculate_ML_to_local_measured_transformation(magnet_meas, points)
            except KeyError:
                ui.logMessage(f'Falha no imã {magnet_meas.name}: frame medido do quadrupolo adjacente ainda não foi calculado.', severity='danger')
                continue

            # creating a new Frame with the transformation from the measured magnet to machine-local
            measMagnetFrame = Frame(magnet_meas.name, transformation)

            # adding it to the frame dictionary
            self.frames['measured'][measMagnetFrame.name] = measMagnetFrame

    def evaluate_magnets_deviations(self) -> bool:
        """ Calculates the deviation of each magnet from its nominal position by comparing
            the measured frame (calculated) and the nominal one (input).
            
            This function expects that both nominal and measured parts of frame dict are populated. """

        # defining header for DataFrame that will hold the deviations
        header = ['Magnet', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']

        deviation_list = []

        # at least one measured and nominal frame has to exist
        if self.frames['measured'] == {} or self.frames['nominal'] == {}:
            return False

        # iterating over measured frames and checking transformation to the corresponding nominal one
        for frame_meas in self.frames['measured'].values():
            
            # ignoring ML
            if (frame_meas.name == 'machine-local'):
                continue

            frame_nom = self.frames['nominal'][frame_meas.name]

            # evaluating transformation between frames and then extracting the deviation from its transformation matrix
            transformation = Transformation.evaluateTransformation(initialFrame=frame_meas, targetFrame=frame_nom)
            deviation = Transformation.individualDeviationsFromEquations(transformation)

            deviation.insert(0, frame_meas.name)

            # acumulating dataframes
            dev = pd.DataFrame([deviation], columns=header)
            deviation_list.append(dev)

        # composing final df
        deviations = pd.concat(deviation_list, ignore_index=True)
        deviations = deviations.set_index(header[0])

        self.deviations = deviations.astype('float32')

        # return positivite flag to allow plotting
        return True

    def evaluate_magnets_long_inter_distances(self, type_of_points: str) -> pd.DataFrame:
        """ Evaluates the longitudinal inter distances between adjance magnets.
        
            This function expects that frames are sorted by the beam trajectory. """

        header = ['Reference', 'Distance (mm)']
        distance_list = []

        frames_keys = list(self.frames[type_of_points].keys())

        for (index, frame) in enumerate(self.frames[type_of_points].values()):

            if ('machine-local' in frame.name):
                continue
            
            # storing first element for latter usage
            if (index == 0):
                first_magnet_frame = frame

            # checking which frame is next
            if (index == len(self.frames[type_of_points]) - 1):
                # first frame of the dictionary
                magnet2_frame = first_magnet_frame
            else:
                # next frame in dict
                magnet2_frame = self.frames[type_of_points][frames_keys[index+1]]

            magnet1_frame = frame

            magnet1_coords = np.array([magnet1_frame.transformation.Tx, magnet1_frame.transformation.Ty, magnet1_frame.transformation.Tz])
            magnet2_coords = np.array([magnet2_frame.transformation.Tx, magnet2_frame.transformation.Ty, magnet2_frame.transformation.Tz])

            # calculating 2D distance
            vector = magnet2_coords - magnet1_coords
            distance = math.sqrt(vector[0]**2 + vector[1]**2)

            # defining name reference for the data and composing df
            distance_ref = magnet1_frame.name + ' x ' + magnet2_frame.name
            distance = pd.DataFrame([[distance_ref, distance]], columns=header)
            distance_list.append(distance)

        distances = pd.concat(distance_list, ignore_index=True)
        distances = distances.set_index(header[0])

        return distances.astype('float32')

    def check_points_coexistance(self, magnet_meas: Magnet, magnet_nom: Magnet) -> bool:
        """ Simple checking if all measured points of a magnet has its nominal corresponding. """

        for point in magnet_meas.points.values():
            if not point.name in magnet_nom.points:
                return False
        return True

    @abstractmethod
    def get_points_to_bestfit(self, magnet_meas: Magnet, magnet_nom: Magnet) -> Dict[str, list]:
        """ Get measured and nominal points from magnet. Points are returned in
            the form of a list of coordinates, which in turn are [x,y,z] lists """

        points_measured = magnet_meas.get_ordered_points_coords()
        points_nominal = magnet_nom.get_ordered_points_coords()

        return {'measured': points_measured, 'nominal': points_nominal}

    @abstractmethod
    def calculate_ML_to_local_measured_transformation(self, magnet: Magnet, points: Dict[str, list]) -> Transformation:
        """ Calculates the transformation between the machine-local frame to 
            the measured frame of the magnet (or local measured frame). """
        
        # defining degress of freedom for the best-fit operation
        dofs = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        # calling method to do the best-fit and evaluate deviation
        deviation = evaluateDeviation(points['measured'], points['nominal'], dofs)

        # as we now have the transformation between magnet-measured and magnet-nominal, to know
        # the relation between machine-local and magnet-measured we have to treat transformations
        # separetally, i.e. local (measured to nominal) and base (nominal to ML)
        local_transf = Transformation(magnet.name, magnet.name, *deviation)
        base_transf = self.frames['nominal'][magnet.name].transformation

        # calculating ML to magnet-measured frame relation from the base and local transformations
        transformation = Transformation.compose_transformations(base_transf, local_transf, 'machine-local', magnet.name)
        
        return transformation

    @abstractmethod
    def sort_frames_by_beam_trajectory(self, type_of_points: str) -> None:
        """ Sort frame dict based on the beam trajectory. Default is a
            alphabetically sorted set of frame names. """

        # sorting in alphabetical order
        sorted_keys = sorted(self.frames[type_of_points])
        sorted_frame_dict = {}
        for key in sorted_keys:
            sorted_frame_dict[key] = self.frames[type_of_points][key]

        self.frames[type_of_points] = sorted_frame_dict

    @abstractmethod
    def populate_magnets_with_points(self, type_of_points: str) -> None:
        """ Creates magnets objects and populates it with loaded points. """

        # iterating over loaded points
        for point_name, coordinate in self.points[type_of_points].iterrows():

            point_info = self.get_point_info(point_name)

            # checks if point's nomenclature matched the expected
            if (not point_info['isCurrentAcceleratorPoint'] or not point_info['isValidPoint']):
                continue

            # creating a new Point object referenced in the ML frame
            point = Point(point_info['pointName'], coordinate['x'], coordinate['y'], coordinate['z'], Frame.machine_local())

            # finding Magnet object or instantiating a new one
            if (point_info['magnetName'] in self.magnets[type_of_points]):
                # reference to the Magnet object
                magnet = self.magnets[type_of_points][point_info['magnetName']]
                # append point into its point list
                magnet.addPoint(point)
            else:
                # instantiate new Magnet object
                magnet = Magnet(point_info['magnetName'], point_info['magnetType'], point_info['location'])
                magnet.addPoint(point)
                # including in data structure
                self.magnets[type_of_points][point_info['magnetName']] = magnet

    @abstractmethod
    def get_point_info(self, point_name: str) -> dict:
        """ Defines the expected template for the point's nomenclature and extracts meta data from it. """
        pass
