from typing import Dict
from geom_entitites.magnet import Magnet
from geom_entitites.point import Point
from geom_entitites.transformation import Transformation
from geom_entitites.frame import Frame

from operations.geometrical import evaluate_deviation_with_bestfit
from accelerator import Accelerator

import numpy as np
import math

class SR(Accelerator):
    def __init__(self):
        super().__init__('SR')

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
                magnet.add_point(point)
                # if type of magnet is B1 and point is measured, insert offset into magnets properties
                if (type_of_points == 'measured' and point_info['magnetType'] == 'dipole-B1'):
                    magnet.shift = self.shiftsB1[point_info['magnetName']]
            else:
                # instantiate new Magnet object
                magnet = Magnet(point_info['magnetName'], point_info['magnetType'], point_info['location'])
                magnet.add_point(point)
                if (type_of_points == 'measured' and point_info['magnetType'] == 'dipole-B1'):
                    magnet.shift = self.shiftsB1[point_info['magnetName']]
                # including in data structure
                self.magnets[type_of_points][point_info['magnetName']] = magnet
        
        # magnets has to be sorted in a special form (quads before B1 in the B03/B11 girders)
        # because the calculation of measured frames of the B1 requires that measured quad
        # has previously been calculated
        self.sort_magnets_with_quads_before_b1(type_of_points)

    def sort_frames_by_beam_trajectory(self, type_of_points: str) -> None:
        """ Sort frame dict based on the beam trajectory. """

        # firstly sort frames in alphabetical order
        sorted_keys = sorted(self.frames[type_of_points])
        sorted_frame_dict = {}
        for key in sorted_keys:
            sorted_frame_dict[key] = self.frames[type_of_points][key]

        # then we have to correct the case of S0xB03, i.e. exchange 
        # B1 magnet with QUAD01, which comes first
        final_frame_dict = {}
        b03_frames = []
        for frameName in sorted_frame_dict:
            frame = sorted_frame_dict[frameName]

            if ('B03-B1' in frame.name):
                b03_frames.append(frame)
                continue

            final_frame_dict[frame.name] = frame

            if ('B03-QUAD01' in frame.name):
                for b03_frame in b03_frames:
                    final_frame_dict[b03_frame.name] = b03_frame
                b03_frames = []

            self.frames[type_of_points] = final_frame_dict

    def sort_magnets_with_quads_before_b1(self, type_of_points: str) -> None:
        """ Sorts the magnet dict firstly in alphabetical order and then
            exchange the order of the QUAD01 with the B1, once further calculations
            of the B1 measured frame requires that the existance of the QUAD01 measured """

        # sorting list of magnet names
        magnet_names_sorted = sorted(self.magnets[type_of_points].keys())

        # exchanguing S0x-B03-B1 with S0x-B03-QUAD01 (same applies to B11)
        for i, magnet_name in enumerate(magnet_names_sorted):
            if 'B03-QUAD01' in magnet_name or 'B11-QUAD01' in magnet_name:
                b1 = f'{magnet_name.split("-")[0]}-{magnet_name.split("-")[1]}-B1-LONG'
                b1_idx = magnet_names_sorted.index(b1)
                magnet_names_sorted[b1_idx] = magnet_names_sorted[i]
                magnet_names_sorted[i] = b1
                
        # reordering magnet's dictionary based on the new order
        self.magnets[type_of_points] = {k: self.magnets[type_of_points][k] for k in magnet_names_sorted}

    def get_points_to_bestfit(self, magnet_meas: Magnet, magnet_nom: Magnet) -> dict:
        """ Get measured and nominal points from magnet. Points are returned in
            the form of a list of coordinates, which in turn are [x,y,z] lists """

        if magnet_meas.type != 'dipole-B2':
            # all magnets types except the B2 requires all 
            # the points with no dof separation
            points_measured = magnet_meas.get_ordered_points_coords()
            points_nominal = magnet_nom.get_ordered_points_coords()
        else:
            # for the B2 we have points for evaluating the level and height (Rx, Rz and Ty)
            # and points to evaluate the rest of the dofs (Tx, Tz and Ry). They will be used
            # individually to perform diferent best-fits and then be composed to reach the
            # position of the measured frame of B2
            points_measured = {'TyRxRz': [], 'TxTzRy': []}
            points_nominal = {'TyRxRz': [], 'TxTzRy': []}

            pointDict = magnet_meas.points

            for pointName in pointDict:
                pointMeas = pointDict[pointName]
                pointNom = magnet_nom.points[pointName]

                # populating lists of points according to the 'LVL' indicator in the name
                if ('LVL' in pointName):
                    # DoFs: ['Ty', 'Rx', 'Rz']
                    points_measured['TyRxRz'].append(pointMeas.coord_list)
                    points_nominal['TyRxRz'].append(pointNom.coord_list)
                else:
                    # DoFs: ['Tx', 'Tz', 'Ry']
                    points_measured['TxTzRy'].append(pointMeas.coord_list)
                    points_nominal['TxTzRy'].append(pointNom.coord_list)

        return {'measured': points_measured, 'nominal': points_nominal}

    def calculate_ML_to_local_measured_transformation(self, magnet: Magnet, points: Dict[str, list]) -> Transformation:
        """ Calculates the transformation between the machine-local frame to 
            the measured frame of the magnet (or local measured frame). """
        
        if (magnet.type == 'dipole-B1'):
            # B1 dipoles needs a special treatment: their measured frame will be a product of 
            # the manipulation of measured frame of the adjacent quad01 magnet and geometrical
            # operations using nominal and measured data, such as the offset of the center of the
            # 2 measured points to the magnetic center of the dipole.

            # points from the magnet will be used because of the operations of frame changuing
            points = magnet.points

            # changuing points' frame: from the current activated frame ("dipole-nominal") to the "quadrupole-measured"
            girder_name = magnet.parent_ref
            frame_from = self.frames['nominal'][f'{magnet.name}']
            frame_to = self.frames['measured'][f'{girder_name}-QUAD01']
            Frame.change_frame(frame_from, frame_to, points)

            # listing the points' coordinates in the new frame for latter usage
            points_transformed = []
            for point in points.values():
                points_transformed.append([point.x, point.y, point.z])

            # changuing back point's frame
            Frame.change_frame(frame_to, frame_from, points)

            # calculating mid point position of the 2 measured B1 points
            mid_point = np.mean(points_transformed, axis=0)

            # defining nominal shift
            # dist = 62.522039008
            dist_from_mid_point_to_magnet_center = 62.4384

            # applying magnet's specific shift to compensate manufacturing variation
            dist_from_mid_point_to_magnet_center -= magnet.shift

            # defining nominal rotation, which is the angle of curvature of the B1
            dRy = 24.044474 * 10**-3  # rad
            if ('B03' in girder_name):
                # differentiating the rotation's signal between B03 and B11 cases
                dRy = - dRy

            # calculating offset in both planar directions
            shift_transv = dist_from_mid_point_to_magnet_center * math.cos(dRy)
            shift_long = dist_from_mid_point_to_magnet_center * math.sin(dRy)
            shift_vert = -228.5 # nominal value

            # calculating shifted point position, which defines the origin of the frame "dipole-measured"
            shifted_point_position = [mid_point[0] - shift_transv, mid_point[1] + shift_vert, mid_point[2] + shift_long]
            # defining a transformation between "dipole-measured" and "quadrupole-measured", which will
            # assume that Tx, Ty, Rx and Rz between the quad and dipole are 0 (ok because they are in the same girder)
            # and use the calculated Tz and Ry between the two
            frame_from = girder_name + '-QUAD01'
            frame_to = magnet.name
            local_transf = Transformation(frame_from, frame_to, 0, 0, shifted_point_position[2], 0, dRy * 10**3, 0)  # mrad
            # base transformation is from machine-local to quad01 measured frame
            base_transf = self.frames['measured'][frame_from].transformation
        else:
            if (magnet.type == 'dipole-B2'):
                # B2 dipoles uses a separation of degrees of freedom to calculate its deviation

                # calculating transformation for the 1st group of points associated with level and height
                dofs = ['Ty', 'Rx', 'Rz']
                dev_tyrxrz = evaluate_deviation_with_bestfit(points['measured']['TyRxRz'], points['nominal']['TyRxRz'], dofs)
            
                # once calculated the 1st transf., transform the 2nd group of points to correct its level and height
                # before using then in the calculation of deviation with the rest of degrees of freedom
                transfPoints = []
                for point in points['nominal']['TxTzRy']:
                    transfPt = Point('n/c', point[0], point[1], point[2])
                    transform = Transformation('n/c', 'n/c', 0, dev_tyrxrz[0], 0, dev_tyrxrz[1], 0, dev_tyrxrz[2])
                    transfPt.transform(transform.transf_matrix, 'n/c')
                    transfPoints.append([transfPt.x, transfPt.y, transfPt.z])

                # calculating the 2nd and last deviation, considering the others dofs and the transformed points
                dofs = ['Tx', 'Tz', 'Ry']
                dev_txtzry = evaluate_deviation_with_bestfit(points['measured']['TxTzRy'], transfPoints, dofs)

                # agregating parcial dofs
                deviation = np.array([dev_txtzry[0], dev_tyrxrz[0], dev_txtzry[1], dev_tyrxrz[1], dev_txtzry[2], dev_tyrxrz[2]])
            else:
                # no dof separation, normal evaluation
                dofs = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
                deviation = evaluate_deviation_with_bestfit(points['measured'], points['nominal'], dofs)

            # defining machine-local to local nominal frame transformation (base)
            # and local nominal to local measured transformation (local)
            local_transf = Transformation(magnet.name, magnet.name, *deviation)
            base_transf = self.frames['nominal'][magnet.name].transformation

        # calculating ML to magnet-measured frame relation from the base and local transformations
        transformation = Transformation.compose_transformations(base_transf, local_transf, 'machine-local', magnet.name)
        return transformation

    def get_point_info(self, point_name: str) -> dict:
        """ Defines the expected template for the point's nomenclature and extracts meta data from it. """
        
        # init dict
        point_info = {}
        point_info["isCurrentAcceleratorPoint"] = True
        point_info["isValidPoint"] = True

        # extract meta data from point name
        details = point_name.split('-')
        sector_id = details[0]
        girder_id = details[1]
        girder_name = sector_id + '-' + girder_id
        magnet_ref = details[2]

        # checking if nomenclature is OK
        if (sector_id[0] != 'S' or girder_id[0] != 'B'):
            point_info["isCurrentAcceleratorPoint"] = False
            return point_info

        if (len(details) < 4):
            point_info["isValidPoint"] = False
            return point_info

        # filtering points that aren't fiducials (especially on B1)
        if ('C01' not in point_name and 'C02' not in point_name and 'C03' not in point_name and 'C04' not in point_name and 'LVL' not in point_name):
            point_info["isValidPoint"] = False
            return point_info

        # treating longitudinal points 
        if (details[len(details)-1] == 'LONG'):
            magnet_name = sector_id + '-' + girder_id + '-' + magnet_ref + '-' + details[len(details)-1]
        else:
            magnet_name = sector_id + '-' + girder_id + '-' + magnet_ref

        # defining magnet type
        if (magnet_ref[:4] == 'QUAD'):
            magnet_type = 'quadrupole'
        elif (magnet_ref[:4] == 'SEXT'):
            magnet_type = 'sextupole'
        elif (magnet_ref == 'B1'):
            magnet_type = 'dipole-B1'
        elif (magnet_ref == 'B2'):
            magnet_type = 'dipole-B2'
        elif (magnet_ref == 'BC'):
            magnet_type = 'dipole-BC'
        else:
            magnet_type = 'other'

        # storing info to return
        point_info["location"] = girder_name
        point_info["magnetType"] = magnet_type
        point_info["pointName"] = point_name
        point_info["magnetName"] = magnet_name

        return point_info

    def calculate_B1_txty_deviations(self) -> None:
        """ Auxiliary method to evaluate the perpendicular (Tx and Ty) deviations of B1 magnets,
            which aren't considered in oficial reports but are worth to be tracked. """

        for magnet in self.magnets['measured'].values():
            if (magnet.type != 'dipole-B1'):
                continue
            
            # extracting measured points and changuing to nominal frame
            meas_points = []
            for point in magnet.points.values():
                # copying for making manipulations
                cp_point = Point.copy_from_point(point)
                # changuing frame to the nominal one
                cp_point.changeFrame(self.frames['nominal'][magnet.name])
                meas_points.append(cp_point.coord_list)
            
            # making calculations to determine the deviation from nominal
            nom_points = [pt.coord_list for pt in self.magnets['nominal'][magnet.name].points]
            avg_meas_points = np.mean(meas_points, axis=0)
            avg_nom_points = np.mean(nom_points, axis=0)
            deviation = avg_meas_points - avg_nom_points
            self.deviations.loc[magnet, ['Tx', 'Ty']] = deviation[:2]

    @staticmethod
    def check_magnet_family(magnet_name: str) -> str:
        """" Auxiliary static method that maps which family a specific
             magnet is part of for reporting purposes """

        sectorA = ['S01', 'S05', 'S09', 'S13', 'S17']
        sectorB = ['S02', 'S04', 'S06', 'S08', 'S10', 'S12', 'S14', 'S16', 'S18', 'S20']
        sectorP = ['S03', 'S07', 'S11', 'S15', 'S19']

        mns = magnet_name.split('-')
        sector, girder, magnet_name = mns[0], mns[1], mns[2]

        # girder influenced by the sector type
        if (girder == 'B02' or girder == 'B03' or girder == 'B11' or girder == 'B01'):
            # high-beta sectors (type A)
            if (sector in sectorA):
                if (girder == 'B02' or girder == 'B01'):
                    # quad type: QFA
                    if ('QUAD01' in magnet_name):
                        family = 'QFA'
                elif (girder == 'B03' or girder == 'B11'):
                    # quad type: QDA
                    if ('QUAD01' in magnet_name):
                        family = 'QDA'
                    # dipole
                    elif ('B1' in magnet_name):
                        family = 'dipole-B1'
            # low-beta sectors (type B)
            elif (sector in sectorB):
                if (girder == 'B01'):
                    # quad type: QFB
                    if ('QUAD01' in magnet_name):
                        family = 'QFB'
                    # quad type: QDB2
                    elif ('QUAD02' in magnet_name):
                        family = 'QDB2'
                elif (girder == 'B02'):
                    # quad type: QFB
                    if ('QUAD02' in magnet_name):
                        family = 'QFB'
                    # quad type: QDB2
                    elif ('QUAD01' in magnet_name):
                        family = 'QDB2'
                elif (girder == 'B03' or girder == 'B11'):
                    # quad type: QDB1
                    if ('QUAD01' in magnet_name):
                        family = 'QDB1'
                    # dipole
                    elif ('B1' in magnet_name):
                        family = 'dipole-B1'
            # low-beta sectors (type P)
            elif (sector in sectorP):
                if (girder == 'B01'):
                    # quad type: QFP
                    if ('QUAD01' in magnet_name):
                        family = 'QFP'
                    # quad type: QDP2
                    elif ('QUAD02' in magnet_name):
                        family = 'QDP2'
                elif (girder == 'B02'):
                    # quad type: QFP
                    if ('QUAD02' in magnet_name):
                        family = 'QFP'
                    # quad type: QDP2
                    elif ('QUAD01' in magnet_name):
                        family = 'QDP2'
                elif (girder == 'B03' or girder == 'B11'):
                    # quad type: QDP1
                    if ('QUAD01' in magnet_name):
                        family = 'QDP1'
                    # dipole
                    elif ('B1' in magnet_name):
                        family = 'dipole-B1'

        elif (girder == 'B04'):
            # quad type: Q1
            if ('QUAD01' in magnet_name):
                family = 'Q1'
            # quad type: Q2
            elif ('QUAD02' in magnet_name):
                family = 'Q2'
        elif (girder == 'B05'):
            # dipole
            if ('B2' in magnet_name):
                family = 'dipole-B2'
        elif (girder == 'B06'):
            # quad type: Q3
            if ('QUAD01' in magnet_name):
                family = 'Q3'
            # quad type: Q4
            elif ('QUAD02' in magnet_name):
                family = 'Q4'
        elif (girder == 'B07'):
            # dipole
            if ('BC' in magnet_name):
                family = 'dipole-BC'
        elif (girder == 'B08'):
            # quad type: Q4
            if ('QUAD01' in magnet_name):
                family = 'Q4'
            # quad type: Q3
            elif ('QUAD02' in magnet_name):
                family = 'Q3'
        elif (girder == 'B09'):
            # dipole
            if ('B2' in magnet_name):
                family = 'dipole-B2'
        elif (girder == 'B10'):
            # quad type: Q2
            if ('QUAD01' in magnet_name):
                family = 'Q2'
            # quad type: Q1
            elif ('QUAD02' in magnet_name):
                family = 'Q1'

        return family