import numpy as np
import pandas as pd
import math


class PointGroup(object):
    """ Método construtor da classe """

    def __init__(self, ptList, typeOfPoints, lookuptable, frame="machine-local"):
        self.ptList = ptList
        self.frame = frame
        self.type = typeOfPoints
        self.lookuptable = lookuptable

    """ Método que gera um dataframe com pontos em coordenadas locais (frames dos berços), a partir
    das coordenadas globais (frame ML) e dos parâmetros de transformação de coordenadas de cada frame """

    def transformToLocalFrame(self, mode='full', convers_type='direct'):
        # alocando um novo Dataframe (baseado no df não-modificado dos pontos) para conter
        # as novas coordenadas locais calculadas
        pts_new = self.ptList.copy()
        # utilizaremos uma cópia do df da lookuptable, pois iremos modificá-lo ao longo das iterações
        lookup_temp = self.lookuptable.copy()

        old_girder = ""
        for i in range(pts_new.iloc[:, 0].size):
            current_girder = pts_new.index[i][:7]
            if (current_girder != old_girder):

                # modo em que nem todos os pontos/berços estão presentes na lista de pontos medidos;
                # nesse caso, o algoritmo vai percorrer a lista até encontrar o berço em questão *
                # *por isso, se faz necessário que a lista esteja ordenada!
                if (mode == 'parcial'):
                    while (current_girder != lookup_temp.index[0]):
                        lookup_temp = lookup_temp.drop(lookup_temp.index[0])

                ref_girder = lookup_temp.index[0]

                # tratativa da exceção dos berços B03 e B11, os quais devem ter suas coordenadas locais calculadas
                # a partir de parâmetros da matriz de transformação do berço anterior e sucessor, respectivamente
                if (current_girder[4:] == 'B03'):
                    # pega os parâmetros de rotação do berço anterior
                    transf_matrix = [lookup_temp.at[ref_girder, 'Tx'], lookup_temp.at[ref_girder, 'Ty'], lookup_temp.at[ref_girder, 'Tz'],
                                     transf_matrix_B02[3], transf_matrix_B02[4], transf_matrix_B02[5]]
                elif (current_girder[4:] == 'B11'):
                    # pega os parâmetros de rotação do berço sucessor
                    if (current_girder == 'S20-B11'):
                        # nesse caso em específico, o sucessor é o S01-B01
                        transf_matrix = [lookup_temp.at[ref_girder, 'Tx'], lookup_temp.at[ref_girder, 'Ty'], lookup_temp.at[ref_girder, 'Tz'],
                                         transf_matrix_S01B01[3], transf_matrix_S01B01[4], transf_matrix_S01B01[5]]
                    else:
                        # no resto dos casos, o sucessor é o próximo da lookup_table
                        transf_matrix = [lookup_temp.at[ref_girder, 'Tx'], lookup_temp.at[ref_girder, 'Ty'], lookup_temp.at[ref_girder, 'Tz'],
                                         lookup_temp.at[lookup_temp.index[1], 'Rx'], lookup_temp.at[lookup_temp.index[1], 'Ry'], lookup_temp.at[lookup_temp.index[1], 'Rz']]
                else:
                    # caso sem exceção, ou seja, todos os parâmetros de transformação de frame são do próprio berço
                    transf_matrix = [lookup_temp.at[ref_girder, 'Tx'], lookup_temp.at[ref_girder, 'Ty'], lookup_temp.at[ref_girder, 'Tz'],
                                     lookup_temp.at[ref_girder, 'Rx'], lookup_temp.at[ref_girder, 'Ry'], lookup_temp.at[ref_girder, 'Rz']]

                # salvando a matriz de transformação dos berços B02, pois serão usados p/ calculo de B03 na próxima iteração
                if (current_girder[4:] == 'B02'):
                    transf_matrix_B02 = transf_matrix

                # salvando a matriz de transf. do primeiro berço (S01-B01), para ser usado no cálculo do S20-B11
                if (current_girder == 'S01-B01'):
                    transf_matrix_S01B01 = transf_matrix

                # tirando da lista de parâm. de transf. a linha do berço que já foi usada
                lookup_temp = lookup_temp.drop(lookup_temp.index[0])

            else:
                # enfatizando que se o berço atual é o mesmo que o da última iteração,
                # vamos usar a mesma matriz de transformação de frame
                transf_matrix = transf_matrix

            # pegando o ponto no frame ML
            p = np.array(
                [[pts_new.iloc[i, 0]], [pts_new.iloc[i, 1]], [pts_new.iloc[i, 2]]])

            if (convers_type == 'direct'):
                # apluicando translação
                p_trans = self.translate(
                    p, transf_matrix[0], transf_matrix[1], transf_matrix[2], convers_type)
                # aplicando rotação
                p_final = self.rotate(
                    p_trans, transf_matrix[3], transf_matrix[4], transf_matrix[5], convers_type)
            else:
                p_rot = self.rotate(
                    p, transf_matrix[3], transf_matrix[4], transf_matrix[5], convers_type)
                p_final = self.translate(
                    p_rot, transf_matrix[0], transf_matrix[1], transf_matrix[2], convers_type)

            # salvando a coordenada do ponto em frame local no Dataframe pts_new
            pts_new.iloc[i, 0:] = [p_final[0, 0], p_final[1, 0], p_final[2, 0]]

            # atualizando a referencia para o ultimo berço
            old_girder = current_girder

        # atualizando propriedade 'frame' do objeto
        self.frame = "local"

        # atualiza lista de pontos do objeto
        self.ptList = pts_new

    """ Método auxiliar para rotacionar um ponto """
    @staticmethod
    def rotate(P, Rx, Ry, Rz, convers_type):
        rot_z = np.array([[np.cos(Rz*10**-3), -np.sin(Rz*10**-3), 0],
                          [np.sin(Rz*10**-3), np.cos(Rz*10**-3), 0], [0, 0, 1]])
        rot_y = np.array([[np.cos(Ry*10**-3), 0, np.sin(Ry*10**-3)],
                          [0, 1, 0], [-np.sin(Ry*10**-3), 0, np.cos(Ry*10**-3)]])
        rot_x = np.array([[1, 0, 0], [0, np.cos(
            Rx*10**-3), -np.sin(Rx*10**-3)], [0, np.sin(Rx*10**-3), np.cos(Rx*10**-3)]])
        ROT = rot_z @ rot_y @ rot_x  # MATRIX MULTIPLICATION!

        if (convers_type == 'direct'):
            P_new = np.transpose(ROT) @ P
        else:
            P_new = ROT @ P

        return P_new

    """ Método auxiliar para transladar um ponto """
    @staticmethod
    def translate(P, Tx, Ty, Tz, convers_type):
        if (convers_type == 'direct'):
            P_new = np.array([[P[0, 0] - Tx], [P[1, 0] - Ty], [P[2, 0] - Tz]])
        else:
            P_new = np.array([[P[0, 0] + Tx], [P[1, 0] + Ty], [P[2, 0] + Tz]])
        return P_new

    """ Método para transformar um ptList de sistema cartesiano para cilíndrico """
    @staticmethod
    def cartesian_to_cylind_coord(points_cart_ML):

        points_cylind_ML = pd.DataFrame(columns=['Girder', 'Theta', 'r', 'z'])

        for i in range(points_cart_ML.iloc[:, 0].size):
            point = points_cart_ML.index[i]
            x, y, z = points_cart_ML.loc[point, 'x'], points_cart_ML.loc[point,
                                                                         'y'], points_cart_ML.loc[point, 'z']

            R = math.sqrt(x**2 + y**2)
            theta = math.atan(y/x) * (180/math.pi)

            # gambiarra para mapear o theta entre 0 e 360°
            if ((x < 0 and y > 0) or (x < 0 and y < 0)):
                theta += 180
            elif (x > 0 and y < 0):
                theta += 360

            data = pd.DataFrame(data=np.array([[point, theta, R, z]]),
                                columns=['Girder', 'Theta', 'r', 'z'])

            points_cylind_ML = points_cylind_ML.append(data, ignore_index=True)

        points_cylind_ML = points_cylind_ML.set_index('Girder')
        return points_cylind_ML.astype('float32')
