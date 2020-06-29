
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def rotate(P, Rx, Ry, Rz):
    rot_z = np.array([[np.cos(Rz*10**-3), -np.sin(Rz*10**-3), 0], [np.sin(Rz*10**-3), np.cos(Rz*10**-3), 0], [0, 0, 1]])
    rot_y = np.array([[np.cos(Ry*10**-3), 0, np.sin(Ry*10**-3)], [0, 1, 0], [-np.sin(Ry*10**-3), 0, np.cos(Ry*10**-3)]])
    rot_x = np.array([[1, 0, 0], [0, np.cos(Rx*10**-3), -np.sin(Rx*10**-3)], [0, np.sin(Rx*10**-3), np.cos(Rx*10**-3)]])
    ROT = rot_z @ rot_y @ rot_x # MATRIX MULTIPLICATION!
    
    P_new = np.transpose(ROT) @ P
    
    return P_new

def translate (P, Tx, Ty, Tz):
    P_new = np.array([[P[0,0] - Tx], [P[1,0] - Ty], [P[2,0] - Tz]])
    return P_new


def generate_localFrames(file_lookup_frame, sheet_lookup_frame, file_nominals, sheet_nominals, is_sorted, mode='full'):
    
    if (is_sorted):
        # extraindo os dados das planilhas e alocando em Dataframes
        lookuptable = pd.read_excel(file_lookup_frame, sheet_name=sheet_lookup_frame)
        pts_ML = pd.read_excel(file_nominals, sheet_name=sheet_nominals, header=None)
        
        # alocando um novo Dataframe (baseado no df não-modificado dos pontos) para conter
        # as novas coordenadas locais calculadas 
        pts_new = pts_ML.copy()
        # utilizaremos uma cópia do df da lookuptable, pois iremos modificá-lo ao longo das iterações
        lookup_temp = lookuptable.copy()
        
        old_girder = ""
        for i in range (pts_new[0].size):
            current_girder = pts_new.loc[i,0][:7]
            if (current_girder != old_girder):
                
                # modo em que nem todos os pontos/berços estão presentes na lista de pontos medidos;
                # nesse caso, o algoritmo vai percorrer a lista até encontrar o berço em questão *
                # *por isso, se faz necessário que a lista esteja ordenada!
                if (mode == 'parcial'):
                    while (current_girder != lookup_temp.at[0,'Girder']):
                        lookup_temp = lookup_temp.drop(0).reset_index(drop=True)
                
                # tratativa da exceção dos berços B03 e B11, os quais devem ter suas coordenadas locais calculadas
                # a partir de parâmetros da matriz de transformação do berço anterior e sucessor, respectivamente
                if (current_girder[4:] == 'B03'):
                    # pega os parâmetros de rotação do berço anterior
                    transf_matrix = [lookup_temp.at[0,'Tx'], lookup_temp.at[0,'Ty'],lookup_temp.at[0,'Tz'],
                                 transf_matrix_B02[3], transf_matrix_B02[4], transf_matrix_B02[5]]
                elif (current_girder[4:] == 'B11'):
                    # pega os parâmetros de rotação do berço sucessor
                    if (current_girder == 'S20-B11'):
                        # nesse caso em específico, o sucessor é o S01-B01
                        transf_matrix = [lookup_temp.at[0,'Tx'], lookup_temp.at[0,'Ty'],lookup_temp.at[0,'Tz'],
                                 transf_matrix_S01B01[3], transf_matrix_S01B01[4], transf_matrix_S01B01[5]]
                    else:
                        # no resto dos casos, o sucessor é o próximo da lookup_table
                        transf_matrix = [lookup_temp.at[0,'Tx'], lookup_temp.at[0,'Ty'],lookup_temp.at[0,'Tz'],
                                     lookup_temp.at[1,'Rx'], lookup_temp.at[1,'Ry'], lookup_temp.at[1,'Rz']]
                else:
                    # caso sem exceção, ou seja, todos os parâmetros de transformação de frame são do próprio berço
                    transf_matrix = [lookup_temp.at[0,'Tx'], lookup_temp.at[0,'Ty'],lookup_temp.at[0,'Tz'],
                                     lookup_temp.at[0,'Rx'], lookup_temp.at[0,'Ry'], lookup_temp.at[0,'Rz']]

                
                # salvando a matriz de transformação dos berços B02, pois serão usados p/ calculo de B03 na próxima iteração
                if (current_girder[4:] == 'B02'):
                    transf_matrix_B02 = transf_matrix
                
                # salvando a matriz de transf. do primeiro berço (S01-B01), para ser usado no cálculo do S20-B11
                if (current_girder == 'S01-B01'):
                    transf_matrix_S01B01 = transf_matrix
                
                # tirando da lista de parâm. de transf. a linha do berço que já foi usada
                lookup_temp = lookup_temp.drop(0).reset_index(drop=True)
                    
            else:
                # enfatizando que se o berço atual é o mesmo que o da última iteração,
                # vamos usar a mesma matriz de transformação de frame
                transf_matrix = transf_matrix
            
            # pegando o ponto no frame ML
            p = np.array([[pts_new.loc[i,1]], [pts_new.loc[i,2]], [pts_new.loc[i,3]]])

            
            # apluicando translação
            p_trans = translate(p, transf_matrix[0], transf_matrix[1], transf_matrix[2])

            
            # aplicando rotação
            p_final = rotate (p_trans, transf_matrix[3], transf_matrix[4], transf_matrix[5])

            
            # salvando a coordenada do ponto em frame local no Dataframe pts_new
            pts_new.loc[i,1:] = [p_final[0,0],p_final[1,0],p_final[2,0]]    
            
            # atualizando a referencia para o ultimo berço
            old_girder = current_girder
            
        # retorna o Dataframe com os pontos nas coordenadas locais
        return pts_new
    
    else:
        # por hora, se o arquivo .xmls base não estiver ordenado o script não funcionará, 
        # então retorna um Dataframe vazio
        return None


def plot_diff_centroids (centroids_diff):
    new_index = np.linspace(0, centroids_diff['x'].size-1, centroids_diff['x'].size)
    centroids_diff['Index'] = new_index
    centroids_diff.insert(0,'Girder',centroids_diff.index)
    centroids_diff.set_index('Index',drop=True,inplace=True)
    centroids_diff.reset_index(drop=True, inplace=True)
    centroids_diff.rename(columns={'x': 'dx [mm]', 'y': 'dy [mm]', 'z': 'dz [mm]'}, inplace=True)

    fig, axs = plt.subplots(3,1, figsize=(18,9), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    tickpos = np.linspace(0,220,21)
    
    centroids_diff.plot.scatter('Girder', 'dx [mm]', c='red', ax=axs[0], title='Horizontal')
    centroids_diff.plot.scatter('Girder', 'dy [mm]', c='limegreen', ax=axs[1], title='Longitudinal')
    centroids_diff.plot.scatter('Girder', 'dz [mm]', c='blue', ax=axs[2], title='Vertical')

    for axes in axs:
        axes.tick_params(axis='x', which='major', direction= 'in', bottom=True, top=True, labelrotation=45,
                         labelsize='small')
        axes.set_xticks(tickpos)
        axes.xaxis.labelpad = 10
        axes.grid(b=True, axis='both', which='major', linestyle='--', alpha=0.5)
        
    plt.minorticks_off()
    
def plot_diff_inOut (centroids_diff):
    new_index = np.linspace(0, centroids_diff['x_in'].size-1, centroids_diff['x_in'].size)
    centroids_diff['Index'] = new_index
    centroids_diff.insert(0,'Girder',centroids_diff.index)
    centroids_diff.set_index('Index',drop=True,inplace=True)
    centroids_diff.reset_index(drop=True, inplace=True)
    centroids_diff.rename(columns={'x_in': 'dx_in [mm]', 'y_in': 'dy_in [mm]', 'z_in': 'dz_in [mm]',
                                   'x_out': 'dx_out [mm]', 'y_out': 'dy_out [mm]', 'z_out': 'dz_out [mm]'}, inplace=True)

    fig, axs = plt.subplots(3,1, figsize=(18,9), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    tickpos = np.linspace(0,220,21)
    
    centroids_diff.plot.scatter('Girder', 'dx_in [mm]', c='red', ax=axs[0], title='Horizontal')
    centroids_diff.plot.scatter('Girder', 'dy_in [mm]', c='limegreen', ax=axs[1], title='Longitudinal')
    centroids_diff.plot.scatter('Girder', 'dz_in [mm]', c='blue', ax=axs[2], title='Vertical')

    for axes in axs:
        axes.tick_params(axis='x', which='major', direction= 'in', bottom=True, top=True, labelrotation=45,
                         labelsize='small')
        axes.set_xticks(tickpos)
        axes.xaxis.labelpad = 10
        axes.grid(b=True, axis='both', which='major', linestyle='--', alpha=0.5)
        
    plt.minorticks_off()
    

def append_values (calc_operation, x_list, y_list, z_list, dataset, current_girder, point_name, index, pts_type):
    
    # criação de listas de coordendas separadas(x, y e z) para cálculo do centróide segundo algumas exceções
    if (current_girder[4:] == 'B03' or current_girder[4:] == 'B11'):
        if (calc_operation == 'centroid'):
            if (len(point_name)==3):
                x_list.append(dataset.loc[index,1])
                z_list.append(dataset.loc[index,3])
            elif (len(point_name) >= 6):
                y_list.append(dataset.loc[index,2])
                
        elif (calc_operation == 'inOut'):
            """ A DEFINIR """
            # temporario...
            if (point_name[-2:] == 'B1'):
                x_list[0].append(dataset.loc[index,1])
                y_list[0].append(dataset.loc[index,2])
            elif (point_name[-2:] == 'MR'):
                x_list[1].append(dataset.loc[index,1])
                y_list[1].append(dataset.loc[index,2])
            else:
                z_list[0].append(dataset.loc[index,3])
                z_list[1].append(dataset.loc[index,3])
            
    elif (current_girder[4:] == 'B05' or current_girder[4:] == 'B09'):
        if (calc_operation == 'centroid'):
            if (pts_type == 'nominal'):
                if (len(point_name) >= 6):
                    x_list.append(dataset.loc[index,1])
                    y_list.append(dataset.loc[index,2])
                    z_list.append(dataset.loc[index,3])
                    
            elif (pts_type == 'measured'):
                if (len(point_name) >= 5):
                    x_list.append(dataset.loc[index,1])
                    y_list.append(dataset.loc[index,2])
                elif (len(point_name) == 4):
                    z_list.append(dataset.loc[index,3])
                    
        elif (calc_operation == 'inOut'):
            if (point_name[-2:] == 'B2'):
                x_list[0].append(dataset.loc[index,1])
                y_list[0].append(dataset.loc[index,2])
            elif (point_name[-2:] == 'MR'):
                x_list[1].append(dataset.loc[index,1])
                y_list[1].append(dataset.loc[index,2])
                
            if (pts_type == 'measured'):
                if (point_name[:2] == 'LV'):
                    z_list[0].append(dataset.loc[index,3])
                    z_list[1].append(dataset.loc[index,3])
            else:
                z_list[0].append(dataset.loc[index,3])
                z_list[1].append(dataset.loc[index,3])
                    
    else:
        if (calc_operation == 'centroid'):
            x_list.append(dataset.loc[index,1])
            y_list.append(dataset.loc[index,2])
            z_list.append(dataset.loc[index,3]) 
        else:
            if (point_name == "C01" or point_name == "C02"):
                x_list[0].append(dataset.loc[index,1])
                y_list[0].append(dataset.loc[index,2])
                z_list[0].append(dataset.loc[index,3]) 
            else:
                x_list[1].append(dataset.loc[index,1])
                y_list[1].append(dataset.loc[index,2])
                z_list[1].append(dataset.loc[index,3]) 
        
        
    return x_list, y_list, z_list

def calculate_centroids (dataset, pts_type):

    # DataFrame que conterá os centroids ao final dessa função
    centroid_df = pd.DataFrame(columns=['Girder', 'x', 'y', 'z'])
    
    # variáveis auxiliares 
    i = 0
    x_list = []
    y_list = []
    z_list = []
    
    while (i<dataset[0].size):
        
        current_girder = dataset.loc[i,0][:7]
        point_name = dataset.loc[i,0][8:]
        
        # insere nas listas temporarias as coordenadas do primeiro ponto do berço
        x_list, y_list, z_list = append_values ('centroid', x_list, y_list, z_list, dataset,
                                                                current_girder, point_name, i, pts_type)
        
        # começando a iteração sobre os outros pontos do mesmo berço
        j = i+1
        while (current_girder == dataset.loc[j,0][:7]):
            point_name = dataset.loc[j,0][8:]
            
            # insere nas listas temporarias as coordenadas do primeiro ponto do berço
            x_list, y_list, z_list = append_values ('centroid', x_list, y_list, z_list, dataset,
                                                                    current_girder, point_name, j, pts_type)


            # tratamento de exceção de menos de 4 pontos medidos nesse berço ...
            # verifica se não é o último ponto
            if ((j+1)<dataset[0].size):
                # verifica se o próximo ponto já é de um berço novo E se o berço atual não é do tipo
                # B05 ou B09 E, finalmente, se existem menos de 4 pontos na lista de coordenadas
                if(dataset.loc[(j+1),0][:7] != current_girder and 
                   current_girder[4:] != 'B05' and current_girder[4:] != 'B09' and 
                   len(x_list) < 4):
                        print("exceção encontrada no berço "+current_girder+": menos de 4 pontos medidos p/ se calcular o X e Z do centroide.")                    
                        positive_list = []
                        negative_list = []
                        for point in x_list:
                            if (point < 0):
                                negative_list.append(point)
                            else:
                                positive_list.append(point)
                        x_list = [np.mean(negative_list), np.mean(positive_list)]
                        """ ATENÇÃO: Eu não deveria fazer esse tratamento também para Y e Z?? """
                    
            j+=1
            if (j>=dataset[0].size):
                break
        
        # cálculo dos centroids
        centr_data = pd.DataFrame(data=np.array([[current_girder, np.mean(x_list).round(4), np.mean(y_list).round(4), np.mean(z_list).round(4)]]),
                                  columns=['Girder', 'x', 'y', 'z'])
        centroid_df = centroid_df.append(centr_data, ignore_index=True)
        
        i = j
        x_list = []
        y_list = []
        z_list = []
    
    # torna os itens da coluna 'girder' nos índices do dataframe,
    # necessário para usar as funções de cálculo aritimético entre dataframes
    centroid_df = centroid_df.set_index('Girder')
    
    # retorna o df com os termos no tipo numérico certo
    return centroid_df.astype('float32')
    

def calculate_inOut (dataset, pts_type):
    # DataFrame que conterá os centroids ao final dessa função
    inout_df = pd.DataFrame(columns=['Girder', 'x_in', 'y_in', 'z_in', 'x_out', 'y_out', 'z_out'])
    
    # variáveis auxiliares 
    i = 0
    x_list = [[],[]] # modelo: [[x_in1, x_in2, ...], [x_out1, x_out2, ...]]
    y_list = [[],[]]
    z_list = [[],[]]
    
    while (i<dataset[0].size):
        
        current_girder = dataset.loc[i,0][:7]
        point_name = dataset.loc[i,0][8:]
        
        # insere nas listas temporarias as coordenadas do primeiro ponto do berço
        x_list, y_list, z_list = append_values ('inOut', x_list, y_list, z_list, dataset,
                                                                current_girder, point_name, i, pts_type)
        # começando a iteração sobre os outros pontos do mesmo berço
        j = i+1
        while (current_girder == dataset.loc[j,0][:7]):
            point_name = dataset.loc[j,0][8:]

            # insere n
            x_list, y_list, z_list = append_values ('inOut', x_list, y_list, z_list, dataset,
                                                                    current_girder, point_name, j, pts_type)

            # tratamento de exceção de menos de 4 pontos medidos nesse berço
            if ((j+1)<dataset[0].size):
                total_points = len(x_list[0]) + len(x_list[1])
                if(dataset.loc[(j+1),0][:7] != current_girder and 
                   current_girder[4:] != 'B05' and current_girder[4:] != 'B09' and 
                   total_points < 4):
                        x_list_in = x_list[0]
                        x_list_out = x_list[1]
                        # verifcação das características das listas de coordenadas em x e y
                        # para computar qual das quinas está faltando, considerando que apenas
                        # 1 ponto está faltando. Modelo que me baseei:
                        # --------------------------------------------------
                        #   [quinas existentes]  | x_list (apenas o sinal)  
                        # --------------------------------------------------
                        #       [1,2,3]          |        [[-,+],[+]]       
                        #       [1,2,4]          |        [[-,+],[-]]       
                        #       [1,3,4]          |        [[-],[+,-]]       
                        #       [2,3,4]          |        [[+],[+,-]]       
                        # --------------------------------------------------
                        if (total_points == 3):
                            error_text = "exceção encontrada no berço "+current_girder+": apenas 3 pontos medidos p/ se calcular a entrada/saída. "
                            if (len(x_list_in) == 1): 
                                if (x_list_in[0] > 0): # caso 4: faltando o C01 do berço
                                    error_text += "Ponto faltante: C01"
                                    # tratativa tipo 1: inserir na sublista 'in' de x_list, y_list e 
                                    # z_list o ponto C0X_new = [C04(x), C02(y), C02(z)]
                                    C0X_new = [x_list[1][1], y_list[0][0], z_list[0][0]]
                                    inout_index = 0
                                    
                                else: # caso 3: faltando o C02
                                    error_text += "Ponto faltante: C02"
                                    # tratativa tipo 2: inserir na sublista 'in' de x_list, y_list e 
                                    # z_list o ponto C0X_new = [C03(x), C01(y), C01(z)]
                                    C0X_new = [x_list[1][0], y_list[0][0], z_list[0][0]]
                                    inout_index = 0
                                    
                            else:
                                if (x_list_out[0] > 0): # caso 1: faltando o C04 do berço
                                    error_text += "Ponto faltante: C04"
                                    # tratativa tipo 3: inserir na sublista 'out' de x_list, y_list e 
                                    # z_list o ponto C0X_new = [C01(x), C03(y), C03(z)]
                                    C0X_new = [x_list[0][0], y_list[0][0], z_list[0][0]]
                                    inout_index = 1
                                    
                                else: # caso 2: faltando o C03
                                    error_text += "Ponto faltante: C03"
                                    # tratativa tipo 4: inserir na sublista 'out' de x_list, y_list e 
                                    # z_list o ponto C0X_new = [C02(x), C04(y), C04(z)]
                                    C0X_new = [x_list[0][1], y_list[0][0], z_list[0][0]]
                                    inout_index = 1
                            
                            x_list[inout_index].append(C0X_new[0])
                            y_list[inout_index].append(C0X_new[1])
                            z_list[inout_index].append(C0X_new[2])
                            
                            print(error_text)
                            
                        elif (len(x_list_in) == 2):
                            error_text = "exceção encontrada no berço "+current_girder+": apenas 2 pontos medidos p/ se calcular a entrada/saída. "
                            
                            """ tratativa para 2 pontos? """
                            
                            print(error_text)
                            
            j+=1
            if (j>=dataset[0].size):
                break
            
        x_list_in = x_list[0]; y_list_in = y_list[0]; z_list_in = z_list[0]
        x_list_out = x_list[1]; y_list_out = y_list[1]; z_list_out = z_list[1]
        
        # cálculo das coordenadas de entrada e saída
        inout_data = pd.DataFrame(data=np.array([[current_girder, np.mean(x_list_in).round(4), np.mean(y_list_in).round(4), np.mean(z_list_in).round(4),
                                                                  np.mean(x_list_out).round(4), np.mean(y_list_out).round(4), np.mean(z_list_out).round(4)]]),
                                  columns=['Girder', 'x_in', 'y_in', 'z_in', 'x_out', 'y_out', 'z_out'])
        inout_df = inout_df.append(inout_data, ignore_index=True)
        
        i = j
        x_list = [[],[]]
        y_list = [[],[]]
        z_list = [[],[]]
    
    # torna os itens da coluna 'girder' nos índices do dataframe,
    # necessário para usar as funções de cálculo aritimético entre dataframes
    inout_df = inout_df.set_index('Girder')
    
    # retorna o df com os termos no tipo numérico certo
    return inout_df.astype('float32')


def calc_df_diff (df1, df2):
    diff = df1.sub(df2)
    return diff


#%%

if __name__ == "__main__":
    
    local_nominals = generate_localFrames("../data/frames_table_fullpre.xlsx", "Planilha4", "../data/SR_nominals.xlsx", "Planilha2", is_sorted=True)
    centroids_nominal = calculate_centroids(local_nominals, 'nominal')
    
    local_measured = generate_localFrames("../data/frames_table_fullpre.xlsx", "Planilha4", "../data/SR_Magnets_Measured.xlsx", "Planilha1", is_sorted=True)
    centroids_measured = calculate_centroids(local_measured, 'measured')
 
    centroids_diff = calc_df_diff (centroids_measured, centroids_nominal)

    inOut_nominal = calculate_inOut(local_nominals, 'nominal')
    inOut_measured = calculate_inOut(local_measured, 'measured')
    
    inOut_diff = calc_df_diff(inOut_nominal, inOut_measured)
    


'''
Próximas implementações:
    - [FEITO] Identificação de entrada e saída de cada berço
    - Implementar cálculo simplificado de rotações; comparar com SA 
    - Importar arquivos e ordenar pela coluna de berços/pontos
    - cálculo de comprimento da máquina: usar método do Henrique
    - corrigir plot horizontal (x-1)
    
    
'''
