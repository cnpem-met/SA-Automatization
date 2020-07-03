
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


""" Função para rotacionar um ponto """
def rotate(P, Rx, Ry, Rz):
    rot_z = np.array([[np.cos(Rz*10**-3), -np.sin(Rz*10**-3), 0], [np.sin(Rz*10**-3), np.cos(Rz*10**-3), 0], [0, 0, 1]])
    rot_y = np.array([[np.cos(Ry*10**-3), 0, np.sin(Ry*10**-3)], [0, 1, 0], [-np.sin(Ry*10**-3), 0, np.cos(Ry*10**-3)]])
    rot_x = np.array([[1, 0, 0], [0, np.cos(Rx*10**-3), -np.sin(Rx*10**-3)], [0, np.sin(Rx*10**-3), np.cos(Rx*10**-3)]])
    ROT = rot_z @ rot_y @ rot_x # MATRIX MULTIPLICATION!
    
    P_new = np.transpose(ROT) @ P
    
    return P_new

""" Função para transladar um ponto """
def translate (P, Tx, Ty, Tz):
    P_new = np.array([[P[0,0] - Tx], [P[1,0] - Ty], [P[2,0] - Tz]])
    return P_new


""" Método que gera um dataframe com pontos em coordenadas locais (frames dos berços), a partir
    das coordenadas globais (frame ML) e dos parâmetros de transformação de coordenadas de cada frame """
def generate_localFrames(lookuptable, pts_ML, mode='full'):
        
    # alocando um novo Dataframe (baseado no df não-modificado dos pontos) para conter
    # as novas coordenadas locais calculadas 
    pts_new = pts_ML.copy()
    # utilizaremos uma cópia do df da lookuptable, pois iremos modificá-lo ao longo das iterações
    lookup_temp = lookuptable.copy()
    
    old_girder = ""
    for i in range (pts_new.iloc[:,0].size):
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
                transf_matrix = [lookup_temp.at[ref_girder,'Tx'], lookup_temp.at[ref_girder,'Ty'],lookup_temp.at[ref_girder,'Tz'],
                             transf_matrix_B02[3], transf_matrix_B02[4], transf_matrix_B02[5]]
            elif (current_girder[4:] == 'B11'):
                # pega os parâmetros de rotação do berço sucessor
                if (current_girder == 'S20-B11'):
                    # nesse caso em específico, o sucessor é o S01-B01
                    transf_matrix = [lookup_temp.at[ref_girder,'Tx'], lookup_temp.at[ref_girder,'Ty'],lookup_temp.at[ref_girder,'Tz'],
                             transf_matrix_S01B01[3], transf_matrix_S01B01[4], transf_matrix_S01B01[5]]
                else:
                    # no resto dos casos, o sucessor é o próximo da lookup_table
                    transf_matrix = [lookup_temp.at[ref_girder,'Tx'], lookup_temp.at[ref_girder,'Ty'],lookup_temp.at[ref_girder,'Tz'],
                                 lookup_temp.at[lookup_temp.index[1],'Rx'], lookup_temp.at[lookup_temp.index[1],'Ry'], lookup_temp.at[lookup_temp.index[1],'Rz']]
            else:
                # caso sem exceção, ou seja, todos os parâmetros de transformação de frame são do próprio berço
                transf_matrix = [lookup_temp.at[ref_girder,'Tx'], lookup_temp.at[ref_girder,'Ty'],lookup_temp.at[ref_girder,'Tz'],
                                 lookup_temp.at[ref_girder,'Rx'], lookup_temp.at[ref_girder,'Ry'], lookup_temp.at[ref_girder,'Rz']]

            
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
        p = np.array([[pts_new.iloc[i,0]], [pts_new.iloc[i,1]], [pts_new.iloc[i,2]]])

        
        # apluicando translação
        p_trans = translate(p, transf_matrix[0], transf_matrix[1], transf_matrix[2])

        
        # aplicando rotação
        p_final = rotate (p_trans, transf_matrix[3], transf_matrix[4], transf_matrix[5])

        
        # salvando a coordenada do ponto em frame local no Dataframe pts_new
        pts_new.iloc[i,0:] = [p_final[0,0],p_final[1,0],p_final[2,0]]    
        
        # atualizando a referencia para o ultimo berço
        old_girder = current_girder
        
    # retorna o Dataframe com os pontos nas coordenadas locais
    return pts_new


"""Função pra plot parcialmente genérica
in: - DataFrame results_df: dataframe de resultados gerado por calculate_angles, centroids etc
    - String analysis_type: "centroid"/"inout"/"angle"
    - Array plots_args_dict: dicionário com dados p/ as propriedade de ordenadas e títulos dos plots.
                                -> Formato: {'y_list' : [var dependente plot 1, var dependente do plot 2, ...], 
                                              'title_list' : [título plot 1,título plot 2, ...]}

                                -> Exemplos: - caso centroid -> {'y_list' : ['x','y'], 'title_list' : ['Horizontal', 'Longitudinal']}
                                              - caso ângulos  -> {'y_list' : ['Roll','Pitch', 'Yaw'], 'title_list' : ['Roll', 'Pitch', 'Yaw']}
                                              - caso inout    -> {'y_list' : ['x_in', 'y_in', 'z_in', 'z_out'], 'title_list' : ['horizontal', 'vertical', 'horizontal', 'vertical']}
"""
def plot_girder_deviation (results_df, analysis_type, plots_args_dict):
    # pegando uma cópia do df original
    results_df = results_df.copy()
    
    # verificando qual o tipo de plot que será, e associando 
    # parâmetros específicos a eles
    if (analysis_type == "centroid"):
        colum1_name = 'x'
        df_colums_dict = {'x': 'dx [mm]', 'y': 'dy [mm]', 'z': 'dz [mm]'}
    elif (analysis_type == 'inout'):
        colum1_name = 'x_in'
        df_colums_dict = {'x_in': 'dx_in [mm]', 'y_in': 'dy_in [mm]', 'z_in': 'dz_in [mm]',
                                   'x_out': 'dx_out [mm]', 'y_out': 'dy_out [mm]', 'z_out': 'dz_out [mm]'}
    elif (analysis_type == 'angle'):
        colum1_name = 'Roll'
        df_colums_dict = {'Roll': 'u_roll [mrad]', 'Pitch': 'u_pitch [mrad]', 'Yaw':'u_yaw [mrad]'}
    
    elif (analysis_type == 'inout_delta'):
        colum1_name = 'delta_x'
        df_colums_dict = {'delta_x': 'dx [mm]', 'delta_y': 'dy [mm]', 'delta_z': 'dz [mm]'}
        
    else:
        print("Falha no plot: tipo de análise não reconhecida.")
        return
    
    """ configurando df """ 
    new_index = np.linspace(0, results_df[colum1_name].size-1, results_df[colum1_name].size)
    results_df['Index'] = new_index
    results_df.insert(0,'Girder',centroids_diff.index)
    results_df.set_index('Index',drop=True,inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    results_df.rename(columns=df_colums_dict, inplace=True)
    
    """ configurando parâmetros do plot """
    # configurando plot de acordo com o número de variáveis que foi passado,
    # ou seja, de acordo com o tamanho da lista 'y_list' recebida
    num_plots = len(plots_args_dict['y_list'])
    # definindo o numero de linhas e colunas do layout do plot
    if (num_plots <= 3):
        grid_subplot = [num_plots,1]
    else:
        grid_subplot = [3, 2]
    # definindo se as absicissas serão compartilhadas ou não
    share_xAxis = 'col'
    if (num_plots > 3 and num_plots % 3 != 0):
        share_xAxis = 'none'
    # criando subplots com os parâmetros gerados acima
    fig, axs = plt.subplots(grid_subplot[0], grid_subplot[1], figsize=(18,9), sharex=share_xAxis)
    plt.subplots_adjust(hspace=0.3)
    tickpos = np.linspace(0,220,21)
    
    """salvando args de entrada"""
    plot_colors = ['red', 'limegreen', 'blue', 'yellow', 'green', 'black']
    # a lista de títulos é direta
    plot_titles = plots_args_dict['title_list']
    # para a lista de colunas do df a serem plotadas, deve-se mapear a lista 'y_list' de entrada
    # em relação ao dict 'df_colums_dict' para estar em conformidade com os novos nomes das colunas
    y_list = []
    for y in plots_args_dict['y_list']:
        if y in df_colums_dict:
            y_list.append(df_colums_dict[y])
    
    """ chamando plots e configurando os seus eixos """
    # caso tenhamos 3 ou menos plots, apenas uma coluna de plots será gerada, enquanto que
    # se for mais de 3, duas serão geradas; por causa disso, a estrutura da lista 'axs' varia
    # entre casos, e portanto foi preciso padronizar em um unico formato em 'axs_new'
    axs_new = []
    if (num_plots <= 3):
        if (num_plots == 1):
            axs_new.append([axs])
        else:
            [axs_new.append([ax, ""]) for ax in axs]
    else:
        axs_new = axs

    for i in range(len(y_list)):
            results_df.plot.scatter('Girder', y_list[i], c=plot_colors[i], ax=axs_new[i%3][int(i/3)], title=plot_titles[i])
    for i in range(len(y_list)):
        axs_new[i%3][int(i/3)].tick_params(axis='x', which='major', direction= 'in', bottom=True, top=True, labelrotation=45,
                         labelsize='small')
        axs_new[i%3][int(i/3)].set_xticks(tickpos)
        axs_new[i%3][int(i/3)].xaxis.labelpad = 10
        axs_new[i%3][int(i/3)].grid(b=True, axis='both', which='major', linestyle='--', alpha=0.5)
    
    # mostrando plots
    plt.minorticks_off()
    

""" Método auxiliar que faz as tratativas de seleção de pontos que irão compor os cálculos 
    dos centroides e das coordenadas de entrada e saída """
def append_values (calc_operation, x_list, y_list, z_list, dataset, current_girder, point_name, index, pts_type):
    
    # criação de listas de coordendas separadas(x, y e z) para cálculo do centróide segundo algumas exceções
    if (current_girder[4:] == 'B03' or current_girder[4:] == 'B11'):
        if (calc_operation == 'centroid'):
            if (len(point_name)==3):
                x_list['pts_name'].append(point_name)
                z_list['pts_name'].append(point_name)
                x_list['pts_val'].append(dataset.iloc[index,0] *-1)
                z_list['pts_val'].append(dataset.iloc[index,2])
            elif (len(point_name) >= 6):
                y_list['pts_name'].append(point_name)
                y_list['pts_val'].append(dataset.iloc[index,1])
                
        elif (calc_operation == 'inOut'):
            """ A DEFINIR """
            # temporario...
            if (point_name[-2:] == 'B1'):
                x_list[0].append(dataset.iloc[index,0] *-1)
                y_list[0].append(dataset.iloc[index,1])
            elif (point_name[-2:] == 'MR'):
                x_list[1].append(dataset.iloc[index,0] *-1)
                y_list[1].append(dataset.iloc[index,1])
            else:
                z_list[0].append(dataset.iloc[index,2])
                z_list[1].append(dataset.iloc[index,2])
            
    elif (current_girder[4:] == 'B05' or current_girder[4:] == 'B09'):
        if (calc_operation == 'centroid'):
            if (pts_type == 'nominal'):
                if (len(point_name) >= 6):
                    x_list['pts_name'].append(point_name)
                    y_list['pts_name'].append(point_name)
                    z_list['pts_name'].append(point_name)
                    x_list['pts_val'].append(dataset.iloc[index,0] *-1)
                    y_list['pts_val'].append(dataset.iloc[index,1])
                    z_list['pts_val'].append(dataset.iloc[index,2])
                    
            elif (pts_type == 'measured'):
                if (len(point_name) >= 5):
                    x_list['pts_name'].append(point_name)
                    y_list['pts_name'].append(point_name)
                    x_list['pts_val'].append(dataset.iloc[index,0] *-1)
                    y_list['pts_val'].append(dataset.iloc[index,1])
                elif (len(point_name) == 4):
                    z_list['pts_name'].append(point_name)
                    z_list['pts_val'].append(dataset.iloc[index,2])
                    
        elif (calc_operation == 'inOut'):
            if (point_name[-2:] == 'B2'):
                x_list[0].append(dataset.iloc[index,0] *-1)
                y_list[0].append(dataset.iloc[index,1])
            elif (point_name[-2:] == 'MR'):
                x_list[1].append(dataset.iloc[index,0] *-1)
                y_list[1].append(dataset.iloc[index,1])
                
            if (pts_type == 'measured'):
                if (point_name[:2] == 'LV'):
                    z_list[0].append(dataset.iloc[index,2])
                    z_list[1].append(dataset.iloc[index,2])
            else:
                z_list[0].append(dataset.iloc[index,2])
                z_list[1].append(dataset.iloc[index,2])
                    
    else:
        if (calc_operation == 'centroid'):
            x_list['pts_name'].append(point_name)
            y_list['pts_name'].append(point_name)
            z_list['pts_name'].append(point_name)
            x_list['pts_val'].append(dataset.iloc[index,0] *-1)
            y_list['pts_val'].append(dataset.iloc[index,1])
            z_list['pts_val'].append(dataset.iloc[index,2]) 
        else:
            if (point_name == "C01" or point_name == "C02"):
                x_list[0].append(dataset.iloc[index,0] *-1)
                y_list[0].append(dataset.iloc[index,1])
                z_list[0].append(dataset.iloc[index,2]) 
            else:
                x_list[1].append(dataset.iloc[index,0] *-1)
                y_list[1].append(dataset.iloc[index,1])
                z_list[1].append(dataset.iloc[index,2]) 
        
        
    return x_list, y_list, z_list


""" Método que calcula os desvios angulares de Roll, Pitch e Yaw de cada berço """
def calculate_angles (dataset):
    
    angles_df = pd.DataFrame(columns=['Girder', 'Roll', 'Pitch', 'Yaw'])
    
    i = 0
    pts_girder = []
    pts_name = []
    
    while (i<dataset.iloc[:,0].size):
        
        current_girder = dataset.index[i][:7]
        point_name = dataset.index[i][8:]
        
        # adiciona na lista todos os pontos de um berço, exceto no caso de B03 e B11,
        # em que só será adicionado os pontos do multipolo
        if (current_girder[4:] == 'B03' or current_girder[4:] == 'B11'):
            if (point_name[-2:] != "B1" and point_name[-2:] != "MR"):
                pts_girder.append([dataset.iloc[i,0], dataset.iloc[i,1], dataset.iloc[i,2]])
                pts_name.append(point_name)
        else:
            pts_girder.append([dataset.iloc[i,0], dataset.iloc[i,1], dataset.iloc[i,2]])
            pts_name.append(point_name)
        
        
        # começando a iteração sobre os outros pontos do mesmo berço
        j = i+1
        while (current_girder == dataset.index[j][:7]):
            point_name = dataset.index[j][8:]
            
            # mesma regra se aplica
            if (current_girder[4:] == 'B03' or current_girder[4:] == 'B11'):
                if (point_name[-2:] != "B1" and point_name[-2:] != "MR"):
                    pts_girder.append([dataset.iloc[j,0], dataset.iloc[j,1], dataset.iloc[j,2]])
                    pts_name.append(point_name)
            else:
               pts_girder.append([dataset.iloc[j,0], dataset.iloc[j,1], dataset.iloc[j,2]])
               pts_name.append(point_name)
            
            j+=1
            
            # se proximo ponto for o ultimo da lista,
            # sai do loop interno
            if (j>=dataset.iloc[:,0].size):
                break
            

        """ tratamento de exceção de menos de 4 pontos medidos nesse berço """
        # verifica se no caso do B1 e Multipolo há menos de 4 pontos
        if ((current_girder[4:] != 'B05' and  current_girder[4:] != 'B09') and len(pts_girder)<4):
            error_txt = "exceção encontrada no berço "+current_girder+": "
            if (len(pts_girder) == 3):
                error_txt += "3 pontos medidos p/ se calcular os erros angulares."
                
                if (current_girder[4:] != 'B11'):
                    pts = ['C01', 'C02', 'C03', 'C04']
                else:
                    pts = ['C09', 'C10', 'C11', 'C12']
                    
                if(not pts[0] in pts_name):
                    # coordenadas do C01/C09 construido: [C04(x), C02(y), C02(z)]
                    # ** assume-se novamente que os pontos estão ordenados!
                    pt_C01_C09 = [pts_girder[2][0], pts_girder[0][1], pts_girder[0][2]]
                    pts_girder.append(pt_C01_C09)
                    pts_name.append(pts[0])
                elif (not pts[1] in pts_name):
                    # coordenadas do C02 construido: [C03(x), C01(y), C01(z)]
                    pt_C02_C10 = [pts_girder[1][0], pts_girder[0][1], pts_girder[0][2]]
                    pts_girder.append(pt_C02_C10)
                    pts_name.append(pts[1])
                elif (not pts[2] in pts_name):
                    # coordenadas do C03 construido: [C02(x), C04(y), C04(z)]
                    pt_C03_C11 = [pts_girder[1][0], pts_girder[2][1], pts_girder[2][2]]
                    pts_girder.append(pt_C03_C11)
                    pts_name.append(pts[2])
                else:
                    # coordenadas do C04 construido: [C01(x), C03(y), C03(z)]
                    pt_C03_C12 = [pts_girder[0][0], pts_girder[2][1], pts_girder[2][2]]
                    pts_girder.append(pt_C03_C12)
                    pts_name.append(pts[3])
               
                    
            elif (len(pts_girder) == 2):
                error_txt += "2 pontos medidos p/ se calcular os erros angulares."
                """ tratamento de exceção para apenas 2 pontos? """
                pass
            
            print(error_txt)
            
        """ calculo dos angulos"""
        if (current_girder[4:] != 'B05' and  current_girder[4:] != 'B09'):
            # operação necessária pois a lista pts_girder pode estar desordenada
            if (current_girder[4:] != 'B11'):
                pts = ['C01', 'C02', 'C03', 'C04']
            else:
                pts = ['C09', 'C10', 'C11', 'C12']
            
            
            index_C1 = pts_name.index(pts[0])
            index_C2 = pts_name.index(pts[1])
            index_C3 = pts_name.index(pts[2])
            index_C4 = pts_name.index(pts[3])
            
            C1 = pts_girder[index_C1] # aqui C1 pode representar tanto C1 (caso dos multipolos) quanto C9 (caso dos B1)
            C2 = pts_girder[index_C2] # o mesmo se aplica para C2, C3 e C4
            C3 = pts_girder[index_C3]
            C4 = pts_girder[index_C4]
            
            roll_tan = (np.mean([C2[2], C3[2]]) - np.mean([C1[2], C4[2]])) / (np.mean([C1[0], C4[0]]) - np.mean([C2[0], C3[0]])) 
            roll = math.atan(roll_tan) *10**3 # em mrad
            
            pitch_tan = (np.mean([C3[2], C4[2]]) - np.mean([C1[2], C2[2]])) / (np.mean([C3[1], C4[1]]) - np.mean([C1[1], C2[1]]))
            pitch = math.atan(pitch_tan) *10**3 # mrad
            
            yaw_tan = (np.mean([C3[0], C4[0]]) - np.mean([C1[0], C2[0]])) / (np.mean([C3[1], C4[1]]) - np.mean([C1[1], C2[1]]))
            yaw = math.atan(yaw_tan) *10**3 # mrad
        
        else: # caso dos B2
        
            # considera-se que os pontos estão ordenados
            error = False
            try:
                B2, B2MR, LV01, LV02, LV03 = pts_girder
            except ValueError:
                print("exceção encontrada no berço "+current_girder+" to tipo B2: menos de 5 pontos no total.\n Calculo de angulos falhou para esse berço")
                error = True
                
            if(error):
                roll = pitch = yaw = -999 # valor para elucidar que houve erro no cálculo para esse berço
            else:
                roll_tan = (np.mean([LV01[2], LV03[2]]) - LV02[2]) / (LV02[0] - np.mean([LV01[0], LV03[0]]))
                roll = math.atan(roll_tan) *10**3 # mrad
                
                pitch_tan = (LV03[2] - LV01[2]) / (LV03[1] - LV01[1])
                pitch = math.atan(pitch_tan) *10**3 # mrad
                
                # transladando B2 para a origem do frame
                B2_new = np.array([[B2[0]], [B2[1]], [B2[2]]])
                B2_new = translate(B2_new, -B2[0], -B2[1], -B2[2])
                B2_new = [B2_new[0,0], B2_new[1,0], B2_new[2,0]]
                
                # aplicando o mesmo shift em B2MR, e rotacionando em 20 mrad
                B2MR_new = np.array([[B2MR[0]], [B2MR[1]], [B2MR[2]]])
                B2MR_new = translate(B2MR_new, -B2[0], -B2[1], -B2[2])
                B2MR_new = rotate(B2MR_new, 0, 0, 20)
                B2MR_new = [B2MR_new[0,0], B2MR_new[1,0], B2MR_new[2,0]]
                
                yaw_tan = (B2MR_new[0] - B2_new[0]) / (B2MR_new[1] - B2_new[1])
                yaw = math.atan(yaw_tan) *10**3 # mrad
                

                
        # salvando angulos em DataFrame
        angle_data = pd.DataFrame(data=np.array([[current_girder, roll, pitch, yaw]]),columns=['Girder', 'Roll', 'Pitch', 'Yaw'])
        angles_df = angles_df.append(angle_data, ignore_index=True)
        
        i = j
        pts_girder = []
        pts_name = []
    
    # torna os itens da coluna 'girder' nos índices do dataframe,
    # necessário para usar as funções de cálculo aritimético entre dataframes
    angles_df = angles_df.set_index('Girder')
    
    # retorna o df com os termos no tipo numérico certo
    return angles_df.astype('float32')

"""" Método para cálculo das coordenadas dos centroides dos berços """
def calculate_centroids (dataset, pts_type):

    # DataFrame que conterá os centroids ao final dessa função
    centroid_df = pd.DataFrame(columns=['Girder', 'x', 'y', 'z'])
    
    # variáveis auxiliares 
    i = 0
    x_list = {'pts_name':[],'pts_val':[]}
    y_list = {'pts_name':[],'pts_val':[]}
    z_list = {'pts_name':[],'pts_val':[]}
    
    while (i<dataset.iloc[:,0].size):
        
        current_girder = dataset.index[i][:7]
        point_name = dataset.index[i][8:]
        
        # insere nas listas temporarias as coordenadas do primeiro ponto do berço
        x_list, y_list, z_list = append_values ('centroid', x_list, y_list, z_list, dataset,
                                                                current_girder, point_name, i, pts_type)
        
        # começando a iteração sobre os outros pontos do mesmo berço
        j = i+1
        while (current_girder == dataset.index[j][:7]):
            point_name = dataset.index[j][8:]
            
            # insere nas listas temporarias as coordenadas do primeiro ponto do berço
            x_list, y_list, z_list = append_values ('centroid', x_list, y_list, z_list, dataset,
                                                                    current_girder, point_name, j, pts_type)


            # tratamento de exceção de menos de 4 pontos medidos nesse berço ...
            # verifica se não é o último ponto
            if ((j+1)<dataset.iloc[:,0].size):
                # verifica se o próximo ponto já é de um berço novo E se o berço atual não é do tipo
                # B05 ou B09 E, finalmente, se existem menos de 4 pontos na lista de coordenadas
                if(dataset.index[j+1][:7] != current_girder and 
                   current_girder[4:] != 'B05' and current_girder[4:] != 'B09' and 
                   len(x_list['pts_name']) < 4):
                        print("exceção encontrada no berço "+current_girder+": menos de 4 pontos medidos p/ se calcular o X e Z do centroide.")                    
                        
                        if (current_girder[4:] == 'B11'):
                            pts_list = ['C09', 'C10', 'C11', 'C12']
                        else:
                            pts_list = ['C01', 'C02', 'C03', 'C04']
                        
                        
                        for pt_name in pts_list:
                            if (not pt_name in x_list['pts_name']):
                                missing_pt = pt_name
                                break
                                
                        
                        if (missing_pt == 'C01' or missing_pt == 'C09'): # x_list = z_list = [C02, C03, C04]
                            x_list['pts_val'] = [np.mean([x_list['pts_val'][0], x_list['pts_val'][1]]), x_list['pts_val'][2]]
                            z_list['pts_val'] = [np.mean([z_list['pts_val'][1], z_list['pts_val'][2]]), z_list['pts_val'][0]]
                        elif (missing_pt == 'C02' or missing_pt == 'C10'): # x_list = z_list = [C01, C03, C04]
                            x_list['pts_val'] = [np.mean([x_list['pts_val'][0], x_list['pts_val'][2]]), x_list['pts_val'][1]]
                            z_list['pts_val'] = [np.mean([z_list['pts_val'][1], z_list['pts_val'][2]]), z_list['pts_val'][0]]
                        elif (missing_pt == 'C03' or missing_pt == 'C11'): # x_list = z_list = [C01, C02, C04]
                            x_list['pts_val'] = [np.mean([x_list['pts_val'][0], x_list['pts_val'][2]]), x_list['pts_val'][1]]
                            z_list['pts_val'] = [np.mean([z_list['pts_val'][0], z_list['pts_val'][1]]), z_list['pts_val'][2]]
                        else: # x_list = z_list = [C01, C02, C03]
                            x_list['pts_val'] = [np.mean([x_list['pts_val'][1], x_list['pts_val'][2]]), x_list['pts_val'][0]]
                            z_list['pts_val'] = [np.mean([z_list['pts_val'][0], z_list['pts_val'][1]]), z_list['pts_val'][2]]

            j+=1
            if (j>=dataset.iloc[:,0].size):
                break
        
        # cálculo dos centroids
        centr_data = pd.DataFrame(data=np.array([[current_girder, np.mean(x_list['pts_val']).round(4), np.mean(y_list['pts_val']).round(4), np.mean(z_list['pts_val']).round(4)]]),
                                  columns=['Girder', 'x', 'y', 'z'])
        centroid_df = centroid_df.append(centr_data, ignore_index=True)
        
        i = j
        x_list = {'pts_name':[],'pts_val':[]}
        y_list = {'pts_name':[],'pts_val':[]}
        z_list = {'pts_name':[],'pts_val':[]}
    
    # torna os itens da coluna 'girder' nos índices do dataframe,
    # necessário para usar as funções de cálculo aritimético entre dataframes
    centroid_df = centroid_df.set_index('Girder')
    
    # retorna o df com os termos no tipo numérico certo
    return centroid_df.astype('float32')
    

""" Método para geração de um dataframe que contenha as coordenadas de entrada
    e saída de cada berço """
def calculate_inOut (dataset, pts_type):
    # DataFrame que conterá os centroids ao final dessa função
    inout_df = pd.DataFrame(columns=['Girder', 'x_in', 'y_in', 'z_in', 'x_out', 'y_out', 'z_out'])
    
    # variáveis auxiliares 
    i = 0
    x_list = [[],[]] # modelo: [[x_in1, x_in2, ...], [x_out1, x_out2, ...]]
    y_list = [[],[]]
    z_list = [[],[]]
    
    while (i<dataset.iloc[:,0].size):
        
        current_girder = dataset.index[i][:7]
        point_name = dataset.index[i][8:]
        
        # insere nas listas temporarias as coordenadas do primeiro ponto do berço
        x_list, y_list, z_list = append_values ('inOut', x_list, y_list, z_list, dataset,
                                                                current_girder, point_name, i, pts_type)
        # começando a iteração sobre os outros pontos do mesmo berço
        j = i+1
        while (current_girder == dataset.index[j][:7]):
            point_name = dataset.index[j][8:]

            # insere n
            x_list, y_list, z_list = append_values ('inOut', x_list, y_list, z_list, dataset,
                                                                    current_girder, point_name, j, pts_type)

            # tratamento de exceção de menos de 4 pontos medidos nesse berço
            if ((j+1)<dataset.iloc[:,0].size):
                total_points = len(x_list[0]) + len(x_list[1])
                if(dataset.index[j+1][:7] != current_girder and 
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
            if (j>=dataset.iloc[:,0].size):
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

""" Método para cálculo entre os deltas entre as coordenadas de entrada de um berço
    e as coord. de saída de seu antecessor """
def calc_delta_inout (inout_df):
    inout_delta = pd.DataFrame(columns=['Girder', 'delta_x', 'delta_y', 'delta_z'])
    
    for i in range (inout_df.iloc[:,0].size):
        cur_girder = inout_df.index[i]
        if (cur_girder == 'S01-B01'):
            ref_girder = 'S20-B11'
        else:
            ref_girder = inout_df.index[i-1]

        data_arg = np.array([[cur_girder, inout_df.loc[cur_girder, 'x_in'] - inout_df.loc[ref_girder, 'x_out'],
                            inout_df.loc[cur_girder, 'y_in'] - inout_df.loc[ref_girder, 'y_out'],
                            inout_df.loc[cur_girder, 'z_in'] - inout_df.loc[ref_girder, 'z_out']]])
                    
        data_df = pd.DataFrame(data=data_arg, columns=['Girder', 'delta_x', 'delta_y', 'delta_z'])
        inout_delta = inout_delta.append(data_df, ignore_index=True)
    
    inout_delta = inout_delta.set_index('Girder')
    return inout_delta.astype('float32')


""" Método para calcular a diferença entre 2 dfs com a mesma estrutura """
def calc_df_diff (df1, df2):
    diff = df1.sub(df2)
    return diff

""" Método que carrega dados em um dataframe a partir de planilhas excel,
    além de ordenar os dados e criar a coluna de índices com os nomes dos berços """
def load_and_sort_dataframe (excel_file_dir, excel_sheet, has_header):
    key_column = 'Girder'
    if (has_header):
        header_val = 0
        names_val = None
    else:
        header_val = None
        names_val = ['Girder', 'x', 'y', 'z']
    
    # extraindo os dados das planilhas e alocando em Dataframes
    df = pd.read_excel(excel_file_dir, sheet_name=excel_sheet, header=header_val, names=names_val)
    df_sorted = df.sort_values(by=key_column)
    df_sorted.reset_index(drop=True, inplace=True)
    df_sorted = df_sorted.set_index(key_column)
    
    return df_sorted


""" Método para atualizar um dataframe que tenha dados obsoletos; a função funciona substituindo as
    linhas do df do 2o argumento (updated_parcial...) nas linhas correspondentes no df do 1o argumento """
def update_dataframe (old_fully_df, updated_parcial_df):
    
    new_fully_df = old_fully_df.copy()
    
    new_fully_df.loc[updated_parcial_df.index] = np.nan
    new_fully_df = new_fully_df.combine_first(updated_parcial_df)
    
    return new_fully_df



""" Main do script... """
if __name__ == "__main__":
    
    # carregando dados das planilhas, e assegurando que os dataframes gerados estejam ordenados
    frameTransf_lookup = load_and_sort_dataframe ("../data/input/frames_table_fullpre.xlsx", "Planilha4", has_header=True)
    ptsML_nominals =  load_and_sort_dataframe ("../data/input/SR_nominals.xlsx", "Planilha2", has_header=False)
    ptsML_measured = load_and_sort_dataframe ("../data/input/SR_Magnets_Measured_rev2.xlsx", "Planilha1", has_header=False)

    # gerando os pontos a partir de coordenadas locais dos berços
    ptsLocal_nominals = generate_localFrames(frameTransf_lookup, ptsML_nominals)
    ptsLocal_measured = generate_localFrames(frameTransf_lookup, ptsML_measured)
    
    # calculando os centroides
    centroids_nominal = calculate_centroids(ptsLocal_nominals, 'nominal')
    centroids_measured = calculate_centroids(ptsLocal_measured, 'measured')
    centroids_diff = calc_df_diff (centroids_measured, centroids_nominal) 
    
    # plotando desvios dos centroides
    plot_args = {'y_list' : ['x', 'y'], 'title_list' : ['horizontal', 'longitudinal']}
    plot_girder_deviation (centroids_diff, 'centroid', plot_args)
    
    # calculando as coordenadas de entrada e saída
    inOut_nominal = calculate_inOut(ptsLocal_nominals, 'nominal')
    inOut_measured = calculate_inOut(ptsLocal_measured, 'measured')
    inOut_diff = calc_df_diff(inOut_measured, inOut_nominal)

    # plotando desvios em in/out
    plot_args = {'y_list' : ['x_in', 'y_in', 'z_in', 'x_out', 'y_out', 'z_out'], 'title_list' : ['horizontal', 'longitudinal', 'vertical', 'horizontal', 'longitudinal', 'vertical',]}
    plot_girder_deviation (inOut_diff, 'inout', plot_args)
    
    # calculando e plotando desvios angulares
    rotational_devs = calculate_angles(ptsLocal_measured)
    plot_args = {'y_list' : ['Roll', 'Pitch'], 'title_list' : ['Roll', 'Pitch']}
    plot_girder_deviation (rotational_devs, 'angle', plot_args)
    
    # calculando e plotando os deltas entre entrada e saída de berços adjacentes
    inout_delta = calc_delta_inout (inOut_diff)
    plot_args = {'y_list' : ['delta_x', 'delta_y', 'delta_z'], 'title_list' : ['dX', 'dY', 'dZ']}
    plot_girder_deviation (inout_delta, 'inout_delta', plot_args)


'''
Próximas implementações:
    - [FEITO] Identificação de entrada e saída de cada berço
    - [FEITO] Implementar cálculo simplificado de rotações
    - [FEITO] Unificar função de plot para aceitar todos os tipos de análise
    - [FEITO] Comparar desvios angulares e centroids calculados pelo script com os calculados pelo SA 
    - [FEITO] Importar arquivos e ordenar pela coluna de berços/pontos
    - [FEITO] padronizar cabeçalhos (presença ou não etc)  dos arquivos .xlxs e dos DataFrames, além dos índices dos DF
    - [FEITO] implementar função que atualiza dataframe a partir de outro df parcial com dados mais atualizados
    - [FEITO] corrigir plot horizontal (x * -1)
    - [FEITO] implementar ferramenta de cálculo de delta entre entrada e saida de berços vizinhos
    - comparar resultados com diferentes métodos de tratativas em casos de pontos faltantes
    - cálculo de comprimento da máquina
    - implementar interface gráfica

'''
