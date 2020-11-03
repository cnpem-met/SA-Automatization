import numpy as np
import matplotlib.pyplot as plt


class Plot(object):

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
    @staticmethod
    def plotGirderDeviation(results_df, analysis_type, plots_args_dict, freezePlot=False):
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
            df_colums_dict = {
                'Roll': 'u_roll [mrad]', 'Pitch': 'u_pitch [mrad]', 'Yaw': 'u_yaw [mrad]'}

        elif (analysis_type == 'inout_delta'):
            colum1_name = 'delta_x'
            df_colums_dict = {
                'delta_x': 'dx [mm]', 'delta_y': 'dy [mm]', 'delta_z': 'dz [mm]'}

        else:
            print("Falha no plot: tipo de análise não reconhecida.")
            return

        """ configurando df """
        new_index = np.linspace(
            0, results_df[colum1_name].size-1, results_df[colum1_name].size)
        results_df['Index'] = new_index
        results_df.insert(0, 'Girder', results_df.index)
        results_df.set_index('Index', drop=True, inplace=True)
        results_df.reset_index(drop=True, inplace=True)
        results_df.rename(columns=df_colums_dict, inplace=True)

        """ configurando parâmetros do plot """
        # configurando plot de acordo com o número de variáveis que foi passado,
        # ou seja, de acordo com o tamanho da lista 'y_list' recebida
        num_plots = len(plots_args_dict['y_list'])
        # definindo o numero de linhas e colunas do layout do plot
        if (num_plots <= 3):
            grid_subplot = [num_plots, 1]
        else:
            grid_subplot = [3, 2]
        # definindo se as absicissas serão compartilhadas ou não
        share_xAxis = 'col'
        if (num_plots > 3 and num_plots % 3 != 0):
            share_xAxis = 'none'
        # criando subplots com os parâmetros gerados acima
        fig, axs = plt.subplots(
            grid_subplot[0], grid_subplot[1], figsize=(18, 9), sharex=share_xAxis)
        plt.subplots_adjust(hspace=0.3)
        tickpos = np.linspace(0, 220, 21)

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
            results_df.plot.scatter(
                'Girder', y_list[i], c=plot_colors[i], ax=axs_new[i % 3][int(i/3)], title=plot_titles[i])
        for i in range(len(y_list)):
            axs_new[i % 3][int(i/3)].tick_params(axis='x', which='major', direction='in', bottom=True, top=True, labelrotation=45,
                                                 labelsize='small')
            axs_new[i % 3][int(i/3)].set_xticks(tickpos)
            axs_new[i % 3][int(i/3)].xaxis.labelpad = 10
            axs_new[i % 3][int(i/3)].grid(b=True, axis='both',
                                          which='major', linestyle='--', alpha=0.5)

        # mostrando plots
        plt.minorticks_off()

        # acrescentando intervalo entre plots
        plt.draw()
        plt.pause(0.001)

        # condicional para definir se a figura irá congelar o andamento do app
        # usado no ultimo plot, pois senão as janelas fecharão assim que o app acabar de executar
        if (not freezePlot):
            plt.ion()
        else:
            plt.ioff()

        # evocando a tela com a figura
        plt.show()
