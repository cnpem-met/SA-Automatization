from numpy.core.numeric import NaN
from entities import (Point, Frame, Transformation)
from accelerators import (SR, Booster, LTB, BTS, FE)
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import matplotlib.offsetbox as offsetbox
from matplotlib.transforms import Affine2D 


class DataUtils():
    @staticmethod
    def generateFrameDict(lookupTableDF):
        frameDict = {}

        # creating base frame
        frameDict['machine-local'] = Frame('machine-local', Transformation(
            'machine-local', 'machine-local', 0, 0, 0, 0, 0, 0))

        # iterate over dataframe
        for frameName, dof in lookupTableDF.iterrows():
            newFrameName = frameName + "-NOMINAL"
            transformation = Transformation(
                'machine-local', newFrameName, dof['Tx'], dof['Ty'], dof['Tz'], dof['Rx'], dof['Ry'], dof['Rz'])

            frame = Frame(newFrameName, transformation)
            frameDict[newFrameName] = frame

        return frameDict

    @staticmethod
    def readExcel(fileDir, dataType, sheetName=0):

        # create header
        if (dataType == 'lookuptable'):
            header = ['frame', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        else:
            header = ['Magnet', 'x', 'y', 'z']

        # reading file
        df = pd.read_excel(fileDir, sheet_name=sheetName,
                           header=None, names=header)

        # checking if it has a bult-in header and droping it
        if (type(df.iloc[0, 1]) is str):
            df = df.drop([0])

        # sorting and indexing df
        df = df.sort_values(by=header[0])
        df.reset_index(drop=True, inplace=True)
        df = df.set_index(header[0])

        return df

    @staticmethod
    def readCSV(fileDir, dataType):
        # create header
        if (dataType == 'lookuptable'):
            header = ['frame', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        else:
            header = ['Magnet', 'x', 'y', 'z']

        # reading csv file with automatic detection of its delimeter
        data = pd.read_csv(fileDir, sep=None, engine='python', names=header)

        # checking if it has a bult-in header and droping it
        if (type(data.iloc[0, 1]) is str):
            data = data.drop([0])

        # sorting and indexing df
        data = data.sort_values(by=header[0])
        data.reset_index(drop=True, inplace=True)
        data = data.set_index(header[0])

        return data

    @staticmethod
    def writeToExcel(fileDir, data):
        data.to_excel(fileDir)

    @staticmethod
    def plotDevitationData(deviations, **plotArgs):  # precisa adaptação pra booster
        # pegando uma cópia do df original
        plotData = deviations.copy()

        accelerator = plotArgs['accelerator']

        # applying filters
        if(accelerator == 'SR' and plotArgs['filtering']['SR']['allDofs']):
            for magnet, dof in plotData.iterrows():
                if('B03-QUAD01' in magnet or 'B11-QUAD01' in magnet or ('QUAD' in magnet and not 'LONG' in magnet)):
                    plotData.at[magnet, 'Tz'] = float('nan')
                elif('B1-LONG' in magnet):
                    plotData.at[magnet, 'Tx'] = float('nan')
                    plotData.at[magnet, 'Ty'] = float('nan')
                    plotData.at[magnet, 'Rx'] = float('nan')
                    plotData.at[magnet, 'Ry'] = float('nan')
                    plotData.at[magnet, 'Rz'] = float('nan')

        if(accelerator == 'booster' and plotArgs['filtering']['booster']['quad02']):
            for magnet, dof in plotData.iterrows():
                if ('QUAD02' in magnet):
                    plotData.at[magnet, :] = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]

        if (accelerator == 'SR'):
            plotTitle = 'Global Alignment Profile - Storage Ring'
            tolerances = [0.08, 0.08, 0.1, 0.3]
            tick_spacing = 18.5
            tick_id = 'Sector'
            bins = [6,6,12,10]
            x_extension = 2
        elif (accelerator == 'booster'):
            plotTitle = 'Global Alignment Profile - Booster'
            tolerances = [0.16, 0.16, 0.4, 0.8]
            tick_spacing = 5
            tick_id = 'Wall'
            bins = [10, 10, 6, 6]
            x_extension = 2
        elif (accelerator == 'LTB'):
            plotTitle = 'Global Alignment Profile - LTB'
            tolerances = [0.16, 0.16, 0.4, 0.8]
            tick_spacing = 1
            tick_id = 'Magnet'
            bins = [10, 10, 6, 6]
            x_extension = 1
        elif (accelerator == 'BTS'):
            plotTitle = 'Global Alignment Profile - BTS'
            tolerances = [0.16, 0.16, 0.4, 0.8]
            tick_spacing = 1
            tick_id = 'Magnet'
            bins = [10, 10, 6, 6]
            x_extension = 1
        elif (accelerator == 'FE'):
            plotTitle = 'Global Alignment Profile - Front Ends'
            tolerances = [0.16, 0.16, 0.4, 0.8]
            tick_spacing = 1
            tick_id = 'Magnet'
            bins = [10, 10, 6, 6]
            x_extension = 1

        # tolerances_booster = [0.16, 0.16, 0.4, 0.8]
        unc = [0.018, 0.043, 0.023, 0.026]

        plot_args = {'y_list': ['Tx', 'Ty', 'Tz', 'Rz'], 'title_list': ['HORIZONTAL', 'VERTICAL', 'LONGITUDINAL', 'ROLL'], 'fig_title': plotTitle}

        xAxisTitle = 'Magnet'

        colum1_name = 'Tx'
        df_colums_dict = {'Tx': 'deviation horiz. [mm]', 'Ty': 'deviation vert. [mm]', 'Tz': 'deviation long. [mm]', 'Rz': 'rotation roll [mrad]'}

        # xAxisTicks = []
        # for i in range(0, plotData.iloc[:, 0].size):
        #     sector = int(i/17) + 1
        #     xAxisTicks.append(str(sector))

        """ configurando df """
        new_index = np.linspace(
            0, plotData[colum1_name].size-1, plotData[colum1_name].size)

        magnetNameList = plotData.index.to_numpy()

        # EXCLUINDO COLUNAS COM DOFS DESNECESSÁRIOS
        plotData = plotData.drop(columns=['Rx', 'Ry'])

        plotData['Index'] = new_index
        plotData.insert(0, xAxisTitle, plotData.index)
        plotData.set_index('Index', drop=True, inplace=True)
        plotData.reset_index(drop=True, inplace=True)
        plotData.rename(columns=df_colums_dict, inplace=True)

        # invertendo sinal das translações transversais
        # plotData["d_transv (mm)"] = -plotData["d_transv (mm)"]

        """ configurando parâmetros do plot """
        # definindo o numero de linhas e colunas do layout do plot
        grid_subplot = [4, 2]
        # grid_subplot = [2, 2]
        # definindo se as absicissas serão compartilhadas ou não
        share_xAxis = 'col'
        # criando subplots com os parâmetros gerados acima

        # fig = plt.figure(figsize=(18, 9))
        fig = plt.figure()
        fig.tight_layout()
        # gs = fig.add_gridspec(4,2, **dict(hspace=0.3, wspace=0.1, width_ratios=[4,1]))
        gs_dev = plt.GridSpec(4,2, top=0.9, hspace=0.3, wspace=0.2, width_ratios=[4,1])
        gs_rot = plt.GridSpec(4,2, bottom=0.07, hspace=0.3, wspace=0.2, width_ratios=[4,1])
        

        axs_plot, axs_hist = [], []
        axs_plot.append(fig.add_subplot(gs_dev[0,0]))
        axs_plot.append(fig.add_subplot(gs_dev[1,0], sharex=axs_plot[0]))
        axs_plot.append(fig.add_subplot(gs_dev[2,0], sharex=axs_plot[0]))
        axs_plot.append(fig.add_subplot(gs_rot[3,0], sharex=axs_plot[0]))

        axs_hist.append(fig.add_subplot(gs_dev[0,1]))
        axs_hist.append(fig.add_subplot(gs_dev[1,1]))
        axs_hist.append(fig.add_subplot(gs_dev[2,1]))
        axs_hist.append(fig.add_subplot(gs_rot[3,1]))

        # plot_dim = axs_hist[3].get_position().bounds

        # for i in range(4):
        #     axs_plot[i].set_position([plot_dim[0], plot_dim[1], plot_dim[2], 0.2*plot_dim[3]])
        #     axs_hist[i].set_position([plot_dim[0], plot_dim[1], plot_dim[2], 0.2*plot_dim[3]])

        # fig, axs = plt.subplots(grid_subplot[0], grid_subplot[1], figsize=(18, 9), sharex=share_xAxis, gridspec_kw=dict(hspace=0.3, wspace=0.1, width_ratios=[4,1]))

        fontBaseSize = 7
        fontScale = 1.2

        # titulo da figura
        bbox_props = dict(boxstyle="square,pad=0.5", fc="white", lw=0.5)
        fig.suptitle(plot_args['fig_title'], y=0.97, color='black', weight='semibold', fontsize=fontScale*(fontBaseSize+6), bbox=bbox_props)
        # fig.set_tight_layout(dict(pad=0))
        # fig.subplots_adjust(left=0.08,bottom=0.08, right=0.88, top=0.88)
        # fig.subplots_adjust(bottom=0.08)

        spacing = int(plotData.iloc[:, 0].size/tick_spacing)
        tickpos = np.linspace(0, plotData.iloc[:, 0].size - tick_spacing, spacing)
        print(spacing, plotData.iloc[:, 0].size)

        tickLabels = []
        if (accelerator == 'SR'):
            [tickLabels.append(f'{str(i+1)}') for i in range(0, len(tickpos))]
            # tickLabels.append('')
        elif (accelerator == 'booster'):
            [tickLabels.append(f'{str(2*i+1)}') for i in range(0, len(tickpos))]
            # tickLabels[-1] = ''

        """salvando args de entrada"""
        pallet = 3
        
        if (pallet == 1):
            plot_colors = ['blue', 'orange', 'green', 'black']
        elif (pallet == 2):
            plot_colors = ['cornflowerblue', 'coral', 'forestgreen', 'black']
        elif (pallet == 3):
            plot_colors = ['royalblue', 'darkorange', 'seagreen', 'black']
        
        
        # a lista de títulos é direta
        plot_titles = plot_args['title_list']

        # para a lista de colunas do df a serem plotadas, deve-se mapear a lista 'y_list' de entrada
        # em relação ao dict 'df_colums_dict' para estar em conformidade com os novos nomes das colunas
        y_list = []
        for y in plot_args['y_list']:
            if y in df_colums_dict:
                y_list.append(df_colums_dict[y])

        x = magnetNameList
        y = []

        for i in range(4):
            y.append(plotData[y_list[i]].to_numpy()) 

        for i in range(len(y)):
            if (plotArgs['filtering'][accelerator]['errorbar']):
                axs_plot[i].errorbar(x, y[i], yerr=unc[i], fmt='o', marker='o', ms=2, mec=plot_colors[i], mfc=plot_colors[i], color='wheat', ecolor='k', elinewidth=0.5, capthick=0.5, capsize=3)
            else:
                k=0
                while (k < len(x)):
                    currGirder = x[k].split('-')[0]+'-'+x[k].split('-')[1]
                    try:
                        nextGirder = x[k+1].split('-')[0]+'-'+x[k+1].split('-')[1]
                    except IndexError:
                        nextGirder = None
                    if(currGirder == nextGirder):
                        # Checking if user set to appear lines between magnets within the same girder
                        if (plotArgs['filtering'][accelerator]['lineWithinGirder']):
                            axs_plot[i].plot(x[k: k+2], y[i][k: k+2], 'o-', color=plot_colors[i], markersize=4)
                            k+=2
                        else:
                            axs_plot[i].plot(x[k], y[i][k], 'o-', color=plot_colors[i], markersize=4)
                            k+=1
                    else:
                        axs_plot[i].plot(x[k], y[i][k], 'o-', color=plot_colors[i], markersize=4)
                        k+=1

            if (i==3):
                axs_plot[i].tick_params(axis='x', which='major', direction='in', bottom=True, top=False, labelrotation=45,labelsize=fontScale*(fontBaseSize+1))
            else:
                axs_plot[i].tick_params(axis='x', which='major', direction='in', bottom=True, top=False, labelbottom=False)

            axs_plot[i].set_xticks(tickpos)

            if (plotArgs['filtering'][accelerator]['reportMode']):
                axs_plot[i].set_xticklabels(tickLabels)
                if (i == 3):
                    axs_plot[i].set_xlabel(tick_id, **dict(fontsize=fontScale*(fontBaseSize+1.5)))

            if(i==3):
                ylabel = 'Deviation [mrad]'
            else:
                ylabel = 'Deviation [mm]'
                if (accelerator == 'SR'):
                    axs_plot[i].set_yticks([-0.1, -0.05, 0, 0.05, 0.1])
                    # axs_plot[i].set_yticklabels([-0.1, -0.5, 0, 0.5, 0.1])
            
            axs_plot[i].grid(b=True, axis='both', which='major', linestyle='--', alpha=0.5)

            axs_plot[i].set_ylabel(ylabel, **dict(fontsize=fontScale*(fontBaseSize+1.5)))
            axs_plot[i].xaxis.labelpad = 10

            # axs_plot[i].set_ylim(-1.42*tolerances[i], 1.42*tolerances[i])
            axs_plot[i].set_xlim(-x_extension, plotData.iloc[:, 0].size)

            bbox_props = dict(boxstyle="square,pad=0.1", fc="white", lw=0)


            axs_plot[i].set_title(plot_titles[i], y=0.92, bbox=bbox_props, **dict(fontsize=fontScale*(fontBaseSize+2), weight=600))

            # CALCULATING STANDARD DEVIATION
            stdDev = np.std(plotData.iloc[:,i+1])
            # stdDevList.append(stdDev)
            avg = np.mean(np.absolute(plotData.iloc[:,i+1]))
            if (i != 3):
                boxTxt = 'Mean Deviation: {:.2f} mm'.format(avg)
            else:
                boxTxt = 'Mean Deviation: {:.2f} mrad'.format(avg)

            # translate_bbox = Affine2D().translate(0,4)

            ob = offsetbox.AnchoredText(boxTxt, loc=1, pad=0.25, borderpad=0.3, prop=dict(fontsize=fontScale*(fontBaseSize+0.7)))
            axs_plot[i].add_artist(ob)

            # Plotting histogram
            axs_hist[i].hist(y[i], bins[i], facecolor=plot_colors[i], ec='white')
            axs_hist[i].set_xticks([-tolerances[i], -stdDev, stdDev, tolerances[i]])
            
            axs_hist[i].tick_params(axis='x', which='major', direction='in', bottom=True, top=False, labelrotation=45,labelsize=fontScale*fontBaseSize)
            axs_hist[i].tick_params(axis='y', labelsize=fontScale*fontBaseSize)

            xlim_init, xlim_end = axs_hist[i].get_xlim()
            axs_hist[i].set_xlim(-1.5*tolerances[i], 1.5*tolerances[i])

            axs_hist[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axs_hist[i].yaxis.set_major_formatter(FormatStrFormatter('%d'))

            # axs_hist[i].axvline(tolerances[i], color=tolerance_color, lw=1.6, alpha=0.6)
            # axs_hist[i].axvline(-tolerances[i], color=tolerance_color, lw=1.6, alpha=0.6)
            ylimits_hist = axs_hist[i].get_ylim()
            x_filled = np.linspace(-tolerances[i], tolerances[i], 10)
            axs_hist[i].fill_between(x_filled, 0, 2*ylimits_hist[1], color='green', alpha=0.06)
            axs_hist[i].set_ylim(ylimits_hist)

            if (i >= 2):
                axs_hist[i].set_xlabel(ylabel, **dict(fontsize=fontScale*(fontBaseSize+1.5)))

            axs_hist[i].set_ylabel('Occurrence', **dict(fontsize=fontScale*(fontBaseSize+1.5)))

            x_filled = np.linspace(-5, len(x)+5, 10)
            axs_plot[i].fill_between(x_filled, tolerances[i], -tolerances[i], color='green', alpha=0.06)

        # plot dos histogramas p/ reunião sobre intervenção [temporário]
        
        # if (accelerator == 'SR'):
        #     bins = [6,6,12,10]
        #     title = 'Destribuição de Desvios - Anel de Armazenamento'
        # else:
        #     bins = [10, 10, 6, 6]
        #     title = 'Destribuição de Desvios - Booster'

        # fontBaseSize = 7
        # fontScale = 1.5

        # fig2, axs2 = plt.subplots(2,2, figsize=(12,8))
        # for i in range(4):
        #     axs2[i%2][int(i/2)].hist(y[i], bins[i], facecolor=plot_colors[i], ec='white')
        #     axs2[i%2][int(i/2)].set_xticks([-tolerances[i], -stdDevList[i], stdDevList[i], tolerances[i]])
        #     axs2[i%2][int(i/2)].axvline(tolerances[i], color=tolerance_color, lw=1.6, alpha=0.6)
        #     axs2[i%2][int(i/2)].axvline(-tolerances[i], color=tolerance_color, lw=1.6, alpha=0.6)
        #     axs2[i%2][int(i/2)].tick_params(axis='x', which='major', direction='in', bottom=True, top=False, labelrotation=45,labelsize=fontScale*fontBaseSize)
        #     axs2[i%2][int(i/2)].tick_params(axis='y', labelsize=fontScale*fontBaseSize)
        #     axs2[i%2][int(i/2)].set_title(plot_titles[i], **dict(fontsize=fontScale*(fontBaseSize+2), weight='semibold'))

        #     xlim_init, xlim_end = axs2[i%2][int(i/2)].get_xlim()
        #     axs2[i%2][int(i/2)].set_xlim(xlim_init - tolerances[i]/5.5, xlim_end + tolerances[i]/5.5)

        #     axs2[i%2][int(i/2)].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #     axs2[i%2][int(i/2)].yaxis.set_major_formatter(FormatStrFormatter('%d'))

        #     mean = np.mean(plotData.iloc[:,i+1])
        #     meanDev = np.mean(np.abs(plotData.iloc[:,i+1]))

        #     if (i != 3):
        #         xLabel = 'Desvio [mm]'
        #         boxTxt = 'Média: {:.2f} mm\nDesvio médio: {:.2f} mm'.format(mean, meanDev)
        #         boxTxt = 'Desvio médio: {:.2f} mm'.format(meanDev)
        #     else:
        #         xLabel = 'Desvio [mrad]'
        #         boxTxt = 'Média: {:.2f} mrad\nDesvio médio: {:.2f} mrad'.format(mean, meanDev)
        #         boxTxt = 'Desvio médio: {:.2f} mrad'.format(meanDev)

        #     axs2[i%2][int(i/2)].set_xlabel(xLabel, **dict(fontsize=fontScale*(fontBaseSize+1.5)))
        #     axs2[i%2][int(i/2)].set_ylabel('Ocorrência', **dict(fontsize=fontScale*(fontBaseSize+1.5)))

        #     ob = offsetbox.AnchoredText(boxTxt, loc=1, prop=dict(fontsize=fontScale*(fontBaseSize-0.5)))
        #     axs2[i%2][int(i/2)].add_artist(ob)

        # fig2.suptitle(title, fontsize=fontScale*(fontBaseSize*2), weight='bold')
        # fig2.subplots_adjust(wspace=0.3, hspace=0.45)

        plt.get_current_fig_manager().window.showMaximized()

        # acrescentando intervalo entre plots
        plt.draw()
        plt.pause(0.001)

        # condicional para definir se a figura irá congelar o andamento do app
        # usado no ultimo plot, pois senão as janelas fecharão assim que o app acabar de executar
        plt.ioff()

        # evocando a tela com a figura
        plt.show()

    @staticmethod
    def plotComparativeDeviation(deviations, **plotArgs):  # precisa adaptação pra booster
        # pegando uma cópia do df original
        plotData = deviations.copy()

        accelerator = plotArgs['accelerator']
        lenghts = plotArgs['len']

        # applying filters
        for magnet, dof in plotData.iterrows():
            if('B03-QUAD01' in magnet or 'B11-QUAD01' in magnet or ('QUAD' in magnet and not 'LONG' in magnet)):
                plotData.at[magnet, 'Tz'] = float('nan')
            elif('B1-LONG' in magnet):
                plotData.at[magnet, 'Tx'] = float('nan')
                plotData.at[magnet, 'Ty'] = float('nan')
                plotData.at[magnet, 'Rx'] = float('nan')
                plotData.at[magnet, 'Ry'] = float('nan')
                plotData.at[magnet, 'Rz'] = float('nan')

        for magnet, dof in plotData.iterrows():
            if ('P' in magnet and 'QUAD02' in magnet):
                plotData.at[magnet, :] = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]

        plotTitle = 'Global Alignment Profile'
        tolerances = [0.16, 0.16, 0.4, 0.8]
        tick_spacing = 17
        x_extension = 2
        unc = [0.018, 0.043, 0.023, 0.026]

        plot_args = {'y_list': ['Tx', 'Ty', 'Tz', 'Rz'], 'title_list': ['HORIZONTAL', 'VERTICAL', 'LONGITUDINAL', 'ROLL'], 'fig_title': plotTitle}

        xAxisTitle = 'Magnet'

        colum1_name = 'Tx'
        df_colums_dict = {
            'Tx': 'deviation horiz. [mm]', 'Ty': 'deviation vert. [mm]', 'Tz': 'deviation long. [mm]', 'Rz': 'rotation roll [mrad]'}

        """ configurando df """
        new_index = np.linspace(0, plotData[colum1_name].size-1, plotData[colum1_name].size)

        magnetNameList = plotData.index.to_numpy()

        # EXCLUINDO COLUNAS COM DOFS DESNECESSÁRIOS
        plotData = plotData.drop(columns=['Rx', 'Ry'])

        plotData['Index'] = new_index
        plotData.insert(0, xAxisTitle, plotData.index)
        plotData.set_index('Index', drop=True, inplace=True)
        plotData.reset_index(drop=True, inplace=True)
        plotData.rename(columns=df_colums_dict, inplace=True)

        fig = plt.figure()
        fig.tight_layout()

        gs_dev = plt.GridSpec(4,1, top=0.9, hspace=0.3)
        
        axs_plot = []
        axs_plot.append(fig.add_subplot(gs_dev[0]))
        axs_plot.append(fig.add_subplot(gs_dev[1], sharex=axs_plot[0]))
        axs_plot.append(fig.add_subplot(gs_dev[2], sharex=axs_plot[0]))
        axs_plot.append(fig.add_subplot(gs_dev[3], sharex=axs_plot[0]))

        fontBaseSize = 7
        fontScale = 1.2

        # titulo da figura
        bbox_props = dict(boxstyle="square,pad=0.5", fc="white", lw=0.5)
        fig.suptitle(plot_args['fig_title'], y=0.97, color='black', weight='semibold', fontsize=fontScale*(fontBaseSize+6), bbox=bbox_props)


        tickpos = np.linspace(0, plotData.iloc[:, 0].size, int(plotData.iloc[:, 0].size/tick_spacing))

        tickLabels = []
        [tickLabels.append(f'{str(i+1)}') for i in range(0, len(tickpos)-1)]


        """salvando args de entrada"""
        plot_colors = ['royalblue', 'darkorange', 'seagreen', 'black']
        
        
        # a lista de títulos é direta
        plot_titles = plot_args['title_list']

        # para a lista de colunas do df a serem plotadas, deve-se mapear a lista 'y_list' de entrada
        # em relação ao dict 'df_colums_dict' para estar em conformidade com os novos nomes das colunas
        y_list = []
        for y in plot_args['y_list']:
            if y in df_colums_dict:
                y_list.append(df_colums_dict[y])

        x = magnetNameList
        y = []

        for i in range(4):
            y.append(plotData[y_list[i]].to_numpy()) 

        for i in range(len(y)):
            if (plotArgs['filtering'][accelerator]['errorbar']):
                axs_plot[i].errorbar(x, y[i], yerr=unc[i], fmt='o', marker='o', ms=2, mec=plot_colors[i], mfc=plot_colors[i], color='wheat', ecolor='k', elinewidth=0.5, capthick=0.5, capsize=3)
            else:
                axs_plot[i].plot(x, y[i], 'o', color=plot_colors[i], ms=2)

            axs_plot[i].set_xticks(tickpos)
            axs_plot[i].set_xticklabels(tickLabels)
                    
            if(i==3):
                ylabel = 'Deviation [mrad]'
            else:
                ylabel = 'Deviation [mm]'

            axs_plot[i].tick_params(axis='x', which='major', direction='in', bottom=True, top=False, labelbottom=False)

            
            axs_plot[i].grid(b=True, axis='both', which='major', linestyle='--', alpha=0.5)

            axs_plot[i].set_ylabel(ylabel, **dict(fontsize=fontScale*(fontBaseSize+1.5)))
            axs_plot[i].xaxis.labelpad = 10

            axs_plot[i].set_ylim(-1.42*tolerances[i], 1.42*tolerances[i])
            axs_plot[i].set_xlim(-x_extension, plotData.iloc[:, 0].size)

            bbox_props = dict(boxstyle="square,pad=0.1", fc="white", lw=0)
            axs_plot[i].set_title(plot_titles[i], y=0.92, bbox=bbox_props, **dict(fontsize=fontScale*(fontBaseSize+2), weight=600))

            axs_plot[i].axvline(x=lenghts[0], color='firebrick', lw='1')
            axs_plot[i].axvline(x=lenghts[1], color='firebrick', lw='1')
            axs_plot[i].axvline(x=lenghts[2], color='firebrick', lw='1')

        axs_plot[0].text(lenghts[0]/2 - 5, 0.16, 'LTB', weight=500, bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.7))
        axs_plot[0].text(lenghts[0] + (lenghts[1]-lenghts[0])/2 - 8, 0.16, 'Booster', weight=500, bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.7))
        axs_plot[0].text(lenghts[1] + (lenghts[2]-lenghts[1])/2 - 5, 0.16, 'BTS', weight=500, bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.7))
        axs_plot[0].text(lenghts[2] + (len(x)-lenghts[2])/2 - 10, 0.16, 'Storage Ring', weight=500, bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.7))

        
        # axs_plot[0].annotate('Booster', xy=(1/8, 0.90), xytext=(1/8, 0.9), xycoords='axes fraction',
        #     fontsize=fontScale*(fontBaseSize+2), ha='center', va='bottom',
        #     bbox=dict(boxstyle='square', fc='white'),
        #     arrowprops=dict(arrowstyle='-[, widthB=14, lengthB=1.5', lw=1.0))

        # axs_plot[0].annotate('Storage Ring', xy=(5/8, 0.90), xytext=(5/8, 0.9), xycoords='axes fraction', 
        #     fontsize=fontScale*(fontBaseSize+2), ha='center', va='bottom',
        #     bbox=dict(boxstyle='square', fc='white'),
        #     arrowprops=dict(arrowstyle='-[, widthB=14.0, lengthB=1.5', lw=1.0))


        plt.get_current_fig_manager().window.showMaximized()

        # acrescentando intervalo entre plots
        plt.draw()
        plt.pause(0.001)

        # condicional para definir se a figura irá congelar o andamento do app
        # usado no ultimo plot, pois senão as janelas fecharão assim que o app acabar de executar
        plt.ioff()

        # evocando a tela com a figura
        plt.show()

    @staticmethod
    def plotRelativeDevitationData(deviations, accelerator):  # precisa adaptação pra booster
        # pegando uma cópia do df original
        plotData = deviations.copy()

        if(isDofsFiltered):
            for magnet, dof in plotData.iterrows():
                if('B03-QUAD01' in magnet or 'B11-QUAD01' in magnet or ('QUAD' in magnet and not 'LONG' in magnet)):
                    plotData.at[magnet, 'Tz'] = float('nan')
                elif('B1-LONG' in magnet):
                    plotData.at[magnet, 'Tx'] = float('nan')
                    plotData.at[magnet, 'Ty'] = float('nan')
                    plotData.at[magnet, 'Rx'] = float('nan')
                    plotData.at[magnet, 'Ry'] = float('nan')
                    plotData.at[magnet, 'Rz'] = float('nan')


        if (accelerator == 'SR'):
            plotTitle = 'Global Alignment Profile - Storage Ring'
            tolerances = [0.02, 0.02, 0.05, 0.15, 0.15, 0.2]
            verticalSpacing = 0.04
        elif (accelerator == 'booster'):
            plotTitle = 'Global Alignment Profile - Booster'
            tolerances = [0.08, 0.08, 0.2, 0.3, 0.3, 0.4]
            verticalSpacing = 0.1
        elif (accelerator == 'LTB'):
            plotTitle = 'Global Alignment Profile - LTB'
        elif (accelerator == 'BTS'):
            plotTitle = 'Global Alignment Profile - BTS'

        plot_args = {'y_list': ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz'], 'title_list': [
            'Transversal', 'Vertical', 'Longitudinal', 'Pitch', 'Yaw', 'Roll'], 'fig_title': plotTitle}

        xAxisTitle = 'Magnet'

        colum1_name = 'Tx'
        df_colums_dict = {
            'Tx': 'deviation transv. [mm]', 'Ty': 'deviation vert. [mm]', 'Tz': 'deviation long. [mm]', 'Rx': 'rotation pitch [mrad]', 'Ry': 'rotation yaw [mrad]', 'Rz': 'rotation roll [mrad]'}

        xAxisTicks = []
        for i in range(0, plotData.iloc[:, 0].size):
            sector = int(i/17) + 1
            xAxisTicks.append(str(sector))

        """ configurando df """
        new_index = np.linspace(
            0, plotData[colum1_name].size-1, plotData[colum1_name].size)

        magnetNameList = plotData.index.to_numpy()

        # print(magnetNameList)

        plotData['Index'] = new_index
        plotData.insert(0, xAxisTitle, plotData.index)
        plotData.set_index('Index', drop=True, inplace=True)
        plotData.reset_index(drop=True, inplace=True)
        plotData.rename(columns=df_colums_dict, inplace=True)

        # invertendo sinal das translações transversais
        # plotData["d_transv (mm)"] = -plotData["d_transv (mm)"]

        """ configurando parâmetros do plot """
        # definindo o numero de linhas e colunas do layout do plot
        grid_subplot = [3, 2]
        # definindo se as absicissas serão compartilhadas ou não
        share_xAxis = 'col'
        # criando subplots com os parâmetros gerados acima
        fig, axs = plt.subplots(
            grid_subplot[0], grid_subplot[1], figsize=(18, 9), sharex=share_xAxis)

        plt.subplots_adjust(hspace=0.3)

        tickpos = np.linspace(
            0, plotData.iloc[:, 0].size, int(plotData.iloc[:, 0].size/17))

        tickLabels = []
        
        # [tickLabels.append(magnetNameList[i].split('-')[0]) for i in range(0, len(tickpos))]
        [tickLabels.append('test'+str(i)) for i in range(0, len(tickpos))]

        """salvando args de entrada"""
        plot_colors = ['red', 'limegreen', 'blue', 'purple', 'green', 'black']
        # a lista de títulos é direta
        plot_titles = plot_args['title_list']

        # titulo da figura
        fig.suptitle(plot_args['fig_title'], fontsize=16)

        # para a lista de colunas do df a serem plotadas, deve-se mapear a lista 'y_list' de entrada
        # em relação ao dict 'df_colums_dict' para estar em conformidade com os novos nomes das colunas
        y_list = []
        for y in plot_args['y_list']:
            if y in df_colums_dict:
                y_list.append(df_colums_dict[y])

        x = magnetNameList
        y = []

        for i in range(6):
            y.append(plotData[y_list[i]].to_numpy()) 

        for i in range(len(y)):
            k=0
            while (k < len(x)):
                currGirder = x[k].split('-')[0]+'-'+x[k].split('-')[1]
                try:
                    nextGirder = x[k+1].split('-')[0]+'-'+x[k+1].split('-')[1]
                except IndexError:
                    nextGirder = None
                if(currGirder == nextGirder):
                    axs[i % 3][int(i/3)].plot(x[k: k+2], y[i][k: k+2], 'o-', color=plot_colors[i])
                    k+=2
                else:
                    axs[i % 3][int(i/3)].plot(x[k], y[i][k], 'o-', color=plot_colors[i])
                    k+=1
            # for j in range(0, len(x), 2):
            #     axs[i % 3][int(i/3)].plot(x[j: j+2], y[i][j: j+2], 'o-', color=plot_colors[i])

            axs[i % 3][int(i/3)].tick_params(axis='x', which='major', direction='in', bottom=True, top=True, labelrotation=45,
                                             labelsize=5)
            axs[i % 3][int(i/3)].set_xticks(tickpos)
            axs[i % 3][int(i/3)].set_xticklabels(tickLabels)
            axs[i % 3][int(i/3)].xaxis.labelpad = 10
            axs[i % 3][int(i/3)].grid(b=True, axis='both',
                                      which='major', linestyle='--', alpha=0.5)

            axs[i % 3][int(i/3)].axhline(tolerances[i], color='green', lw=1.5, alpha=0.4)
            axs[i % 3][int(i/3)].axhline(-tolerances[i], color='green', lw=1.5, alpha=0.4)

            axs[i % 3][int(i/3)].axhline(tolerances[i]*2, color='yellow', lw=1.5, alpha=0.4)
            axs[i % 3][int(i/3)].axhline(-tolerances[i]*2, color='yellow', lw=1.5, alpha=0.4)

            ylim_bottom, ylim_top = axs[i % 3][int(i/3)].get_ylim()
            axs[i % 3][int(i/3)].set_ylim(ylim_bottom - verticalSpacing, ylim_top + verticalSpacing)

            axs[i % 3][int(i/3)].set_xlim(0, plotData.iloc[:, 0].size)

            axs[i % 3][int(i/3)].set_title(plot_titles[i])

            axs[i % 3][int(i/3)].fill_between(plotData.iloc[:, 0], tolerances[i], -tolerances[i], color='green', alpha=0.2)
            axs[i % 3][int(i/3)].fill_between(plotData.iloc[:, 0], 2*tolerances[i], tolerances[i], color='yellow', alpha=0.2)
            axs[i % 3][int(i/3)].fill_between(plotData.iloc[:, 0], -tolerances[i], -2*tolerances[i], color='yellow', alpha=0.2)

        # mostrando plots
        # plt.minorticks_off()

        # acrescentando intervalo entre plots
        plt.draw()
        plt.pause(0.001)

        # condicional para definir se a figura irá congelar o andamento do app
        # usado no ultimo plot, pois senão as janelas fecharão assim que o app acabar de executar
        plt.ioff()

        # evocando a tela com a figura
        plt.show()

    @staticmethod
    def transformToLocalFrame(objectDict, frameDict, accelerator):
        magnetsNotComputed = ""
        for objectName in objectDict:
            magnetDict = objectDict[objectName]
                
            for magnetName in magnetDict:
                magnet = magnetDict[magnetName]
                targetFrame = magnetName + "-NOMINAL"
                try:
                    magnet.transformFrame(frameDict, targetFrame)
                except KeyError:
                    magnetsNotComputed += magnet.name + ','

        # at least one frame was not computed
        if (magnetsNotComputed != ""):
            print("[Transformação p/ frames locais] Imãs não computados: " +
                  magnetsNotComputed[:-1])
            pass

    @staticmethod
    def debug_transformToLocalFrame(objectDict, frameDict, accelerator):
        magnetsNotComputed = ""
        for objectName in objectDict:
            magnetDict = objectDict[objectName]
                
            for magnetName in magnetDict:
                magnet = magnetDict[magnetName]

                separetedName = magnetName.split('-')
                sectorId = separetedName[0]
                girderId = separetedName[1]

                if (girderId != 'B05' and girderId != 'B09' and girderId != 'B07'):
                    targetFrame = f"{sectorId}-{girderId}-QUAD01-NOMINAL"
                else:
                    targetFrame = magnetName + "-NOMINAL"

                try:
                    magnet.transformFrame(frameDict, targetFrame)
                except KeyError:
                    magnetsNotComputed += magnet.name + ','

        # at least one frame was not computed
        if (magnetsNotComputed != ""):
            print("[Transformação p/ frames locais] Imãs não computados: " +
                  magnetsNotComputed[:-1])
            pass

    @staticmethod
    def transformToMachineLocal(girderDict, frameDict):
        for girderName in girderDict:
            magnetDict = girderDict[girderName]
            for magnetName in magnetDict:
                magnet = magnetDict[magnetName]
                magnet.transformFrame(frameDict, "machine-local")

    @staticmethod
    def debug_translatePoints(girderDict, frameDict):
        output = ""

        for girderName in girderDict:
            magnetDict = girderDict[girderName]
            for magnetName in magnetDict:
                magnet = magnetDict[magnetName]
                pointDict = magnet.pointDict

                
                for pointName in pointDict:
                    point = pointDict[pointName]

                    cp_point = Point.copyFromPoint(point)

                    sectorID = pointName.split('-')[0]
                    girderID = pointName.split('-')[1]
                    magnetID = pointName.split('-')[2]

                    # print(f"{cp_point.name}: [{cp_point.x}, {cp_point.y}, {cp_point.z}]")

                    # vertical translating
                    if ('QUAD' in magnetID):
                        cp_point.y -= 0.0125
                    elif ('SEXT' in magnetID):
                        cp_point.y += 0.017 + 0.0125
                    elif ('B1' in magnetID):
                        cp_point.y += 0.0125

                    # horizontal translating
                    if ('QUAD' in magnetID):
                        if ('C02' in cp_point.name or 'C03' in cp_point.name):
                            cp_point.x -= 0.0125
                        elif ('C01' in cp_point.name or 'C04' in cp_point.name):
                            cp_point.x += 0.0125
                    elif ('SEXT' in magnetID):
                        if ('C02' in cp_point.name or 'C03' in cp_point.name):
                            cp_point.x += 0.017 + 0.0125
                        elif ('C01' in cp_point.name or 'C04' in cp_point.name):
                            cp_point.x += 0.0125
                    elif ('B1' in magnetID):
                        cp_point.x += 0.0125

                    # applying specific shifts to B1's
                    shiftsB1 = {"S01-B03-B1": -0.21939260387942738,"S01-B11-B1": 0.5027928637375751,"S02-B03-B1": -0.6566497181853421,"S02-B11-B1": 0.3949965569748386,"S03-B03-B1": -0.20433956473073067,"S03-B11-B1": 0.43980701894961527,"S04-B03-B1": -0.24083142212426623,"S04-B11-B1": 0.044734592439588994,"S05-B03-B1": -0.5419523768496219,"S05-B11-B1": 0.18519311704547903,"S06-B03-B1": 0.06556785208046989,"S06-B11-B1": 0.2463624895503429,"S07-B03-B1": -0.11493942111696498,"S07-B11-B1": 0.1979572509557599,"S08-B03-B1": -0.19108205778576348,"S08-B11-B1": 0.10247298117068482,"S09-B03-B1": -0.12550137421514052,"S09-B11-B1": 0.06038905678307316,"S10-B03-B1": 0.08284427370889347,"S10-B11-B1": 0.4413268321516668,"S11-B03-B1": -0.08184888494565712,"S11-B11-B1": 0.08674365614044177,"S12-B03-B1": -0.3405172535192946,"S12-B11-B1": -0.2162778490154338,"S13-B03-B1": -0.20894238262729203,"S13-B11-B1": 0.007992350452042274,"S14-B03-B1": -0.44218076120701255,"S14-B11-B1": 0.19238108862685266,"S15-B03-B1": -0.14013324602614574,"S15-B11-B1": 0.16677316354694938,"S16-B03-B1": -0.8252640711741677,"S16-B11-B1": -0.056585429443245516,"S17-B03-B1": -0.542567297776479,"S17-B11-B1": 0.1909879411927733,"S18-B03-B1": -0.1966650964553054,"S18-B11-B1": 0.15873723593284694,"S19-B03-B1": -0.4565826348706068,"S19-B11-B1": 0.2918019854017899,"S20-B03-B1": -0.4598210056558685,"S20-B11-B1": 0.5146069215769487} 

                    if ('B1' in magnetID):
                        # transforming from current frame to its own B1 frame
                        initialFrame = f"{sectorID}-{girderID}-QUAD01-NOMINAL"
                        targetFrame = f"{sectorID}-{girderID}-B1-NOMINAL"
                        transformation = Transformation.evaluateTransformation(frameDict, initialFrame, targetFrame)
                        cp_point.transform(transformation, targetFrame)

                        # applying shift
                        magnetName = f"{sectorID}-{girderID}-B1"
                        cp_point.x -= shiftsB1[magnetName]

                    # transforming back to machine-local
                    initialFrame = f"{sectorID}-{girderID}-{magnetID}-NOMINAL"
                    if (girderID != 'B05' and girderID != 'B09' and girderID != 'B07'):
                        if (magnetID != 'B1'):
                            initialFrame = f"{sectorID}-{girderID}-QUAD01-NOMINAL"

                    transformation = Transformation.evaluateTransformation(frameDict, initialFrame, 'machine-local')
                    cp_point.transform(transformation, 'machine-local')

                    # print(f"{cp_point.name}: [{cp_point.x}, {cp_point.y}, {cp_point.z}]")
                    
                    # acrescentando a tag de nomes dos imãs que definem a longitudinal
                    if ('B01-QUAD01' in cp_point.name):
                        cp_point.name += '-LONG'
                    elif ('B02' in cp_point.name):
                        if (sectorID == 'S01' or sectorID == 'S05' or sectorID == 'S09' or sectorID == 'S13' or sectorID == 'S17'):
                            if('QUAD01' in cp_point.name):
                                cp_point.name += '-LONG'
                        else:
                            if (len(pointName.split('-')) == 3):
                                if ('QUAD01' in cp_point.name):
                                    cp_point.name = f"{sectorID}-{girderID}-QUAD02-LONG"
                                elif ('QUAD02' in cp_point.name):
                                    cp_point.name = f"{sectorID}-{girderID}-QUAD01"
                            else:
                                if('QUAD02' in cp_point.name):
                                    cp_point.name += '-LONG'

                    elif ('B04-QUAD02' in cp_point.name):
                        cp_point.name += '-LONG'
                    elif ('B06-QUAD02' in cp_point.name):
                        cp_point.name += '-LONG'
                    elif ('B08-QUAD01' in cp_point.name):
                        cp_point.name += '-LONG'
                    elif ('B10-QUAD01' in cp_point.name):
                        cp_point.name += '-LONG'
                    elif ('B03' in girderID or 'B11' in girderID):
                        if ('B1' in magnetID):
                            cp_point.name += '-LONG'

                    newPoint = f"{cp_point.name}, {cp_point.x}, {cp_point.y}, {cp_point.z}\n"
                    output += newPoint

        # writing the modified list of points in output file 
        with open('../data/output/anel-novos_pontos.txt', 'w') as f:
            f.write(output)

    @staticmethod
    def printDictData(objectDict, mode='console', fileName='dataDict'):
        output = ""
        for objectName in objectDict:
            output += objectName + '\n'
            magnetDict = objectDict[objectName]
            for magnet in magnetDict:
                pointDict = magnetDict[magnet].pointDict
                # shift = magnetDict[magnet].shift
                output += '\t'+magnet+'\n'
                # output += '\t\tshift: '+str(shift)+'\n'
                output += '\t\tpointlist:\n'
                for point in pointDict:
                    pt = pointDict[point]
                    output += '\t\t\t' + pt.name + ' ' + \
                        str(pt.x) + ' ' + str(pt.y) + \
                        ' ' + str(pt.z) + '\n'

        if (mode == 'console'):
            print(output)
        else:
            with open(f'../data/output/{fileName}.txt', 'w') as f:
                f.write(output)

    @staticmethod
    def printFrameDict(frameDict):
        for frameName in frameDict:
            frame = frameDict[frameName]
            print(frame.name)
            print('\t(Tx, Ty, Tz, Rx, Ry, Rz): ('+str(frame.transformation.Tx)+', '+str(frame.transformation.Ty)+', ' +
                  str(frame.transformation.Tz)+', '+str(frame.transformation.Rx)+', '+str(frame.transformation.Ry)+', '+str(frame.transformation.Rz)+')')

    @staticmethod
    def calculateMagnetsDeviations(frameDict):
        header = ['Magnet', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        deviationList = []

        for frameName in frameDict:
            frame = frameDict[frameName]

            if ('MEASURED' in frame.name or frame.name == 'machine-local'):
                continue

            magnetName = ""
            splitedFrameName = frame.name.split('-')
            for i in range(len(splitedFrameName) - 1):
                magnetName += (splitedFrameName[i]+'-')

            frameFrom = magnetName + 'MEASURED'
            frameTo = magnetName + 'NOMINAL'

            try:
                transformation = Transformation.evaluateTransformation(frameDict, frameFrom, frameTo)
                dev = Transformation.individualDeviationsFromEquations(transformation)
            except KeyError:
                dev = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]
                pass

            dev.insert(0, magnetName[:-1])

            deviation = pd.DataFrame([dev], columns=header)
            deviationList.append(deviation)

        deviations = pd.concat(deviationList, ignore_index=True)
        deviations = deviations.set_index(header[0])

        # retorna o df com os termos no tipo numérico certo
        return deviations.astype('float32')

    @staticmethod
    def calculateMagnetsRelativeDeviations(deviationDF):
        relDev = deviationDF.copy()
        relDevList = []

        k = 0

        while k < len(relDev.iloc[:,0]):
            currGirder = relDev[k].split('-')[0] + '-' + relDev[k].split('-')[1]

            try:
                nextGirder = relDev[k+1].split('-')[0] + '-' + relDev[k+1].split('-')[1]
            except IndexError:
                nextGirder = None
            
            if(currGirder == nextGirder):
                k += 2
                continue
            elif (nextGirder):
                diff = [(relDev.at[k+1, 'Tx'] - relDev.at[k, 'Tx']), (relDev.at[k+1, 'Ty'] - relDev.at[k, 'Ty'])]
                deviation = pd.DataFrame([diff], columns=['Magnet', 'Tx', 'Ty'])
                relDevList.append(deviation)
            
            k += 1

        deviations = pd.concat(relDevList, ignore_index=True)
        deviations = deviations.set_index('Magnet')

        # retorna o df com os termos no tipo numérico certo
        return deviations.astype('float32')

    @staticmethod
    def calculateMagnetsDeviations2(frameDict):
        header = ['Magnet', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        deviationList = []

        for frameName in frameDict:
            frame = frameDict[frameName]

            if ('NOMINAL' in frame.name or frame.name == 'machine-local'):
                continue

            magnetName = ""
            splitedFrameName = frame.name.split('-')
            for i in range(len(splitedFrameName) - 1):
                magnetName += (splitedFrameName[i]+'-')

            frameFrom = magnetName + 'NOMINAL'
            frameTo = frame.name

            try:
                transformation = Transformation.evaluateTransformation(
                    frameDict, frameFrom, frameTo)
            except KeyError:
                print("[Cálculo dos desvios] Desvio do imã " +
                      magnetName[:-1] + "não calculado.")
                continue

            dev = Transformation.individualDeviationsFromEquations(
                transformation)

            dev.insert(0, magnetName[:-1])

            deviation = pd.DataFrame([dev], columns=header)
            deviationList.append(deviation)

        deviations = pd.concat(deviationList, ignore_index=True)
        deviations = deviations.set_index(header[0])

        # retorna o df com os termos no tipo numérico certo
        return deviations.astype('float32')

    @staticmethod
    # FALTA ADAPTAÇÃO PARA O BOOSTER
    def sortFrameDictByBeamTrajectory(frameDict, accelerator="SR"):
        # sorting in alphabetical order
        keySortedList = sorted(frameDict)
        sortedFrameDict = {}
        for key in keySortedList:
            sortedFrameDict[key] = frameDict[key]

        if(accelerator == 'SR'):
            finalFrameDict = {}
            b1FrameList = []
            # correcting the case of S0xB03
            for frameName in sortedFrameDict:
                frame = sortedFrameDict[frameName]

                if ('B03-B1' in frame.name):
                    b1FrameList.append(frame)
                    continue

                finalFrameDict[frame.name] = frame

                if ('B03-QUAD01-NOMINAL' in frame.name):
                    for b1Frame in b1FrameList:
                        finalFrameDict[b1Frame.name] = b1Frame
                    b1FrameList = []
        else:
            # incluir exceções do booster aqui
            finalFrameDict = sortedFrameDict

        return finalFrameDict

    @staticmethod
    def calculateEuclidianDistance(params, *args):

        x0 = args[0]
        x_ref = args[1]
        dofs = args[2].copy()
        dofs_backup = args[2]

        x0 = np.array(x0)
        x_ref = np.array(x_ref)

        (Tx, Ty, Tz, Rx, Ry, Rz) = (0, 0, 0, 0, 0, 0)
        # ** assume-se que os parametros estão ordenados
        for param in params:
            if 'Tx' in dofs:
                Tx = param
                dofs.pop(dofs.index('Tx'))
            elif 'Ty' in dofs:
                Ty = param
                dofs.pop(dofs.index('Ty'))
            elif 'Tz' in dofs:
                Tz = param
                dofs.pop(dofs.index('Tz'))
            elif 'Rx' in dofs:
                Rx = param
                dofs.pop(dofs.index('Rx'))
            elif 'Ry' in dofs:
                Ry = param
                dofs.pop(dofs.index('Ry'))
            elif 'Rz' in dofs:
                Rz = param
                dofs.pop(dofs.index('Rz'))

        # inicializando variável para cálculo do(s) valor a ser minimizado
        diff = []

        for i in range(np.shape(x0)[0]):

            rot_z = np.array([[np.cos(Rz*10**-3), -np.sin(Rz*10**-3), 0],
                              [np.sin(Rz*10**-3), np.cos(Rz*10**-3), 0], [0, 0, 1]])
            rot_y = np.array([[np.cos(Ry*10**-3), 0, np.sin(Ry*10**-3)],
                              [0, 1, 0], [-np.sin(Ry*10**-3), 0, np.cos(Ry*10**-3)]])
            rot_x = np.array([[1, 0, 0], [0, np.cos(
                Rx*10**-3), -np.sin(Rx*10**-3)], [0, np.sin(Rx*10**-3), np.cos(Rx*10**-3)]])
            ROT = rot_z @ rot_y @ rot_x
            xr = np.dot(ROT, x0[i])

            xt = xr + np.array([Tx, Ty, Tz])

            if 'Tx' in dofs_backup:
                diff.append(((x_ref[i, 0]-xt[0])**2).sum())
            if 'Ty' in dofs_backup:
                diff.append(((x_ref[i, 1]-xt[1])**2).sum())
            if 'Tz' in dofs_backup:
                diff.append(((x_ref[i, 2]-xt[2])**2).sum())

        return np.sqrt(np.sum(diff))

    @staticmethod
    def evaluateDeviation(ptsMeas, ptsRef, dofs):

        # inicializando array com parâmetros a serem manipulados durante as iterações da minimização
        params = np.zeros(len(dofs))

        # aplicando a operação de minimização para achar os parâmetros de transformação
        deviation = minimize(fun=DataUtils.calculateEuclidianDistance, x0=params, args=(ptsMeas, ptsRef, dofs),\
                             method='SLSQP', options={'ftol': 1e-10, 'disp': False})['x']

        # invertendo o sinal do resultado p/ adequação
        deviation = [dof*(-1) for dof in deviation]

        return deviation

    @staticmethod
    def generateMeasuredFrames(objectDictMeas, objectDictNom, frameDict, accelerator, ui, typeOfMagnets='', isTypeOfMagnetsIgnored=True):
        for objectName in objectDictMeas:
            magnetDict = objectDictMeas[objectName]
            for magnetName in magnetDict:
                magnet = magnetDict[magnetName]
                pointDict = magnet.pointDict

                if (magnet.type == typeOfMagnets or isTypeOfMagnetsIgnored):
                    if (accelerator == 'SR'):
                        # creating lists for measured and nominal points
                        try:
                            pointList = SR.appendPoints(pointDict, objectDictNom, magnet, objectName)
                        except KeyError:
                            ui.logMessage('Falha no imã ' + magnet.name +': o nome de 1 ou mais pontos nominais não correspondem aos medidos.', severity="danger")
                            continue
                        # calculating the transformation from the nominal points to the measured ones
                        try:
                            localTransformation = SR.calculateLocalDeviationsByTypeOfMagnet(magnet, objectName, pointDict, frameDict, pointList)
                        except KeyError:
                            ui.logMessage('Falha no imã ' + magnet.name +': frame medido do quadrupolo adjacente ainda não foi calculado.', severity='danger')
                            continue
                    elif (accelerator == 'booster'):
                        try:
                            pointList = Booster.appendPoints(pointDict, objectDictNom, magnet, objectName)
                            localTransformation = Booster.calculateLocalDeviations(magnet, pointList)
                        except KeyError:
                            ui.logMessage('Falha no imã ' + magnet.name +': o nome de 1 ou mais pontos nominais não correspondem aos medidos.', severity='danger')
                            continue
                    elif (accelerator == 'LTB'):
                        try:
                            pointList = LTB.appendPoints(pointDict, objectDictNom, magnet, objectName)
                            localTransformation = LTB.calculateLocalDeviations(magnet, pointList)
                        except KeyError:
                            ui.logMessage('Falha no imã ' + magnet.name +': o nome de 1 ou mais pontos nominais não correspondem aos medidos.', severity='danger')
                            continue
                    elif (accelerator == 'BTS'):
                        try:
                            pointList = BTS.appendPoints(pointDict, objectDictNom, magnet, objectName)
                            localTransformation = BTS.calculateLocalDeviations(magnet, pointList)
                        except KeyError:
                            ui.logMessage('Falha no imã ' + magnet.name +': o nome de 1 ou mais pontos nominais não correspondem aos medidos.', severity='danger')
                            continue
                    elif (accelerator == 'FE'):
                        try:
                            pointList = FE.appendPoints(pointDict, objectDictNom, magnet, objectName)
                            localTransformation = FE.calculateLocalDeviations(magnet, pointList)
                        except KeyError:
                            ui.logMessage('Falha no imã ' + magnet.name +': o nome de 1 ou mais pontos nominais não correspondem aos medidos.', severity='danger')
                            continue
                    try:
                        # referencing the transformation from the frame of nominal magnet to the machine-local
                        baseTransformation = frameDict[localTransformation.frameFrom].transformation
                    except KeyError:
                        ui.logMessage('Falha no imã ' + magnet.name + ': frame nominal não foi computado; checar tabela com transformações.', severity='danger')
                        continue

                    # calculating the homogeneous matrix of the transformation from the frame of the measured magnet to the machine-local
                    transfMatrix = baseTransformation.transfMatrix @ localTransformation.transfMatrix

                    # updating the homogeneous matrix and frameFrom of the already created Transformation
                    transformation = localTransformation
                    transformation.transfMatrix = transfMatrix
                    transformation.frameFrom = 'machine-local'

                    transformation.Tx = transfMatrix[0, 3]
                    transformation.Ty = transfMatrix[1, 3]
                    transformation.Tz = transfMatrix[2, 3]
                    transformation.Rx = baseTransformation.Rx + localTransformation.Rx
                    transformation.Ry = baseTransformation.Ry + localTransformation.Ry
                    transformation.Rz = baseTransformation.Rz + localTransformation.Rz

                    # creating a new Frame with the transformation from the measured magnet to machine-local
                    measMagnetFrame = Frame(transformation.frameTo, transformation)

                    # adding it to the frame dictionary
                    frameDict[measMagnetFrame.name] = measMagnetFrame

    @staticmethod
    def calculateMagnetsDistances(frameDict, dataType):
        header = ['Reference', 'Distance (mm)']

        if (dataType == 'measured'):
            ignoreTag = 'NOMINAL'
            firstIndex = 0
            lastIndexShift = 3
        else:
            ignoreTag = 'MEASURED'
            firstIndex = 1
            lastIndexShift = 2

        distancesList = []
        frameDictKeys = list(frameDict.keys())
        for (index, frameName) in enumerate(frameDict):
            frame = frameDict[frameName]

            if (ignoreTag in frame.name or 'machine-local' in frame.name):
                continue

            if (index == firstIndex):
                firstMagnetsFrame = frame

            if (index == len(frameDict) - lastIndexShift):
                # first frame of the dictionary
                magnet2frame = firstMagnetsFrame
            else:
                # next measured frame
                magnet2frame = frameDict[frameDictKeys[index+2]]

            magnet1frame = frame

            magnet1coord = np.array([magnet1frame.transformation.Tx,
                                     magnet1frame.transformation.Ty, magnet1frame.transformation.Tz])
            magnet2coord = np.array([magnet2frame.transformation.Tx,
                                     magnet2frame.transformation.Ty, magnet2frame.transformation.Tz])

            vector = magnet2coord - magnet1coord
            distance = math.sqrt(vector[0]**2 + vector[1]**2)

            splitedName1 = magnet1frame.name.split('-')
            splitedName2 = magnet2frame.name.split('-')

            magnet1frameName = splitedName1[0] + '-' + \
                splitedName1[1] + '- ' + splitedName1[2]
            magnet2frameName = splitedName2[0] + '-' + \
                splitedName2[1] + '- ' + splitedName2[2]

            distanceReference = magnet1frameName + ' x ' + magnet2frameName

            distanceDF = pd.DataFrame(
                [[distanceReference, distance]], columns=header)
            distancesList.append(distanceDF)

        distances = pd.concat(distancesList, ignore_index=True)
        distances = distances.set_index(header[0])

        # retorna o df com os termos no tipo numérico certo
        return distances.astype('float32')

    @staticmethod
    def generateReport2(frameDict, accelerator):
        computedMagnets = ''
        missingMagnets = ''

        for frameName in frameDict:
            frame = frameDict[frameName]

            if ('MEASURED' in frame.name or frame.name == 'machine-local'):
                continue
            
            if (accelerator == 'SR'):
                try:
                    magnetName = frame.name.split('-')[0] + '-' + frame.name.split('-')[1] + '-' + frame.name.split('-')[2]+ '-' + frame.name.split('-')[3]+'\n'
                except IndexError:
                    magnetName = frame.name.split('-')[0] + '-' + frame.name.split('-')[1] + '-' + frame.name.split('-')[2]+'\n'
            else:
                magnetName = frame.name.split('-')[0] + '-' + frame.name.split('-')[1]+'\n'
            
            measuredFrame = magnetName[:-1] + '-' + 'MEASURED'

            if (not measuredFrame in frameDict):
                missingMagnets += magnetName
            else:
                computedMagnets += magnetName

        report = f'IMÃS FALTANTES\n {missingMagnets}\n\nIMÃS JÁ COMPUTADOS\n {computedMagnets}'
        
        return report

    @staticmethod
    def generateReport(frameDict, accelerator):
        computedMagnets = ''
        missingMagnets = ''

        for frameName in frameDict:
            frame = frameDict[frameName]

            if ('MEASURED' in frame.name or frame.name == 'machine-local'):
                continue
            
            nameParts = []
            [nameParts.append(frame.name.split('-')[i]) for i in range (len(frame.name.split('-')) - 1)]

            magnetName = ''
            for name in nameParts:
                magnetName +=  f'-{name}'

            magnetName = magnetName[1:]
            measuredFrame = magnetName + '-MEASURED'

            if (not measuredFrame in frameDict):
                missingMagnets += f'{magnetName}\n'
            else:
                computedMagnets += f'{magnetName}\n'

        report = f'IMÃS FALTANTES\n {missingMagnets}\n\nIMÃS JÁ COMPUTADOS\n {computedMagnets}'
        
        return report