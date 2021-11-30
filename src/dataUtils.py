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
from matplotlib import ticker


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
    def writeToExcel(fileDir, data, accelerator, report_type='deviations'):
        # adding extra column to hold quadrupole types
        if (report_type == 'deviations' and accelerator == 'SR'):
            data.insert(1, 'magnet-family', np.empty(len(data.iloc[:,0]), dtype=str), True)
            for magnet, dof in data.iterrows():
                magnetFamily = SR.checkMagnetFamily(magnet)
                data.at[magnet, 'magnet-family'] = magnetFamily

        data.to_excel(fileDir)

    @staticmethod
    def plotDevitationData(deviations, **plotArgs):  # precisa adaptação pra booster
        # making a copy of the dataframes
        plotData = []
        for data in deviations:
            plotData.append(data.copy())

        accelerator = plotArgs['accelerator']

        # font-sizing
        fontBaseSize = 7
        fontScale = 1.2

        # initializing sheet object
        fig = plt.figure(tight_layout=True)
        # fig = plt.figure(figsize=(3.5,12))
        # fig = plt.figure(figsize=(8,7))
        # fig = plt.figure(figsize=(8,7)) # histograms

        # defining sheet layout and axes
        # gs_dev = plt.GridSpec(4,2, top=0.9, hspace=0.3, wspace=0.15, width_ratios=[4,1])
        # gs_rot = plt.GridSpec(4,2, bottom=0.07, hspace=0.3, wspace=0.15, width_ratios=[4,1])
        gs_dev = plt.GridSpec(4,2, top=0.9, hspace=0.2, wspace=0.1, width_ratios=[4,1])
        gs_rot = plt.GridSpec(4,2, bottom=0.09, hspace=0.2, wspace=0.1, width_ratios=[4,1])
        
        # gs_dev = plt.GridSpec(2,2, top=0.94, bottom=0.1, hspace=0.4, wspace=0.4) # histograms
        axs = [[],[]]

        if(accelerator == 'SR'):
            # applying filters
            if (plotArgs['filtering']['SR']['allDofs']):
                for magnet, dof in plotData[0].iterrows():
                    if('B03-QUAD01' in magnet or 'B11-QUAD01' in magnet or ('QUAD' in magnet and not 'LONG' in magnet)):
                        plotData[0].at[magnet, 'Tz'] = float('nan')
                    elif('B1-LONG' in magnet):
                        plotData[0].at[magnet, 'Tx'] = float('nan')
                        plotData[0].at[magnet, 'Ty'] = float('nan')
                        plotData[0].at[magnet, 'Rx'] = float('nan')
                        plotData[0].at[magnet, 'Ry'] = float('nan')
                        plotData[0].at[magnet, 'Rz'] = float('nan')
            # defining general properties
            plotTitle = 'Global Alignment Profile - Storage Ring'
            tolerances = [0.08, 0.08, 0.1, 0.3]
            tick_spacing = 18.5
            tick_id = 'Sector'
            bins = [6,6,12,10]
            x_extension = 2
            num_columns = 1
            
            axs[0].append(fig.add_subplot(gs_dev[0,0]))
            axs[0].append(fig.add_subplot(gs_dev[1,0], sharex=axs[0][0]))
            axs[0].append(fig.add_subplot(gs_dev[2,0], sharex=axs[0][0]))
            axs[0].append(fig.add_subplot(gs_rot[3,0], sharex=axs[0][0]))
            axs[1].append(fig.add_subplot(gs_dev[0,1]))
            axs[1].append(fig.add_subplot(gs_dev[1,1]))
            axs[1].append(fig.add_subplot(gs_dev[2,1]))
            axs[1].append(fig.add_subplot(gs_rot[3,1]))

            # fig.tight_layout()

        if(accelerator == 'booster'):
            # applying filter
            if (plotArgs['filtering']['booster']['quad02']):
                for magnet, dof in plotData[0].iterrows():
                    if ('QUAD02' in magnet):
                        plotData[0].at[magnet, :] = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]
            # defining general properties
            plotTitle = 'Global Alignment Profile - Booster'
            tolerances = [0.16, 0.16, 0.4, 0.8]
            tick_spacing = 5
            tick_id = 'Sector'
            bins = [10, 10, 6, 6]
            x_extension = 2
            num_columns = 1
            axs[0].append(fig.add_subplot(gs_dev[0,0]))
            axs[0].append(fig.add_subplot(gs_dev[1,0], sharex=axs[0][0]))
            axs[0].append(fig.add_subplot(gs_dev[2,0], sharex=axs[0][0]))
            axs[0].append(fig.add_subplot(gs_rot[3,0], sharex=axs[0][0]))
            axs[1].append(fig.add_subplot(gs_dev[0,1]))
            axs[1].append(fig.add_subplot(gs_dev[1,1]))
            axs[1].append(fig.add_subplot(gs_dev[2,1]))
            axs[1].append(fig.add_subplot(gs_rot[3,1]))
            
        if (accelerator == 'transport-lines'):
            # defining sheet layout and axes
            num_columns = len(deviations)
            axs_raw = fig.subplots(4,num_columns, sharex='col', gridspec_kw=dict(top=0.85, hspace=0.2, wspace=0.2, width_ratios=[1 for i in range(num_columns)]))
            axs = [axs_raw[:, i] for i in range(num_columns)]
            # defining general properties
            accelerator = 'LTB'
            plotTitle = 'Global Alignment Profile - Transport Lines'
            tolerances = [0.16, 0.16, 0.4, 0.8]
            tick_spacing = 1
            tick_id = 'Magnet'
            bins = [10, 10, 6, 6]
            x_extension = 1
            # defining sheet layout and axes
            bbox_props = dict(boxstyle="square,pad=0.5", fc="white", lw=0.5)
            fig.text(0.26, 0.9, '            LTB            ', weight='semibold', fontsize=fontScale*(fontBaseSize+3), bbox=bbox_props)
            fig.text(0.675, 0.9, '            BTS            ', weight='semibold', fontsize=fontScale*(fontBaseSize+3), bbox=bbox_props)

        if (accelerator == 'FE'):
            # defining sheet layout and axes
            num_columns = len(deviations)
            axs_raw = fig.subplots(4,num_columns, sharex='col', gridspec_kw=dict(top=0.85, hspace=0.2, wspace=0.35, width_ratios=[1 for i in range(num_columns)]))
            axs = [axs_raw[:, i] for i in range(num_columns)]
            # defining general properties
            plotTitle = 'Global Alignment Profile - Front Ends'
            tolerances = [0.16, 0.16, 0.4, 0.8]
            tick_spacing = 1
            tick_id = 'Magnet'
            bins = [10, 10, 6, 6]
            x_extension = 1
            # defining sheet layout and axes
            bbox_props = dict(boxstyle="square,pad=0.5", fc="white", lw=0.5)
            col_names = plotArgs['FElist']
            normalized_names = DataUtils.normalizeTextInBox(col_names)
            # name_xpos = [0.175, 0.465, 0.74] # for subset 1
            name_xpos = [0.182, 0.46, 0.743] # for subset 2
            fig.text(name_xpos[0], 0.895, normalized_names[0], weight='semibold', fontsize=fontScale*(fontBaseSize+3), bbox=bbox_props)
            fig.text(name_xpos[1], 0.895, normalized_names[1], weight='semibold', fontsize=fontScale*(fontBaseSize+3), bbox=bbox_props)
            fig.text(name_xpos[2], 0.895, normalized_names[2], weight='semibold', fontsize=fontScale*(fontBaseSize+3), bbox=bbox_props)
            

        # Tolerances calculated with Monte Carlo analysis
        unc = [0.018, 0.043, 0.023, 0.026]

        # mapping df's columns to its respective plot titles and y-axis title
        plot_args = {'y_list': ['Tx', 'Ty', 'Tz', 'Rz'], 'title_list': ['HORIZONTAL', 'VERTICAL', 'LONGITUDINAL', 'ROLL'], 'fig_title': plotTitle}
        df_colums_dict = {'Tx': 'deviation horiz. [mm]', 'Ty': 'deviation vert. [mm]', 'Tz': 'deviation long. [mm]', 'Rz': 'rotation roll [mrad]'}
        xAxisTitle = 'Magnet'

        # initializing list that will hold the lists with magnets' names 
        magnetNameList = []
        
        for index, data in enumerate(plotData):
            magnetNameList.append(data.index.to_numpy())

            # excluding columns that are not being considered for plotting
            data = data.drop(columns=['Rx', 'Ry'])

            # manipulating df to be more plot friendly
            new_index = np.linspace(0, data['Tx'].size-1, data['Tx'].size)
            data['Index'] = new_index
            data.insert(0, xAxisTitle, data.index)
            data.set_index('Index', drop=True, inplace=True)
            data.reset_index(drop=True, inplace=True)
            data.rename(columns=df_colums_dict, inplace=True)
            plotData[index] = data        

        # figure's title
        bbox_props = dict(boxstyle="square,pad=0.5", fc="white", lw=0.5)
        fig_title = plot_args['fig_title']
        # fig.suptitle(fig_title, y=0.97, color='black', weight='semibold', fontsize=fontScale*(fontBaseSize+6), bbox=bbox_props)

        # setting x axis' tick position
        spacing = int(plotData[0].iloc[:, 0].size/tick_spacing)
        tickpos = np.linspace(0, plotData[0].iloc[:, 0].size - tick_spacing, spacing)

        # defining x axis' labels for report mode
        tickLabels = []
        if (accelerator == 'SR'):
            [tickLabels.append(f'{str(i+1)}') for i in range(0, len(tickpos))]
        elif (accelerator == 'booster'):
            [tickLabels.append(f'{str(2*i+1)}') for i in range(0, len(tickpos))]

        # definig color's pallet for all the plots
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

        # creating data structures to be called in plot's function
        x = magnetNameList
        y = [[] for i in range(num_columns)]
    
        for column in range(len(plotData)):
            for row in range(4):
                y[column].append(plotData[column][y_list[row]].to_numpy()) 

        for column in range(len(y)):
            for row in range(4):
                DataUtils.plotDispersion(axs[column][row], x[column], y[column][row], row, unc[row], plot_colors[row], tickpos, tickLabels, tick_id, fontScale, fontBaseSize, x_extension, accelerator, tolerances[row], plot_titles[row], plotData[column], plotArgs['filtering'][accelerator]['lineWithinGirder'],  plotArgs['filtering'][accelerator]['errorbar'],  plotArgs['filtering'][accelerator]['reportMode'] )
                
                # calculating standard deviation and plotting histograms for some accelerators
                if (accelerator == 'SR' or accelerator == 'booster'):
                    stdDev = np.std(plotData[0].iloc[:,row+1])
                    DataUtils.plotHistogram(axs[1][row], y[column][row], stdDev, row, plot_colors[row], tolerances[row], fontScale, fontBaseSize, bins[row])

        plt.get_current_fig_manager().window.showMaximized()

        # acrescentando intervalo entre plots
        plt.draw()
        plt.pause(0.001)

        # condicional para definir se a figura irá congelar o andamento do app
        # usado no ultimo plot, pois senão as janelas fecharão assim que o app acabar de executar
        plt.ioff()

        plt.savefig("../data/output/SR-plot.png", dpi=150)

        # evocando a tela com a figura
        plt.show()

    @staticmethod
    def plotDispersion(ax, x, y, i, uncertainty, plot_color, tickpos, tickLabels, tick_id, fontScale, fontBaseSize, x_extension, accelerator, tolerances, plot_title, plotData, showLineBetweenGirders, showErrorbar, isReportMode ):
        if (showErrorbar):
            _, caps, bars = ax.errorbar(x, y, yerr=uncertainty, fmt='o', marker='o', ms=2, mec=plot_color, mfc=plot_color, color='wheat', ecolor='k', elinewidth=0.5, capthick=0.5, capsize=3)
            
            # loop through bars and caps and set the alpha value
            [bar.set_alpha(0.7) for bar in bars]
            [cap.set_alpha(0.7) for cap in caps]

            # for j in range(len(x)-1):
            #     if (accelerator == 'SR'):
            #         unc = SR.getUncertaintyAndTranslationsPerMagnet(x[j], 'unc', i)
            #     else:
            #         uncertainties = [0.018, 0.043, 0.023, 0.026]
            #         unc = uncertainties[i]
            #     ax.errorbar(x[j:j+1], y[j:j+1], yerr=unc, marker='o', ms=2, mec=plot_color, mfc=plot_color, color='wheat', ecolor='k', elinewidth=0.5, capthick=0.5, capsize=3)

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
                    if (showLineBetweenGirders):
                        ax.plot(x[k: k+2], y[k: k+2], 'o-', color=plot_color, markersize=4)
                        k+=2
                    else:
                        ax.plot(x[k], y[k], 'o-', color=plot_color, markersize=4)
                        k+=1
                else:
                    ax.plot(x[k], y[k], 'o-', color=plot_color, markersize=4)
                    k+=1

        ax.set_xticks(tickpos)
        # changuing x-axis' labels to be less specific (i.e. not the magnets' id) if report mode is on
        if (isReportMode):
            ax.set_xticklabels(tickLabels)
            if (i == 3):
                ax.set_xlabel(tick_id, **dict(fontsize=fontScale*(fontBaseSize+5)))#3
        
        # defining plot properties that depends on which plot it is referencing to
        if(i==3):
            # y-axis' title
            ylabel = 'Error [mrad]'
            # configuring x-axis' labels
            # ax.tick_params(axis='x', which='major', direction='in', bottom=True, top=False, labelrotation=45,labelsize=fontScale*(fontBaseSize+2))
            ax.tick_params(axis='x', labelsize=fontScale*(fontBaseSize+2.8))#1.8
            # text to be shown in mean deviation's box
            boxTxt = 'Mean deviation: {:.2f} mrad'
            # ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
            # ax.set_ylim(-0.88, 0.88)
        else:
            ylabel = 'Error [mm]'
            ax.tick_params(axis='x', which='major', direction='in', bottom=True, top=False, labelbottom=False)
            boxTxt = 'Mean deviation: {:.2f} mm'
            # defining specific y-scale for some accelerators
            # if (i != 2):
            #     ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
            #     if (i==0):
            #         ax.set_ylim(-0.22, 0.22)
            #     else:
            #         ax.set_ylim(-0.26, 0.26)
            # else:
            #     ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
            #     ax.set_ylim(-0.44, 0.44)

        # filling the tolerance zone with a color
        x_filled = np.linspace(-5, len(x)+5, 10)
        ax.fill_between(x_filled, tolerances, -tolerances, color='green', alpha=0.1)
        
        # generic plot properties
        ax.grid(b=True, axis='both', which='major', linestyle='--', alpha=0.5)
        ax.tick_params(axis='y', labelsize=fontScale*(fontBaseSize+3))#2
        ax.set_ylabel(ylabel, **dict(fontsize=fontScale*(fontBaseSize+3.5)))#2.5
        # ax.xaxis.labelpad = 10
        ax.xaxis.labelpad = 5
        ax.set_xlim(-x_extension, plotData.iloc[:, 0].size)

        # checking if highest point is conflicting with text box and correcting it iteratively
        # y_lims = ax.get_ylim()
        # y_range = y_lims[1] - y_lims[0]
        # estimated_box_bottomypos = y_lims[1] - y_range/4.5
        # highest_pt = max(y)
        # if (highest_pt + uncertainty >= estimated_box_bottomypos):
        #     ax.set_ylim(y_lims[0]*1.01, y_lims[1]*1.02)
        #     isConflicting = True
        #     while (isConflicting):
        #         y_lims = ax.get_ylim()
        #         y_range = y_lims[1] - y_lims[0]
        #         estimated_box_bottomypos = y_lims[1] - y_range/4.5
        #         if (highest_pt + uncertainty >= estimated_box_bottomypos):
        #             ax.set_ylim(y_lims[0]*1.01, y_lims[1]*1.02)
        #             isConflicting = True
        #         else:
        #             isConflicting = False
        if (i==0 or i==1):
            ax.set_ylim(-0.115, 0.115)
        elif (i==2):
            ax.set_ylim(-0.142, 0.142)
        elif  (i==3):
            ax.set_ylim(-0.45, 0.45)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # setting plots titles within white boxes
        bbox_props = dict(boxstyle="square,pad=0.1", fc="white", lw=0)
        ax.set_title(plot_title, y=0.92, bbox=bbox_props, **dict(fontsize=fontScale*(fontBaseSize+5), weight=600))#4

        # calculating the absolute mean and showing it inside a box in plot
        avg = np.mean(np.absolute(plotData.iloc[:,i+1]))
        ob = offsetbox.AnchoredText(boxTxt.format(avg), loc=1, pad=0.25, borderpad=0.5, prop=dict(fontsize=fontScale*(fontBaseSize+4), weight=550))#3
        ax.add_artist(ob)

    @staticmethod
    def plotHistogram(ax, data, stdDev, i, plot_color, tolerances, fontScale, fontBaseSize, bin, plot_title=None):
        data = data[~np.isnan(data)]
        hist, bins = np.histogram(data, bins=bin)
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2

        total = len(data)
        hist_perc = [occ/total*100 for occ in hist]
        ax.bar(center, hist_perc, align='center', width=width, facecolor=plot_color, ec='white')

        if (plot_title):
            # setting plots titles within white boxes
            bbox_props = dict(boxstyle="square,pad=0.1", fc="white", lw=0)
            ax.set_title(plot_title, y=0.96, bbox=bbox_props, **dict(fontsize=fontScale*(fontBaseSize+2), weight=600))#2

        ax.set_xticks([-1.5*tolerances, -tolerances, -0.5*tolerances, 0, 0.5*tolerances, tolerances, 1.5*tolerances])

        # ax.tick_params(axis='x', which='major', direction='in', bottom=True, top=False, labelrotation=45,labelsize=fontScale*(fontBaseSize+1.8))
        ax.tick_params(axis='x', labelsize=fontScale*(fontBaseSize+2.8))#1.8
        ax.tick_params(axis='y', labelsize=fontScale*(fontBaseSize+2.8))

        ax.set_xlim(-1.3*tolerances, 1.3*tolerances)

        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

        ylimits_hist = ax.get_ylim()
        x_filled = np.linspace(-tolerances, tolerances, 10)
        ax.fill_between(x_filled, 0, 2*ylimits_hist[1], color='green', alpha=0.1)
        ax.set_ylim(ylimits_hist)

        unit = 'mm'
        if (i == 3):
            ax.set_xlabel('Error [mrad]', **dict(fontsize=fontScale*(fontBaseSize+3.8)))#2.8
            unit = 'mrad'
        elif (i == 2):
            ax.set_xlabel('Error [mm]', **dict(fontsize=fontScale*(fontBaseSize+3.8)))
        
        ax.xaxis.labelpad = 5

        ax.set_ylabel('%', **dict(fontsize=fontScale*(fontBaseSize+3.8)))

        y_lim = ax.get_ylim()
        ax.set_ylim(y_lim[0], y_lim[1]*1.3)

        ax.grid(b=True, axis='y', which='major', linestyle='--', alpha=0.7)

        # box_txt = r'$\sigma = {:.2f}\hspace\{{}\}{}$'.format(stdDev, 0.2, unit)
        # box_txt = r'$\sigma = $' + '{:.2f} {}'.format(stdDev, unit)
        box_txt = 'SD = {:.2f} {}'.format(stdDev, unit)
        ob = offsetbox.AnchoredText(box_txt, loc=1, pad=0.25, borderpad=0.5, prop=dict(fontsize=fontScale*(fontBaseSize+3.5), weight=550))#2.5
        ax.add_artist(ob)

    # @staticmethod
    # def plotComparativeDeviation(deviations, **plotArgs):  # precisa adaptação pra booster
    #     # pegando uma cópia do df original
    #     plotData = deviations.copy()

    #     accelerator = plotArgs['accelerator']
    #     lenghts = plotArgs['len']

    #     # applying filters
    #     for magnet, dof in plotData.iterrows():
    #         if('B03-QUAD01' in magnet or 'B11-QUAD01' in magnet or ('QUAD' in magnet and not 'LONG' in magnet)):
    #             plotData.at[magnet, 'Tz'] = float('nan')
    #         elif('B1-LONG' in magnet):
    #             plotData.at[magnet, 'Tx'] = float('nan')
    #             plotData.at[magnet, 'Ty'] = float('nan')
    #             plotData.at[magnet, 'Rx'] = float('nan')
    #             plotData.at[magnet, 'Ry'] = float('nan')
    #             plotData.at[magnet, 'Rz'] = float('nan')

    #     for magnet, dof in plotData.iterrows():
    #         if ('P' in magnet and 'QUAD02' in magnet):
    #             plotData.at[magnet, :] = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]

    #     plotTitle = 'Global Alignment Profile'
    #     tolerances = [0.16, 0.16, 0.4, 0.8]
    #     tick_spacing = 17
    #     x_extension = 2
    #     unc = [0.018, 0.043, 0.023, 0.026]

    #     plot_args = {'y_list': ['Tx', 'Ty', 'Tz', 'Rz'], 'title_list': ['HORIZONTAL', 'VERTICAL', 'LONGITUDINAL', 'ROLL'], 'fig_title': plotTitle}

    #     xAxisTitle = 'Magnet'

    #     colum1_name = 'Tx'
    #     df_colums_dict = {
    #         'Tx': 'deviation horiz. [mm]', 'Ty': 'deviation vert. [mm]', 'Tz': 'deviation long. [mm]', 'Rz': 'rotation roll [mrad]'}

    #     """ configurando df """
    #     new_index = np.linspace(0, plotData[colum1_name].size-1, plotData[colum1_name].size)

    #     magnetNameList = plotData.index.to_numpy()

    #     # EXCLUINDO COLUNAS COM DOFS DESNECESSÁRIOS
    #     plotData = plotData.drop(columns=['Rx', 'Ry'])

    #     plotData['Index'] = new_index
    #     plotData.insert(0, xAxisTitle, plotData.index)
    #     plotData.set_index('Index', drop=True, inplace=True)
    #     plotData.reset_index(drop=True, inplace=True)
    #     plotData.rename(columns=df_colums_dict, inplace=True)

    #     fig = plt.figure()
    #     fig.tight_layout()

    #     gs_dev = plt.GridSpec(4,1, top=0.9, hspace=0.3)
        
    #     axs_plot = []
    #     axs_plot.append(fig.add_subplot(gs_dev[0]))
    #     axs_plot.append(fig.add_subplot(gs_dev[1], sharex=axs_plot[0]))
    #     axs_plot.append(fig.add_subplot(gs_dev[2], sharex=axs_plot[0]))
    #     axs_plot.append(fig.add_subplot(gs_dev[3], sharex=axs_plot[0]))

    #     fontBaseSize = 7
    #     fontScale = 1.2

    #     # titulo da figura
    #     bbox_props = dict(boxstyle="square,pad=0.5", fc="white", lw=0.5)
    #     fig.suptitle(plot_args['fig_title'], y=0.97, color='black', weight='semibold', fontsize=fontScale*(fontBaseSize+6), bbox=bbox_props)


    #     tickpos = np.linspace(0, plotData.iloc[:, 0].size, int(plotData.iloc[:, 0].size/tick_spacing))

    #     tickLabels = []
    #     [tickLabels.append(f'{str(i+1)}') for i in range(0, len(tickpos)-1)]


    #     """salvando args de entrada"""
    #     plot_colors = ['royalblue', 'darkorange', 'seagreen', 'black']
        
        
    #     # a lista de títulos é direta
    #     plot_titles = plot_args['title_list']

    #     # para a lista de colunas do df a serem plotadas, deve-se mapear a lista 'y_list' de entrada
    #     # em relação ao dict 'df_colums_dict' para estar em conformidade com os novos nomes das colunas
    #     y_list = []
    #     for y in plot_args['y_list']:
    #         if y in df_colums_dict:
    #             y_list.append(df_colums_dict[y])

    #     x = magnetNameList
    #     y = []

    #     for i in range(4):
    #         y.append(plotData[y_list[i]].to_numpy()) 

    #     for i in range(len(y)):
    #         if (plotArgs['filtering'][accelerator]['errorbar']):
    #             axs_plot[i].errorbar(x, y[i], yerr=unc[i], fmt='o', marker='o', ms=2, mec=plot_colors[i], mfc=plot_colors[i], color='wheat', ecolor='k', elinewidth=0.5, capthick=0.5, capsize=3)
    #         else:
    #             axs_plot[i].plot(x, y[i], 'o', color=plot_colors[i], ms=2)

    #         axs_plot[i].set_xticks(tickpos)
    #         axs_plot[i].set_xticklabels(tickLabels)
                    
    #         if(i==3):
    #             ylabel = 'Deviation [mrad]'
    #         else:
    #             ylabel = 'Deviation [mm]'

    #         axs_plot[i].tick_params(axis='x', which='major', direction='in', bottom=True, top=False, labelbottom=False)

            
    #         axs_plot[i].grid(b=True, axis='both', which='major', linestyle='--', alpha=0.5)

    #         axs_plot[i].set_ylabel(ylabel, **dict(fontsize=fontScale*(fontBaseSize+1.5)))
    #         axs_plot[i].xaxis.labelpad = 10

    #         axs_plot[i].set_ylim(-1.42*tolerances[i], 1.42*tolerances[i])
    #         axs_plot[i].set_xlim(-x_extension, plotData.iloc[:, 0].size)

    #         bbox_props = dict(boxstyle="square,pad=0.1", fc="white", lw=0)
    #         axs_plot[i].set_title(plot_titles[i], y=0.92, bbox=bbox_props, **dict(fontsize=fontScale*(fontBaseSize+2), weight=600))

    #         axs_plot[i].axvline(x=lenghts[0], color='firebrick', lw='1')
    #         axs_plot[i].axvline(x=lenghts[1], color='firebrick', lw='1')
    #         axs_plot[i].axvline(x=lenghts[2], color='firebrick', lw='1')

    #     axs_plot[0].text(lenghts[0]/2 - 5, 0.16, 'LTB', weight=500, bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.7))
    #     axs_plot[0].text(lenghts[0] + (lenghts[1]-lenghts[0])/2 - 8, 0.16, 'Booster', weight=500, bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.7))
    #     axs_plot[0].text(lenghts[1] + (lenghts[2]-lenghts[1])/2 - 5, 0.16, 'BTS', weight=500, bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.7))
    #     axs_plot[0].text(lenghts[2] + (len(x)-lenghts[2])/2 - 10, 0.16, 'Storage Ring', weight=500, bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.7))

        
    #     # axs_plot[0].annotate('Booster', xy=(1/8, 0.90), xytext=(1/8, 0.9), xycoords='axes fraction',
    #     #     fontsize=fontScale*(fontBaseSize+2), ha='center', va='bottom',
    #     #     bbox=dict(boxstyle='square', fc='white'),
    #     #     arrowprops=dict(arrowstyle='-[, widthB=14, lengthB=1.5', lw=1.0))

    #     # axs_plot[0].annotate('Storage Ring', xy=(5/8, 0.90), xytext=(5/8, 0.9), xycoords='axes fraction', 
    #     #     fontsize=fontScale*(fontBaseSize+2), ha='center', va='bottom',
    #     #     bbox=dict(boxstyle='square', fc='white'),
    #     #     arrowprops=dict(arrowstyle='-[, widthB=14.0, lengthB=1.5', lw=1.0))


    #     plt.get_current_fig_manager().window.showMaximized()

    #     # acrescentando intervalo entre plots
    #     plt.draw()
    #     plt.pause(0.001)

    #     # condicional para definir se a figura irá congelar o andamento do app
    #     # usado no ultimo plot, pois senão as janelas fecharão assim que o app acabar de executar
    #     plt.ioff()

    #     # evocando a tela com a figura
    #     plt.show()

    # @staticmethod
    # def plotRelativeDevitationData(deviations, accelerator):  # precisa adaptação pra booster
    #     # pegando uma cópia do df original
    #     plotData = deviations.copy()

    #     if(isDofsFiltered):
    #         for magnet, dof in plotData.iterrows():
    #             if('B03-QUAD01' in magnet or 'B11-QUAD01' in magnet or ('QUAD' in magnet and not 'LONG' in magnet)):
    #                 plotData.at[magnet, 'Tz'] = float('nan')
    #             elif('B1-LONG' in magnet):
    #                 plotData.at[magnet, 'Tx'] = float('nan')
    #                 plotData.at[magnet, 'Ty'] = float('nan')
    #                 plotData.at[magnet, 'Rx'] = float('nan')
    #                 plotData.at[magnet, 'Ry'] = float('nan')
    #                 plotData.at[magnet, 'Rz'] = float('nan')


    #     if (accelerator == 'SR'):
    #         plotTitle = 'Global Alignment Profile - Storage Ring'
    #         tolerances = [0.02, 0.02, 0.05, 0.15, 0.15, 0.2]
    #         verticalSpacing = 0.04
    #     elif (accelerator == 'booster'):
    #         plotTitle = 'Global Alignment Profile - Booster'
    #         tolerances = [0.08, 0.08, 0.2, 0.3, 0.3, 0.4]
    #         verticalSpacing = 0.1
    #     elif (accelerator == 'LTB'):
    #         plotTitle = 'Global Alignment Profile - LTB'
    #     elif (accelerator == 'BTS'):
    #         plotTitle = 'Global Alignment Profile - BTS'

    #     plot_args = {'y_list': ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz'], 'title_list': [
    #         'Transversal', 'Vertical', 'Longitudinal', 'Pitch', 'Yaw', 'Roll'], 'fig_title': plotTitle}

    #     xAxisTitle = 'Magnet'

    #     colum1_name = 'Tx'
    #     df_colums_dict = {
    #         'Tx': 'deviation transv. [mm]', 'Ty': 'deviation vert. [mm]', 'Tz': 'deviation long. [mm]', 'Rx': 'rotation pitch [mrad]', 'Ry': 'rotation yaw [mrad]', 'Rz': 'rotation roll [mrad]'}

    #     xAxisTicks = []
    #     for i in range(0, plotData.iloc[:, 0].size):
    #         sector = int(i/17) + 1
    #         xAxisTicks.append(str(sector))

    #     """ configurando df """
    #     new_index = np.linspace(
    #         0, plotData[colum1_name].size-1, plotData[colum1_name].size)

    #     magnetNameList = plotData.index.to_numpy()

    #     # print(magnetNameList)

    #     plotData['Index'] = new_index
    #     plotData.insert(0, xAxisTitle, plotData.index)
    #     plotData.set_index('Index', drop=True, inplace=True)
    #     plotData.reset_index(drop=True, inplace=True)
    #     plotData.rename(columns=df_colums_dict, inplace=True)

    #     # invertendo sinal das translações transversais
    #     # plotData["d_transv (mm)"] = -plotData["d_transv (mm)"]

    #     """ configurando parâmetros do plot """
    #     # definindo o numero de linhas e colunas do layout do plot
    #     grid_subplot = [3, 2]
    #     # definindo se as absicissas serão compartilhadas ou não
    #     share_xAxis = 'col'
    #     # criando subplots com os parâmetros gerados acima
    #     fig, axs = plt.subplots(
    #         grid_subplot[0], grid_subplot[1], figsize=(18, 9), sharex=share_xAxis)

    #     plt.subplots_adjust(hspace=0.3)

    #     tickpos = np.linspace(
    #         0, plotData.iloc[:, 0].size, int(plotData.iloc[:, 0].size/17))

    #     tickLabels = []
        
    #     # [tickLabels.append(magnetNameList[i].split('-')[0]) for i in range(0, len(tickpos))]
    #     [tickLabels.append('test'+str(i)) for i in range(0, len(tickpos))]

    #     """salvando args de entrada"""
    #     plot_colors = ['red', 'limegreen', 'blue', 'purple', 'green', 'black']
    #     # a lista de títulos é direta
    #     plot_titles = plot_args['title_list']

    #     # titulo da figura
    #     fig.suptitle(plot_args['fig_title'], fontsize=16)

    #     # para a lista de colunas do df a serem plotadas, deve-se mapear a lista 'y_list' de entrada
    #     # em relação ao dict 'df_colums_dict' para estar em conformidade com os novos nomes das colunas
    #     y_list = []
    #     for y in plot_args['y_list']:
    #         if y in df_colums_dict:
    #             y_list.append(df_colums_dict[y])

    #     x = magnetNameList
    #     y = []

    #     for i in range(6):
    #         y.append(plotData[y_list[i]].to_numpy()) 

    #     for i in range(len(y)):
    #         k=0
    #         while (k < len(x)):
    #             currGirder = x[k].split('-')[0]+'-'+x[k].split('-')[1]
    #             try:
    #                 nextGirder = x[k+1].split('-')[0]+'-'+x[k+1].split('-')[1]
    #             except IndexError:
    #                 nextGirder = None
    #             if(currGirder == nextGirder):
    #                 axs[i % 3][int(i/3)].plot(x[k: k+2], y[i][k: k+2], 'o-', color=plot_colors[i])
    #                 k+=2
    #             else:
    #                 axs[i % 3][int(i/3)].plot(x[k], y[i][k], 'o-', color=plot_colors[i])
    #                 k+=1
    #         # for j in range(0, len(x), 2):
    #         #     axs[i % 3][int(i/3)].plot(x[j: j+2], y[i][j: j+2], 'o-', color=plot_colors[i])

    #         axs[i % 3][int(i/3)].tick_params(axis='x', which='major', direction='in', bottom=True, top=True, labelrotation=45,
    #                                          labelsize=5)
    #         axs[i % 3][int(i/3)].set_xticks(tickpos)
    #         axs[i % 3][int(i/3)].set_xticklabels(tickLabels)
    #         axs[i % 3][int(i/3)].xaxis.labelpad = 10
    #         axs[i % 3][int(i/3)].grid(b=True, axis='both',
    #                                   which='major', linestyle='--', alpha=0.5)

    #         axs[i % 3][int(i/3)].axhline(tolerances[i], color='green', lw=1.5, alpha=0.4)
    #         axs[i % 3][int(i/3)].axhline(-tolerances[i], color='green', lw=1.5, alpha=0.4)

    #         axs[i % 3][int(i/3)].axhline(tolerances[i]*2, color='yellow', lw=1.5, alpha=0.4)
    #         axs[i % 3][int(i/3)].axhline(-tolerances[i]*2, color='yellow', lw=1.5, alpha=0.4)

    #         ylim_bottom, ylim_top = axs[i % 3][int(i/3)].get_ylim()
    #         axs[i % 3][int(i/3)].set_ylim(ylim_bottom - verticalSpacing, ylim_top + verticalSpacing)

    #         axs[i % 3][int(i/3)].set_xlim(0, plotData.iloc[:, 0].size)

    #         axs[i % 3][int(i/3)].set_title(plot_titles[i])

    #         axs[i % 3][int(i/3)].fill_between(plotData.iloc[:, 0], tolerances[i], -tolerances[i], color='green', alpha=0.2)
    #         axs[i % 3][int(i/3)].fill_between(plotData.iloc[:, 0], 2*tolerances[i], tolerances[i], color='yellow', alpha=0.2)
    #         axs[i % 3][int(i/3)].fill_between(plotData.iloc[:, 0], -tolerances[i], -2*tolerances[i], color='yellow', alpha=0.2)

    #     # mostrando plots
    #     # plt.minorticks_off()

    #     # acrescentando intervalo entre plots
    #     plt.draw()
    #     plt.pause(0.001)

    #     # condicional para definir se a figura irá congelar o andamento do app
    #     # usado no ultimo plot, pois senão as janelas fecharão assim que o app acabar de executar
    #     plt.ioff()

    #     # evocando a tela com a figura
    #     plt.show()

    @staticmethod
    def separateFEsData(deviations, frontendList):
        # separatedDeviations = {fe: None for fe in frontendList}
        separatedDeviations = []
        # filtering and saving dfs
        for fe in frontendList:
            filtered_fe = deviations.filter(like=fe, axis=0)
            # separatedDeviations[fe] = filtered_fe
            separatedDeviations.append(filtered_fe)
        return separatedDeviations

    @staticmethod
    def normalizeTextInBox(namesList):
        targetNumOfChar = 27
        normalizedNames = []
        for name in namesList:
            # total characters of the name
            lenght = len(name)
            # number of empty characteres of normalized name
            diff = targetNumOfChar - lenght
            numOfSideEmptyChar = int(diff/2)
            # building text
            normName = " "*numOfSideEmptyChar + name + " "*numOfSideEmptyChar
            normalizedNames.append(normName)
        return normalizedNames

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
            #print("[Transformação p/ frames locais] Imãs não computados: " + magnetsNotComputed[:-1])
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
        elif(accelerator == 'booster'):
            finalFrameDict = {}
            oddWallDict = []
            evenWallDict = []
            # correcting the case of S0xB03
            for frameName in sortedFrameDict:
                frame = sortedFrameDict[frameName]
                try:
                    wall_num = int(frame.name.split('-')[0][1:])
                except ValueError:
                    wall_num = 0
                    
                # even walls
                if (wall_num % 2 == 0):
                    if ('DIP' in frame.name):
                        evenWallDict.append(frame)
                        continue
                # odd walls
                else:
                    if ('DIP' in frame.name):
                        oddWallDict.append(frame)
                        continue

                finalFrameDict[frame.name] = frame

                if ('QUAD01-NOMINAL' in frame.name and wall_num % 2 == 0):
                    for frame in evenWallDict:
                        finalFrameDict[frame.name] = frame
                    evenWallDict = []

                if ('QUAD01-NOMINAL' in frame.name and wall_num % 2 != 0):
                    for frame in oddWallDict:
                        finalFrameDict[frame.name] = frame
                    oddWallDict = []
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
                    # tf = transfMatrix
                    # transformation = Transformation('machine-local', localTransformation.frameTo, tf[0], tf[1], tf[2], tf[3], tf[4], tf[5])
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