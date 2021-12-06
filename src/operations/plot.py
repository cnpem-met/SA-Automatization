from typing import List
import pandas as pd
import config

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.offsetbox as offsetbox

def plot_magnets_absolute_deviation(deviations: List[pd.DataFrame], **plot_args):
    """ Plots magnets absolute deviations. """

    # making a copy of the dataframes
    plot_data = []
    for data in deviations:
        plot_data.append(data.copy())


    # font-sizing
    fontBaseSize = 7
    fontScale = 1.2

    # initializing sheet object
    fig = plt.figure(tight_layout=True)

    # defining sheet layout and axes
    gs_dev = plt.GridSpec(4,2, top=0.9, hspace=0.2, wspace=0.1, width_ratios=[4,1])
    gs_rot = plt.GridSpec(4,2, bottom=0.09, hspace=0.2, wspace=0.1, width_ratios=[4,1])
    axs = [[],[]]

    # defining accelerator's specific settings and actions
    accelerator = plot_args['accelerator']
    if(accelerator == 'SR'):
        # applying filters
        if (plot_args['filtering']['SR']['allDofs']):
            for magnet, dof in plot_data[0].iterrows():
                if('B03-QUAD01' in magnet or 'B11-QUAD01' in magnet or ('QUAD' in magnet and not 'LONG' in magnet)):
                    plot_data[0].at[magnet, 'Tz'] = float('nan')
                elif('B1-LONG' in magnet):
                    plot_data[0].at[magnet, 'Tx'] = float('nan')
                    plot_data[0].at[magnet, 'Ty'] = float('nan')
                    plot_data[0].at[magnet, 'Rx'] = float('nan')
                    plot_data[0].at[magnet, 'Ry'] = float('nan')
                    plot_data[0].at[magnet, 'Rz'] = float('nan')
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

    if(accelerator == 'booster'):
        # applying filter
        if (plot_args['filtering']['booster']['quad02']):
            for magnet, dof in plot_data[0].iterrows():
                if ('QUAD02' in magnet):
                    plot_data[0].at[magnet, :] = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]
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
        
    if (accelerator == 'LTB' or accelerator == 'BTS'):
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
        col_names = plot_args['FElist']
        normalized_names = normalizeTextInBox(col_names)
        # name_xpos = [0.175, 0.465, 0.74] # for subset 1
        name_xpos = [0.182, 0.46, 0.743] # for subset 2
        fig.text(name_xpos[0], 0.895, normalized_names[0], weight='semibold', fontsize=fontScale*(fontBaseSize+3), bbox=bbox_props)
        fig.text(name_xpos[1], 0.895, normalized_names[1], weight='semibold', fontsize=fontScale*(fontBaseSize+3), bbox=bbox_props)
        fig.text(name_xpos[2], 0.895, normalized_names[2], weight='semibold', fontsize=fontScale*(fontBaseSize+3), bbox=bbox_props)
        

    # Uncertainties for magnet points inside radiation shielding calculated with Monte Carlo analysis per degree of freedom
    magnets_uncertainties = config.POINTS_UNCERTAINTIES

    # mapping df's columns to its respective plot titles and y-axis title
    plot_naming_details = {'y_list': ['Tx', 'Ty', 'Tz', 'Rz'], 'title_list': ['HORIZONTAL', 'VERTICAL', 'LONGITUDINAL', 'ROLL'], 'fig_title': plotTitle}
    df_colums_dict = {'Tx': 'deviation horiz. [mm]', 'Ty': 'deviation vert. [mm]', 'Tz': 'deviation long. [mm]', 'Rz': 'rotation roll [mrad]'}
    x_axis_title = 'Magnet'

    # list that will hold magnet's names
    magnet_name_list = []

    # cleaning data for plotting
    for index, data in enumerate(plot_data):
        magnet_name_list.append(data.index.to_numpy())

        # excluding columns that are not being considered for plotting
        data = data.drop(columns=['Rx', 'Ry'])

        # manipulating df to be more plot friendly
        new_index = np.linspace(0, data['Tx'].size-1, data['Tx'].size)
        data['Index'] = new_index
        data.insert(0, x_axis_title, data.index)
        data.set_index('Index', drop=True, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data.rename(columns=df_colums_dict, inplace=True)
        plot_data[index] = data        

    # figure's title
    bbox_props = dict(boxstyle="square,pad=0.5", fc="white", lw=0.5)
    fig_title = plot_naming_details['fig_title']

    # setting x axis' tick position
    spacing = int(plot_data[0].iloc[:, 0].size/tick_spacing)
    tickpos = np.linspace(0, plot_data[0].iloc[:, 0].size - tick_spacing, spacing)

    # defining x axis' labels for report mode
    tickLabels = []
    if (accelerator == 'SR'):
        [tickLabels.append(f'{str(i+1)}') for i in range(0, len(tickpos))]
    elif (accelerator == 'booster'):
        [tickLabels.append(f'{str(2*i+1)}') for i in range(0, len(tickpos))]

    # definig color's pallet for all the plots
    plot_colors = ['royalblue', 'darkorange', 'seagreen', 'black']
    
    # title list is directly accessed
    plot_titles = plot_naming_details['title_list']

    # for the df's columns to be plotted, we have to map 'y_list' with 'df_columns_dict' 
    y_list = []
    for y in plot_naming_details['y_list']:
        if y in df_colums_dict:
            y_list.append(df_colums_dict[y])

    # creating data structures to be called in plot's function
    x = magnet_name_list
    y = [[] for i in range(num_columns)]

    # considering that more than one dataset can be plotted at once, make room for more than 1 plot
    for column in range(len(plot_data)):
        for row in range(4):
            y[column].append(plot_data[column][y_list[row]].to_numpy()) 

    # plot dispersion and in some cases the histogram
    for column in range(len(y)):
        for row in range(4):
            plot_dispersion(axs[column][row], x[column], y[column][row], row, magnets_uncertainties[row], plot_colors[row], tickpos, tickLabels, tick_id, fontScale, fontBaseSize, x_extension, accelerator, tolerances[row], plot_titles[row], plot_data[column], plot_args['filtering'][accelerator]['lineWithinGirder'],  plot_args['filtering'][accelerator]['errorbar'],  plot_args['filtering'][accelerator]['reportMode'] )
            
            # calculating standard deviation and plotting histograms for some accelerators
            if (accelerator == 'SR' or accelerator == 'booster'):
                stdDev = np.std(plot_data[0].iloc[:,row+1])
                plot_histogram(axs[1][row], y[column][row], stdDev, row, plot_colors[row], tolerances[row], fontScale, fontBaseSize, bins[row])

    # some needed calls
    plt.get_current_fig_manager().window.showMaximized()
    plt.draw()
    plt.pause(0.001)
    plt.ioff()
    plt.savefig(f"{config.OUTPUT_PATH}/SR-plot.png", dpi=150)
    plt.show()

def plot_dispersion(ax, x, y, i, uncertainty, plot_color, tickpos, tick_labels, tick_id, font_scale, font_base_size, x_extension, accelerator, tolerances, plot_title, plot_data, show_line_between_girders, show_errorbar, is_report_mode ):
    """ Calls the low level plot methods. """

    # simply plotting data with errobars
    if (show_errorbar):
        _, caps, bars = ax.errorbar(x, y, yerr=uncertainty, fmt='o', marker='o', ms=2, mec=plot_color, mfc=plot_color, color='wheat', ecolor='k', elinewidth=0.5, capthick=0.5, capsize=3)
        
        # loop through bars and caps and set the alpha value
        [bar.set_alpha(0.7) for bar in bars]
        [cap.set_alpha(0.7) for cap in caps]
    else:
        k=0
        while (k < len(x)):
            currGirder = x[k].split('-')[0]+'-'+x[k].split('-')[1]
            try:
                nextGirder = x[k+1].split('-')[0]+'-'+x[k+1].split('-')[1]
            except IndexError:
                nextGirder = None
            if(currGirder == nextGirder):
                if (show_line_between_girders):
                    # show lines between magnets within the same girder
                    ax.plot(x[k: k+2], y[k: k+2], 'o-', color=plot_color, markersize=4)
                    k+=2
                else:
                    # only plot the points with no line
                    ax.plot(x[k], y[k], 'o-', color=plot_color, markersize=4)
                    k+=1
            else:
                # only plot the points with no line
                ax.plot(x[k], y[k], 'o-', color=plot_color, markersize=4)
                k+=1

    ax.set_xticks(tickpos)
    # changuing x-axis' labels to be less specific (i.e. not the magnets' id) if report mode is on
    if (is_report_mode):
        ax.set_xticklabels(tick_labels)
        if (i == 3):
            ax.set_xlabel(tick_id, **dict(fontsize=font_scale*(font_base_size+5)))#3
    
    # defining plot properties that depends on which plot it is referencing to
    if(i==3):
        # y-axis' title
        ylabel = 'Error [mrad]'
        # configuring x-axis' labels
        ax.tick_params(axis='x', labelsize=font_scale*(font_base_size+2.8))#1.8
        # text to be shown in mean deviation's box
        boxTxt = 'Mean deviation: {:.2f} mrad'
    else:
        ylabel = 'Error [mm]'
        ax.tick_params(axis='x', which='major', direction='in', bottom=True, top=False, labelbottom=False)
        boxTxt = 'Mean deviation: {:.2f} mm'

    # filling the tolerance zone with a color
    x_filled = np.linspace(-5, len(x)+5, 10)
    ax.fill_between(x_filled, tolerances, -tolerances, color='green', alpha=0.1)
    
    # generic plot properties
    ax.grid(b=True, axis='both', which='major', linestyle='--', alpha=0.5)
    ax.tick_params(axis='y', labelsize=font_scale*(font_base_size+3))#2
    ax.set_ylabel(ylabel, **dict(fontsize=font_scale*(font_base_size+3.5)))#2.5
    # ax.xaxis.labelpad = 10
    ax.xaxis.labelpad = 5
    ax.set_xlim(-x_extension, plot_data.iloc[:, 0].size)

    # checking if highest point is conflicting with text box and correcting it iteratively
    y_lims = ax.get_ylim()
    y_range = y_lims[1] - y_lims[0]
    estimated_box_bottomypos = y_lims[1] - y_range/4.5
    highest_pt = max(y)
    if (highest_pt + uncertainty >= estimated_box_bottomypos):
        ax.set_ylim(y_lims[0]*1.01, y_lims[1]*1.02)
        isConflicting = True
        while (isConflicting):
            y_lims = ax.get_ylim()
            y_range = y_lims[1] - y_lims[0]
            estimated_box_bottomypos = y_lims[1] - y_range/4.5
            if (highest_pt + uncertainty >= estimated_box_bottomypos):
                ax.set_ylim(y_lims[0]*1.01, y_lims[1]*1.02)
                isConflicting = True
            else:
                isConflicting = False

    # # SR oficial report y lims
    # if (i==0 or i==1):
    #     ax.set_ylim(-0.115, 0.115)
    # elif (i==2):
    #     ax.set_ylim(-0.142, 0.142)
    # elif  (i==3):
    #     ax.set_ylim(-0.45, 0.45)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # setting plots titles within white boxes
    bbox_props = dict(boxstyle="square,pad=0.1", fc="white", lw=0)
    ax.set_title(plot_title, y=0.92, bbox=bbox_props, **dict(fontsize=font_scale*(font_base_size+5), weight=600))#4

    # calculating the absolute mean and showing it inside a box in plot
    avg = np.mean(np.absolute(plot_data.iloc[:,i+1]))
    ob = offsetbox.AnchoredText(boxTxt.format(avg), loc=1, pad=0.25, borderpad=0.5, prop=dict(fontsize=font_scale*(font_base_size+4), weight=550))#3
    ax.add_artist(ob)

def plot_histogram(ax, data, stdDev, i, plot_color, tolerances, fontScale, fontBaseSize, bin, plot_title=None):
    """ Calls low level method to plot histogram """

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

    box_txt = 'SD = {:.2f} {}'.format(stdDev, unit)
    ob = offsetbox.AnchoredText(box_txt, loc=1, pad=0.25, borderpad=0.5, prop=dict(fontsize=fontScale*(fontBaseSize+3.5), weight=550))#2.5
    ax.add_artist(ob)

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