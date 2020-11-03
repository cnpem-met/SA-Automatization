from pointGroup import PointGroup
from girderGroup import GirderGroup
from dataOperator import DataOperator
from plot import Plot

if __name__ == "__main__":
    # carregando dados das planilhas
    lookuptable_MLtoLocal = DataOperator.loadFromExcel(
        "../data/input/frames_table_fullpre.xlsx", "Planilha4", has_header=True)
    rawPtsNominal = DataOperator.loadFromExcel(
        "../data/input/SR_nominals.xlsx", "Planilha2", has_header=False)
    rawPtsMeasured = DataOperator.loadFromExcel(
        "../data/input/SR_Magnets_Measured_rev2.xlsx", "Planilha1", has_header=False)

    # gerando grupos de pontos
    ptsNominal = PointGroup(rawPtsNominal, "nominal", lookuptable_MLtoLocal)
    ptsMeasured = PointGroup(rawPtsMeasured, "measured", lookuptable_MLtoLocal)

    # transformando para frame local
    ptsNominal.transformToLocalFrame()
    ptsMeasured.transformToLocalFrame()

    # gerando grupos de berços
    girderNominal = GirderGroup(ptsNominal)
    girderMeasured = GirderGroup(ptsMeasured)

    # calculando centroides
    girderNominal.computeCentroids()
    girderMeasured.computeCentroids()

    # calculando entrada e saida
    girderNominal.computeInOut()
    girderMeasured.computeInOut()

    # calculando desvios (pelo método antigo)
    diffCentroids = GirderGroup.evalDiff_pointToPoint(
        girderNominal.centroids, girderMeasured.centroids)
    diffInOut = GirderGroup.evalDiff_pointToPoint(
        girderNominal.inOut, girderMeasured.inOut)

    # calculando delta entre entrada e saída
    deltaInOut = GirderGroup.calc_delta_inout(diffInOut)

    # calculando angulos (pelo método antigo)
    rotationalDeviations = GirderGroup.calculate_angles(ptsMeasured.ptList)

    # sessão de Plots
    plot_args = {'y_list': ['x', 'z'],
                 'title_list': ['horizontal', 'vertical']}
    Plot.plotGirderDeviation(diffCentroids, 'centroid', plot_args)

    plot_args = {'y_list': ['x_in', 'y_in', 'z_in', 'x_out', 'y_out', 'z_out'], 'title_list': [
        'horizontal', 'longitudinal', 'vertical', 'horizontal', 'longitudinal', 'vertical', ]}
    Plot.plotGirderDeviation(diffInOut, 'inout', plot_args)

    plot_args = {'y_list': ['Roll', 'Pitch'], 'title_list': ['Roll', 'Pitch']}
    Plot.plotGirderDeviation(rotationalDeviations, 'angle', plot_args)

    plot_args = {'y_list': ['delta_x', 'delta_y',
                            'delta_z'], 'title_list': ['dX', 'dY', 'dZ']}
    Plot.plotGirderDeviation(deltaInOut, 'inout_delta',
                             plot_args, freezePlot=True)
