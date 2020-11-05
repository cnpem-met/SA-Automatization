from pointGroup import PointGroup
from girderGroup import GirderGroup
from dataOperator import DataOperator
from plot import Plot

if __name__ == "__main__":
    # carregando dados das planilhas
    lookuptable_MLtoLocal = DataOperator.loadFromExcel(
        "../data/input/frames_table_fullpre.xlsx", "Planilha4", has_header=True)
    rawPtsNominal = DataOperator.loadFromExcel(
        "../data/input/SR_nominals_completo.xlsx", "Planilha2", has_header=False)
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

    # computando os centroides
    girderNominal.computeCentroids()
    girderMeasured.computeCentroids()

    # calculando desvios dos centroides
    diffCentroids = GirderGroup.evalDiff_pointToPoint(
        girderNominal.centroids, girderMeasured.centroids)

    print(diffCentroids)

    # calculando desvios angulares
    diffRotational = GirderGroup.calculate_angles(girderMeasured)

    # calculando desvios de todos dof baseado no best-fit de todos os pontos
    diffAllDoFs = GirderGroup.evalDiff_bestFit(
        girderNominal.pointGroup.ptList, girderMeasured.pointGroup.ptList)

    # salvando resultado em excel
    # DataOperator.saveToExcel(
    #     '../data/output/all-dofs-deviations.xlsx', diffAllDoFs)

    # ----- plots -----

    # rotações (ponto a ponto)
    plot_args = {'y_list': ['Roll', 'Pitch', 'Yaw'],
                 'title_list': ['Rx', 'Ry', 'Rz'], 'fig_title': 'Análise ponto a ponto'}
    Plot.plotGirderDeviation(diffRotational, 'angle', plot_args)

    # translações (ponto a ponto, centroides)
    plot_args = {'y_list': ['x', 'y', 'z'],
                 'title_list': ['Tx', 'Ty', 'Tz'], 'fig_title': 'Análise ponto a ponto'}
    Plot.plotGirderDeviation(diffCentroids, 'centroid', plot_args)

    # translações e rotações (bestfit)
    plot_args = {'y_list': ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz'], 'title_list': [
        'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz'], 'fig_title': 'Análise por Bestfit'}
    Plot.plotGirderDeviation(diffAllDoFs, 'allDoFs',
                             plot_args, freezePlot=True)
