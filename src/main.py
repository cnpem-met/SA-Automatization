from ui import Ui
from PyQt5 import QtWidgets
import sys

from facModel import FacModel
from dataOperator import DataOperator


class App(QtWidgets.QApplication):
    def __init__(self):
        # Initializing the application and its UI
        super(App, self).__init__(sys.argv)

        self.initAppVariables()

        # Calling the UI component
        self.ui = Ui(self)

        # Changuing the default window theme
        self.setStyle('Fusion')

        # Calling the app to execute
        sys.exit(self.exec_())

    def initAppVariables(self):
        self.lookupTable = None
        self.measured = None
        self.nominals = None
        self.girderDictMeas = None
        self.girderDictNom = None

        self.shiftsB1 = None

        # flags
        self.isNominalsLoaded = False
        self.isMeasuredLoaded = False
        self.isLookuptableLoaded = False
        self.isInternalDataProcessed = False

        # loaded filenames
        self.ptsNomFileName = ""
        self.ptsMeasFileName = ""
        self.lookupFileName = ""


class Ui(QtWidgets.QMainWindow):
    def __init__(self, app):
        # Initializing the inherited class
        super(Ui, self).__init__()

        # Load the .ui file
        uic.loadUi('gui.ui', self)

        # Keeping a reference to the app component
        self.app = app

        # creating the event listeners for buttons and other widgets
        self.createEventListeners()

        # Show the screen
        self.show()

    def createEventListeners(self):
        # toolbar actions
        self.actionNominals.triggered.connect(self.loadNominalsFile)
        self.actionMeasured.triggered.connect(self.loadMeasuredFile)
        self.actionLookup.triggered.connect(self.loadLookuptableFile)
        self.actionSaveEnv.triggered.connect(self.saveEnv)
        self.actionLoadEnv.triggered.connect(self.loadEnv)

        # buttons
        self.btnPlotAbsolute.clicked.connect(self.plotAbsolute)
        self.expMachModel.clicked.connect(self.exportMachineModel)

    def exportMachineModel(self):
        distancesNom = Analysis.generateLongitudinalDistances(
            self.app.girderDictNom)

        distancesMeas = Analysis.generateLongitudinalDistances(
            self.app.girderDictMeas)

        DataUtils.writeToExcel(
            "../data/output/distances-measured.xlsx", distancesNom)
        DataUtils.writeToExcel(
            "../data/output/distances-nominal.xlsx", distancesMeas)

    def logMessage(self, message):
        self.logbook.append(message)

    def saveEnv(self):
        with open("config.json", "w") as file:
            config = {'ptsNomFileName': self.app.ptsNomFileName,
                      'ptsMeasFileName': self.app.ptsMeasFileName, 'lookupFileName': self.app.lookupFileName, 'shiftsB1': self.app.shiftsB1}
            json.dump(config, file)

    def loadEnv(self):
        with open("config.json", "r") as file:
            config = json.load(file)

        self.loadNominalsFile(filePath=config['ptsNomFileName'])
        self.loadMeasuredFile(filePath=config['ptsMeasFileName'])
        self.loadLookuptableFile(filePath=config['lookupFileName'])

    def openFileNameDialog(self):
        dialog = QWidget()

        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            dialog, "Selecione o arquivo", "", "Excel Sheet (*.xlsx)", options=options)

        return fileName

    def loadNominalsFile(self, filePath=None):

        filepath = filePath or self.openFileNameDialog()

        if(not filepath):
            return

        rawPtsNominal = DataUtils.readExcel(
            filepath, 'points')
        # gerando grupos de pontos
        self.app.nominals = PointGroup(
            rawPtsNominal, "nominal")

        self.isNominalsLoaded = True
        self.frameNom.setStyleSheet(
            "background-color:green; border-radius:10px;")

        self.ptsNomFileName = filepath

        self.logMessage("Nominal's file loaded")

        self.checkAllFilesAndProcessData()

    def loadMeasuredFile(self, filePath=None):

        filepath = filePath or self.openFileNameDialog()

        if(not filepath):
            return

        rawPtsMeasured = DataUtils.readExcel(
            filepath, 'points')

        self.app.measured = PointGroup(
            rawPtsMeasured, "measured")

        self.isMeasuredLoaded = True
        self.frameMeas.setStyleSheet(
            "background-color:green; border-radius:10px;")

        self.ptsMeasFileName = filepath

        self.logMessage("Measured's file loaded")

        self.checkAllFilesAndProcessData()

    def loadLookuptableFile(self, filePath=None):

        filepath = filePath or self.openFileNameDialog()

        if(not filepath):
            return
        try:
            # carregando dados das planilhas
            self.lookuptable_MLtoLocal = DataUtils.readExcel(
                filepath, 'lookuptable')
        except KeyError:
            # erro quando planilha alvo não é a de índice 0 dentro do arquivo
            print("erro")
            return

        if (self.isMeasuredLoaded):
            self.app.measured.lookuptable = self.lookuptable_MLtoLocal

        if(self.isNominalsLoaded):
            self.app.nominals.lookuptable = self.lookuptable_MLtoLocal

        self.isLookuptableLoaded = True
        self.frameLookup.setStyleSheet(
            "background-color:green; border-radius:10px;")

        self.lookupFileName = filepath

        self.logMessage("Lookuptable's file loaded")

        self.app.processEvents()
        self.checkAllFilesAndProcessData()

    def checkAllFilesAndProcessData(self):
        # se todos os arquivos tiverem sido carregados
        if((self.isNominalsLoaded and self.isMeasuredLoaded and self.isLookuptableLoaded) and not self.isInternalDataProcessed):
            self.logMessage("Transforming points to Local frame...")
            self.processInternalData()
            self.isInternalDataProcessed = True
            self.logMessage("Ploting and exporting ready to go")

    def processInternalData(self):
        # operação sobre pontos carregados
        self.transformPointsToLocalFrame()
        self.generateGirderObjects()
        # self.computeCentroids()

        # habilita botões de plot
        self.btnPlotAbsolute.setEnabled(True)
        self.btnExpAbsolute.setEnabled(True)
        self.expMachModel.setEnabled(True)

    def transformPointsToLocalFrame(self):
        # transformando para frame local
        self.app.nominals.transformToLocalFrame()
        self.app.measured.transformToLocalFrame()

        # print(self.app.nominals.ptList[:50])

    def generateGirderObjects(self):
        # gerando grupos de berços
        self.girderNominal = GirderGroup(self.app.nominals)
        self.girderMeasured = GirderGroup(self.app.measured)

    def computeCentroids(self):
       # computando os centroides
        self.girderNominal.computeCentroids()
        self.girderMeasured.computeCentroids()

    def plotAbsolute(self):
        self.logMessage("Plotting the absolute alignment profile...")
        self.app.processEvents()

        # checando se todos os arquivos foram devidamente carregados
        if(not self.isNominalsLoaded or not self.isMeasuredLoaded or not self.isLookuptableLoaded):
            self.logMessage("error: not all required files were loaded")
            return

        # calculando desvios de todos dof de cada berço com o best-fit de seus pontos
        diffAllDoFs = GirderGroup.evalDiff_bestFit(
            self.girderNominal.pointGroup.ptList, self.girderMeasured.pointGroup.ptList)

        # definindo as propriedades do plot
        plot_args = {'y_list': ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz'], 'title_list': [
            'Transversal', 'Longitudinal', 'Vertical', 'Pitch', 'Roll', 'Yaw'], 'fig_title': 'Global Alignment Profile - Storage Ring'}

        # chamando o plot
        Plot.plotGirderDeviation(diffAllDoFs, 'allDoFs',
                                 plot_args, freezePlot=True)


if __name__ == "__main__":
    App()

""" 
PENDÊNCIAS
- /feito/ correção nomenclatura
- /feito/ correção sinal bestgit
- /feito/padronização do título e eixos
- plot de tolerancias (especificá-las)
- [modelo liu - long.] avaliação de distancias entre dipolos e quadrupolos 
- [modelo liu - vert.] avaliação individual de cada quadrupolo e dipolo
- estimar série de fourier
- substrair para obter alinhamento relativo
"""
