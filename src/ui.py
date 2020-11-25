from pointGroup import PointGroup
from girderGroup import GirderGroup
from dataOperator import DataOperator
from facModel import FacModel

from plot import Plot

# imports de bibliotecas
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import (QFileDialog, QWidget)
import sys
import json


class Ui(QtWidgets.QMainWindow):
    def __init__(self, app):
        # Initializing the inherited class
        super(Ui, self).__init__()

        # Load the .ui file
        uic.loadUi('gui.ui', self)

        # Keeping a reference to the app component
        self.app = app

        # init app variables
        self.initAppVariables()

        # creating the event listeners for buttons and other widgets
        self.createEventListeners()

        # Show the screen
        self.show()

    def initAppVariables(self):
        # variables to be shared between internal methods
        self.ptsNominal = None
        self.ptsMeasured = None
        self.girderNominal = None
        self.girderMeasured = None

        # flags
        self.isNominalsLoaded = False
        self.isMeasuredLoaded = False
        self.isLookuptableLoaded = False
        self.isInternalDataProcessed = False

        # loaded filenames
        self.ptsNomFileName = ""
        self.ptsMeasFileName = ""
        self.lookupFileName = ""

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
        fac = FacModel(self.girderMeasured, self.girderNominal)
        distMeas = fac.generateLongitudinalDistances('measured')
        distNom = fac.generateLongitudinalDistances('nominal')

        DataOperator.saveToExcel(
            "../data/output/distances-measured.xlsx", distMeas)
        DataOperator.saveToExcel(
            "../data/output/distances-nominal.xlsx", distNom)

    def logMessage(self, message):
        self.logbook.append(message)

    def saveEnv(self):
        with open("config.json", "w") as file:
            config = {'ptsNomFileName': self.ptsNomFileName,
                      'ptsMeasFileName': self.ptsMeasFileName, 'lookupFileName': self.lookupFileName}
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

        rawPtsNominal = DataOperator.loadFromExcel(
            filepath, 0, has_header=False)
        # gerando grupos de pontos
        self.ptsNominal = PointGroup(
            rawPtsNominal, "nominal")

        if(self.isLookuptableLoaded):
            self.ptsNominal.lookuptable = self.lookuptable_MLtoLocal

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

        rawPtsMeasured = DataOperator.loadFromExcel(
            filepath, 0, has_header=False)

        self.ptsMeasured = PointGroup(
            rawPtsMeasured, "measured")

        if(self.isLookuptableLoaded):
            self.ptsMeasured.lookuptable = self.lookuptable_MLtoLocal

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
            self.lookuptable_MLtoLocal = DataOperator.loadFromExcel(
                filepath, 0, has_header=True)
        except KeyError:
            # erro quando planilha alvo não é a de índice 0 dentro do arquivo
            print("erro")
            return

        if (self.isMeasuredLoaded):
            self.ptsMeasured.lookuptable = self.lookuptable_MLtoLocal

        if(self.isNominalsLoaded):
            self.ptsNominal.lookuptable = self.lookuptable_MLtoLocal

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
        self.ptsNominal.transformToLocalFrame()
        self.ptsMeasured.transformToLocalFrame()

        # print(self.ptsNominal.ptList[:50])

    def generateGirderObjects(self):
        # gerando grupos de berços
        self.girderNominal = GirderGroup(self.ptsNominal)
        self.girderMeasured = GirderGroup(self.ptsMeasured)

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
