# imports de bibliotecas
from accelerators import (Booster, SR)
from dataUtils import (DataUtils)

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import (QFileDialog, QWidget)
from PyQt5.QtGui import (QColor)
import sys
import json
from functools import partial


class App(QtWidgets.QApplication):
    def __init__(self):
        # Initializing the application and its UI
        super(App, self).__init__(sys.argv)

        self.initAppVariables()

        # Calling the UI component
        self.ui = Ui(self)

        # Changuing the default window theme
        # self.setStyle('Fusion')

        # Calling the app to execute
        sys.exit(self.exec_())

    def initAppVariables(self):
        self.lookupTable = {}
        self.measured = {}
        self.nominals = {}
        self.entitiesDictMeas = {}
        self.entitiesDictNom = {}
        self.frameDict = {}

        self.shiftsB1 = None

        # flags
        self.isNominalsLoaded = {"SR": False, "booster": False}
        self.isMeasuredLoaded = {"SR": False, "booster": False}
        self.isLookuptableLoaded = {"SR": False, "booster": False}
        self.isInternalDataProcessed = {"SR": False, "booster": False}

        # loaded filenames
        self.ptsNomFileName = {}
        self.ptsMeasFileName = {}
        self.lookupFileName = {}


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
        self.actionNom_SR.triggered.connect(partial(self.loadNominalsFile, 'SR'))
        self.actionNom_booster.triggered.connect(partial(self.loadNominalsFile, 'booster'))
        
        self.actionMeas_SR.triggered.connect(partial(self.loadMeasuredFile, 'SR'))
        self.actionMeas_booster.triggered.connect(partial(self.loadMeasuredFile, 'booster'))

        self.actionFrames_SR.triggered.connect(partial(self.loadLookuptableFile, 'SR'))
        self.actionFrames_booster.triggered.connect(partial(self.loadLookuptableFile, 'booster'))

        self.actionSaveEnv.triggered.connect(self.saveEnv)
        self.actionLoadEnv.triggered.connect(self.loadEnv)

        # buttons
        self.plotAbsolute_SR.clicked.connect(partial(self.plotAbsolute, 'SR'))
        self.plotAbsolute_booster.clicked.connect(partial(self.plotAbsolute, 'booster'))
        
        # self.expMachModel.clicked.connect(self.exportMachineModel)

    # def exportMachineModel(self):
    #     distancesNom = Analysis.generateLongitudinalDistances(
    #         self.app.entitiesDictNom)

    #     distancesMeas = Analysis.generateLongitudinalDistances(
    #         self.app.entitiesDictMeas)

    #     DataUtils.writeToExcel(
    #         "../data/output/distances-measured.xlsx", distancesNom)
    #     DataUtils.writeToExcel(
    #         "../data/output/distances-nominal.xlsx", distancesMeas)

    def logMessage(self, message, severity='normal'):
        
        if (severity != 'normal'):

            if (severity == 'danger'):
                fontWeight = 63
                color = 'red'
            elif (severity == 'alert'):
                fontWeight = 57
                color = 'yellow'
            elif (severity == 'sucess'):
                fontWeight = 57
                color = 'green'

            # saving properties
            fw = self.logbook.fontWeight()
            tc = self.logbook.textColor()

            self.logbook.setFontWeight(fontWeight)
            self.logbook.setTextColor(QColor(color))
            self.logbook.append(message)

            self.logbook.setFontWeight(fw)
            self.logbook.setTextColor(tc)
        else:
            self.logbook.append(message)

        

    def saveEnv(self):
        with open("config.json", "w") as file:
            config = {'ptsNomFileName': {"booster": self.app.ptsNomFileName['booster'], "SR": self.app.ptsNomFileName['SR']},
                      'ptsMeasFileName': {"booster": self.app.ptsMeasFileName['booster'], "SR": self.app.ptsMeasFileName['SR']},
                      'lookupTable': {"booster": self.app.lookupTable['booster'], "SR": self.app.lookupTable['SR']},
                      'shiftsB1': self.app.shiftsB1}
            json.dump(config, file)

    def loadEnv(self):
        with open("config.json", "r") as file:
            config = json.load(file)

        self.loadNominalsFile(
            'booster', filePath=config['ptsNomFileName']['booster'])
        self.loadNominalsFile('SR', filePath=config['ptsNomFileName']['SR'])

        self.loadMeasuredFile(
            'booster', filePath=config['ptsMeasFileName']['booster'])
        self.loadMeasuredFile('SR', filePath=config['ptsMeasFileName']['SR'])

        self.loadLookuptableFile(
            'booster', filePath=config['lookupFileName']['booster'])
        self.loadLookuptableFile('SR', filePath=config['lookupFileName']['SR'])

    def openFileNameDialog(self, typeOfFile, accelerator):
        dialog = QWidget()

        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            dialog, "Selecione o arquivo - " + typeOfFile + " - " + accelerator, "", filter="All files (*)", options=options)

        return fileName

    def loadNominalsFile(self, accelerator, filePath=None):

        filepath = filePath or self.openFileNameDialog('pontos nominais', accelerator)

        if(not filepath):
            return

        fileName = filepath.split('/')[len(filepath.split('/')) - 1]
        fileExtension = fileName.split('.')[1]

        if (fileExtension == 'txt' or fileExtension == 'csv' or fileExtension == 'TXT' or fileExtension == 'CSV'):
            nominals = DataUtils.readCSV(filepath, 'points')
        elif (fileExtension == 'xlsx'):
            nominals = DataUtils.readExcel(
                filepath, 'points')
        else:
            self.logMessage("Formato do arquivo "+fileName+" não suportado.", severity="alert")
            return

        # gerando grupos de pontos
        self.app.nominals[accelerator] = nominals

        self.app.isNominalsLoaded[accelerator] = True
        
        if (accelerator == 'SR'):
            self.ledNom_SR.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileNom_SR.setText(fileName)
            self.fileNom_SR.setToolTip(fileName)
        elif (accelerator == 'booster'):
            self.ledNom_booster.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileNom_booster.setText(fileName)
            self.fileNom_booster.setToolTip(fileName)



        self.app.ptsNomFileName[accelerator] = filepath

        self.logMessage("Arquivo nominal do "+accelerator+" carregado.")

        self.checkAllFilesAndProcessData(accelerator)

    def loadMeasuredFile(self, accelerator, filePath=None):

        filepath = filePath or self.openFileNameDialog('pontos medidos', accelerator)

        if(not filepath):
            return

        fileName = filepath.split('/')[len(filepath.split('/')) - 1]
        fileExtension = fileName.split('.')[1]

        if (fileExtension == 'txt' or fileExtension == 'csv' or fileExtension == 'TXT' or fileExtension == 'CSV'):
            measured = DataUtils.readCSV(filepath, 'points')
        elif (fileExtension == 'xlsx'):
            measured = DataUtils.readExcel(
                filepath, 'points')
        else:
            self.logMessage("Formato do arquivo "+fileName+" não suportado.", severity="alert")
            return

        self.app.measured[accelerator] = measured

        self.app.isMeasuredLoaded[accelerator] = True
        
        if (accelerator == 'SR'):
            self.ledMeas_SR.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileMeas_SR.setText(fileName)
            self.fileMeas_SR.setToolTip(fileName)
        elif (accelerator == 'booster'):
            self.ledMeas_booster.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileMeas_booster.setText(fileName)
            self.fileMeas_booster.setToolTip(fileName)

        self.app.ptsMeasFileName[accelerator] = filepath

        self.logMessage("Arquivo com medidos do "+accelerator+" carregado.")

        self.checkAllFilesAndProcessData(accelerator)

    def loadLookuptableFile(self, accelerator, filePath=None):

        filepath = filePath or self.openFileNameDialog('lookup de frames', accelerator)

        if(not filepath):
            return

        fileName = filepath.split('/')[len(filepath.split('/')) - 1]
        fileExtension = fileName.split('.')[1]

        if (fileExtension == 'txt' or fileExtension == 'csv' or fileExtension == 'TXT' or fileExtension == 'CSV'):
            lookuptable = DataUtils.readCSV(filepath, 'lookuptable')
        elif (fileExtension == 'xlsx'):
            lookuptable = DataUtils.readExcel(
                filepath, 'lookuptable')
        else:
            self.logMessage("Format of file "+fileName+"not supported.")
            return

        self.app.lookupTable[accelerator] = lookuptable

        self.app.isLookuptableLoaded[accelerator] = True
        
        if (accelerator == 'SR'):
            self.ledFrames_SR.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileFrames_SR.setText(fileName)
            self.fileFrames_SR.setToolTip(fileName)
        elif (accelerator == 'booster'):
            self.ledFrames_booster.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileFrames_booster.setText(fileName)
            self.fileFrames_booster.setToolTip(fileName)

        self.app.lookupFileName[accelerator] = filepath

        self.logMessage("Lookuptable de frames do " + accelerator + " carregada.")

        self.checkAllFilesAndProcessData(accelerator)

    def checkAllFilesAndProcessData(self, accelerator):
        # se todos os arquivos tiverem sido carregados
        if((self.app.isNominalsLoaded[accelerator] and self.app.isMeasuredLoaded[accelerator] and self.app.isLookuptableLoaded[accelerator]) and not self.app.isInternalDataProcessed[accelerator]):
            self.logMessage("Transformando pontos do " + accelerator + " para os frames locais...")
            self.app.processEvents()
            self.processInternalData(accelerator)
            self.app.isInternalDataProcessed[accelerator] = True
            self.logMessage("Dados do "+accelerator+" prontos.", severity="sucess")

    def processInternalData(self, accelerator):
        # creating frame's dict for the particular accelerator
        self.app.frameDict[accelerator] = DataUtils.generateFrameDict(
            self.app.lookupTable[accelerator])

        if (accelerator == 'booster'):
            # try:
            #     entitiesDictNom = Booster.createObjectsStructure(self.app.nominals['booster'])
            # except IndexError:
            #     self.logMessage('Erro ao carregar pontos nominais do booster. Checar se todos os pontos seguem a nomenclatura correta.', severity='danger')
            #     self.ledNom_booster.setStyleSheet("background-color:yellow; border-radius:7px;")
            #     return
            entitiesDictNom = Booster.createObjectsStructure(self.app.nominals['booster'])
            entitiesDictMeas = Booster.createObjectsStructure(self.app.measured['booster'])
        elif (accelerator == 'SR'):
            entitiesDictNom = SR.createObjectsStructure(
                self.app.nominals['SR'])
            entitiesDictMeas = SR.createObjectsStructure(
                self.app.measured['SR'])

        self.app.entitiesDictNom[accelerator] = entitiesDictNom
        self.app.entitiesDictMeas[accelerator] = entitiesDictMeas

        DataUtils.transformToLocalFrame(
            self.app.entitiesDictNom[accelerator], self.app.frameDict[accelerator], accelerator)
        DataUtils.transformToLocalFrame(
            self.app.entitiesDictMeas[accelerator], self.app.frameDict[accelerator], accelerator)

        self.generateMeasuredFrames(accelerator)

        self.app.frameDict[accelerator] = DataUtils.sortFrameDictByBeamTrajectory(
            self.app.frameDict[accelerator], accelerator)

        # habilita botões de plot
        if(accelerator == 'SR'):
            self.plotAbsolute_SR.setEnabled(True)
            self.expAbsolute_SR.setEnabled(True)
            self.expMachModel_SR.setEnabled(True)
        elif (accelerator == 'booster'):
            self.plotAbsolute_booster.setEnabled(True)
            self.expAbsolute_booster.setEnabled(True)

    def generateMeasuredFrames(self, accelerator):
        if (accelerator == 'SR'):
            DataUtils.generateMeasuredFrames(
                'quadrupole', self.app.entitiesDictMeas[accelerator], self.app.entitiesDictNom[accelerator], self.app.frameDict[accelerator], 'SR', self)
            DataUtils.generateMeasuredFrames(
                'dipole-B1', self.app.entitiesDictMeas[accelerator], self.app.entitiesDictNom[accelerator], self.app.frameDict[accelerator], 'SR', self)
            DataUtils.generateMeasuredFrames(
                'dipole-B2', self.app.entitiesDictMeas[accelerator], self.app.entitiesDictNom[accelerator], self.app.frameDict[accelerator], 'SR', self)
            DataUtils.generateMeasuredFrames(
                'dipole-BC', self.app.entitiesDictMeas[accelerator], self.app.entitiesDictNom[accelerator], self.app.frameDict[accelerator], 'SR', self)
            return

        if (accelerator == 'booster'):
            DataUtils.generateMeasuredFrames(
                'quadrupole', self.app.entitiesDictMeas[accelerator], self.app.entitiesDictNom[accelerator], self.app.frameDict[accelerator], 'booster', self)
            DataUtils.generateMeasuredFrames(
                'dipole-booster', self.app.entitiesDictMeas[accelerator], self.app.entitiesDictNom[accelerator], self.app.frameDict[accelerator], 'booster', self)
            return

    def plotAbsolute(self, accelerator):
        self.logMessage(
            "Plotando o perfil de alinhamento absoluto do "+accelerator+"...")
        self.app.processEvents()

        # checando se todos os arquivos foram devidamente carregados
        if(not self.app.isNominalsLoaded[accelerator] or not self.app.isMeasuredLoaded[accelerator] or not self.app.isLookuptableLoaded[accelerator]):
            self.logMessage(
                "Erro ao plotar gráfico: nem todos os arquivos foram carregados")
            return

        magnetsDeviations = DataUtils.calculateMagnetsDeviations(
            self.app.frameDict[accelerator])

        DataUtils.plotDevitationData(magnetsDeviations, accelerator)


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
