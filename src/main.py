# imports de bibliotecas
from accelerators import (Booster, SR, LTB, BTS, FE)
from dataUtils import (DataUtils)

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import (QFileDialog, QWidget, QMessageBox, QDialog, QDialogButtonBox, QVBoxLayout, QTextEdit)
from PyQt5.QtGui import (QColor)
import sys
import json
import pandas as pd
from functools import partial

# globals
DEBUG_MODE = False


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
        self.isNominalsLoaded = {"SR": False, "booster": False, "LTB": False, "BTS": False, "FE": False}
        self.isMeasuredLoaded = {"SR": False, "booster": False, "LTB": False, "BTS": False, "FE": False}
        self.isLookuptableLoaded = {"SR": False, "booster": False, "LTB": False, "BTS": False, "FE": False}
        self.isInternalDataProcessed = {"SR": False, "booster": False, "LTB": False, "BTS": False, "FE": False}

        # loaded filenames
        self.ptsNomFileName = {}
        self.ptsMeasFileName = {}
        self.lookupFileName = {}

        # report info
        self.report = {}


class Ui(QtWidgets.QMainWindow):
    def __init__(self, app):
        # Initializing the inherited class
        super(Ui, self).__init__()

        # Load the .ui file
        uic.loadUi('gui.ui', self)

        # Keeping a reference to the app component
        self.app = app

        # Check the mode of the session: DEBUG_MODE => run specific functions to test something
        if(DEBUG_MODE):
            self.runDebugFunctions()
            return

        # creating the event listeners for buttons and other widgets
        self.createEventListeners()

        # Show the screen
        self.show()
        
    def createEventListeners(self):
        # toolbar actions
        self.actionNom_SR.triggered.connect(partial(self.loadNominalsFile, 'SR'))
        self.actionNom_booster.triggered.connect(partial(self.loadNominalsFile, 'booster'))
        self.actionNom_LTB.triggered.connect(partial(self.loadNominalsFile, 'LTB'))
        self.actionNom_BTS.triggered.connect(partial(self.loadNominalsFile, 'BTS'))
        self.actionNom_FE.triggered.connect(partial(self.loadNominalsFile, 'FE'))
        
        self.actionMeas_SR.triggered.connect(partial(self.loadMeasuredFile, 'SR'))
        self.actionMeas_booster.triggered.connect(partial(self.loadMeasuredFile, 'booster'))
        self.actionMeas_LTB.triggered.connect(partial(self.loadMeasuredFile, 'LTB'))
        self.actionMeas_BTS.triggered.connect(partial(self.loadMeasuredFile, 'BTS'))
        self.actionMeas_FE.triggered.connect(partial(self.loadMeasuredFile, 'FE'))

        self.actionFrames_SR.triggered.connect(partial(self.loadLookuptableFile, 'SR'))
        self.actionFrames_booster.triggered.connect(partial(self.loadLookuptableFile, 'booster'))
        self.actionFrames_LTB.triggered.connect(partial(self.loadLookuptableFile, 'LTB'))
        self.actionFrames_BTS.triggered.connect(partial(self.loadLookuptableFile, 'BTS'))
        self.actionFrames_FE.triggered.connect(partial(self.loadLookuptableFile, 'FE'))

        self.actionSaveEnv.triggered.connect(self.saveEnv)
        self.actionLoadEnv.triggered.connect(self.loadEnv)

        # plot buttons
        self.plotAbsolute_SR.clicked.connect(partial(self.plotAbsolute, 'SR'))
        self.plotAbsolute_booster.clicked.connect(partial(self.plotAbsolute, 'booster'))
        self.plotAbsolute_LTB.clicked.connect(partial(self.plotAbsolute, 'LTB'))
        self.plotAbsolute_BTS.clicked.connect(partial(self.plotAbsolute, 'BTS'))
        self.plotAbsolute_FE.clicked.connect(partial(self.plotAbsolute, 'FE'))
        self.plotComparative.clicked.connect(self.plotComparativeDev)

        # report buttons
        self.reportProgress_booster.clicked.connect(partial(self.showReportPopup, 'booster'))
        self.reportProgress_SR.clicked.connect(partial(self.showReportPopup, 'SR'))
        
        self.expLongDist_SR.clicked.connect(partial(self.exportLongitudinalDistances, 'SR'))
        self.expAbsolute_SR.clicked.connect(partial(self.exportAbsoluteDeviations, 'SR'))
        # DataUtils.debug_translatePoints(self.app.entitiesDictNom['SR'], self.app.frameDict['SR'])

    def runDebugFunctions(self):
        self.loadEnv()

        # creating frame's dict for the particular accelerator
        self.app.frameDict['SR'] = DataUtils.generateFrameDict(self.app.lookupTable['SR'])

        entitiesDictNom = SR.createObjectsStructure(self.app.nominals['SR'])

        self.app.entitiesDictNom['SR'] = entitiesDictNom

        DataUtils.debug_transformToLocalFrame(
            self.app.entitiesDictNom['SR'], self.app.frameDict['SR'], 'SR')

    def exportLongitudinalDistances(self, accelerator):
        longDistMeas = DataUtils.calculateMagnetsDistances(self.app.frameDict[accelerator], 'measured')
        longDistNom = DataUtils.calculateMagnetsDistances(self.app.frameDict[accelerator], 'nominal')

        DataUtils.writeToExcel("../data/output/distances-measured.xlsx", longDistMeas, accelerator)
        DataUtils.writeToExcel("../data/output/distances-nominal.xlsx", longDistNom, accelerator)

    def exportAbsoluteDeviations(self, accelerator):
        # checando se todos os arquivos foram devidamente carregados
        if(not self.app.isNominalsLoaded[accelerator] or not self.app.isMeasuredLoaded[accelerator] or not self.app.isLookuptableLoaded[accelerator]):
            self.logMessage("Erro ao plotar gráfico: nem todos os arquivos foram carregados")
            return

        magnetsDeviations = DataUtils.calculateMagnetsDeviations(self.app.frameDict[accelerator])
        DataUtils.writeToExcel(f"../data/output/deviations-{accelerator}.xlsx", magnetsDeviations, accelerator)

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
            config = {'ptsNomFileName': {"booster": self.app.ptsNomFileName['booster'], "SR": self.app.ptsNomFileName['SR'], "LTB": self.app.ptsNomFileName['LTB'], "BTS": self.app.ptsNomFileName['BTS'], "FE": self.app.ptsNomFileName['FE']},
                      'ptsMeasFileName': {"booster": self.app.ptsMeasFileName['booster'], "SR": self.app.ptsMeasFileName['SR'], "LTB": self.app.ptsMeasFileName['LTB'], "BTS": self.app.ptsMeasFileName['BTS'], "FE": self.app.ptsMeasFileName['FE']},
                      'lookupTable': {"booster": self.app.lookupTable['booster'], "SR": self.app.lookupTable['SR'], "LTB": self.app.lookupTable['LTB'], "BTS": self.app.lookupTable['BTS'], "FE": self.app.lookupTable['FE']},
                      'shiftsB1': self.app.shiftsB1}
            json.dump(config, file)

    def loadEnv(self):
        with open("config.json", "r") as file:
            config = json.load(file)

        self.app.shiftsB1 = config['shiftsB1']

        self.loadNominalsFile('booster', filePath=config['ptsNomFileName']['booster'])
        self.loadNominalsFile('SR', filePath=config['ptsNomFileName']['SR'])
        self.loadNominalsFile('LTB', filePath=config['ptsNomFileName']['LTB'])
        self.loadNominalsFile('BTS', filePath=config['ptsNomFileName']['BTS'])
        self.loadNominalsFile('FE', filePath=config['ptsNomFileName']['FE'])

        self.loadMeasuredFile('booster', filePath=config['ptsMeasFileName']['booster'])
        self.loadMeasuredFile('SR', filePath=config['ptsMeasFileName']['SR'])
        self.loadMeasuredFile('LTB', filePath=config['ptsMeasFileName']['LTB'])
        self.loadMeasuredFile('BTS', filePath=config['ptsMeasFileName']['BTS'])
        self.loadMeasuredFile('FE', filePath=config['ptsMeasFileName']['FE'])

        self.loadLookuptableFile('booster', filePath=config['lookupFileName']['booster'])
        self.loadLookuptableFile('SR', filePath=config['lookupFileName']['SR'])
        self.loadLookuptableFile('LTB', filePath=config['lookupFileName']['LTB'])
        self.loadLookuptableFile('BTS', filePath=config['lookupFileName']['BTS'])
        self.loadLookuptableFile('FE', filePath=config['lookupFileName']['FE'])

    def openFileNameDialog(self, typeOfFile, accelerator):
        dialog = QWidget()

        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            dialog, "Selecione o arquivo - " + typeOfFile + " - " + accelerator, "", filter="All files (*)", options=options)

        return fileName

    def loadNominalsFile(self, accelerator, filePath=None):

        if(filePath == ""):
            return

        filepath = filePath or self.openFileNameDialog('pontos nominais', accelerator)

        if(not filepath or filePath == ""):
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
        elif (accelerator == 'LTB'):
            self.ledNom_LTB.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileNom_LTB.setText(fileName)
            self.fileNom_LTB.setToolTip(fileName)
        elif (accelerator == 'BTS'):
            self.ledNom_BTS.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileNom_BTS.setText(fileName)
            self.fileNom_BTS.setToolTip(fileName)
        elif (accelerator == 'FE'):
            self.ledNom_FE.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileNom_FE.setText(fileName)
            self.fileNom_FE.setToolTip(fileName)

        self.app.ptsNomFileName[accelerator] = filepath

        self.logMessage("Arquivo nominal do "+accelerator+" carregado.")

        self.checkAllFilesAndProcessData(accelerator)

    def loadMeasuredFile(self, accelerator, filePath=None):
        
        if(filePath == ""):
            return

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
        elif (accelerator == 'LTB'):
            self.ledMeas_LTB.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileMeas_LTB.setText(fileName)
            self.fileMeas_LTB.setToolTip(fileName)
        elif (accelerator == 'BTS'):
            self.ledMeas_BTS.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileMeas_BTS.setText(fileName)
            self.fileMeas_BTS.setToolTip(fileName)
        elif (accelerator == 'FE'):
            self.ledMeas_FE.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileMeas_FE.setText(fileName)
            self.fileMeas_FE.setToolTip(fileName)

        self.app.ptsMeasFileName[accelerator] = filepath

        self.logMessage("Arquivo com medidos do "+accelerator+" carregado.")

        self.checkAllFilesAndProcessData(accelerator)

    def loadLookuptableFile(self, accelerator, filePath=None):
        
        if(filePath == ""):
            return

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
        elif (accelerator == 'LTB'):
            self.ledFrames_LTB.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileFrames_LTB.setText(fileName)
            self.fileFrames_LTB.setToolTip(fileName)
        elif (accelerator == 'BTS'):
            self.ledFrames_BTS.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileFrames_BTS.setText(fileName)
            self.fileFrames_BTS.setToolTip(fileName)
        elif (accelerator == 'FE'):
            self.ledFrames_FE.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileFrames_FE.setText(fileName)
            self.fileFrames_FE.setToolTip(fileName)

        self.app.lookupFileName[accelerator] = filepath

        self.logMessage("Lookuptable de frames do " + accelerator + " carregada.")

        self.checkAllFilesAndProcessData(accelerator)

    def checkAllFilesAndProcessData(self, accelerator):
        # se todos os arquivos tiverem sido carregados
        if((self.app.isNominalsLoaded[accelerator] and self.app.isMeasuredLoaded[accelerator] and self.app.isLookuptableLoaded[accelerator])):
            self.logMessage("Transformando pontos do " + accelerator + " para os frames locais...")
            self.app.processEvents()
            self.processInternalData(accelerator)
            self.app.isInternalDataProcessed[accelerator] = True
            self.logMessage("Dados do "+accelerator+" prontos.", severity="sucess")

    def processInternalData(self, accelerator):
        # creating frame's dict for the particular accelerator
        self.app.frameDict[accelerator] = DataUtils.generateFrameDict(self.app.lookupTable[accelerator])

        if (accelerator == 'booster'):
            entitiesDictNom = Booster.createObjectsStructure(self.app.nominals['booster'])
            entitiesDictMeas = Booster.createObjectsStructure(self.app.measured['booster'])
        elif (accelerator == 'SR'):
            entitiesDictNom = SR.createObjectsStructure(self.app.nominals['SR'])
            entitiesDictMeas = SR.createObjectsStructure(self.app.measured['SR'], shiftsB1=self.app.shiftsB1)
        elif (accelerator == 'LTB'):
            entitiesDictNom = LTB.createObjectsStructure(self.app.nominals['LTB'])
            entitiesDictMeas = LTB.createObjectsStructure(self.app.measured['LTB'])
        elif (accelerator == 'BTS'):
            entitiesDictNom = BTS.createObjectsStructure(self.app.nominals['BTS'])
            entitiesDictMeas = BTS.createObjectsStructure(self.app.measured['BTS'])
        elif (accelerator == 'FE'):
            entitiesDictNom = FE.createObjectsStructure(self.app.nominals['FE'])
            entitiesDictMeas = FE.createObjectsStructure(self.app.measured['FE'])

        self.app.entitiesDictNom[accelerator] = entitiesDictNom
        self.app.entitiesDictMeas[accelerator] = entitiesDictMeas

        DataUtils.transformToLocalFrame(self.app.entitiesDictNom[accelerator], self.app.frameDict[accelerator], accelerator)
        DataUtils.transformToLocalFrame(self.app.entitiesDictMeas[accelerator], self.app.frameDict[accelerator], accelerator)

        self.generateMeasuredFrames(accelerator)

        self.app.frameDict[accelerator] = DataUtils.sortFrameDictByBeamTrajectory(self.app.frameDict[accelerator], accelerator)

        self.app.report[accelerator] = DataUtils.generateReport(self.app.frameDict[accelerator], accelerator)

        # DataUtils.printFrameDict(self.app.frameDict[accelerator])
        DataUtils.printDictData(self.app.entitiesDictMeas[accelerator])

        # habilita botões de plot
        if(accelerator == 'SR'):
            self.plotAbsolute_SR.setEnabled(True)
            self.expAbsolute_SR.setEnabled(True)
            # self.expMachModel_SR.setEnabled(True)
            self.reportProgress_SR.setEnabled(True)
            self.expLongDist_SR.setEnabled(True)
        elif (accelerator == 'booster'):
            self.plotAbsolute_booster.setEnabled(True)
            self.expAbsolute_booster.setEnabled(True)
            self.reportProgress_booster.setEnabled(True)
        elif (accelerator == 'LTB'):
            self.plotAbsolute_LTB.setEnabled(True)
            self.expAbsolute_LTB.setEnabled(True)
        elif (accelerator == 'BTS'):
            self.plotAbsolute_BTS.setEnabled(True)
            self.expAbsolute_BTS.setEnabled(True)
        elif (accelerator == 'FE'):
            self.plotAbsolute_FE.setEnabled(True)
            self.expAbsolute_FE.setEnabled(True)

    def generateMeasuredFrames(self, accelerator):
        if (accelerator == 'SR'):
            typesOfMagnets = ['quadrupole', 'dipole-B1', 'dipole-B2', 'dipole-BC']
            for magnetType in typesOfMagnets:
                DataUtils.generateMeasuredFrames(self.app.entitiesDictMeas[accelerator], self.app.entitiesDictNom[accelerator],\
                                                self.app.frameDict[accelerator], 'SR', self, typeOfMagnets=magnetType, isTypeOfMagnetsIgnored=False)
        else:
            DataUtils.generateMeasuredFrames(self.app.entitiesDictMeas[accelerator], self.app.entitiesDictNom[accelerator],\
                                             self.app.frameDict[accelerator], accelerator, self)
        return
        
        

    def plotAbsolute(self, accelerator):
        self.logMessage("Plotando o perfil de alinhamento absoluto do "+accelerator+"...")
        self.app.processEvents()

        # checando se todos os arquivos foram devidamente carregados
        if(not self.app.isNominalsLoaded[accelerator] or not self.app.isMeasuredLoaded[accelerator] or not self.app.isLookuptableLoaded[accelerator]):
            self.logMessage("Erro ao plotar gráfico: nem todos os arquivos foram carregados")
            return

        magnetsDeviations = []

        if (accelerator == 'LTB' or accelerator == 'BTS'):
            accelerator = 'transport-lines'
            magnetsDeviations.append(DataUtils.calculateMagnetsDeviations(self.app.frameDict['LTB']))
            magnetsDeviations.append(DataUtils.calculateMagnetsDeviations(self.app.frameDict['BTS']))
        elif (accelerator == 'FE'):
            magnetsDeviations.append(DataUtils.separateFEsData(DataUtils.calculateMagnetsDeviations(self.app.frameDict['FE']), ['MAN', 'EMA', 'CAT']))
            magnetsDeviations.append(DataUtils.separateFEsData(DataUtils.calculateMagnetsDeviations(self.app.frameDict['FE']), ['IPE', 'CAR', 'MOG']))

        else:
            magnetsDeviations.append(DataUtils.calculateMagnetsDeviations(self.app.frameDict[accelerator]))
            if (accelerator == 'SR'):
                # appending in df B1's transversal deviations
                # magnetsDeviations[0] = SR.calcB1PerpendicularTranslations(magnetsDeviations[0], self.app.entitiesDictMeas['SR'], self.app.entitiesDictNom['SR'], self.app.frameDict['SR'])
                # magnetsDeviations[0] = SR.addTranslationsFromMagnetMeasurements(magnetsDeviations[0])
                pass

        plotArgs = {'accelerator': accelerator, 'filtering':\
            {'SR': {'allDofs':  self.plotAllDofs_SR.isChecked(), 'lineWithinGirder': self.lineWithinGirder_SR.isChecked(), 'reportMode': self.reportMode_SR.isChecked(), 'errorbar': self.errorbar_SR.isChecked()},\
            'booster': {'quad02': self.plotQuad02_booster.isChecked(), 'lineWithinGirder': False, 'reportMode': self.reportMode_booster.isChecked(), 'errorbar': self.errorbar_booster.isChecked()},\
            'LTB': {'lineWithinGirder': False, 'reportMode': False, 'errorbar': self.errorbar_LTB.isChecked()},\
            'BTS': {'lineWithinGirder': False, 'reportMode': False, 'errorbar': self.errorbar_BTS.isChecked()},\
            'FE': {'lineWithinGirder': False, 'reportMode': False, 'errorbar': self.errorbar_FE.isChecked()}}}

        

        if (accelerator == 'FE'):
            plotArgs['FElist'] = ['MANACÁ', 'EMA', 'CATERETÊ']
            DataUtils.plotDevitationData(magnetsDeviations[0], **plotArgs)
            plotArgs['FElist'] = ['IPÊ', 'CARNAÚBA', 'MOGNO']
            DataUtils.plotDevitationData(magnetsDeviations[1], **plotArgs)
        else:
            DataUtils.plotDevitationData(magnetsDeviations, **plotArgs)

    def plotComparativeDev(self):
        self.logMessage("Plotando o perfil de alinhamento de todo os aceleradores...")
        self.app.processEvents()

        magnetsDeviations_LTB = DataUtils.calculateMagnetsDeviations(self.app.frameDict['LTB'])
        magnetsDeviations_booster = DataUtils.calculateMagnetsDeviations(self.app.frameDict['booster'])
        magnetsDeviations_BTS = DataUtils.calculateMagnetsDeviations(self.app.frameDict['BTS'])
        magnetsDeviations_SR = DataUtils.calculateMagnetsDeviations(self.app.frameDict['SR'])
        deviation = [magnetsDeviations_LTB, magnetsDeviations_booster, magnetsDeviations_BTS, magnetsDeviations_SR]

        sizes = [magnetsDeviations_LTB.iloc[:, 0].size, magnetsDeviations_booster.iloc[:, 0].size, magnetsDeviations_BTS.iloc[:, 0].size]
        lenghts = [sizes[0], sizes[1] + sizes[0], sizes[2] + sizes[1] + sizes[0]]

        deviations = pd.concat(deviation)

        plotArgs = {'accelerator': 'SR', 'len': lenghts, 'filtering':\
            {'SR': {'allDofs':  self.plotAllDofs_SR.isChecked(), 'lineWithinGirder': self.lineWithinGirder_SR.isChecked(), 'reportMode': self.reportMode_SR.isChecked(), 'errorbar': self.errorbar_SR.isChecked()},\
            'booster': {'quad02': self.plotQuad02_booster.isChecked(), 'lineWithinGirder': False, 'reportMode': self.reportMode_booster.isChecked(), 'errorbar': self.errorbar_booster.isChecked()},\
            'LTB': {'lineWithinGirder': False, 'reportMode': False, 'errorbar': self.errorbar_LTB.isChecked()},\
            'BTS': {'lineWithinGirder': False, 'reportMode': False, 'errorbar': self.errorbar_BTS.isChecked()}}}

        DataUtils.plotComparativeDeviation(deviations, **plotArgs)

    def plotRelative(self, accelerator):
        self.logMessage("Plotando o perfil de alinhamento relativo do "+accelerator+"...")
        self.app.processEvents()

        # checando se todos os arquivos foram devidamente carregados
        if(not self.app.isNominalsLoaded[accelerator] or not self.app.isMeasuredLoaded[accelerator] or not self.app.isLookuptableLoaded[accelerator]):
            self.logMessage("Erro ao plotar gráfico: nem todos os arquivos foram carregados")
            return


        magnetsDeviations = DataUtils.calculateMagnetsRelativeDeviations(
            self.app.frameDict[accelerator])

        # DataUtils.writeToExcel('../data/output/sr-devs.xlsx', magnetsDeviations)

        DataUtils.plotRelativeDevitationData(magnetsDeviations, accelerator)

    def showReportPopup(self, accelerator):

        dialog = QDialog()
        dialog.setWindowTitle(f'Status - {accelerator}')

        buttons = QDialogButtonBox.Ok

        buttonBox = QDialogButtonBox(buttons)
        buttonBox.accepted.connect(dialog.accept)

        messageBox = QTextEdit()
        messageBox.setReadOnly(True)
        messageBox.append(self.app.report[accelerator])

        layout = QVBoxLayout()

        layout.addWidget(messageBox)
        layout.addWidget(buttonBox)

        dialog.setLayout(layout)

        dialog.exec()

if __name__ == "__main__":
    App()

""" 
PENDÊNCIAS
- segregar plots dos FEs
- corrigir ordem dos imãs do Booster, LTB, BTS e FE
- LIMPAR CÓDIGO!!
- subir versão limpa no github
- [modelo liu - long.] avaliação de distancias entre dipolos e quadrupolos 
- [modelo liu - vert.] avaliação individual de cada quadrupolo e dipolo
- estimar série de fourier
- substrair para obter alinhamento relativo
"""
