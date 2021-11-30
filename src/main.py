# imports de bibliotecas
from accelerators import (Booster, SR, LTB, BTS, FE)
from dataUtils import DataUtils
from ui import Ui

from PyQt5.QtWidgets import QApplication

import sys
import json
import pandas as pd

class App(QApplication):
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

    def runDebugFunctions(self):
        self.loadEnv()

        # creating frame's dict for the particular accelerator
        self.frameDict['SR'] = DataUtils.generateFrameDict(self.lookupTable['SR'])

        entitiesDictNom = SR.createObjectsStructure(self.nominals['SR'])

        self.entitiesDictNom['SR'] = entitiesDictNom

        DataUtils.debug_transformToLocalFrame(self.entitiesDictNom['SR'], self.frameDict['SR'], 'SR')

    def loadEnv(self):
        with open("config.json", "r") as file:
            config = json.load(file)

        self.shiftsB1 = config['shiftsB1']

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

    def exportLongitudinalDistances(self, accelerator):
        longDistMeas = DataUtils.calculateMagnetsDistances(self.frameDict[accelerator], 'measured')
        longDistNom = DataUtils.calculateMagnetsDistances(self.frameDict[accelerator], 'nominal')

        DataUtils.writeToExcel("../data/output/distances-measured.xlsx", longDistMeas, accelerator)
        DataUtils.writeToExcel("../data/output/distances-nominal.xlsx", longDistNom, accelerator)

    def exportAbsoluteDeviations(self, accelerator):
        # checando se todos os arquivos foram devidamente carregados
        if(not self.isNominalsLoaded[accelerator] or not self.isMeasuredLoaded[accelerator] or not self.isLookuptableLoaded[accelerator]):
            self.ui.logMessage("Erro ao plotar gráfico: nem todos os arquivos foram carregados")
            return

        magnetsDeviations = DataUtils.calculateMagnetsDeviations(self.frameDict[accelerator])
        DataUtils.writeToExcel(f"../data/output/deviations-{accelerator}.xlsx", magnetsDeviations, accelerator)

    def saveEnv(self):
        with open("config.json", "w") as file:
            config = {'ptsNomFileName': {"booster": self.ptsNomFileName['booster'], "SR": self.ptsNomFileName['SR'], "LTB": self.ptsNomFileName['LTB'], "BTS": self.ptsNomFileName['BTS'], "FE": self.ptsNomFileName['FE']},
                      'ptsMeasFileName': {"booster": self.ptsMeasFileName['booster'], "SR": self.ptsMeasFileName['SR'], "LTB": self.ptsMeasFileName['LTB'], "BTS": self.ptsMeasFileName['BTS'], "FE": self.ptsMeasFileName['FE']},
                      'lookupTable': {"booster": self.lookupTable['booster'], "SR": self.lookupTable['SR'], "LTB": self.lookupTable['LTB'], "BTS": self.lookupTable['BTS'], "FE": self.lookupTable['FE']},
                      'shiftsB1': self.shiftsB1}
            json.dump(config, file)

    def loadNominalsFile(self, accelerator, filePath=None):

        if(filePath == ""):
            return

        filepath = filePath or self.ui.openFileNameDialog('pontos nominais', accelerator)

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
            self.ui.logMessage("Formato do arquivo "+fileName+" não suportado.", severity="alert")
            return

        # gerando grupos de pontos
        self.nominals[accelerator] = nominals

        self.isNominalsLoaded[accelerator] = True
        self.ptsNomFileName[accelerator] = filepath
        
        self.ui.update_nominals_indicator(accelerator, fileName)
        self.ui.logMessage("Arquivo nominal do "+accelerator+" carregado.")

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
            self.ui.logMessage("Formato do arquivo "+fileName+" não suportado.", severity="alert")
            return

        self.measured[accelerator] = measured

        self.isMeasuredLoaded[accelerator] = True

        self.ui.update_measured_indicator(accelerator, fileName)

        self.ptsMeasFileName[accelerator] = filepath

        self.ui.logMessage("Arquivo com medidos do "+accelerator+" carregado.")

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
            self.ui.logMessage("Format of file "+fileName+"not supported.")
            return

        self.lookupTable[accelerator] = lookuptable

        self.isLookuptableLoaded[accelerator] = True
        
        self.ui.update_lookup_indicator(accelerator, fileName)

        self.lookupFileName[accelerator] = filepath

        self.ui.logMessage("Lookuptable de frames do " + accelerator + " carregada.")

        self.checkAllFilesAndProcessData(accelerator)

    def checkAllFilesAndProcessData(self, accelerator):
        # se todos os arquivos tiverem sido carregados
        if((self.isNominalsLoaded[accelerator] and self.isMeasuredLoaded[accelerator] and self.isLookuptableLoaded[accelerator])):
            self.ui.logMessage("Transformando pontos do " + accelerator + " para os frames locais...")
            self.processEvents()
            self.processInternalData(accelerator)
            self.isInternalDataProcessed[accelerator] = True
            self.ui.logMessage("Dados do "+accelerator+" prontos.", severity="sucess")

    def processInternalData(self, accelerator):
        # creating frame's dict for the particular accelerator
        self.frameDict[accelerator] = DataUtils.generateFrameDict(self.lookupTable[accelerator])

        if (accelerator == 'booster'):
            entitiesDictNom = Booster.createObjectsStructure(self.nominals['booster'])
            entitiesDictMeas = Booster.createObjectsStructure(self.measured['booster'])
        elif (accelerator == 'SR'):
            entitiesDictNom = SR.createObjectsStructure(self.nominals['SR'])
            entitiesDictMeas = SR.createObjectsStructure(self.measured['SR'], shiftsB1=self.shiftsB1)
        elif (accelerator == 'LTB'):
            entitiesDictNom = LTB.createObjectsStructure(self.nominals['LTB'])
            entitiesDictMeas = LTB.createObjectsStructure(self.measured['LTB'])
        elif (accelerator == 'BTS'):
            entitiesDictNom = BTS.createObjectsStructure(self.nominals['BTS'])
            entitiesDictMeas = BTS.createObjectsStructure(self.measured['BTS'])
        elif (accelerator == 'FE'):
            entitiesDictNom = FE.createObjectsStructure(self.nominals['FE'])
            entitiesDictMeas = FE.createObjectsStructure(self.measured['FE'])

        self.entitiesDictNom[accelerator] = entitiesDictNom
        self.entitiesDictMeas[accelerator] = entitiesDictMeas

        DataUtils.transformToLocalFrame(self.entitiesDictNom[accelerator], self.frameDict[accelerator], accelerator)
        DataUtils.transformToLocalFrame(self.entitiesDictMeas[accelerator], self.frameDict[accelerator], accelerator)

        self.generateMeasuredFrames(accelerator)

        self.frameDict[accelerator] = DataUtils.sortFrameDictByBeamTrajectory(self.frameDict[accelerator], accelerator)

        self.report[accelerator] = DataUtils.generateReport(self.frameDict[accelerator], accelerator)

        # DataUtils.printFrameDict(self.frameDict[accelerator])
        DataUtils.printDictData(self.entitiesDictMeas[accelerator])

        self.ui.enable_plot_button(accelerator)

    def generateMeasuredFrames(self, accelerator):
        if (accelerator == 'SR'):
            typesOfMagnets = ['quadrupole', 'dipole-B1', 'dipole-B2', 'dipole-BC']
            for magnetType in typesOfMagnets:
                DataUtils.generateMeasuredFrames(self.entitiesDictMeas[accelerator], self.entitiesDictNom[accelerator],\
                                                self.frameDict[accelerator], 'SR', self, typeOfMagnets=magnetType, isTypeOfMagnetsIgnored=False)
        else:
            DataUtils.generateMeasuredFrames(self.entitiesDictMeas[accelerator], self.entitiesDictNom[accelerator],\
                                             self.frameDict[accelerator], accelerator, self)
        return
        
    def plotAbsolute(self, accelerator):
        self.ui.logMessage("Plotando o perfil de alinhamento absoluto do "+accelerator+"...")
        self.processEvents()

        # checando se todos os arquivos foram devidamente carregados
        if(not self.isNominalsLoaded[accelerator] or not self.isMeasuredLoaded[accelerator] or not self.isLookuptableLoaded[accelerator]):
            self.ui.logMessage("Erro ao plotar gráfico: nem todos os arquivos foram carregados")
            return

        magnetsDeviations = []

        if (accelerator == 'LTB' or accelerator == 'BTS'):
            accelerator = 'transport-lines'
            magnetsDeviations.append(DataUtils.calculateMagnetsDeviations(self.frameDict['LTB']))
            magnetsDeviations.append(DataUtils.calculateMagnetsDeviations(self.frameDict['BTS']))
        elif (accelerator == 'FE'):
            magnetsDeviations.append(DataUtils.separateFEsData(DataUtils.calculateMagnetsDeviations(self.frameDict['FE']), ['MAN', 'EMA', 'CAT']))
            magnetsDeviations.append(DataUtils.separateFEsData(DataUtils.calculateMagnetsDeviations(self.frameDict['FE']), ['IPE', 'CAR', 'MOG']))

        else:
            magnetsDeviations.append(DataUtils.calculateMagnetsDeviations(self.frameDict[accelerator]))
            if (accelerator == 'SR'):
                # appending in df B1's transversal deviations
                # magnetsDeviations[0] = SR.calcB1PerpendicularTranslations(magnetsDeviations[0], self.entitiesDictMeas['SR'], self.entitiesDictNom['SR'], self.frameDict['SR'])
                # magnetsDeviations[0] = SR.addTranslationsFromMagnetMeasurements(magnetsDeviations[0])
                pass

        plotArgs = self.ui.get_plot_args_from_ui(accelerator) 

        if (accelerator == 'FE'):
            plotArgs['FElist'] = ['MANACÁ', 'EMA', 'CATERETÊ']
            DataUtils.plotDevitationData(magnetsDeviations[0], **plotArgs)
            plotArgs['FElist'] = ['IPÊ', 'CARNAÚBA', 'MOGNO']
            DataUtils.plotDevitationData(magnetsDeviations[1], **plotArgs)
        else:
            DataUtils.plotDevitationData(magnetsDeviations, **plotArgs)

    def plotComparativeDev(self):
        self.ui.logMessage("Plotando o perfil de alinhamento de todo os aceleradores...")
        self.processEvents()

        magnetsDeviations_LTB = DataUtils.calculateMagnetsDeviations(self.frameDict['LTB'])
        magnetsDeviations_booster = DataUtils.calculateMagnetsDeviations(self.frameDict['booster'])
        magnetsDeviations_BTS = DataUtils.calculateMagnetsDeviations(self.frameDict['BTS'])
        magnetsDeviations_SR = DataUtils.calculateMagnetsDeviations(self.frameDict['SR'])
        deviation = [magnetsDeviations_LTB, magnetsDeviations_booster, magnetsDeviations_BTS, magnetsDeviations_SR]

        sizes = [magnetsDeviations_LTB.iloc[:, 0].size, magnetsDeviations_booster.iloc[:, 0].size, magnetsDeviations_BTS.iloc[:, 0].size]
        lenghts = [sizes[0], sizes[1] + sizes[0], sizes[2] + sizes[1] + sizes[0]]

        deviations = pd.concat(deviation)

        plotArgs = self.ui.get_plot_args_from_ui('SR')

        DataUtils.plotComparativeDeviation(deviations, **plotArgs)

    def plotRelative(self, accelerator):
        self.ui.logMessage("Plotando o perfil de alinhamento relativo do "+accelerator+"...")
        self.processEvents()

        # checando se todos os arquivos foram devidamente carregados
        if(not self.isNominalsLoaded[accelerator] or not self.isMeasuredLoaded[accelerator] or not self.isLookuptableLoaded[accelerator]):
            self.ui.logMessage("Erro ao plotar gráfico: nem todos os arquivos foram carregados")
            return


        magnetsDeviations = DataUtils.calculateMagnetsRelativeDeviations(
            self.frameDict[accelerator])

        # DataUtils.writeToExcel('../data/output/sr-devs.xlsx', magnetsDeviations)
        DataUtils.plotRelativeDevitationData(magnetsDeviations, accelerator)


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
