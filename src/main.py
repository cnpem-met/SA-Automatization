# imports de bibliotecas
import config

from accelerator import Accelerator
from accelerator.booster import Booster
from accelerator.storage_ring import SR
from accelerator.ltb import LTB
from accelerator.bts import BTS
from accelerator.frontend import FE

from operations.geometrical import calculateMagnetsRelativeDeviations
from operations.misc import generateReport, separateFEsData
from operations.plot import plotDevitationData, plotComparativeDeviation, plotRelativeDevitationData
from operations.io import writeToExcel, readExcel, readCSV
from ui.ui import Ui

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

        self.accelerators = {"SR": SR('SR'), "booster": Booster('booster'), "LTB": LTB('LTB'), "BTS": BTS('BTS'), "FE": FE('FE')}


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

    def loadEnv(self):
        # with open("config.json", "r") as file:
        #     config = json.load(file)

        self.accelerators['SR'].shiftsB1 = config.shiftsB1

        self.loadNominalsFile('booster', filePath=config.ptsNomFileName['booster'])
        self.loadNominalsFile('SR', filePath=config.ptsNomFileName['SR'])
        self.loadNominalsFile('LTB', filePath=config.ptsNomFileName['LTB'])
        self.loadNominalsFile('BTS', filePath=config.ptsNomFileName['BTS'])
        self.loadNominalsFile('FE', filePath=config.ptsNomFileName['FE'])

        self.loadMeasuredFile('booster', filePath=config.ptsMeasFileName['booster'])
        self.loadMeasuredFile('SR', filePath=config.ptsMeasFileName['SR'])
        self.loadMeasuredFile('LTB', filePath=config.ptsMeasFileName['LTB'])
        self.loadMeasuredFile('BTS', filePath=config.ptsMeasFileName['BTS'])
        self.loadMeasuredFile('FE', filePath=config.ptsMeasFileName['FE'])

        self.loadLookuptableFile('booster', filePath=config.lookupFileName['booster'])
        self.loadLookuptableFile('SR', filePath=config.lookupFileName['SR'])
        self.loadLookuptableFile('LTB', filePath=config.lookupFileName['LTB'])
        self.loadLookuptableFile('BTS', filePath=config.lookupFileName['BTS'])
        self.loadLookuptableFile('FE', filePath=config.lookupFileName['FE'])

    def exportLongitudinalDistances(self, acc_name):
        longDistMeas = self.accelerators[acc_name].evaluate_magnets_long_inter_distances('measured')
        longDistNom = self.accelerators[acc_name].evaluate_magnets_long_inter_distances('nominal')

        writeToExcel("../data/output/long-distances-measured.xlsx", longDistMeas, acc_name)
        writeToExcel("../data/output/long-distances-nominal.xlsx", longDistNom, acc_name)

    def exportAbsoluteDeviations(self, acc_name):
        writeToExcel(f"../data/output/deviations-{acc_name}.xlsx", self.accelerators[acc_name].deviations, acc_name)

    def saveEnv(self):
        # with open("config.json", "w") as file:
        #     config = {'ptsNomFileName': {"booster": self.ptsNomFileName['booster'], "SR": self.ptsNomFileName['SR'], "LTB": self.ptsNomFileName['LTB'], "BTS": self.ptsNomFileName['BTS'], "FE": self.ptsNomFileName['FE']},
        #               'ptsMeasFileName': {"booster": self.ptsMeasFileName['booster'], "SR": self.ptsMeasFileName['SR'], "LTB": self.ptsMeasFileName['LTB'], "BTS": self.ptsMeasFileName['BTS'], "FE": self.ptsMeasFileName['FE']},
        #               'lookupTable': {"booster": self.lookupTable['booster'], "SR": self.lookupTable['SR'], "LTB": self.lookupTable['LTB'], "BTS": self.lookupTable['BTS'], "FE": self.lookupTable['FE']},
        #               'shiftsB1': self.accelerators['SR'].shiftsB1}
        #     json.dump(config, file)
        pass

    def loadNominalsFile(self, acc_name, filePath=None):

        if(filePath == ""):
            return

        filepath = filePath or self.ui.openFileNameDialog('pontos nominais', acc_name)

        if(not filepath or filePath == ""):
            return

        fileName = filepath.split('/')[len(filepath.split('/')) - 1]
        fileExtension = fileName.split('.')[1]

        if (fileExtension == 'txt' or fileExtension == 'csv' or fileExtension == 'TXT' or fileExtension == 'CSV'):
            nominals = readCSV(filepath, 'points')
        elif (fileExtension == 'xlsx'):
            nominals = readExcel(
                filepath, 'points')
        else:
            self.ui.logMessage("Formato do arquivo "+fileName+" não suportado.", severity="alert")
            return

        # gerando grupos de pontos
        self.nominals[acc_name] = nominals
        self.accelerators[acc_name].points['nominal'] = nominals

        self.isNominalsLoaded[acc_name] = True
        self.ptsNomFileName[acc_name] = filepath
        
        self.ui.update_nominals_indicator(acc_name, fileName)
        self.ui.logMessage("Arquivo nominal do "+acc_name+" carregado.")

        self.checkAllFilesAndProcessData(acc_name)

    def loadMeasuredFile(self, accelerator, filePath=None):
        
        if(filePath == ""):
            return

        filepath = filePath or self.openFileNameDialog('pontos medidos', accelerator)

        if(not filepath):
            return

        fileName = filepath.split('/')[len(filepath.split('/')) - 1]
        fileExtension = fileName.split('.')[1]

        if (fileExtension == 'txt' or fileExtension == 'csv' or fileExtension == 'TXT' or fileExtension == 'CSV'):
            measured = readCSV(filepath, 'points')
        elif (fileExtension == 'xlsx'):
            measured = readExcel(
                filepath, 'points')
        else:
            self.ui.logMessage("Formato do arquivo "+fileName+" não suportado.", severity="alert")
            return

        self.measured[accelerator] = measured
        self.accelerators[accelerator].points['measured'] = measured

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
            lookuptable = readCSV(filepath, 'lookuptable')
        elif (fileExtension == 'xlsx'):
            lookuptable = readExcel(
                filepath, 'lookuptable')
        else:
            self.ui.logMessage("Format of file "+fileName+"not supported.")
            return

        self.lookupTable[accelerator] = lookuptable
        self.accelerators[accelerator].ml_to_nominal_frame_mapping = lookuptable

        self.isLookuptableLoaded[accelerator] = True
        
        self.ui.update_lookup_indicator(accelerator, fileName)

        self.lookupFileName[accelerator] = filepath

        self.ui.logMessage("Lookuptable de frames do " + accelerator + " carregada.")

        self.checkAllFilesAndProcessData(accelerator)

    def checkAllFilesAndProcessData(self, acc_name):
        # se todos os arquivos tiverem sido carregados
        if((self.isNominalsLoaded[acc_name] and self.isMeasuredLoaded[acc_name] and self.isLookuptableLoaded[acc_name])):
            self.ui.logMessage("Transformando pontos do " + acc_name + " para os frames locais...")
            self.processEvents()

            accelerator = self.accelerators[acc_name]
            self.processInternalData(accelerator)

            self.isInternalDataProcessed[acc_name] = True
            self.ui.logMessage("Dados do "+acc_name+" prontos.", severity="sucess")

    def processInternalData(self, accelerator: Accelerator):

            accelerator.create_nominal_frames()
            accelerator.populate_magnets_with_points('nominal')
            accelerator.populate_magnets_with_points('measured')

            accelerator.change_to_local_frame('nominal')
            accelerator.change_to_local_frame('measured')

            accelerator.calculate_measured_frames(self.ui, magnet_types_to_ignore=['sextupole'])

            accelerator.sort_frames_by_beam_trajectory('nominal')
            accelerator.sort_frames_by_beam_trajectory('measured')

            self.report[accelerator.name] = generateReport(accelerator.frames['measured'], accelerator.frames['nominal'])

            if accelerator.evaluate_magnets_deviations():
                self.ui.enable_plot_button(accelerator.name)
            else:
                self.ui.logMessage(f'Plot não foi possível para {accelerator.name} devido a ausência de frames medidos e/ou nominais.', severity='danger')

    def plotAbsolute(self, acc_name):
        self.ui.logMessage("Plotando o perfil de alinhamento absoluto do "+acc_name+"...")
        self.processEvents()

        deviations = []

        if(acc_name == 'LTB' or acc_name == 'BTS'):
            deviations.append(self.accelerators['LTB'].deviations)
            deviations.append(self.accelerators['BTS'].deviations)
        elif(acc_name == 'FE'):
            deviations.append(separateFEsData(self.accelerators[acc_name].deviations, ['MAN', 'EMA', 'CAT']))
            deviations.append(separateFEsData(self.accelerators[acc_name].deviations, ['IPE', 'CAR', 'MOG']))
        else:
            deviations.append(self.accelerators[acc_name].deviations)
            if (acc_name == 'SR'):
                # appending in df B1's transversal deviations
                # deviations[0] = SR.calcB1PerpendicularTranslations(deviations[0], self.entitiesDictMeas['SR'], self.entitiesDictNom['SR'], self.frameDict['SR'])
                # deviations[0] = SR.addTranslationsFromMagnetMeasurements(deviations[0])
                pass

        plotArgs = self.ui.get_plot_args_from_ui(acc_name)

        if (acc_name == 'FE'):
            plotArgs['FElist'] = ['MANACÁ', 'EMA', 'CATERETÊ']
            plotDevitationData(deviations[0], **plotArgs)
            plotArgs['FElist'] = ['IPÊ', 'CARNAÚBA', 'MOGNO']
            plotDevitationData(deviations[1], **plotArgs)
        else:
            plotDevitationData(deviations, **plotArgs)

    def plotComparativeDev(self):
        self.ui.logMessage("Plotando o perfil de alinhamento de todo os aceleradores...")
        self.processEvents()

        deviation = [self.accelerators['LTB'].deviations, self.accelerators['booster'].deviations, self.accelerators['BTS'].deviations, self.accelerators['SR'].deviations]

        sizes = [deviation[0].iloc[:, 0].size, deviation[1].iloc[:, 0].size, deviation[2].iloc[:, 0].size]
        lenghts = [sizes[0], sizes[1] + sizes[0], sizes[2] + sizes[1] + sizes[0]]

        deviations = pd.concat(deviation)

        plotArgs = self.ui.get_plot_args_from_ui('SR')

        plotComparativeDeviation(deviations, **plotArgs)

    def plotRelative(self, accelerator):
        self.ui.logMessage("Plotando o perfil de alinhamento relativo do "+accelerator+"...")
        self.processEvents()

        # checando se todos os arquivos foram devidamente carregados
        if(not self.isNominalsLoaded[accelerator] or not self.isMeasuredLoaded[accelerator] or not self.isLookuptableLoaded[accelerator]):
            self.ui.logMessage("Erro ao plotar gráfico: nem todos os arquivos foram carregados")
            return

        magnetsDeviations = calculateMagnetsRelativeDeviations(self.frameDict[accelerator])

        # writeToExcel('../data/output/sr-devs.xlsx', magnetsDeviations)
        plotRelativeDevitationData(magnetsDeviations, accelerator)

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
