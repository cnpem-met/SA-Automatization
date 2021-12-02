# imports de bibliotecas
from re import A
from accelerator.booster import Booster
from accelerator.storage_ring import SR
from accelerator.ltb import LTB
from accelerator.bts import BTS
from accelerator.frontend import FE

from geom_entitites.frame import Frame
from operations.geometrical import calculateMagnetsDeviations, calculateMagnetsDistances, calculateMagnetsRelativeDeviations, debug_transformToLocalFrame, transformToLocalFrame
from operations.misc import generateReport, printDictData, printFrameDict, separateFEsData, sortFrameDictByBeamTrajectory
from operations.plot import plotDevitationData, plotComparativeDeviation, plotRelativeDevitationData
from operations.io import generateFrameDict, writeToExcel, readExcel, readCSV
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

        self.accelerators = {"SR": SR(), "booster": None, "LTB": None, "BTS": None, "FE": None}


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
        self.frameDict['SR'] = generateFrameDict(self.lookupTable['SR'])

        entitiesDictNom = SR.createObjectsStructure(self.nominals['SR'])

        self.entitiesDictNom['SR'] = entitiesDictNom

        debug_transformToLocalFrame(self.entitiesDictNom['SR'], self.frameDict['SR'], 'SR')

    # ref: limpar
    def loadEnv(self):
        with open("config.json", "r") as file:
            config = json.load(file)

        self.accelerators['SR'].shiftsB1 = config['shiftsB1']

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
        longDistMeas = calculateMagnetsDistances(self.frameDict[accelerator], 'measured')
        longDistNom = calculateMagnetsDistances(self.frameDict[accelerator], 'nominal')

        writeToExcel("../data/output/distances-measured.xlsx", longDistMeas, accelerator)
        writeToExcel("../data/output/distances-nominal.xlsx", longDistNom, accelerator)

    def exportAbsoluteDeviations(self, accelerator):
        # checando se todos os arquivos foram devidamente carregados
        if(not self.isNominalsLoaded[accelerator] or not self.isMeasuredLoaded[accelerator] or not self.isLookuptableLoaded[accelerator]):
            self.ui.logMessage("Erro ao plotar gráfico: nem todos os arquivos foram carregados")
            return

        magnetsDeviations = calculateMagnetsDeviations(self.frameDict[accelerator])
        writeToExcel(f"../data/output/deviations-{accelerator}.xlsx", magnetsDeviations, accelerator)

    def saveEnv(self):
        with open("config.json", "w") as file:
            config = {'ptsNomFileName': {"booster": self.ptsNomFileName['booster'], "SR": self.ptsNomFileName['SR'], "LTB": self.ptsNomFileName['LTB'], "BTS": self.ptsNomFileName['BTS'], "FE": self.ptsNomFileName['FE']},
                      'ptsMeasFileName': {"booster": self.ptsMeasFileName['booster'], "SR": self.ptsMeasFileName['SR'], "LTB": self.ptsMeasFileName['LTB'], "BTS": self.ptsMeasFileName['BTS'], "FE": self.ptsMeasFileName['FE']},
                      'lookupTable': {"booster": self.lookupTable['booster'], "SR": self.lookupTable['SR'], "LTB": self.lookupTable['LTB'], "BTS": self.lookupTable['BTS'], "FE": self.lookupTable['FE']},
                      'shiftsB1': self.accelerators['SR'].shiftsB1}
            json.dump(config, file)

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
        self.accelerators[accelerator].lookupTable = lookuptable

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
        # self.frameDict[accelerator] = generateFrameDict(self.lookupTable[accelerator])

        if (accelerator == 'SR'):
            self.accelerators[accelerator].generateFrameDict()
            self.accelerators[accelerator].createObjectsStructure('nominal')
            self.accelerators[accelerator].createObjectsStructure('measured')

            # printDictData(self.accelerators[accelerator].girders['nominal'])

            self.accelerators[accelerator].transformToLocalFrame('nominal')
            self.accelerators[accelerator].transformToLocalFrame('measured')

            self.accelerators[accelerator].generateMeasuredFrames(self.ui)

            # printFrameDict(self.accelerators['SR'].frames['nominal'])
            # printFrameDict(self.accelerators['SR'].frames['measured'])

            self.accelerators[accelerator].sortFrameDictByBeamTrajectory('nominal')
            self.accelerators[accelerator].sortFrameDictByBeamTrajectory('measured')

            self.report[accelerator] = generateReport(self.accelerators[accelerator].frames['measured'], accelerator)

            self.ui.enable_plot_button(accelerator)

        else:
            return

        # if (accelerator == 'SR'):
        #     entitiesDictNom = SR.createObjectsStructure(self.nominals['SR'])
        #     entitiesDictMeas = SR.createObjectsStructure(self.measured['SR'])
        # elif (accelerator == 'booster'):
        #     entitiesDictNom = Booster.createObjectsStructure(self.nominals['booster'])
        #     entitiesDictMeas = Booster.createObjectsStructure(self.measured['booster'])
        # elif (accelerator == 'LTB'):
        #     entitiesDictNom = LTB.createObjectsStructure(self.nominals['LTB'])
        #     entitiesDictMeas = LTB.createObjectsStructure(self.measured['LTB'])
        # elif (accelerator == 'BTS'):
        #     entitiesDictNom = BTS.createObjectsStructure(self.nominals['BTS'])
        #     entitiesDictMeas = BTS.createObjectsStructure(self.measured['BTS'])
        # elif (accelerator == 'FE'):
        #     entitiesDictNom = FE.createObjectsStructure(self.nominals['FE'])
        #     entitiesDictMeas = FE.createObjectsStructure(self.measured['FE'])

        # self.entitiesDictNom[accelerator] = entitiesDictNom
        # self.entitiesDictMeas[accelerator] = entitiesDictMeas

        

    def generateMeasuredFrames(self, accelerator):
        if (accelerator == 'SR'):
            typesOfMagnets = ['quadrupole', 'dipole-B1', 'dipole-B2', 'dipole-BC']
            for magnetType in typesOfMagnets:
                self.calculateMeasuredFrames(self.entitiesDictMeas[accelerator], self.entitiesDictNom[accelerator],\
                                                self.frameDict[accelerator], 'SR', typeOfMagnets=magnetType, isTypeOfMagnetsIgnored=False)
        else:
            self.calculateMeasuredFrames(self.entitiesDictMeas[accelerator], self.entitiesDictNom[accelerator],\
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
            magnetsDeviations.append(calculateMagnetsDeviations(self.frameDict['LTB']))
            magnetsDeviations.append(calculateMagnetsDeviations(self.frameDict['BTS']))
        elif (accelerator == 'FE'):
            magnetsDeviations.append(separateFEsData(calculateMagnetsDeviations(self.frameDict['FE']), ['MAN', 'EMA', 'CAT']))
            magnetsDeviations.append(separateFEsData(calculateMagnetsDeviations(self.frameDict['FE']), ['IPE', 'CAR', 'MOG']))

        else:
            magnetsDeviations.append(calculateMagnetsDeviations(self.accelerators[accelerator].frames))
            if (accelerator == 'SR'):
                # appending in df B1's transversal deviations
                # magnetsDeviations[0] = SR.calcB1PerpendicularTranslations(magnetsDeviations[0], self.entitiesDictMeas['SR'], self.entitiesDictNom['SR'], self.frameDict['SR'])
                # magnetsDeviations[0] = SR.addTranslationsFromMagnetMeasurements(magnetsDeviations[0])
                pass

        plotArgs = self.ui.get_plot_args_from_ui(accelerator)

        if (accelerator == 'FE'):
            plotArgs['FElist'] = ['MANACÁ', 'EMA', 'CATERETÊ']
            plotDevitationData(magnetsDeviations[0], **plotArgs)
            plotArgs['FElist'] = ['IPÊ', 'CARNAÚBA', 'MOGNO']
            plotDevitationData(magnetsDeviations[1], **plotArgs)
        else:
            plotDevitationData(magnetsDeviations, **plotArgs)

    def plotComparativeDev(self):
        self.ui.logMessage("Plotando o perfil de alinhamento de todo os aceleradores...")
        self.processEvents()

        magnetsDeviations_LTB = calculateMagnetsDeviations(self.frameDict['LTB'])
        magnetsDeviations_booster = calculateMagnetsDeviations(self.frameDict['booster'])
        magnetsDeviations_BTS = calculateMagnetsDeviations(self.frameDict['BTS'])
        magnetsDeviations_SR = calculateMagnetsDeviations(self.frameDict['SR'])
        deviation = [magnetsDeviations_LTB, magnetsDeviations_booster, magnetsDeviations_BTS, magnetsDeviations_SR]

        sizes = [magnetsDeviations_LTB.iloc[:, 0].size, magnetsDeviations_booster.iloc[:, 0].size, magnetsDeviations_BTS.iloc[:, 0].size]
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

    def calculateMeasuredFrames(self, objectDictMeas, objectDictNom, frameDict, accelerator, typeOfMagnets='', isTypeOfMagnetsIgnored=True):
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
                            self.ui.logMessage('Falha no imã ' + magnet.name +': o nome de 1 ou mais pontos nominais não correspondem aos medidos.', severity="danger")
                            continue
                        # calculating the transformation from the nominal points to the measured ones
                        try:
                            localTransformation = SR.calculateLocalDeviationsByTypeOfMagnet(magnet, objectName, pointDict, frameDict, pointList)
                        except KeyError:
                            self.ui.logMessage('Falha no imã ' + magnet.name +': frame medido do quadrupolo adjacente ainda não foi calculado.', severity='danger')
                            continue
                    elif (accelerator == 'booster'):
                        try:
                            pointList = Booster.appendPoints(pointDict, objectDictNom, magnet, objectName)
                            localTransformation = Booster.calculateLocalDeviations(magnet, pointList)
                        except KeyError:
                            self.ui.logMessage('Falha no imã ' + magnet.name +': o nome de 1 ou mais pontos nominais não correspondem aos medidos.', severity='danger')
                            continue
                    elif (accelerator == 'LTB'):
                        try:
                            pointList = LTB.appendPoints(pointDict, objectDictNom, magnet, objectName)
                            localTransformation = LTB.calculateLocalDeviations(magnet, pointList)
                        except KeyError:
                            self.ui.logMessage('Falha no imã ' + magnet.name +': o nome de 1 ou mais pontos nominais não correspondem aos medidos.', severity='danger')
                            continue
                    elif (accelerator == 'BTS'):
                        try:
                            pointList = BTS.appendPoints(pointDict, objectDictNom, magnet, objectName)
                            localTransformation = BTS.calculateLocalDeviations(magnet, pointList)
                        except KeyError:
                            self.ui.logMessage('Falha no imã ' + magnet.name +': o nome de 1 ou mais pontos nominais não correspondem aos medidos.', severity='danger')
                            continue
                    elif (accelerator == 'FE'):
                        try:
                            pointList = FE.appendPoints(pointDict, objectDictNom, magnet, objectName)
                            localTransformation = FE.calculateLocalDeviations(magnet, pointList)
                        except KeyError:
                            self.ui.logMessage('Falha no imã ' + magnet.name +': o nome de 1 ou mais pontos nominais não correspondem aos medidos.', severity='danger')
                            continue
                    try:
                        # referencing the transformation from the frame of nominal magnet to the machine-local
                        baseTransformation = frameDict[localTransformation.frameFrom].transformation
                    except KeyError:
                        self.ui.logMessage('Falha no imã ' + magnet.name + ': frame nominal não foi computado; checar tabela com transformações.', severity='danger')
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
