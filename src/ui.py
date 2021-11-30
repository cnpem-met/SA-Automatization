from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QWidget, QMessageBox, QDialog, QDialogButtonBox, QVBoxLayout, QTextEdit)
from PyQt5.QtGui import (QColor)
from PyQt5 import uic
from functools import partial

# globals
DEBUG_MODE = False

class Ui(QMainWindow):
    def __init__(self, app):
        # Initializing the inherited class
        super(Ui, self).__init__()

        # Load the .ui file
        uic.loadUi('gui.ui', self)

        # Keeping a reference to the app component
        self.app = app

        # Check the mode of the session: DEBUG_MODE => run specific functions to test something
        if(DEBUG_MODE):
            self.app.runDebugFunctions()
            return

        # creating the event listeners for buttons and other widgets
        self.createEventListeners()

        # Show the screen
        self.show()
        
    def createEventListeners(self):
        # toolbar actions
        self.actionNom_SR.triggered.connect(partial(self.app.loadNominalsFile, 'SR'))
        self.actionNom_booster.triggered.connect(partial(self.app.loadNominalsFile, 'booster'))
        self.actionNom_LTB.triggered.connect(partial(self.app.loadNominalsFile, 'LTB'))
        self.actionNom_BTS.triggered.connect(partial(self.app.loadNominalsFile, 'BTS'))
        self.actionNom_FE.triggered.connect(partial(self.app.loadNominalsFile, 'FE'))
        
        self.actionMeas_SR.triggered.connect(partial(self.app.loadMeasuredFile, 'SR'))
        self.actionMeas_booster.triggered.connect(partial(self.app.loadMeasuredFile, 'booster'))
        self.actionMeas_LTB.triggered.connect(partial(self.app.loadMeasuredFile, 'LTB'))
        self.actionMeas_BTS.triggered.connect(partial(self.app.loadMeasuredFile, 'BTS'))
        self.actionMeas_FE.triggered.connect(partial(self.app.loadMeasuredFile, 'FE'))

        self.actionFrames_SR.triggered.connect(partial(self.app.loadLookuptableFile, 'SR'))
        self.actionFrames_booster.triggered.connect(partial(self.app.loadLookuptableFile, 'booster'))
        self.actionFrames_LTB.triggered.connect(partial(self.app.loadLookuptableFile, 'LTB'))
        self.actionFrames_BTS.triggered.connect(partial(self.app.loadLookuptableFile, 'BTS'))
        self.actionFrames_FE.triggered.connect(partial(self.app.loadLookuptableFile, 'FE'))

        self.actionSaveEnv.triggered.connect(self.app.saveEnv)
        self.actionLoadEnv.triggered.connect(self.app.loadEnv)

        # plot buttons
        self.plotAbsolute_SR.clicked.connect(partial(self.app.plotAbsolute, 'SR'))
        self.plotAbsolute_booster.clicked.connect(partial(self.app.plotAbsolute, 'booster'))
        self.plotAbsolute_LTB.clicked.connect(partial(self.app.plotAbsolute, 'LTB'))
        self.plotAbsolute_BTS.clicked.connect(partial(self.app.plotAbsolute, 'BTS'))
        self.plotAbsolute_FE.clicked.connect(partial(self.app.plotAbsolute, 'FE'))
        self.plotComparative.clicked.connect(self.app.plotComparativeDev)

        # report buttons
        self.reportProgress_booster.clicked.connect(partial(self.showReportPopup, 'booster'))
        self.reportProgress_SR.clicked.connect(partial(self.showReportPopup, 'SR'))
        
        self.expLongDist_SR.clicked.connect(partial(self.app.exportLongitudinalDistances, 'SR'))
        self.expAbsolute_SR.clicked.connect(partial(self.app.exportAbsoluteDeviations, 'SR'))
        # DataUtils.debug_translatePoints(self.app.entitiesDictNom['SR'], self.app.frameDict['SR'])

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

    def openFileNameDialog(self, typeOfFile, accelerator):
        dialog = QWidget()

        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            dialog, "Selecione o arquivo - " + typeOfFile + " - " + accelerator, "", filter="All files (*)", options=options)

        return fileName

    def update_nominals_indicator(self, accelerator, fileName):
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

    def update_measured_indicator(self, accelerator, fileName):
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

    def update_lookup_indicator(self, accelerator, fileName):
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

    def enable_plot_button(self, accelerator):
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

    def get_plot_args_from_ui(self, accelerator):
        return {'accelerator': accelerator, 'filtering':\
            {'SR': {'allDofs':  self.plotAllDofs_SR.isChecked(), 'lineWithinGirder': self.lineWithinGirder_SR.isChecked(), 'reportMode': self.reportMode_SR.isChecked(), 'errorbar': self.errorbar_SR.isChecked()},\
            'booster': {'quad02': self.plotQuad02_booster.isChecked(), 'lineWithinGirder': False, 'reportMode': self.reportMode_booster.isChecked(), 'errorbar': self.errorbar_booster.isChecked()},\
            'LTB': {'lineWithinGirder': False, 'reportMode': False, 'errorbar': self.errorbar_LTB.isChecked()},\
            'BTS': {'lineWithinGirder': False, 'reportMode': False, 'errorbar': self.errorbar_BTS.isChecked()},\
            'FE': {'lineWithinGirder': False, 'reportMode': False, 'errorbar': self.errorbar_FE.isChecked()}}}

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
