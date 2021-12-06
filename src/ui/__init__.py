from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QWidget, QDialog, QDialogButtonBox, QVBoxLayout, QTextEdit)
from PyQt5.QtGui import (QColor)
from PyQt5 import uic
from functools import partial


class Ui(QMainWindow):
    def __init__(self, app):
        # Initializing the inherited class
        super().__init__()

        # Load the .ui file
        uic.loadUi('gui.ui', self)

        # Keeping a reference to the app component
        self.app = app

        # creating the event listeners for buttons and other widgets
        self.create_event_listeners()

        # Show the screen
        self.show()
        
    def create_event_listeners(self) -> None:
        """ Connects UI action elements to its corresponding methods. """

        # toolbar actions
        self.actionNom_SR.triggered.connect(partial(self.app.load_nominals_file, 'SR'))
        self.actionNom_booster.triggered.connect(partial(self.app.load_nominals_file, 'booster'))
        self.actionNom_LTB.triggered.connect(partial(self.app.load_nominals_file, 'LTB'))
        self.actionNom_BTS.triggered.connect(partial(self.app.load_nominals_file, 'BTS'))
        self.actionNom_FE.triggered.connect(partial(self.app.load_nominals_file, 'FE'))
        
        self.actionMeas_SR.triggered.connect(partial(self.app.load_measured_file, 'SR'))
        self.actionMeas_booster.triggered.connect(partial(self.app.load_measured_file, 'booster'))
        self.actionMeas_LTB.triggered.connect(partial(self.app.load_measured_file, 'LTB'))
        self.actionMeas_BTS.triggered.connect(partial(self.app.load_measured_file, 'BTS'))
        self.actionMeas_FE.triggered.connect(partial(self.app.load_measured_file, 'FE'))

        self.actionFrames_SR.triggered.connect(partial(self.app.loadLookuptableFile, 'SR'))
        self.actionFrames_booster.triggered.connect(partial(self.app.loadLookuptableFile, 'booster'))
        self.actionFrames_LTB.triggered.connect(partial(self.app.loadLookuptableFile, 'LTB'))
        self.actionFrames_BTS.triggered.connect(partial(self.app.loadLookuptableFile, 'BTS'))
        self.actionFrames_FE.triggered.connect(partial(self.app.loadLookuptableFile, 'FE'))

        self.actionSaveEnv.triggered.connect(self.app.save_env)
        self.actionLoadEnv.triggered.connect(self.app.load_env)

        # plot buttons
        self.plotAbsolute_SR.clicked.connect(partial(self.app.plot_absolute_deviations, 'SR'))
        self.plotAbsolute_booster.clicked.connect(partial(self.app.plot_absolute_deviations, 'booster'))
        self.plotAbsolute_LTB.clicked.connect(partial(self.app.plot_absolute_deviations, 'LTB'))
        self.plotAbsolute_BTS.clicked.connect(partial(self.app.plot_absolute_deviations, 'BTS'))
        self.plotAbsolute_FE.clicked.connect(partial(self.app.plot_absolute_deviations, 'FE'))
        # self.plotComparative.clicked.connect(self.app.plotComparativeDev)

        # report buttons
        self.reportProgress_booster.clicked.connect(partial(self.show_report_popup, 'booster'))
        self.reportProgress_SR.clicked.connect(partial(self.show_report_popup, 'SR'))
        
        self.expLongDist_SR.clicked.connect(partial(self.app.export_longitudinal_distances, 'SR'))
        self.expAbsolute_SR.clicked.connect(partial(self.app.export_absolute_deviations, 'SR'))

    def log_message(self, message: str, severity: str = 'normal') -> None:
        """ Logs message in the text area of the UI. """
        
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

    def open_file_dialog(self, type_of_file: str, acc_name: str) -> str:
        """ Prompts a file dialog to pick a generic file. """
        
        dialog = QWidget()
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(dialog, f"Selecione o arquivo - {type_of_file} - {acc_name}", "", filter="All files (*)", options=options)

        return file_name

    def update_nominals_indicator(self, acc_name: str, file_name: str) -> None:
        """ Updates UI nominals indicators.  """

        if (acc_name == 'SR'):
            self.ledNom_SR.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileNom_SR.setText(file_name)
            self.fileNom_SR.setToolTip(file_name)
        elif (acc_name == 'booster'):
            self.ledNom_booster.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileNom_booster.setText(file_name)
            self.fileNom_booster.setToolTip(file_name)
        elif (acc_name == 'LTB'):
            self.ledNom_LTB.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileNom_LTB.setText(file_name)
            self.fileNom_LTB.setToolTip(file_name)
        elif (acc_name == 'BTS'):
            self.ledNom_BTS.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileNom_BTS.setText(file_name)
            self.fileNom_BTS.setToolTip(file_name)
        elif (acc_name == 'FE'):
            self.ledNom_FE.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileNom_FE.setText(file_name)
            self.fileNom_FE.setToolTip(file_name)

    def update_measured_indicator(self, acc_name: str, file_name: str) -> None:
        """ Updates UI measured indicators.  """

        if (acc_name == 'SR'):
            self.ledMeas_SR.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileMeas_SR.setText(file_name)
            self.fileMeas_SR.setToolTip(file_name)
        elif (acc_name == 'booster'):
            self.ledMeas_booster.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileMeas_booster.setText(file_name)
            self.fileMeas_booster.setToolTip(file_name)
        elif (acc_name == 'LTB'):
            self.ledMeas_LTB.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileMeas_LTB.setText(file_name)
            self.fileMeas_LTB.setToolTip(file_name)
        elif (acc_name == 'BTS'):
            self.ledMeas_BTS.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileMeas_BTS.setText(file_name)
            self.fileMeas_BTS.setToolTip(file_name)
        elif (acc_name == 'FE'):
            self.ledMeas_FE.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileMeas_FE.setText(file_name)
            self.fileMeas_FE.setToolTip(file_name)

    def update_lookup_indicator(self, acc_name: str, file_name: str) -> None:
        """ Updates UI lookup table indicators.  """

        if (acc_name == 'SR'):
            self.ledFrames_SR.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileFrames_SR.setText(file_name)
            self.fileFrames_SR.setToolTip(file_name)
        elif (acc_name == 'booster'):
            self.ledFrames_booster.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileFrames_booster.setText(file_name)
            self.fileFrames_booster.setToolTip(file_name)
        elif (acc_name == 'LTB'):
            self.ledFrames_LTB.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileFrames_LTB.setText(file_name)
            self.fileFrames_LTB.setToolTip(file_name)
        elif (acc_name == 'BTS'):
            self.ledFrames_BTS.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileFrames_BTS.setText(file_name)
            self.fileFrames_BTS.setToolTip(file_name)
        elif (acc_name == 'FE'):
            self.ledFrames_FE.setStyleSheet("background-color:green; border-radius:7px;")
            self.fileFrames_FE.setText(file_name)
            self.fileFrames_FE.setToolTip(file_name)

    def enable_plot_button(self, acc_name: str) -> None:
        """ Changes the state of plot buttons, which enables user to click.  """

        # habilita botões de plot
        if(acc_name == 'SR'):
            self.plotAbsolute_SR.setEnabled(True)
            self.expAbsolute_SR.setEnabled(True)
            # self.expMachModel_SR.setEnabled(True)
            self.reportProgress_SR.setEnabled(True)
            self.expLongDist_SR.setEnabled(True)
        elif (acc_name == 'booster'):
            self.plotAbsolute_booster.setEnabled(True)
            self.expAbsolute_booster.setEnabled(True)
            self.reportProgress_booster.setEnabled(True)
        elif (acc_name == 'LTB'):
            self.plotAbsolute_LTB.setEnabled(True)
            self.expAbsolute_LTB.setEnabled(True)
        elif (acc_name == 'BTS'):
            self.plotAbsolute_BTS.setEnabled(True)
            self.expAbsolute_BTS.setEnabled(True)
        elif (acc_name == 'FE'):
            self.plotAbsolute_FE.setEnabled(True)
            self.expAbsolute_FE.setEnabled(True)

    def get_plot_args_from_ui(self, acc_name: str) -> dict:
        """ Compiles plot arguments selected by the user in the UI. """

        return {'accelerator': acc_name, 'filtering':\
            {'SR': {'allDofs':  self.plotAllDofs_SR.isChecked(), 'lineWithinGirder': self.lineWithinGirder_SR.isChecked(), 'reportMode': self.reportMode_SR.isChecked(), 'errorbar': self.errorbar_SR.isChecked()},\
            'booster': {'quad02': self.plotQuad02_booster.isChecked(), 'lineWithinGirder': False, 'reportMode': self.reportMode_booster.isChecked(), 'errorbar': self.errorbar_booster.isChecked()},\
            'LTB': {'lineWithinGirder': False, 'reportMode': False, 'errorbar': self.errorbar_LTB.isChecked()},\
            'BTS': {'lineWithinGirder': False, 'reportMode': False, 'errorbar': self.errorbar_BTS.isChecked()},\
            'FE': {'lineWithinGirder': False, 'reportMode': False, 'errorbar': self.errorbar_FE.isChecked()}}}

    def show_report_popup(self, acc_name: str) -> None:
        """ Prompts the report dialog. """

        dialog = QDialog()
        dialog.setWindowTitle(f'Status - {acc_name}')

        buttons = QDialogButtonBox.Ok

        buttonBox = QDialogButtonBox(buttons)
        buttonBox.accepted.connect(dialog.accept)

        messageBox = QTextEdit()
        messageBox.setReadOnly(True)
        messageBox.append(self.app.report[acc_name])

        layout = QVBoxLayout()

        layout.addWidget(messageBox)
        layout.addWidget(buttonBox)

        dialog.setLayout(layout)

        dialog.exec()
