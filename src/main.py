import config

from accelerator import Accelerator
from accelerator.booster import Booster
from accelerator.storage_ring import SR
from accelerator.ltb import LTB
from accelerator.bts import BTS
from accelerator.frontend import FE

from operations.misc import generate_report, separate_frontend_data
from operations.plot import plot_magnets_absolute_deviation
from operations.io import write_to_excel, read_excel, read_csv

from ui import Ui

from PyQt5.QtWidgets import QApplication
import sys

class App(QApplication):
    def __init__(self):
        # Initializing the application and its UI
        super(App, self).__init__(sys.argv)

        self.init_app_variables()

        # Calling the UI component
        self.ui = Ui(self)

        # Calling the app to execute
        sys.exit(self.exec_())

    def init_app_variables(self) -> None:
        """ Initializes app level variables. """
        
        # init classes specific to each accelerator and stores in a dict for easy access
        self.accelerators = {"SR": SR(), "booster": Booster(), "LTB": LTB(), "BTS": BTS(), "FE": FE()}

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

    def load_env(self) -> None:
        """ Loads environment info and files from the config file. """

        # with open("config.json", "r") as file:
        #     config = json.load(file)

        # loads B1 shifts and save on SR class parameter
        self.accelerators['SR'].shiftsB1 = config.OFFSETS_B1

        # loads files containing the list of all the nominal points 
        self.load_nominals_file('booster', file_path=config.NOM_POINTS_FILE_NAMES['booster'])
        self.load_nominals_file('SR', file_path=config.NOM_POINTS_FILE_NAMES['SR'])
        self.load_nominals_file('LTB', file_path=config.NOM_POINTS_FILE_NAMES['LTB'])
        self.load_nominals_file('BTS', file_path=config.NOM_POINTS_FILE_NAMES['BTS'])
        self.load_nominals_file('FE', file_path=config.NOM_POINTS_FILE_NAMES['FE'])

        # loads files containing the list of all the measured points 
        self.load_measured_file('booster', file_path=config.MEAS_POINTS_FILE_NAMES['booster'])
        self.load_measured_file('SR', file_path=config.MEAS_POINTS_FILE_NAMES['SR'])
        self.load_measured_file('LTB', file_path=config.MEAS_POINTS_FILE_NAMES['LTB'])
        self.load_measured_file('BTS', file_path=config.MEAS_POINTS_FILE_NAMES['BTS'])
        self.load_measured_file('FE', file_path=config.MEAS_POINTS_FILE_NAMES['FE'])

        # loads files containing the mapping between the machine-local and magnet-specific nominal frames
        self.loadLookuptableFile('booster', file_path=config.FRAMES_LOOKUPTABLE_FILE_NAMES['booster'])
        self.loadLookuptableFile('SR', file_path=config.FRAMES_LOOKUPTABLE_FILE_NAMES['SR'])
        self.loadLookuptableFile('LTB', file_path=config.FRAMES_LOOKUPTABLE_FILE_NAMES['LTB'])
        self.loadLookuptableFile('BTS', file_path=config.FRAMES_LOOKUPTABLE_FILE_NAMES['BTS'])
        self.loadLookuptableFile('FE', file_path=config.FRAMES_LOOKUPTABLE_FILE_NAMES['FE'])

    def export_longitudinal_distances(self, acc_name: str) -> None:
        """ Saves the longitudinal inter distance between adjacent magnets to a .xlsx file.  """

        long_dist_meas = self.accelerators[acc_name].evaluate_magnets_long_inter_distances('measured')
        long_dist_nom = self.accelerators[acc_name].evaluate_magnets_long_inter_distances('nominal')

        write_to_excel(f'{config.OUTPUT_PATH}/long-distances-measured.xlsx', long_dist_meas, acc_name)
        write_to_excel(f'{config.OUTPUT_PATH}/long-distances-nominal.xlsx', long_dist_nom, acc_name)

    def export_absolute_deviations(self, acc_name: str) -> None:
        """ Saves the overall deviations of each magnet to a .xlsx file. """

        write_to_excel(f"{config.OUTPUT_PATH}/deviations-{acc_name}.xlsx", self.accelerators[acc_name].deviations, acc_name)

    def save_env(self) -> None:
        """ Saves environment variables to file. This function is currently disabled because no json file
            is being used in the current version of the program, but can be reinserted in the future. """

        self.ui.log_message('Variáveis de ambiente não são mais salvas, recurso desabilitado.', severity='danger')

        # with open("config.json", "w") as file:
        #     config = {'ptsNomFileName': {"booster": self.ptsNomFileName['booster'], "SR": self.ptsNomFileName['SR'], "LTB": self.ptsNomFileName['LTB'], "BTS": self.ptsNomFileName['BTS'], "FE": self.ptsNomFileName['FE']},
        #               'ptsMeasFileName': {"booster": self.ptsMeasFileName['booster'], "SR": self.ptsMeasFileName['SR'], "LTB": self.ptsMeasFileName['LTB'], "BTS": self.ptsMeasFileName['BTS'], "FE": self.ptsMeasFileName['FE']},
        #               'lookupTable': {"booster": self.lookupTable['booster'], "SR": self.lookupTable['SR'], "LTB": self.lookupTable['LTB'], "BTS": self.lookupTable['BTS'], "FE": self.lookupTable['FE']},
        #               'shiftsB1': self.accelerators['SR'].shiftsB1}
        #     json.dump(config, file)

    def load_nominals_file(self, acc_name: str, file_path:str = None) -> None:
        """ Loads files with list of nominal points and saves it in the form
            of a DataFrame into the associated accelerator object. """

        # ignoring empty fields in env file
        if(file_path == ""):
            return

        # prompting a file dialog
        filepath = file_path or self.ui.open_file_dialog('pontos nominais', acc_name)

        # ignoring if user not selected any file
        if(not filepath or file_path == ""):
            return

        # treating the filepath
        file_name = filepath.split('/')[len(filepath.split('/')) - 1]
        file_extension = file_name.split('.')[1]

        # reading file
        if (file_extension == 'txt' or file_extension == 'csv' or file_extension == 'TXT' or file_extension == 'CSV'):
            nominals = read_csv(filepath, 'points')
        elif (file_extension == 'xlsx'):
            nominals = read_excel(filepath, 'points')
        else:
            self.ui.log_message(f"Formato do arquivo {file_name} não suportado.", severity="alert")
            return

        # saving data into accelerator
        self.accelerators[acc_name].points['nominal'] = nominals

        # updating app flags
        self.isNominalsLoaded[acc_name] = True
        self.ptsNomFileName[acc_name] = filepath
        
        # ui calls
        self.ui.update_nominals_indicator(acc_name, file_name)
        self.ui.log_message("Arquivo nominal do "+acc_name+" carregado.")

        # checks if all required data were loaded
        self.chack_all_files_and_process(acc_name)

    def load_measured_file(self, acc_name: str, file_path=None) -> None:
        """ Loads files with list of measured points and saves it in the form
            of a DataFrame into the associated accelerator object. """

        if(file_path == ""):
            return

        filepath = file_path or self.openFileNameDialog('pontos medidos', acc_name)

        if(not filepath):
            return

        file_name = filepath.split('/')[len(filepath.split('/')) - 1]
        file_extension = file_name.split('.')[1]

        if (file_extension == 'txt' or file_extension == 'csv' or file_extension == 'TXT' or file_extension == 'CSV'):
            measured = read_csv(filepath, 'points')
        elif (file_extension == 'xlsx'):
            measured = read_excel(filepath, 'points')
        else:
            self.ui.log_message("Formato do arquivo "+file_name+" não suportado.", severity="alert")
            return

        self.accelerators[acc_name].points['measured'] = measured

        self.isMeasuredLoaded[acc_name] = True
        self.ptsMeasFileName[acc_name] = filepath

        self.ui.update_measured_indicator(acc_name, file_name)
        self.ui.log_message("Arquivo com medidos do "+acc_name+" carregado.")

        self.chack_all_files_and_process(acc_name)

    def loadLookuptableFile(self, acc_name: str, file_path=None) -> None:
        """ Loads files with list of mapping between the machine-local and the
            magnet-specific nominal frame, and saves it in the form
            of a DataFrame into the associated accelerator object. """

        if(file_path == ""):
            return

        filepath = file_path or self.openFileNameDialog('lookup de frames', acc_name)

        if(not filepath):
            return

        file_name = filepath.split('/')[len(filepath.split('/')) - 1]
        file_extension = file_name.split('.')[1]

        if (file_extension == 'txt' or file_extension == 'csv' or file_extension == 'TXT' or file_extension == 'CSV'):
            lookuptable = read_csv(filepath, 'lookuptable')
        elif (file_extension == 'xlsx'):
            lookuptable = read_excel(filepath, 'lookuptable')
        else:
            self.ui.log_message("Format of file "+file_name+"not supported.")
            return

        self.accelerators[acc_name].ml_to_nominal_frame_mapping = lookuptable

        self.isLookuptableLoaded[acc_name] = True
        self.lookupFileName[acc_name] = filepath
        
        self.ui.update_lookup_indicator(acc_name, file_name)
        self.ui.log_message("Lookuptable de frames do " + acc_name + " carregada.")

        self.chack_all_files_and_process(acc_name)

    def chack_all_files_and_process(self, acc_name: str) -> None:
        """ Checks the internal flags to see if all files needed to process the 
            raw data were loaded, and if so triggers the processing.  """

        if((self.isNominalsLoaded[acc_name] and self.isMeasuredLoaded[acc_name] and self.isLookuptableLoaded[acc_name])):
            self.ui.log_message(f"Transformando pontos do {acc_name} para os frames locais...")
            self.processEvents()

            accelerator = self.accelerators[acc_name]
            self.process_raw_data(accelerator)

            self.isInternalDataProcessed[acc_name] = True
            self.ui.log_message(f"Dados do {acc_name} prontos.", severity="sucess")

    def process_raw_data(self, accelerator: Accelerator) -> None:
        """ Handles the flux of calls for calculating the accelerator's magnets deviations. """

        # nominal frames are first created, based on mapping from input file
        accelerator.create_nominal_frames()
        
        # magnets are then populated with points that were also loaded from input files
        accelerator.populate_magnets_with_points('nominal')
        accelerator.populate_magnets_with_points('measured')

        # all nominal and measured points are transformed to be referenced in its magnets nominal frame
        accelerator.change_to_local_frame('nominal')
        accelerator.change_to_local_frame('measured')

        # measured frames are then calculated
        accelerator.calculate_measured_frames(self.ui, magnet_types_to_ignore=['sextupole'])

        # frames needs to be correctly sorted to present deviations in the right order
        accelerator.sort_frames_by_beam_trajectory('nominal')
        accelerator.sort_frames_by_beam_trajectory('measured')

        # finally, calculates the magnets deviations
        if accelerator.evaluate_magnets_deviations():
            self.ui.enable_plot_button(accelerator.name)
        else:
            self.ui.log_message(f'Plot não foi possível para {accelerator.name} devido a ausência de frames medidos e/ou nominais.', severity='danger')

        # update report for user control
        self.report[accelerator.name] = generate_report(accelerator.frames['measured'], accelerator.frames['nominal'])

    def plot_absolute_deviations(self, acc_name: str) -> None:
        """ Call the plot function to display absolute deviation data in a predefined template. """

        self.ui.log_message(f"Plotando o perfil de alinhamento absoluto do {acc_name}...")
        self.processEvents()

        deviations = []

        if(acc_name == 'LTB' or acc_name == 'BTS'):
            deviations.append(self.accelerators['LTB'].deviations)
            deviations.append(self.accelerators['BTS'].deviations)
        elif(acc_name == 'FE'):
            deviations.append(separate_frontend_data(self.accelerators[acc_name].deviations, ['MAN', 'EMA', 'CAT']))
            deviations.append(separate_frontend_data(self.accelerators[acc_name].deviations, ['IPE', 'CAR', 'MOG']))
        else:
            deviations.append(self.accelerators[acc_name].deviations)

        plotArgs = self.ui.get_plot_args_from_ui(acc_name)

        if (acc_name == 'FE'):
            plotArgs['FElist'] = ['MANACÁ', 'EMA', 'CATERETÊ']
            plot_magnets_absolute_deviation(deviations[0], **plotArgs)
            plotArgs['FElist'] = ['IPÊ', 'CARNAÚBA', 'MOGNO']
            plot_magnets_absolute_deviation(deviations[1], **plotArgs)
        else:
            plot_magnets_absolute_deviation(deviations, **plotArgs)


if __name__ == "__main__":
    App()

""" 
PENDÊNCIAS
- corrigir ordem dos imãs do Booster, LTB, BTS e FE
- [modelo liu - long.] avaliação de distancias entre dipolos e quadrupolos 
- [modelo liu - vert.] avaliação individual de cada quadrupolo e dipolo
- estimar série de fourier
- substrair para obter alinhamento relativo
"""
