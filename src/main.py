from ui import Ui
from PyQt5 import QtWidgets
import sys


class App(QtWidgets.QApplication):
    def __init__(self):
        # Initializing the application and its UI
        super(App, self).__init__(sys.argv)

        # Calling the UI component
        self.ui = Ui(self)

        # Changuing the default window theme
        self.setStyle('Fusion')

        # Calling the app to execute
        sys.exit(self.exec_())


if __name__ == "__main__":
    App()

""" 
PENDÊNCIAS
- correção nomenclatura
- correção sinal bestgit
- padronização do título e eixos
- plot de tolerancias (especificá-las)
- [modelo liu - long.] avaliação de distancias entre dipolos e quadrupolos 
- [modelo liu - vert.] avaliação individual de cada quadrupolo e dipolo
- estimar série de fourier
- substrair para obter alinhamento relativo
"""
