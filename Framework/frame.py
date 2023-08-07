
import os
import sys
import UI.MainWindow as win
from PyQt5.QtWidgets import QApplication, QMainWindow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = win.Ui_MainWindow()
    win.show()
    sys.exit(app.exec_())

