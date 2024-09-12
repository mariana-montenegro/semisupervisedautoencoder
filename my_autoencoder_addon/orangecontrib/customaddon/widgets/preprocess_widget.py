from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data import Table, Domain, ContinuousVariable
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QLabel
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit
import numpy as np

class TestSizeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Test Size")
        
        self.layout = QFormLayout(self)
        self.test_size_input = QLineEdit(self)
        self.test_size_input.setText("0.3")
        self.layout.addRow("Test Size (between 0 and 1):", self.test_size_input)
        
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox)

    def get_test_size(self):
        try:
            return float(self.test_size_input.text())
        except ValueError:
            return None

class OWPreprocessing(OWWidget):
    name = "Preprocessing"
    description = "Preprocess the data and prepare train and test datasets."
    icon = "widget_icons/Preprocessing.svg"

    class Inputs:
        input_data = Input("Data", Table)

    class Outputs:
        data_train = Output("Train Data", Table)
        data_test = Output("Test Data", Table)

    def __init__(self):
        super().__init__()

        self.description_label = QLabel("This widget preprocesses the data for training and testing.")
        self.controlArea.layout().addWidget(self.description_label)

        self.set_test_size_button = QPushButton("Set Test Size")
        self.set_test_size_button.clicked.connect(self.open_test_size_dialog)
        self.controlArea.layout().addWidget(self.set_test_size_button)

        self.preprocess_button = QPushButton("Preprocess Data")
        self.preprocess_button.clicked.connect(self.manual_preprocess)
        self.controlArea.layout().addWidget(self.preprocess_button)

        self.info_label = QLabel("Waiting for data...")
        self.controlArea.layout().addWidget(self.info_label)

        self.data = None
        self.data_train = None
        self.data_test = None
        self.test_size = 0.3  # default test size

    @Inputs.input_data
    def set_input_data(self, data):
        if data is not None:
            self.info_label.setText("Data received. Ready to preprocess.")
            self.data = data
        else:
            self.info_label.setText("No data received.")

    def open_test_size_dialog(self):
        dialog = TestSizeDialog(self)
        if dialog.exec_():
            test_size = dialog.get_test_size()
            if test_size is not None and 0 < test_size < 1:
                self.test_size = test_size
                self.info_label.setText(f"Test size set to {self.test_size}")
            else:
                self.info_label.setText("Invalid test size. Please enter a value between 0 and 1.")

    def manual_preprocess(self):
        if self.data is not None:
            self.info_label.setText("Preprocessing data...")
            self.preprocess_data(self.data)
        else:
            self.info_label.setText("No data loaded to preprocess.")

    def preprocess_data(self, data):
        test_size = self.test_size

        # Ensure the data has a class variable
        class_var = data.domain.class_var
        if class_var is None:
            self.info_label.setText("Data does not have a class variable.")
            return

        # Use Orange's built-in methods for filtering
        filtered_data = data[data.Y != 'CTR']

        # Perform one-hot encoding
        onehot = onehot()
        encoded_data = onehot(filtered_data)

        # Split data into train and test sets
        data_train, data_test = encoded_data.split(1 - test_size)

        self.data_train = data_train
        self.data_test = data_test

        self.info_label.setText("Data ready: Train and Test sets created")
        self.Outputs.data_train.send(self.data_train)
        self.Outputs.data_test.send(self.data_test)
