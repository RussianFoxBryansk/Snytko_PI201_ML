import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from neuron import OurNeuralNetwork

def prepare_data(file_path):
    np_dataset = np.array(pd.read_excel(file_path))
    np_y = np_dataset[:, 3].reshape(-1, 1)
    np_x1 = np_dataset[:, 0].reshape(-1, 1)
    np_x2 = np_dataset[:, 1].reshape(-1, 1)
    np_x3 = np_dataset[:, 2].reshape(-1, 1)

    labelencoder = LabelEncoder()
    np_y = labelencoder.fit_transform(np_y.ravel())  # Убедитесь, что это одномерный массив
    data = np.hstack((np_x1, np_x2, np_x3))
    all_y_trues = np_y
    return data, all_y_trues

def train_model(data, all_y_trues):
    network = OurNeuralNetwork()
    network.train(data, all_y_trues)
    network.save_weights('neuron_weights.txt')

if __name__ == "__main__":
    data, all_y_trues = prepare_data('DATASET.XLSX')
    train_model(data, all_y_trues)
