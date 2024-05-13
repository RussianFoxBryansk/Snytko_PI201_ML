
import pickle


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


labelencoder = LabelEncoder()




np_dataset = np.array(pd.read_excel('DATASET.XLSX'))



np_y = np_dataset[:, 3].reshape(-1, 1)
np_x1 = np_dataset[:, 0].reshape(-1, 1)
np_x2 = np_dataset[:, 1].reshape(-1, 1)
np_x3 = np_dataset[:, 2].reshape(-1, 1)

np_x3 = labelencoder.fit_transform(np_x3).reshape(-1, 1)


sumx = np.hstack((np_x1, np_x2,np_x3))


model = LinearRegression()
model.fit(sumx, np_y)


with open('HeightWeightGender=FootSize','wb') as pkl:
    pickle.dump(model, pkl)

