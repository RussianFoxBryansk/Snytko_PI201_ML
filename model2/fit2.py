import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)  # Установите большее значение, например, 1000
label_encoder = LabelEncoder()

df = pd.read_excel("DATASET.xlsx")

df["Окрас"] = label_encoder.fit_transform(df["Окрас"])
df["Порода"] = label_encoder.fit_transform(df["Порода"])

X = df.drop(["Порода"], axis=1)
Y = df["Порода"]
X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,test_size=0.2,random_state=3)

model.fit(X_train1, Y_train1)

import pickle

with open('Cat.ai','wb') as pkl:
    pickle.dump(model, pkl)

