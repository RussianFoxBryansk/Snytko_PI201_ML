import pickle
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify

app = Flask(__name__)


menu = [{"name": "Главная", "url": "/"},
        {"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"}]


###########################################################
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from math import sqrt
data = pd.read_excel('DATASET.xlsx')

label_encoder=LabelEncoder()
data["Окрас"]=label_encoder.fit_transform(data["Окрас"])
data["Порода"]=label_encoder.fit_transform(data["Порода"])

shuffle_index = np.random.permutation(data.shape[0])
req_data = data.iloc[shuffle_index]

train_size = int(req_data.shape[0]*0.7)
train_df = req_data.iloc[:train_size,:]
test_df = req_data.iloc[train_size:,:]
train = train_df.values
test = test_df.values
y_true = test[:,-1]
print('Train_Shape: ',train_df.shape)
print('Test_Shape: ',test_df.shape)


def euclidean_distance(x_test, x_train):
    distance = 0
    for i in range(len(x_test)-1):
        distance += (x_test[i]-x_train[i])**2
        return sqrt(distance)

def get_neighbors(x_test, x_train, num_neighbors):
    distances = []
    data = []
    for i in x_train:
        distances.append(euclidean_distance(x_test,i))
        data.append(i)
        distances = np.array(distances)
        data = np.array(data)
        sort_indexes = distances.argsort()             #argsort() функция возвращает индексы путем сортировки данных о расстояниях в порядке возрастания
        data = data[sort_indexes]                      #изменяем наши данные на основе отсортированных индексов, чтобы мы могли получить ближайших соседей
        return data[:num_neighbors]

def prediction(x_test, x_train, num_neighbors):
    classes = []
    neighbors = get_neighbors(x_test, x_train, num_neighbors)
    for i in neighbors:
        classes.append(i[-1])
        predicted = max(classes, key=classes.count)
        return predicted

def accuracy(y_true, y_pred):
    num_correct = 0
    for i in range(len(y_true)):
        if y_true[i]==y_pred[i]:
            num_correct+=1
            accuracy = num_correct/len(y_true)
            return accuracy

y_pred = []
for i in test:
    y_pred.append(prediction(i, train, 5))

accuracy = accuracy(y_true, y_pred)



#############################################################


model2 = pickle.load(open('model2/Cat.ai', 'rb'))
model = pickle.load(open('model/HeightWeightGender=FootSize', 'rb'))


@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные Снытко Русланом Николаевичем", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод ближайших соседей (собственный)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = prediction(X_new, train, 3)
        if pred == 0:
            preds = '0 - Британская'
        elif pred == 1:
            preds = '1 - Персидская'
        else:
            preds = '2 - Сиамская'
        return render_template('lab1.html', title="Метод ближайших соседей (собственный)", menu=menu,
                               class_model = "Это: " + preds)

@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu)
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = str(model2.predict(X_new)[0])
        if pred == 0:
            preds = '0 - Британская'
        elif pred == 1:
            preds = '1 - Персидская'
        else:
            preds = '2 - Сиамская'
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu,
                               class_model = "Это: " + preds)

@app.route("/p_lab3", methods=['POST', 'GET'])
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html', title="Линейная регрессия", menu=menu)
    elif request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3'])]])
        preds = str('{:.2f}'.format(model.predict(X_new)[0][0]))
        return render_template('lab3.html', title="Линейная регрессия", menu=menu, class_model="Это: " + str(preds))


@app.route('/api', methods=[ 'GET'])
def get_sort():
    X_new = np.array([[float(request.args.get('list1')),
                           float(request.args.get('list2')),
                           float(request.args.get('list3'))]])
    pred = str('{:.2f}'.format(model.predict(X_new)[0][0]))

    return jsonify(sort=pred)

@app.route('/api_v2', methods=['GET'])
def get_sort_v2():
    request_data = request.get_json()
    X_new = np.array([[float(request_data['list1']),
                           float(request_data['list2']),
                           float(request_data['list3'])]])
    pred = str('{:.2f}'.format(model.predict(X_new)[0][0]))

    return jsonify(sort=pred)

if __name__ == "__main__":
    app.run(debug=True)
