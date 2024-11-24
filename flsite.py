import os
import pickle
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from model3.neuron import OurNeuralNetwork

app = Flask(__name__)


menu = [{"name": "Главная", "url": "/"},
        {"name": "Лаб1", "url": "p_knn"},
        {"name": "Лаб2", "url": "p_lab2"},
        {"name": "Лаб3", "url": "p_lab3"},
        {"name": "Лаб4", "url": "neuron1"},
        {"name": "Лаб5", "url": "api_reg_tf?distance=600&transport_mode=1&cost=3500"},
        {"name": "Лаб6", "url": "api_class_tf?distance=600&transport_mode=1&cost=3500"}]


###########################################################
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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


model_reg = tf.keras.models.load_model('model3/regression_model.h5')
model_class = tf.keras.models.load_model('model3/classification_model.h5')
model_fm = tf.keras.models.load_model('model4/fashion_mnist_model.h5')
model_mm = tf.keras.models.load_model('model4/flowers_4_model.h5')
model_my = tf.keras.models.load_model('model4/my_model.h5')

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


new_neuron = OurNeuralNetwork()
try:
    new_neuron.load_weights('model3/neuron_weights.txt')
    print("Найден файл 'neuron_weights.txt' .")
except FileNotFoundError:
    print("Ошибка: файл 'neuron_weights.txt' не найден.")

@app.route("/neuron1", methods=['POST', 'GET'])
def p_lab4():
    if request.method == 'GET':
        return render_template('lab4.html', title="Первый нейрон", menu=menu, class_model='')

    if request.method == 'POST':
        try:
            # Получаем данные из формы
            height = float(request.form['height'])  # Рост
            weight = float(request.form['weight'])  # Вес
            size = float(request.form['size'])  # Размер

            # Формируем входные данные для нейронной сети
            X_new = np.array([[height, weight, size]])  # Убедитесь, что 3 параметра
            print(X_new)
            # Получаем предсказания от нейронной сети

            predictions = np.apply_along_axis(new_neuron.feedforward, 1, X_new)

            # Определяем пол на основе предсказания
            gender = 'Мужчина' if predictions[0] >= 0.5 else 'Женщина'

            print("Предсказанные значения:", predictions, gender)

            return render_template('lab4.html', title="Первый нейрон", menu=menu,
                                   class_model="Это: " + gender)

        except Exception as e:
            print("Ошибка:", e)  # Логируем ошибку
            return render_template('lab4.html', title="Первый нейрон", menu=menu,
                                   class_model="Произошла ошибка: " + str(e))


@app.route('/api_reg_tf', methods=['GET'])
def predict_regression():
    # Получение данных из запроса "http://localhost:5000/api_reg_tf?distance=600&transport_mode=1&cost=3500"
    distance = float(request.args.get('distance'))
    transport_mode = int(request.args.get('transport_mode'))
    cost = float(request.args.get('cost'))

    # Формирование входных данных для модели
    input_data = np.array([[distance, transport_mode, cost]])

    # Предсказание времени в пути
    predictions = model_reg.predict(input_data)

    return jsonify(time_needed=str(predictions[0][0]))

@app.route('/api_class_tf', methods=['get'])
def predict_classification():
    # Получение данных из запроса http://localhost:5000/api_class_tf?distance=600&transport_mode=1&cost=3500
    distance = float(request.args.get('distance'))
    transport_mode = int(request.args.get('transport_mode'))
    cost = float(request.args.get('cost'))

    # Формирование входных данных для модели
    input_data = np.array([[distance, transport_mode, cost]])

    # Предсказание
    predictions = model_class.predict(input_data)
    # Интерпретация предсказания
    if predictions[0][0] >= 0.5:
        result = 'Эконом'
    else:
        result = 'Бизнес'

    # Установите флаг для корректного отображения кириллицы в JSON
    app.config['JSON_AS_ASCII'] = False

    # Возврат результата
    return jsonify(result=result)


@app.route("/tf_mnis", methods=['GET', 'POST'])
def tf_mnis():
    if request.method == 'GET':
        return render_template('lab5.html')

    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                raise Exception("Файл не найден")
            file = request.files['file']

            if file.filename == '':
                raise Exception("Файл не выбран")
            if file and allowed_file(file.filename):
                file_path = os.path.join('static', file.filename)
                file.save(file_path)

                img = image.load_img(file_path, target_size=(28, 28), color_mode='grayscale')
                x = image.img_to_array(img)
                x = x.reshape(1, 784)  # Нормализация для модели
                x /= 255.0  # Нормализация данных

                prediction = model_fm.predict(x)  # Предсказание на основе загруженного изображения
                predicted_class_index = np.argmax(prediction)

                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                predicted_class_name = class_names[predicted_class_index]

                return render_template('lab5.html', class_name=predicted_class_name, image_path=file.filename)

            else:
                raise Exception("Неподдерживаемый файл")
        except Exception as e:
            print("Ошибка:", e)
            return render_template('lab5.html', error=str(e))


# Функция для загрузки названий классов из файла
def load_class_names(file_path):
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

# Загружаем названия классов один раз при старте приложения
class_names = load_class_names('model4/class_names.txt')

@app.route("/my_mnis", methods=['GET', 'POST'])
def my_mnis():
    if request.method == 'GET':
        return render_template('lab6.html')

    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                raise Exception("Файл не найден")
            file = request.files['file']

            if file.filename == '':
                raise Exception("Файл не выбран")

            if file and allowed_file(file.filename):
                file_path = os.path.join('static', file.filename)
                file.save(file_path)

                # Загрузка и предварительная обработка изображения
                img = image.load_img(file_path, target_size=(180, 180))  # Корректный размер
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)  # Создаем пакет из 1 изображения
                x /= 255.0  # Нормализация

                # Получаем выбранную модель
                selected_model = request.form['model']

                # Предсказание с использованием выбранной модели
                if selected_model == 'model1':
                    prediction = model_mm.predict(x)  # Используем MobileNetV2
                    model_name = "MobileNetV2"
                elif selected_model == 'model2':
                    prediction = model_my.predict(x)  # Используем собственный нейрон
                    model_name = "Собственный нейрон"

                # Определяем класс и вероятность из предсказания
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = class_names[predicted_class_index]
                prediction_probability = np.max(prediction)  # Максимальная вероятность

                # Выводим информацию в консоль
                print(f"Модель: {model_name}, Предсказанный класс: {predicted_class_name}, Вероятность: {prediction_probability:.4f}")

                return render_template('lab6.html', class_name=predicted_class_name, image_path=file.filename)

            else:
                raise Exception("Неподдерживаемый файл")
        except Exception as e:
            print("Ошибка:", e)
            return render_template('lab6.html', error=str(e))



# Функция для проверки типов файлов
def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


if __name__ == "__main__":
    app.run(debug=True)
