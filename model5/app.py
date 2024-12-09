from flask import Flask, request, render_template, jsonify
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import os

app = Flask(__name__)

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

@app.route('/', methods=['GET'])
def index():
    return render_template('lab7.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Открытие изображения
    image = Image.open(io.BytesIO(file.read()))

    # Выполнение обнаружения
    results = model(image)

    # Список классов, которые нас интересуют
    desired_classes = ['person', 'car']
    confidence_threshold = 0.5

    # Фильтрация результатов по классам и вероятности
    filtered_results = results.pandas().xyxy[0]
    filtered_results = filtered_results[(filtered_results['name'].isin(desired_classes)) & (filtered_results['confidence'] >= confidence_threshold)]

    # Преобразование изображения в массив NumPy
    filtered_image = np.array(image)

    # Преобразование результатов в формат, подходящий для отображения
    for _, row in filtered_results.iterrows():
        label = row['name']
        conf = row['confidence']
        xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
        color = (0, 255, 0)  # Зеленый цвет для рамки
        cv2.rectangle(filtered_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(filtered_image, f'{label} {conf:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Сохранение отфильтрованного изображения
    output_image_path = os.path.join('static', 'filtered_image.png')
    cv2.imwrite(output_image_path, filtered_image)

    # Сохранение результатов в CSV-файл
    results_csv_path = "results.csv"
    filtered_results.to_csv(results_csv_path, index=False)

    # Возвращаем результаты в шаблон
    return render_template('lab7.html', filtered_image_path=output_image_path, detections=filtered_results.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
