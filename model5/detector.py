import torch
import cv2
import numpy as np
import pandas
from PIL import Image
import requests
import ultralytics

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Загрузка изображения
image = Image.open('1.png')

# Выполнение обнаружения
results = model(image)

# Отображение результатов
results.show()

# Список классов, которые нас интересуют
desired_classes = ['person', 'car']

# Порог вероятности
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

# Отображение отфильтрованного изображения
cv2.imshow("Filtered Image", filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Сохранение отфильтрованного изображения
cv2.imwrite("filtered_image.png", filtered_image)

# Сохранение результатов в CSV-файл
filtered_results.to_csv("results.csv", index=False)
