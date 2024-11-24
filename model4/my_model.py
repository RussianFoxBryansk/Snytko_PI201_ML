import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
import pathlib

# Извлечение данных из zip-архива
with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('dataset_market')

# Установка пути к директории с данными
data_dir = pathlib.Path("dataset_market/dataset")

# Установка параметров
batch_size = 32
img_height = 180
img_width = 180

# Используем ImageDataGenerator для аугментации данных
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Создаем обучающий и валидационный наборы данных
train_ds = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training',
    seed=123
)

val_datagen = ImageDataGenerator(validation_split=0.2, rescale=1. / 255)

val_ds = val_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    seed=123
)

class_names = list(train_ds.class_indices.keys())

# Открываем файл для записи
with open('class_names.txt', 'w') as file:
    for class_name in class_names:
        file.write(class_name + '\n')

print("Список классов сохранен в class_names.txt")

# Создание простой модели CNN
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

# Обучение модели
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Оценка модели на валидационном наборе
test_loss, test_accuracy = model.evaluate(val_ds)
print("Доля верных ответов на тестовых данных, в процентах:", round(test_accuracy * 100, 4))

# Визуализация результатов обучения
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Сохраняем модель
model.save("my_model.h5")

# Загружаем тестовое изображение
img = tf.keras.utils.load_img(
    "i.webp", target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Создаем пакет

# Предсказание класса изображения
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(score)
print(
    f"Это изображение похоже на {class_names[np.argmax(score)]} с вероятностью {100 * np.max(score):.2f} процентов."
)
