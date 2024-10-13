# Импорт зависимостей
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

# Подготовим данные для обучения
# Расстояние в км, способ перемещения (0 - авто, 1 - поезд, 2 - автобус), затраты в рублях
data = np.array([[50, 0, 1000],   # 50 км, авто, 1000₽
                 [100, 0, 2000],  # 100 км, авто, 2000₽
                 [200, 1, 1500],  # 200 км, поезд, 1500₽
                 [300, 1, 3000],  # 300 км, поезд, 3000₽
                 [500, 2, 2500]], # 500 км, автобус, 2500₽
                dtype=float)

# Метки классов: 1 - "эконом", 0 - "бизнес"
y_class = np.array([1, 1, 0, 0, 1])  

for i, row in enumerate(data):
    distance, transport_mode, cost = row
    print(f"Поездка на {distance} км с помощью {'авто' if transport_mode == 0 else 'поезда' if transport_mode == 1 else 'автобуса'} "
          f"обойдется примерно в {cost}₽ и классифицируется как {'эконом' if y_class[i] == 1 else 'бизнес'}.")

# Создание модели для классификации
model_class = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=[data.shape[1]]),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Один выход для бинарной классификации
])

model_class.compile(optimizer='adam', loss='binary_crossentropy')

# Обучение модели
model_class.fit(data, y_class, epochs=100, batch_size=1, verbose=1)
print("Завершили тренировку модели классификации")

# Прогноз
test_data = np.array([[600, 1, 3500]])  # 600 км, поезд, 3500₽
y_pred_class = model_class.predict(test_data)

# Интерпретация предсказания
predicted_label = 'Эконом' if y_pred_class[0][0] >= 0.5 else 'Бизнес'
print(f"Модель предсказала, что поездка на 600 км на поезде с затратами 3500₽ будет классифицирована как: {predicted_label}")

# Сохранение модели для классификации
model_class.save('classification_model.h5')
