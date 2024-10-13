# Импорт зависимостей
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

# Подготовим данные для тренировки
# Расстояние в км, способ перемещения (0 - авто, 1 - поезд, 2 - автобус), затраты в рублях
data = np.array([[50, 0, 1000],   # 50 км, авто, 1000₽
                 [100, 0, 2000],  # 100 км, авто, 2000₽
                 [200, 1, 1500],  # 200 км, поезд, 1500₽
                 [300, 1, 3000],  # 300 км, поезд, 3000₽
                 [500, 2, 2500]], # 500 км, автобус, 2500₽
                dtype=float)

# Здесь мы будем предсказывать время в пути как целевую переменную для регрессии
time_needed = np.array([1, 2, 3, 5, 8], dtype=float)  

# Создание модели для регрессии
model_reg = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=[data.shape[1]]),  # Входной слой
    tf.keras.layers.Dense(5, activation='relu'),  # Скрывательный слой
    tf.keras.layers.Dense(1, activation='linear')  # Один выход для регрессии
])

model_reg.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
history = model_reg.fit(data, time_needed, epochs=500, verbose=1)
print("Завершили тренировку модели регрессии")



# Прогноз времени для поездки на 600 км, поезд, 3500₽
test_data = np.array([[600, 1, 3500]])  # 600 км, поезд, 3500₽
predicted_time = model_reg.predict(test_data)
print(f"Модель предсказала, что поездка на 600 км на поезде с затратами 3500₽ займет около {predicted_time[0][0]} часов.")

# Сохранение модели для регрессии
model_reg.save('regression_model.h5')

