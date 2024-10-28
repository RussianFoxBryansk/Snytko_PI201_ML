import numpy as np
from neuron import OurNeuralNetwork


def load_and_test_model(test_data):
    model = OurNeuralNetwork()
    try:
        model.load_weights('neuron_weights.txt')
    except FileNotFoundError:
        print("Ошибка: файл 'neuron_weights.txt' не найден.")
        return

    predictions = np.apply_along_axis(model.feedforward, 1, test_data)

    # Преобразуем предсказания в классы
    gender = ['Мужчина' if p >= 0.5 else 'Женщина' for p in predictions]

    print("Предсказанные значения:", predictions)
    print("Классы:", gender)


if __name__ == "__main__":
    # Убедитесь, что размер test_data соответствует ожидаемым входным данным
    test_data = np.array([[175, 70, 42]])  # Пример тестовых данных
    load_and_test_model(test_data)

