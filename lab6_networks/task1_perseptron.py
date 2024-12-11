import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Предобработка данных
# Преобразуем в вектор длины 784 и нормализуем
x_train = x_train.reshape((60000, 784)).astype("float32") / 255.0 #x1 коолиетво образов x2 количество пикселей
x_test = x_test.reshape((10000, 784)).astype("float32") / 255.0

# Преобразуем метки в one-hot векторы
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Создаем модель многослойного персептрона
model = keras.Sequential([ #модель Керас, создаёт нейронку как последовательность слоёв
    layers.Dense(512, activation='relu', input_shape=(784,)), #x1 кол-во нейронов x2 функция активации x3форма входа
    #одномерный вектор дляной 784
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=2)

# Оценка на тестовых данных
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Точность", test_acc)
