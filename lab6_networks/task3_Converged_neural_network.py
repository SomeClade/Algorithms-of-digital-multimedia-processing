from tensorflow import keras
from tensorflow.keras import layers

# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Для сверточных сетей нужно привести данные к форме (кол-во_образцов, высота, ширина, каналы)
x_train = x_train.reshape((60000, 28, 28, 1)).astype("float32") / 255.0 # создаётся 4 мерный массив, где x1 - количество образов
# x2 x3 - размеры изображений x4 - канальность преобразуем в float32 для точности и делим на 255 чтобы перейти к диапозону от 0 до 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype("float32") / 255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Создадим простую CNN
model = keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', #Adaptive Moment Estimation
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=2)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Точность:", test_acc)
