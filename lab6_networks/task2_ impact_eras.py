from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((60000, 784)).astype("float32") / 255.0
x_test = x_test.reshape((10000, 784)).astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# Функция для создания и обучения модели
def build_and_train_model(epochs):
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Засекаем время обучения
    import time
    start_time = time.time()
    model.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=2)
    train_time = time.time() - start_time

    # Засекаем время оценки (скорость работы сети)
    start_time = time.time()
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    eval_time = time.time() - start_time

    return test_acc, train_time, eval_time


# Сравним, например, 5, 10 и 20 эпох
for epochs in [5, 10, 20]:
    acc, train_time, eval_time = build_and_train_model(epochs)
    print(f"Эпохи: {epochs}")
    print(f"Точность {acc:.4f}")
    print(f"Время тренировки: {train_time:.2f} s")
    print(f"Время оценки: {eval_time:.4f} s")
    print("-" * 30)
