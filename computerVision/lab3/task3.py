import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


def weighted_median(image, weights):
    size = weights.shape[0]
    padding = size // 2
    expanded = np.pad(image, padding, mode="edge")
    result = np.zeros_like(image)

    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            window = expanded[row:row + size, column:column + size]
            # Вес задает число повторений, а не множитель яркости.
            values = np.repeat(window.ravel(), weights.ravel())
            values.sort()
            result[row, column] = values[len(values) // 2]
    return result


def rank_filter(image, size, rank):
    padding = size // 2
    expanded = np.pad(image, padding, mode="edge")
    result = np.zeros_like(image)

    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            window = expanded[row:row + size, column:column + size]
            values = np.sort(window, axis=None)
            result[row, column] = values[rank - 1]
    return result


def wiener_filter(image, size):
    values = image.astype(float)
    mean = cv2.blur(values, (size, size), borderType=cv2.BORDER_REPLICATE)
    mean_square = cv2.blur(values ** 2, (size, size), borderType=cv2.BORDER_REPLICATE)
    variance = np.maximum(mean_square - mean ** 2, 0)

    # Дисперсию шума оцениваем без исходного изображения.
    noise_variance = variance.mean()
    gain = np.maximum(variance - noise_variance, 0) / np.maximum(variance, 0.00000001)
    result = mean + gain * (values - mean)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def adaptive_median(image, max_size):
    padding = max_size // 2
    expanded = np.pad(image, padding, mode="edge")
    # Без подходящего окна оставляем исходный пиксель.
    result = image.copy()

    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            for size in range(3, max_size + 1, 2):
                radius = size // 2
                center_row = row + padding
                center_column = column + padding
                window = expanded[center_row - radius:center_row + radius + 1,
                                  center_column - radius:center_column + radius + 1]
                minimum = int(window.min())
                maximum = int(window.max())
                median = int(np.median(window))

                # Если медиана сама выброс, увеличиваем окно.
                if minimum < median < maximum:
                    pixel = int(image[row, column])
                    if not minimum < pixel < maximum:
                        result[row, column] = median
                    break
    return result


os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

original = cv2.imread("results/task1_original.png", cv2.IMREAD_GRAYSCALE)
noise_names = ["gaussian", "salt", "pepper", "salt_pepper", "speckle", "poisson"]
noise_titles = ["Гаусс", "Соль", "Перец", "Соль-перец", "Спекл", "Пуассон"]
measurements = []

weights3 = np.ones((3, 3), dtype=int)
weights3[1, 1] = 3
weights5 = np.ones((5, 5), dtype=int)
weights5[2, 2] = 9

for noise_index in range(len(noise_names)):
    noise_name = noise_names[noise_index]
    noisy = cv2.imread("results/task1_" + noise_name + ".png", cv2.IMREAD_GRAYSCALE)
    print("Обрабатываем шум:", noise_titles[noise_index], flush=True)

    methods = [
        ("median_3", "Медиана 3x3", cv2.medianBlur(noisy, 3)),
        ("median_5", "Медиана 5x5", cv2.medianBlur(noisy, 5)),
        ("weighted_3", "Взвешенная 3x3, центр = 3", weighted_median(noisy, weights3)),
        ("weighted_5", "Взвешенная 5x5, центр = 9", weighted_median(noisy, weights5))
    ]

    # Низкий ранг затемняет, высокий — осветляет.
    for size, rank in [(3, 3), (3, 7), (5, 7), (5, 19)]:
        name = "rank_" + str(size) + "_r" + str(rank)
        title = "Ранговый " + str(size) + "x" + str(size) + ", ранг = " + str(rank)
        methods.append((name, title, rank_filter(noisy, size, rank)))

    methods.append(("wiener_3", "Винер 3x3", wiener_filter(noisy, 3)))
    methods.append(("wiener_5", "Винер 5x5", wiener_filter(noisy, 5)))
    methods.append(("adaptive_5", "Адаптивная: до 5x5", adaptive_median(noisy, 5)))
    methods.append(("adaptive_9", "Адаптивная: до 9x9", adaptive_median(noisy, 9)))

    for name, title, filtered in methods:
        mse = np.mean((original.astype(float) - filtered.astype(float)) ** 2)
        measurements.append([noise_name, name, round(float(mse), 4)])
        cv2.imwrite("results/task3_" + noise_name + "_" + name + ".png", filtered)
        print(noise_titles[noise_index], "/", title, ": MSE =", round(float(mse), 2))

    for start in range(0, len(methods), 4):
        images = [original, noisy]
        titles = ["Исходное изображение", "С шумом: " + noise_titles[noise_index]]
        for name, title, filtered in methods[start:start + 4]:
            mse = np.mean((original.astype(float) - filtered.astype(float)) ** 2)
            images.append(filtered)
            titles.append(title + "\nMSE = " + str(round(float(mse), 1)))

        plt.figure(figsize=(12, 7))
        for index in range(len(images)):
            plt.subplot(2, 3, index + 1)
            plt.imshow(images[index], cmap="gray", vmin=0, vmax=255)
            plt.title(titles[index])
            plt.axis("off")
        plt.tight_layout()
        plt.savefig("results/task3_" + noise_name + "_comparison_" +
                    str(start // 4 + 1) + ".png", dpi=120)

with open("results/task3_metrics.csv", "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["noise", "method", "mse"])
    writer.writerows(measurements)

plt.show()
