import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


def contraharmonic(image, size, order):
    # Ноль заменяем малым числом для отрицательных степеней.
    values = np.maximum(image.astype(float), 0.000001)

    # В отношении средних общий делитель сокращается.
    numerator = cv2.blur(values ** (order + 1), (size, size),
                         borderType=cv2.BORDER_REPLICATE)
    denominator = cv2.blur(values ** order, (size, size),
                           borderType=cv2.BORDER_REPLICATE)
    result = numerator / denominator
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

original = cv2.imread("results/task1_original.png", cv2.IMREAD_GRAYSCALE)
noise_names = ["gaussian", "salt", "pepper", "salt_pepper", "speckle", "poisson"]
noise_titles = ["Гаусс", "Соль", "Перец", "Соль-перец", "Спекл", "Пуассон"]
measurements = []

for noise_index in range(len(noise_names)):
    noise_name = noise_names[noise_index]
    noisy = cv2.imread("results/task1_" + noise_name + ".png", cv2.IMREAD_GRAYSCALE)

    gaussian3 = cv2.GaussianBlur(noisy, (3, 3), 0.8, borderType=cv2.BORDER_REPLICATE)
    gaussian7 = cv2.GaussianBlur(noisy, (7, 7), 1.6, borderType=cv2.BORDER_REPLICATE)
    methods = [("gaussian_3", "Гаусс 3x3, sigma = 0.8", gaussian3),
               ("gaussian_7", "Гаусс 7x7, sigma = 1.6", gaussian7)]

    # Q < 0 — соль, Q > 0 — перец, Q = 0 — усреднение.
    for size in [3, 5]:
        for order in [-1.5, 0, 1.5]:
            filtered = contraharmonic(noisy, size, order)
            name = "contra_" + str(size) + "_q" + str(order)
            title = "Контргарм. " + str(size) + "x" + str(size) + ", Q = " + str(order)
            methods.append((name, title, filtered))

    for name, title, filtered in methods:
        mse = np.mean((original.astype(float) - filtered.astype(float)) ** 2)
        measurements.append([noise_name, name, round(float(mse), 4)])
        cv2.imwrite("results/task2_" + noise_name + "_" + name + ".png", filtered)
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
        plt.savefig("results/task2_" + noise_name + "_comparison_" +
                    str(start // 4 + 1) + ".png", dpi=120)

with open("results/task2_metrics.csv", "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["noise", "method", "mse"])
    writer.writerows(measurements)

plt.show()
