import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

original = cv2.imread("images/architecture.png", cv2.IMREAD_GRAYSCALE)
height, width = original.shape

factor = 320 / max(height, width)
original = cv2.resize(original, (round(width * factor), round(height * factor)),
                      interpolation=cv2.INTER_AREA)
values = original.astype(float)
random = np.random.default_rng(42)

# Отрицательный шум сохраняем до перевода в uint8.
gaussian_noise = random.normal(0, 22, original.shape)
gaussian = values + gaussian_noise

salt = original.copy()
salt[random.random(original.shape) < 0.12] = 255

pepper = original.copy()
pepper[random.random(original.shape) < 0.12] = 0

probability = random.random(original.shape)
salt_pepper = original.copy()
salt_pepper[probability < 0.10] = 0
salt_pepper[probability > 0.90] = 255

# Спекл: сила шума зависит от яркости.
speckle = values + values * random.normal(0, 0.28, original.shape)

# 45 — максимальное среднее число фотонов.
poisson = random.poisson(values / 255 * 45) / 45 * 255

noisy_images = [gaussian, salt, pepper, salt_pepper, speckle, poisson]
noise_names = ["gaussian", "salt", "pepper", "salt_pepper", "speckle", "poisson"]
noise_titles = ["Гаусс: sigma = 22", "Соль: 12%", "Перец: 12%",
                "Соль-перец: 20%", "Спекл: sigma = 0.28", "Пуассон: уровень 45"]
measurements = []

cv2.imwrite("results/task1_original.png", original)
for index in range(len(noisy_images)):
    noisy_images[index] = np.clip(np.rint(noisy_images[index]), 0, 255).astype(np.uint8)
    cv2.imwrite("results/task1_" + noise_names[index] + ".png", noisy_images[index])
    mse = np.mean((values - noisy_images[index].astype(float)) ** 2)
    measurements.append([noise_names[index], round(float(mse), 4)])
    print(noise_titles[index], ": MSE =", round(float(mse), 2))

for start in range(0, len(noisy_images), 3):
    images = [original] + noisy_images[start:start + 3]
    titles = ["Исходное изображение"] + noise_titles[start:start + 3]
    plt.figure(figsize=(10, 8))
    for index in range(4):
        plt.subplot(2, 2, index + 1)
        plt.imshow(images[index], cmap="gray", vmin=0, vmax=255)
        plt.title(titles[index])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/task1_comparison_" + str(start // 3 + 1) + ".png", dpi=120)

with open("results/task1_metrics.csv", "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["noise", "mse"])
    writer.writerows(measurements)

plt.show()
