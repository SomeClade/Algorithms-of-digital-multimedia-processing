# ЛР1. Задача 1. Гистограммы и изменение контраста.
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

gray = cv2.imread("images/cat.png")

# Сжимаем его до 80..169, чтобы получить слабый контраст.
low = (gray.astype(float) * 0.35 + 80).astype(np.uint8)
minimum = int(low.min())
maximum = int(low.max())

bright = np.clip(low.astype(int) + 50, 0, 255).astype(np.uint8)

# Линейное растяжение и растяжение со степенью 0.5.
if maximum > minimum:
    normalized = (low.astype(float) - minimum) / (maximum - minimum)
    stretched = np.rint(255 * normalized).astype(np.uint8)
    gamma = np.rint(255 * normalized ** 0.5).astype(np.uint8)
else:
    stretched = low.copy()
    gamma = low.copy()

p = np.zeros_like(low, dtype=float)
for channel in range(3):
    histogram = cv2.calcHist(
        [low], [channel], None, [256], [0, 256]
    ).ravel()
    cdf = np.cumsum(histogram.astype(float)) / (low.shape[0] * low.shape[1])
    p[:, :, channel] = cdf[low[:, :, channel]]

uniform = np.rint((maximum - minimum) * p + minimum).astype(np.uint8)

# Защита от log(0), когда вероятность равна 1.
p_safe = np.clip(p, 0, 1 - 0.000001)
i_min = minimum / 255

# Экспоненциальное преобразование, alpha = 3.
values = i_min - np.log(1 - p_safe) / 3
exponential = np.rint(255 * np.clip(values, 0, 1)).astype(np.uint8)

# Преобразование Рэлея, alpha = 0.3.
values = i_min + np.sqrt(2 * 0.3 ** 2 * np.log(1 / (1 - p_safe)))
rayleigh = np.rint(255 * np.clip(values, 0, 1)).astype(np.uint8)

power = np.rint(255 * p ** (2 / 3)).astype(np.uint8)
hyperbolic = np.rint(255 * 0.04 ** p).astype(np.uint8)

equalized = low.copy()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_result = low.copy()

for channel in range(3):
    equalized[:, :, channel] = cv2.equalizeHist(low[:, :, channel])
    clahe_result[:, :, channel] = clahe.apply(low[:, :, channel])

lut = np.arange(256, dtype=float)
if maximum > minimum:
    lut = np.clip((lut - minimum) / (maximum - minimum), 0, 1)
    lut = np.rint(255 * lut ** 0.5).astype(np.uint8)
else:
    lut = lut.astype(np.uint8)
lut_result = cv2.LUT(low, lut)
difference = np.abs(gamma.astype(int) - lut_result.astype(int))
print("Максимальное отличие LUT от формулы:", difference.max())
cv2.imwrite("results/task1_lut.png", lut_result)

images = [gray, low, bright, stretched, gamma, uniform,
          exponential, rayleigh, power, hyperbolic, equalized, clahe_result]
titles = ["Исходное цветное", "Слабый контраст", "Яркость +50",
          "Линейное растяжение", "Растяжение, степень 0.5", "Равномерное",
          "Экспоненциальное", "Рэлея", "Степень 2/3",
          "Гиперболическое", "OpenCV equalizeHist", "OpenCV CLAHE"]
names = ["gray", "low", "bright", "stretched", "gamma", "uniform",
         "exponential", "rayleigh", "power", "hyperbolic", "equalized", "clahe"]

# Сохраняем изображения и печатаем данные для сравнения.
for i in range(len(images)):
    cv2.imwrite("results/task1_" + names[i] + ".png", images[i])
    print(titles[i], ": минимум =", images[i].min(),
          "максимум =", images[i].max(),
          "средняя яркость =", round(float(images[i].mean()), 2))

for start in range(0, len(images), 3):
    plt.figure(figsize=(12, 6))
    for column in range(3):
        i = start + column
        plt.subplot(2, 3, column + 1)

        # Отображаем цветное изображение.
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis("off")

        plt.subplot(2, 3, column + 4)

        # Строим гистограммы для синего, зелёного и красного каналов.
        for channel, color in enumerate(["b", "g", "r"]):
            hist = cv2.calcHist(
                [images[i]], [channel], None, [256], [0, 256]
            )
            plt.plot(hist, color=color)

        plt.xlim(0, 255)
        plt.xlabel("Яркость")
        plt.ylabel("Число пикселей")
        plt.grid()

    plt.tight_layout()
    number = start // 3 + 1
    plt.savefig("results/task1_comparison_" + str(number) + ".png", dpi=150)

plt.show()
