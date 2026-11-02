import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

original = cv2.imread("images/city.png")
height, width = original.shape[:2]

center_x = (width - 1) / 2
center_y = (height - 1) / 2

scale = max(width, height) / 2
coefficient = 0.25

# Строим координаты всех пикселей и переносим начало координат в центр.
grid_y, grid_x = np.indices((height, width), dtype=np.float32)
normal_x = (grid_x - center_x) / scale
normal_y = (grid_y - center_y) / scale

# Чем дальше точка от центра, тем сильнее меняются координаты выборки.
radius_squared = normal_x ** 2 + normal_y ** 2
distortion_factor = 1 + coefficient * radius_squared
distortion_x = center_x + scale * normal_x * distortion_factor
distortion_y = center_y + scale * normal_y * distortion_factor

# Для каждого пикселя результата remap берет цвет по рассчитанным координатам источника.
distorted = cv2.remap(original, distortion_x, distortion_y, cv2.INTER_LINEAR)


inverse_x = normal_x.copy()
before_error = np.abs(original.astype(float) - distorted.astype(float)).mean()
inverse_y = normal_y.copy()
for iteration in range(15):
    radius_squared = inverse_x ** 2 + inverse_y ** 2
    correction_factor = 1 + coefficient * radius_squared
    inverse_x = normal_x / correction_factor
    inverse_y = normal_y / correction_factor

# Возвращаем нормированные координаты в пиксели и исправляем искаженный снимок.
correction_x = center_x + scale * inverse_x
correction_y = center_y + scale * inverse_y
corrected = cv2.remap(distorted, correction_x, correction_y, cv2.INTER_LINEAR)

# Я говорил что такое безумие?!!
after_error = np.abs(original.astype(float) - corrected.astype(float)).mean()
print("Коэффициент бочкообразного искажения:", coefficient)
print("Средняя абсолютная ошибка до коррекции:", round(float(before_error), 2))
print("Средняя абсолютная ошибка после коррекции:", round(float(after_error), 2))

images = [original, distorted, corrected]
titles = ["Исходное изображение", "Бочкообразная дисторсия", "После коррекции"]
names = ["original", "distorted", "corrected"]

plt.figure(figsize=(13, 4))
for index in range(3):
    cv2.imwrite("results/task2_" + names[index] + ".png", images[index])
    plt.subplot(1, 3, index + 1)
    plt.imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB))
    plt.title(titles[index])
    plt.axis("off")

plt.tight_layout()
plt.savefig("results/task2_comparison.png", dpi=150)
plt.show()
