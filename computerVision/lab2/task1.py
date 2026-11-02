import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

original = cv2.imread("images/city.png")
height, width = original.shape[:2]

# Последний столбец матрицы задает перенос: 60 по x и 40 по y.
shift_matrix = np.float32([[1, 0, 60], [0, 1, 40]])
shifted = cv2.warpAffine(original, shift_matrix, (width + 60, height + 40))

# Меняем направление y.
reflection_x = np.float32([[1, 0, 0], [0, -1, height - 1]])
reflected_x = cv2.warpAffine(original, reflection_x, (width, height),
                             flags=cv2.INTER_NEAREST)

# Теперь меняем направление x
reflection_y = np.float32([[-1, 0, width - 1], [0, 1, 0]])
reflected_y = cv2.warpAffine(original, reflection_y, (width, height),
                             flags=cv2.INTER_NEAREST)

# Уменьшаем обе координаты
scale_matrix = np.float32([[0.7, 0, 0], [0, 0.7, 0]])
scaled = cv2.warpAffine(original, scale_matrix,
                        (round(width * 0.7), round(height * 0.7)))

# Поворот вокруг центра 30 градусов
center = ((width - 1) / 2, (height - 1) / 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 30, 1)
cosine = abs(rotation_matrix[0, 0])
sine = abs(rotation_matrix[0, 1])

rotated_width = int(np.ceil(width * cosine + height * sine))
rotated_height = int(np.ceil(width * sine + height * cosine))

# [0, 2] отвечает за сдвиг по x, [1, 2] — по y.
rotation_matrix[0, 2] += (rotated_width - width) / 2
rotation_matrix[1, 2] += (rotated_height - height) / 2

rotated = cv2.warpAffine(original, rotation_matrix,
                         (rotated_width, rotated_height))

affine_matrix = np.float32([[1.3, 0, 0], [0, 0.75, 0]])
affine_scaled = cv2.warpAffine(original, affine_matrix,
                              (round(width * 1.3), round(height * 0.75)))

# Скос по x: x_new = x + 0.35 * y
shear_x_matrix = np.float32([[1, 0.35, 0], [0, 1, 0]])
sheared_x = cv2.warpAffine(original, shear_x_matrix,
                           (width + int(np.ceil(height * 0.35)), height))

# Скос по y: y_new = y + 0.25 * x
shear_y_matrix = np.float32([[1, 0, 0], [0.25, 1, 0]])
sheared_y = cv2.warpAffine(original, shear_y_matrix,
                           (width, height + int(np.ceil(width * 0.25))))

# Задаем четыре исходных угла и места, куда каждый из них должен перейти.
source_points = np.float32([[0, 0], [width - 1, 0],
                            [width - 1, height - 1], [0, height - 1]])
target_points = np.float32([[40, 30], [width - 41, 5],
                            [width - 1, height - 31], [10, height - 1]])
projective_matrix = cv2.getPerspectiveTransform(source_points, target_points)
projective = cv2.warpPerspective(original, projective_matrix, (width, height))

grid_y, grid_x = np.indices((height, width), dtype=np.float32)

polynomial_x = grid_x + 0.001 * (grid_y - center[1]) ** 2
polynomial_y = grid_y + 0.0005 * (grid_x - center[0]) ** 2
polynomial = cv2.remap(original, polynomial_x, polynomial_y, cv2.INTER_LINEAR)

sinusoidal_x = grid_x + 20 * np.sin(2 * np.pi * grid_y / 90)
sinusoidal = cv2.remap(original, sinusoidal_x, grid_y, cv2.INTER_LINEAR)

images = [original, shifted, reflected_x, reflected_y, scaled, rotated,
          affine_scaled, sheared_x, sheared_y, projective, polynomial, sinusoidal]
titles = ["Исходное изображение", "Перенос: x +60, y +40", "Отражение относительно Ox",
          "Отражение относительно Oy", "Масштаб 0.7", "Поворот на 30 градусов",
          "Масштаб: x 1.3, y 0.75", "Скос по Ox: 0.35", "Скос по Oy: 0.25",
          "Проективное преобразование", "Полиномиальное преобразование",
          "Синусоидальное преобразование"]
names = ["original", "shifted", "reflected_x", "reflected_y", "scaled", "rotated",
         "affine_scaled", "sheared_x", "sheared_y", "projective", "polynomial", "sinusoidal"]

for index in range(len(images)):
    cv2.imwrite("results/task1_" + names[index] + ".png", images[index])
    print(titles[index], ": размер", images[index].shape[1], "x", images[index].shape[0])

for start in range(0, len(images), 3):
    plt.figure(figsize=(12, 4))
    for column in range(3):
        index = start + column
        plt.subplot(1, 3, column + 1)
        plt.imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB))
        plt.title(titles[index])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/task1_comparison_" + str(start // 3 + 1) + ".png", dpi=150)

plt.show()
