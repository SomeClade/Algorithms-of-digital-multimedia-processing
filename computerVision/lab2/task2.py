import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

original = cv2.imread("images/barrel.png")
height, width = original.shape[:2]

# Матрица камеры приближенная, параметры подобраны по прямым краям корпуса.
camera_matrix = np.array([
    [width, 0, (width - 1) / 2],
    [0, width, (height - 1) / 2],
    [0, 0, 1]
], dtype=float)

# Отрицательный k1 описывает бочкообразную дисторсию в модели OpenCV.
dist_coeffs = np.array([-0.14, 0, 0, 0, 0], dtype=float)

# Последний ноль убирает черные края ценой небольшой обрезки изображения.
new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (width, height), 0)
corrected = cv2.undistort(original, camera_matrix, dist_coeffs, None, new_camera_matrix)

print("Коэффициент дисторсии k1:", dist_coeffs[0])
cv2.imwrite("results/task2_original.png", original)

images = [original, corrected]
titles = ["Фото с бочкообразной дисторсией", "После коррекции"]
names = ["distorted", "corrected"]

plt.figure(figsize=(14, 4))
for index in range(2):
    cv2.imwrite("results/task2_" + names[index] + ".png", images[index])
    plt.subplot(1, 2, index + 1)
    plt.imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB))
    plt.title(titles[index])
    plt.axis("off")

plt.tight_layout()
plt.savefig("results/task2_comparison.png", dpi=150)
plt.show()
