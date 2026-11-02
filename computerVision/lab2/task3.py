import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

image1 = cv2.imread("images/telegram_top.png")
image2 = cv2.imread("images/telegram_bottom.png")

template_height = 20

# Для поиска достаточно яркости
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
template = gray2[:template_height, :]

# В каждой позиции вычисляется сходство полосы с первым фрагментом.
correlation = cv2.matchTemplate(gray1, template, cv2.TM_CCOEFF_NORMED)
minimum, score, minimum_position, position = cv2.minMaxLoc(correlation)

offset_y = position[1]
width = image1.shape[1]

canvas_height = max(image1.shape[0], offset_y + image2.shape[0])

# Последний столбец матрицы задает перенос: по x — 0, по y — найденное смещение.
translation_matrix = np.float32([[1, 0, 0], [0, 1, offset_y]])
aligned = cv2.warpAffine(image2, translation_matrix, (width, canvas_height))

result = aligned.copy()
result[:image1.shape[0], :] = image1

overlap_height = min(image1.shape[0] - offset_y, image2.shape[0])
print("Смещение второго фрагмента по Oy:", offset_y)
print("Высота области перекрытия:", overlap_height)
print("Коэффициент совпадения:", round(score, 6))
print("Размер склеенного изображения:", result.shape[1], "x", result.shape[0])
print("Матрица перевода второго фрагмента в систему первого:")
print(translation_matrix)

images = [image1, image2, aligned, result]
titles = ["Первый фрагмент", "Второй фрагмент",
          "Второй в системе первого", "Автоматическая склейка"]
names = ["part1", "part2", "aligned", "stitched"]

plt.figure(figsize=(10, 8))
for index in range(4):
    cv2.imwrite("results/task3_" + names[index] + ".png", images[index])
    plt.subplot(2, 2, index + 1)
    plt.imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB))
    plt.title(titles[index])
    plt.axis("off")

plt.tight_layout()
plt.savefig("results/task3_comparison.png", dpi=150)
plt.show()
