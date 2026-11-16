import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

original = cv2.imread("images/rice.png", cv2.IMREAD_GRAYSCALE)


threshold, binary = cv2.threshold(original, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

eroded = cv2.erode(binary, kernel)
dilated = cv2.dilate(binary, kernel)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
removed = cv2.subtract(binary, opened)
added = cv2.subtract(closed, opened)

print("Порог Оцу:", threshold)
print("Удалено пикселей при открытии:", np.count_nonzero(removed))
print("Добавлено пикселей при закрытии:", np.count_nonzero(added))

images = [original, binary, eroded, dilated, opened, closed]
names = ["original", "binary", "eroded", "dilated", "opened", "closed"]
titles = ["Фотография риса", "Бинаризация по Оцу", "Эрозия: эллипс 3 × 3",
          "Дилатация: эллипс 3 × 3", "Открытие: удаление выступов",
          "Закрытие после открытия"]

plt.figure(figsize=(12, 8))
for index in range(len(images)):
    cv2.imwrite("results/task1_" + names[index] + ".png", images[index])
    plt.subplot(2, 3, index + 1)
    plt.imshow(images[index], cmap="gray", vmin=0, vmax=255)
    plt.title(titles[index])
    plt.axis("off")
plt.tight_layout()
plt.savefig("results/task1_comparison.png", dpi=150)

changes = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
changes[removed > 0] = [255, 0, 0]
changes[added > 0] = [0, 180, 255]
plt.figure(figsize=(12, 5))
for index, image in enumerate([binary, closed, changes]):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image[55:195, 115:270], cmap="gray", vmin=0, vmax=255,
               interpolation="nearest")
    plt.title(["До обработки", "После обработки",
               "Красный: удалено; голубой: добавлено"][index])
    plt.axis("off")
plt.tight_layout()
plt.savefig("results/task1_details.png", dpi=180)
cv2.imwrite("results/task1_removed.png", removed)
cv2.imwrite("results/task1_added.png", added)

plt.show()
