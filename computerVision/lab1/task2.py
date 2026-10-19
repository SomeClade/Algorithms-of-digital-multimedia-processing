# ЛР1. Задача 2. Проекции и границы ракушки.
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

color = cv2.imread("images/seashell.png")

gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

# сумма проекция по осям
projection_x = np.sum(gray, axis=0)
projection_y = np.sum(gray, axis=1)

# Белый фон становится 0, ракушка становится 255.
# Порог 245 выбран для светлого фона данного изображения.
threshold, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
mask_x = np.sum(mask // 255, axis=0)
mask_y = np.sum(mask // 255, axis=1)

# Ненулевые проекции маски показывают, где есть объект.
columns = np.where(mask_x > 0)[0]
rows = np.where(mask_y > 0)[0]
if len(columns) == 0 or len(rows) == 0:
    raise SystemExit("Объект не найден. Проверьте порог бинаризации.")

left = int(columns[0])
right = int(columns[-1])
top = int(rows[0])
bottom = int(rows[-1])
print("Границы ракушки: x от", left, "до", right, ", y от", top, "до", bottom)

# найденная область на изображении.
marked = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
cv2.rectangle(marked, (left, top), (right, bottom), (255, 0, 0), 2)

plt.figure(figsize=(13, 7))
plt.subplot(2, 3, 1)
plt.imshow(marked)
plt.title("Найденные границы ракушки")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.plot(projection_x)
plt.title("Проекция яркости на Ox")
plt.xlabel("Столбец x")
plt.ylabel("Сумма яркости")
plt.grid()

plt.subplot(2, 3, 3)
plt.plot(projection_y)
plt.title("Проекция яркости на Oy")
plt.xlabel("Строка y")
plt.ylabel("Сумма яркости")
plt.grid()

plt.subplot(2, 3, 4)
plt.imshow(mask, cmap="gray", vmin=0, vmax=255)
plt.title("Маска ракушки")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.plot(mask_x)
plt.axvline(left, color="red")
plt.axvline(right, color="red")
plt.title("Проекция маски на Ox")
plt.xlabel("Столбец x")
plt.ylabel("Пикселей объекта")
plt.grid()

plt.subplot(2, 3, 6)
plt.plot(mask_y)
plt.axvline(top, color="red")
plt.axvline(bottom, color="red")
plt.title("Проекция маски на Oy")
plt.xlabel("Строка y")
plt.ylabel("Пикселей объекта")
plt.grid()

plt.tight_layout()
plt.savefig("results/task2_projections.png", dpi=150)
cv2.imwrite("results/task2_mask.png", mask)
np.savetxt("results/task2_projection_x.txt", projection_x, fmt="%d")
np.savetxt("results/task2_projection_y.txt", projection_y, fmt="%d")
plt.show()
