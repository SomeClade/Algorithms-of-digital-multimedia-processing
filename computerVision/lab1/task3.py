import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

gray = cv2.imread("images/barcode.png", cv2.IMREAD_GRAYSCALE)

# Берём строку посередине изображения: она пересекает штрихи.
line_y = gray.shape[0] // 2
profile = gray[line_y, :]
print("Строка профиля:", line_y)
print("Число точек профиля:", len(profile))
print("Минимальная и максимальная яркость:", profile.min(), profile.max())

plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
plt.axhline(line_y, color="red")
plt.title("Штрихкод и линия построения профиля")
plt.xlabel("Столбец x")
plt.ylabel("Строка y")

plt.subplot(2, 1, 2)
plt.plot(profile, color="black")
plt.title("Профиль вдоль горизонтальной линии")
plt.xlabel("Столбец x")
plt.ylabel("Яркость")
plt.xlim(0, gray.shape[1] - 1)
plt.ylim(-5, 260)
plt.grid()

plt.tight_layout()
plt.savefig("results/task3_profile.png", dpi=150)
np.savetxt("results/task3_profile.txt", profile, fmt="%d")
plt.show()
