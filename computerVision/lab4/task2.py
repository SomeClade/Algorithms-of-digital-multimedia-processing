import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

original = cv2.imread("images/coins.jpg")

gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
eroded = cv2.erode(binary, erosion_kernel)
dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
separated = cv2.dilate(eroded, dilation_kernel)

before_count = cv2.connectedComponents(binary)[0] - 1
after_count, labels = cv2.connectedComponents(separated)
after_count -= 1
print("Компонент до разделения:", before_count)
print("Объектов после разделения:", after_count)

# Внутренняя граница A - erosion(A).
boundary_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
boundary = cv2.subtract(separated, cv2.erode(separated, boundary_kernel))
contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
overlay = original.copy()
cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
for index, contour in enumerate(contours):
    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        x = round(moments["m10"] / moments["m00"])
        y = round(moments["m01"] / moments["m00"])
        cv2.putText(overlay, str(index + 1), (x - 7, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1,
                    cv2.LINE_AA)

images = [cv2.cvtColor(original, cv2.COLOR_BGR2RGB), binary, eroded,
          separated, boundary, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)]
titles = ["Исходная фотография", "Бинарное изображение: " + str(before_count) + " компоненты",
          "Эрозия: эллипс 21 × 21", "Дилатация 7 × 7: " + str(after_count) + " объекта",
          "Морфологические границы", "Контуры разделенных объектов"]

for name, image in zip(["binary", "eroded", "separated", "boundary", "contours"],
                       [binary, eroded, separated, boundary, overlay]):
    cv2.imwrite("results/task2_" + name + ".png", image)
np.save("results/task2_labels.npy", labels)

plt.figure(figsize=(12, 10))
for index in range(len(images)):
    plt.subplot(2, 3, index + 1)
    plt.imshow(images[index], cmap="gray", vmin=0, vmax=255)
    plt.title(titles[index])
    plt.axis("off")
plt.tight_layout()
plt.savefig("results/task2_comparison.png", dpi=150)

plt.show()
