import os
import cv2
import matplotlib.pyplot as plt


os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

image_settings = [
    ("morph_holes", "Пемза: множество пор и отверстий", 115),
    ("morph_noise", "шум камеры: ISO 12800", 95)
]
kernel_sizes = [3, 15, 151]
kernel_shape = cv2.MORPH_ELLIPSE

for image_name, image_title, threshold in image_settings:
    original = cv2.imread("images/" + image_name + ".png")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite("results/" + image_name + "_gray.png", gray)
    cv2.imwrite("results/" + image_name + "_binary.png", binary)
    print(image_title, ": порог =", threshold)

    for size in kernel_sizes:
        kernel = cv2.getStructuringElement(kernel_shape, (size, size))
        eroded = cv2.erode(binary, kernel)
        dilated = cv2.dilate(binary, kernel)

        # Открытие убирает мелкие белые точки: сначала эрозия, потом дилатация.
        opened = cv2.dilate(eroded, kernel)
        # Закрытие заполняет черные дырки: порядок обратный.
        closed = cv2.erode(dilated, kernel)

        method_names = ["eroded", "dilated", "opened", "closed"]
        results = [eroded, dilated, opened, closed]
        for method_name, result in zip(method_names, results):
            cv2.imwrite("results/" + image_name + "_" + method_name + "_" + str(size) + ".png", result)

        images = [cv2.cvtColor(original, cv2.COLOR_BGR2RGB), binary] + results
        titles = [
            "Скачанная фотография", "Порог: " + str(threshold),
            "Эрозия", "Дилатация", "Открытие\nэрозия → дилатация",
            "Закрытие\nдилатация → эрозия"
        ]
        figure = plt.figure(figsize=(12, 9))
        figure.suptitle(image_title + ": ядро " + str(size) + "x" + str(size))
        for index in range(len(images)):
            plt.subplot(2, 3, index + 1)
            plt.imshow(images[index], cmap="gray", vmin=0, vmax=255)
            plt.title(titles[index])
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout(rect=(0, 0, 1, 0.95), h_pad=5)
        plt.savefig("results/" + image_name + "_comparison_" + str(size) + ".png", dpi=150)

plt.show()
