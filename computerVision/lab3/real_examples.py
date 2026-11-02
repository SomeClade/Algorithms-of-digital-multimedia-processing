import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def median_with_kernel(image, kernel):
    size = kernel.shape[0]
    padding = size // 2
    expanded = cv2.copyMakeBorder(image, padding, padding, padding, padding,
                                  cv2.BORDER_REPLICATE)
    result = np.zeros_like(image)
    mask = kernel != 0

    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            window = expanded[row:row + size, column:column + size]
            # Нули маски исключаем из выборки, каналы считаем отдельно.
            result[row, column] = np.median(window[mask], axis=0)
    return result


os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

image_names = ["real_noisy", "real_blur"]
image_titles = ["Настоящий шум: ночной Гонконг", "Настоящий смаз: движение камеры"]

gaussian_settings = [(3, 0.8), (5, 1.2), (7, 1.6)]
median_sizes = [3, 5, 7]
kernel_size = 5
kernel_shapes = [
    ("square", "Квадрат", cv2.MORPH_RECT),
    ("cross", "Крест", cv2.MORPH_CROSS),
    ("ellipse", "Круг (эллипс)", cv2.MORPH_ELLIPSE)
]

for image_index in range(len(image_names)):
    image_name = image_names[image_index]
    original = cv2.imread("images/" + image_name + ".png")
    images = [original]
    method_names = ["original"]
    method_titles = ["download image"]

    for size, sigma in gaussian_settings:
        images.append(cv2.GaussianBlur(original, (size, size), sigma))
        method_names.append("gaussian_" + str(size))
        method_titles.append("Гаусс " + str(size) + "x" + str(size) + ", sigma " + str(sigma))

    for size in median_sizes:
        images.append(cv2.medianBlur(original, size))
        method_names.append("median_" + str(size))
        method_titles.append("Медиана " + str(size) + "x" + str(size))

    figure = plt.figure(figsize=(16, 9))
    figure.suptitle(image_titles[image_index])
    for index in range(len(images)):
        cv2.imwrite("results/" + image_name + "_" + method_names[index] + ".png", images[index])
        plt.subplot(2, 4, index + 1)
        plt.imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB))
        plt.title(method_titles[index])
        plt.axis("off")

    plt.tight_layout(rect=(0, 0.06, 1, 0.95), h_pad=4)
    plt.savefig("results/" + image_name + "_comparison.png", dpi=150)
    print(image_titles[image_index], ": сравнено настроек —", len(images) - 1)

    kernel_images = [original]
    kernel_titles = ["download image"]
    for kernel_name, kernel_title, shape in kernel_shapes:
        kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
        filtered = median_with_kernel(original, kernel)
        kernel_images.append(filtered)
        kernel_titles.append(kernel_title + " " + str(kernel_size) + "x" + str(kernel_size))
        cv2.imwrite("results/" + image_name + "_kernel_" + kernel_name + "_" +
                    str(kernel_size) + ".png", filtered)
        print(image_titles[image_index], "/", kernel_title, ": готово", flush=True)

    figure = plt.figure(figsize=(10, 10))
    figure.suptitle(image_titles[image_index] + ": формы медианного ядра")
    for index in range(len(kernel_images)):
        plt.subplot(2, 2, index + 1)
        plt.imshow(cv2.cvtColor(kernel_images[index], cv2.COLOR_BGR2RGB))
        plt.title(kernel_titles[index])
        plt.axis("off")

    plt.tight_layout(rect=(0, 0.06, 1, 0.95), h_pad=4)
    plt.savefig("results/" + image_name + "_kernels_comparison.png", dpi=150)

plt.show()
