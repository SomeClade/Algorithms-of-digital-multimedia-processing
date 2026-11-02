import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_edges(image):
    smoothed = cv2.GaussianBlur(image, (5, 5), 1, borderType=cv2.BORDER_REPLICATE)

    # CV_32F сохраняет отрицательные значения производных.
    roberts_x = np.float32([[1, 0], [0, -1]])
    roberts_y = np.float32([[0, 1], [-1, 0]])
    gradient_x = cv2.filter2D(smoothed, cv2.CV_32F, roberts_x,
                             borderType=cv2.BORDER_REPLICATE)
    gradient_y = cv2.filter2D(smoothed, cv2.CV_32F, roberts_y,
                             borderType=cv2.BORDER_REPLICATE)
    roberts = cv2.magnitude(gradient_x, gradient_y)

    prewitt_x = np.float32([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.float32([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    gradient_x = cv2.filter2D(smoothed, cv2.CV_32F, prewitt_x,
                             borderType=cv2.BORDER_REPLICATE)
    gradient_y = cv2.filter2D(smoothed, cv2.CV_32F, prewitt_y,
                             borderType=cv2.BORDER_REPLICATE)
    prewitt = cv2.magnitude(gradient_x, gradient_y)

    gradient_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3,
                           borderType=cv2.BORDER_REPLICATE)
    gradient_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3,
                           borderType=cv2.BORDER_REPLICATE)
    sobel = cv2.magnitude(gradient_x, gradient_y)

    laplacian = np.abs(cv2.Laplacian(smoothed, cv2.CV_32F, ksize=1,
                                     borderType=cv2.BORDER_REPLICATE))

    # Нормализация только для показа: масштабы отклика у операторов разные.
    results = []
    for response in [roberts, prewitt, sobel, laplacian]:
        normalized = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)
        results.append(np.rint(normalized).astype(np.uint8))

    results.append(cv2.Canny(smoothed, 50, 120, L2gradient=True))
    return results


os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

image_names = ["architecture", "galaxies"]
image_titles = ["Плотная городская застройка", "Hubble eXtreme Deep Field"]
method_names = ["roberts", "prewitt", "sobel", "laplacian", "canny"]
method_titles = ["Робертс", "Превитт", "Собел", "Лаплас", "Кэнни: 50 / 120"]

for image_index in range(len(image_names)):
    image_name = image_names[image_index]
    original = cv2.imread("images/" + image_name + ".png", cv2.IMREAD_GRAYSCALE)
    edges = find_edges(original)
    images = [original] + edges
    titles = ["Исходное изображение"] + method_titles
    cv2.imwrite("results/task4_" + image_name + "_original.png", original)

    for name, edge in zip(method_names, edges):
        cv2.imwrite("results/task4_" + image_name + "_" + name + ".png", edge)
        print(image_titles[image_index], "/", name, ": ненулевых пикселей", np.count_nonzero(edge))

    figure = plt.figure(figsize=(12, 8))
    figure.suptitle(image_titles[image_index])
    for index in range(6):
        plt.subplot(2, 3, index + 1)
        plt.imshow(images[index], cmap="gray", vmin=0, vmax=255)
        plt.title(titles[index])
        plt.axis("off")
    plt.tight_layout(rect=(0, 0.09, 1, 0.95), h_pad=4)
    plt.savefig("results/task4_" + image_name + "_comparison.png", dpi=150)

plt.show()
