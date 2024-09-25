import cv2
import numpy as np
from task2_normalize_gaussian_matrix import gaussian_kernel, normalize_kernel


def manual_convolution(image, kernel):
    image_height, image_width = image.shape
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    # Добавляем отступы для того, чтобы обработать края изображения
    padded_image = np.pad(image, pad, mode='constant')

    # Создаем новое изображение для результата
    output_image = np.zeros_like(image)

    # Проходим по каждому пикселю изображения
    for i in range(image_height):
        for j in range(image_width):
            # Извлекаем участок изображения, соответствующий размеру ядра
            region = padded_image[i:i + kernel_size, j:j + kernel_size]

            # Применяем свёртку: умножаем ядро на соответствующий участок изображения и суммируем
            output_value = np.sum(region * kernel)

            # Записываем результат в новый массив
            output_image[i, j] = output_value

    return output_image


def apply_gaussian_filter_manual(image, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = normalize_kernel(kernel)
    return manual_convolution(image, kernel)


if __name__ == "__main__":
    img = cv2.imread('images/img1.jpg', cv2.IMREAD_GRAYSCALE)  # Загружаем изображение в оттенках серого
    filtered_img = apply_gaussian_filter_manual(img, 5, 100.0)
    cv2.imwrite('filtered_image_manual.jpg', filtered_img)
