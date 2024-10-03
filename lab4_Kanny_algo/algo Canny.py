import cv2
import numpy as np
import math

# Задание 1: Чтение изображения, преобразование в черно-белый формат и применение размытия по Гауссу
def process_image(path, kernel_size=5, sigma_x=10, sigma_y=10, size_x=640, size_y=640):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (size_x, size_y))
    cv2.imshow("Gray Image", img_resized)

    img_gaussian = cv2.GaussianBlur(img_resized, (kernel_size, kernel_size), sigmaX=sigma_x, sigmaY=sigma_y)
    cv2.imshow("Gaussian Blur Image", img_gaussian)

    # Задание 2: Вычисление и вывод матриц длин и углов градиентов
    grads = calculate_gradients(img_gaussian)
    grad_lengths = calculate_gradient_lengths(grads)
    grad_angles = calculate_gradient_angles(grads)
    print("Gradient Lengths:\n", grad_lengths)
    print("Gradient Angles:\n", grad_angles)

    # Задание 3: Подавление немаксимумов
    suppressed_img = suppress_non_maximums(grad_lengths, grad_angles)
    cv2.imshow('Non-Maximum Suppression Image', suppressed_img)

    # Задание 4: Двойная пороговая фильтрация
    edge_img = double_threshold_filter(img_gaussian, suppressed_img, grad_lengths, lower_ratio=0.1, upper_ratio=0.25)
    cv2.imshow('Edge Detection Image', edge_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Вычисление градиентов по Sobel оператору
def calculate_gradients(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return sobel_x, sobel_y

# Вычисление длин градиентов
def calculate_gradient_lengths(grads):
    sobel_x, sobel_y = grads
    return np.sqrt(np.square(sobel_x) + np.square(sobel_y))

# Вычисление углов градиентов
def calculate_gradient_angles(grads):
    sobel_x, sobel_y = grads
    return np.arctan2(sobel_y, sobel_x) * (180.0 / np.pi)  # преобразование в градусы

# Подавление немаксимумов
def suppress_non_maximums(grad_lengths, grad_angles):
    height, width = grad_lengths.shape
    suppressed = np.zeros_like(grad_lengths)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            angle = grad_angles[y, x]
            q = 255
            r = 255

            # Углы приближаем к ближайшему направлению (0, 45, 90, 135 градусов)
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = grad_lengths[y, x + 1]
                r = grad_lengths[y, x - 1]
            elif 22.5 <= angle < 67.5:
                q = grad_lengths[y + 1, x - 1]
                r = grad_lengths[y - 1, x + 1]
            elif 67.5 <= angle < 112.5:
                q = grad_lengths[y + 1, x]
                r = grad_lengths[y - 1, x]
            elif 112.5 <= angle < 157.5:
                q = grad_lengths[y - 1, x - 1]
                r = grad_lengths[y + 1, x + 1]

            if grad_lengths[y, x] >= q and grad_lengths[y, x] >= r:
                suppressed[y, x] = grad_lengths[y, x]
            else:
                suppressed[y, x] = 0

    return suppressed

# Двойная пороговая фильтрация
def double_threshold_filter(img, suppressed_img, grad_lengths, lower_ratio=0.1, upper_ratio=0.25):
    max_gradient = np.max(grad_lengths)
    low_threshold = max_gradient * lower_ratio
    high_threshold = max_gradient * upper_ratio

    edges = np.zeros_like(img)

    strong_edge = 255
    weak_edge = 75

    strong_row, strong_col = np.where(suppressed_img >= high_threshold)
    weak_row, weak_col = np.where((suppressed_img < high_threshold) & (suppressed_img >= low_threshold))

    edges[strong_row, strong_col] = strong_edge
    edges[weak_row, weak_col] = weak_edge

    for y in range(1, img.shape[0] - 1):
        for x in range(1, img.shape[1] - 1):
            if edges[y, x] == weak_edge:
                if ((edges[y + 1, x - 1:x + 2] == strong_edge).any() or
                        (edges[y - 1, x - 1:x + 2] == strong_edge).any() or
                        (edges[y, [x - 1, x + 1]] == strong_edge).any()):
                    edges[y, x] = strong_edge
                else:
                    edges[y, x] = 0

    return edges

# Запуск программы с изображением
process_image("img1.jpg")
