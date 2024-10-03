import cv2
import numpy as np


# Задание 1: Чтение изображения, преобразование в черно-белый формат и применение размытия по Гауссу
def process_image(path, kernel_size=5, sigma_x=10, sigma_y=10, size_x=640, size_y=640):
    """
    kernel_size: размер ядра для размытия по Гауссу. Чем больше значение, тем сильнее размытие (значение должно быть нечетным).
    sigma_x, sigma_y: стандартные отклонения для размытия по Гауссу по осям X и Y. Чем больше значения, тем сильнее размытие.
    size_x, size_y: размеры изображения после изменения. Определяют, каким размером будет изображение на экране.
    """

    # Чтение изображения с заданного пути
    img = cv2.imread(path)

    # Преобразование изображения в черно-белый формат (градации серого)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Изменение размера изображения на заданные размеры
    img_resized = cv2.resize(img_gray, (size_x, size_y))

    # Отображение черно-белого изображения
    cv2.imshow("Gray Image", img_resized)

    # Применение размытия по Гауссу. Здесь kernel_size определяет размеры фильтра, а sigma_x и sigma_y — степень размытия.
    img_gaussian = cv2.GaussianBlur(img_resized, (kernel_size, kernel_size), sigmaX=sigma_x, sigmaY=sigma_y)

    # Отображение размытого изображения
    cv2.imshow("Gaussian Blur Image", img_gaussian)

    # Задание 2: Вычисление и вывод матриц длин и углов градиентов
    grads = calculate_gradients(img_gaussian)  # Вычисление градиентов по X и Y с помощью оператора Собеля
    grad_lengths = calculate_gradient_lengths(grads)  # Вычисление длин градиентов
    grad_angles = calculate_gradient_angles(grads)  # Вычисление углов градиентов

    # Вывод матриц длин и углов градиентов в консоль
    print("Gradient Lengths:\n", grad_lengths)
    print("Gradient Angles:\n", grad_angles)

    # Задание 3: Подавление немаксимумов для выделения четких контуров
    suppressed_img = suppress_non_maximums(grad_lengths, grad_angles)
    cv2.imshow('Non-Maximum Suppression Image', suppressed_img)  # Отображение изображения после подавления немаксимумов

    # Задание 4: Двойная пороговая фильтрация для выделения краев
    edge_img = double_threshold_filter(img_gaussian, suppressed_img, grad_lengths, lower_ratio=0.1, upper_ratio=0.25)
    cv2.imshow('Edge Detection Image', edge_img)  # Отображение изображения с найденными краями

    # Ожидание нажатия клавиши для завершения программы
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Вычисление градиентов по оператору Собеля
def calculate_gradients(img):
    """
    Вычисляет горизонтальные и вертикальные градиенты с помощью оператора Собеля.
    Используются значения по осям X и Y, чтобы выявить изменение интенсивности.
    """
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Градиент по оси X
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Градиент по оси Y
    return sobel_x, sobel_y


# Вычисление длин градиентов
def calculate_gradient_lengths(grads):
    """
    Вычисляет длины градиентов, используя формулу длины вектора: sqrt(Gx^2 + Gy^2).
    grads: список с двумя матрицами, представляющими градиенты по осям X и Y.
    Возвращает матрицу длин градиентов для каждого пикселя.
    """
    sobel_x, sobel_y = grads
    return np.sqrt(np.square(sobel_x) + np.square(sobel_y))


# Вычисление углов градиентов
def calculate_gradient_angles(grads):
    """
    Вычисляет углы градиентов по каждому пикселю.
    Используется функция arctan2 для вычисления угла между осью X и вектором градиента.
    """
    sobel_x, sobel_y = grads
    return np.arctan2(sobel_y, sobel_x) * (180.0 / np.pi)  # Преобразование углов из радиан в градусы


# Подавление немаксимумов
def suppress_non_maximums(grad_lengths, grad_angles):
    """
    Подавляет немаксимальные значения в градиентах. Этот этап необходим для тонкого выделения контуров.
    grad_lengths: матрица длин градиентов
    grad_angles: матрица углов градиентов
    Возвращает изображение, где остаются только точки с наибольшими значениями в направлениях градиентов.
    """
    height, width = grad_lengths.shape
    suppressed = np.zeros_like(grad_lengths)  # Пустая матрица для результата

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            angle = grad_angles[y, x]  # Получаем угол градиента
            q = 255  # Предполагаемое соседнее значение в направлении градиента
            r = 255  # Другое соседнее значение

            # Углы округляются до 0, 45, 90, и 135 градусов
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

            # Подавляем значения, которые не являются локальными максимумами
            if grad_lengths[y, x] >= q and grad_lengths[y, x] >= r:
                suppressed[y, x] = grad_lengths[y, x]
            else:
                suppressed[y, x] = 0

    return suppressed


# Двойная пороговая фильтрация
def double_threshold_filter(img, suppressed_img, grad_lengths, lower_ratio=0.1, upper_ratio=0.25):
    """
    Выполняет двойную пороговую фильтрацию для выделения краев.
    lower_ratio: нижний порог (выраженный в долях от максимального градиента).
    upper_ratio: верхний порог (тоже выраженный в долях от максимального градиента).
    Все значения градиентов, превышающие верхний порог, считаются сильными краями, а те, что находятся между верхним и нижним — слабыми.
    Возвращает изображение с выделенными сильными и слабыми краями.
    """
    max_gradient = np.max(grad_lengths)
    low_threshold = max_gradient * lower_ratio  # Нижний порог
    high_threshold = max_gradient * upper_ratio  # Верхний порог

    edges = np.zeros_like(img)  # Пустая матрица для хранения результатов фильтрации

    strong_edge = 255  # Значение для сильных краев
    weak_edge = 75  # Значение для слабых краев

    # Находим сильные и слабые края
    strong_row, strong_col = np.where(suppressed_img >= high_threshold)
    weak_row, weak_col = np.where((suppressed_img < high_threshold) & (suppressed_img >= low_threshold))

    # Отмечаем сильные и слабые края на изображении
    edges[strong_row, strong_col] = strong_edge
    edges[weak_row, weak_col] = weak_edge

    # Проверка соседей слабых краев: если они граничат с сильными, то помечаем их как сильные
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


process_image("img1.jpg")

'''
Gradient Lengths (Длины градиентов):

Матрица значений градиентов яркости для каждого пикселя.
Большинство значений в матрице равно 0. Это означает, что для многих пикселей изменение яркости между соседними пикселями незначительно (отсутствие границы).
Для некоторых пикселей градиент не равен нулю, например:
Значение 4 во второй строке первой колонки показывает резкое изменение яркости в этой точке.
Другие значения, такие как 3.16 и 1.41, указывают на более слабое изменение яркости в соответствующих точках.
'''