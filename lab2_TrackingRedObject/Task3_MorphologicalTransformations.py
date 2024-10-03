import cv2
import numpy as np

# Функция для получения маски в зависимости от выбранного цвета
def get_color_mask(hsv_frame, color):
    if color == 'red':
        lower_red_1 = np.array([0, 120, 70])
        upper_red_1 = np.array([10, 255, 255])

        lower_red_2 = np.array([170, 120, 70])
        upper_red_2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv_frame, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv_frame, lower_red_2, upper_red_2)

        # Объединение двух масок
        return cv2.bitwise_or(mask1, mask2)

    elif color == 'blue':
        lower_blue = np.array([102, 68, 89])
        upper_blue = np.array([120, 255, 255])
        return cv2.inRange(hsv_frame, lower_blue, upper_blue)

    elif color == 'green':
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        return cv2.inRange(hsv_frame, lower_green, upper_green)

    else:
        return np.zeros_like(hsv_frame[:, :, 0])  # Возвращаем пустую маску


# Открытие камеры
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

selected_color = 'red'

# Структурный элемент для морфологических операций (размер 5x5)
kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Ошибка при получении кадра")
        break

    # Преобразование изображения в цветовое пространство HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Получаем маску для выбранного цвета
    color_mask = get_color_mask(hsv_frame, selected_color)

    # Применение морфологических операций
    opening = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    opened_output = cv2.bitwise_and(frame, frame, mask=opening)
    closed_output = cv2.bitwise_and(frame, frame, mask=closing)

    cv2.imshow('Original Image', frame)
    cv2.imshow('Opened Image (Noise Removed)', opened_output)
    cv2.imshow('Closed Image (Gaps Filled)', closed_output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        selected_color = 'red'
    elif key == ord('g'):
        selected_color = 'green'
    elif key == ord('b'):
        selected_color = 'blue'

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()

"""
Морфологические преобразования:

Открытие (cv2.MORPH_OPEN): удаляет шумы, состоящие из небольших белых точек. Применяется сначала сужение (erode), затем расширение (dilate).
Закрытие (cv2.MORPH_CLOSE): заполняет небольшие разрывы внутри объектов. Применяется сначала расширение (dilate), затем сужение (erode).
Ключевые функции:

cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel): Применяет операцию открытия.
cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel): Применяет операцию закрытия.
Структурный элемент kernel определяет, как будут выполняться эти операции (размер 5x5).

"""