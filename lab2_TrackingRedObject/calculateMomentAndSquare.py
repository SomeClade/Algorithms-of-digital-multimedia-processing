import cv2
import numpy as np

# Функция для получения маски в зависимости от выбранного цвета
def get_color_mask(hsv_frame, color):
    if color == 'red':
        # Диапазон красного цвета
        lower_red_1 = np.array([0, 120, 70])
        upper_red_1 = np.array([10, 255, 255])

        lower_red_2 = np.array([170, 120, 70])
        upper_red_2 = np.array([180, 255, 255])

        # Применение фильтра для выделения красного
        mask1 = cv2.inRange(hsv_frame, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv_frame, lower_red_2, upper_red_2)

        # Объединение двух масок
        return cv2.bitwise_or(mask1, mask2)

    elif color == 'blue':
        # Диапазон синего цвета
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        return cv2.inRange(hsv_frame, lower_blue, upper_blue)

    elif color == 'green':
        # Диапазон зеленого цвета
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

selected_color = 'red'  # Начальный цвет фильтрации

while True:
    ret, frame = cap.read()

    if not ret:
        print("Ошибка при получении кадра")
        break

    # Преобразование изображения в цветовое пространство HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Получаем маску для выбранного цвета
    color_mask = get_color_mask(hsv_frame, selected_color)

    # Находим моменты изображения
    moments = cv2.moments(color_mask)

    # Площадь объекта (момент m00)
    area = moments['m00']

    # Если площадь объекта больше порогового значения (для исключения шумов)
    if area > 1000:
        # Вычисляем координаты центра масс объекта
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # Отображаем центр объекта и его площадь
        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)  # Рисуем центр массы
        cv2.putText(frame, f"Area: {int(area)}", (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Отображение исходного изображения с центром объекта
    cv2.imshow('Original Image with Object Moments', frame)

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
