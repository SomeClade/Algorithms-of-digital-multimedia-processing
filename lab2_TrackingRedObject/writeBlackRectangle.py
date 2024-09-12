import cv2
import numpy as np

def get_color_mask(hsv_frame, color):
    if color == 'red':
        # Диапазон красного цвета
        lower_red_1 = np.array([0, 120, 70])
        upper_red_1 = np.array([10, 255, 255])

        lower_red_2 = np.array([170, 120, 70])
        upper_red_2 = np.array([180, 255, 255])

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

selected_color = 'red'

while True:
    ret, frame = cap.read()

    if not ret:
        print("Ошибка при получении кадра")
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

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

        # Находим контуры объекта для построения ограничивающего прямоугольника
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Находим самый большой контур (считаем, что это объект)
            largest_contour = max(contours, key=cv2.contourArea)

            # Вычисляем ограничивающий прямоугольник
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Рисуем черный прямоугольник вокруг объекта
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

            # Отображаем центр объекта
            cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)  # Рисуем центр массы

    # Отображение исходного изображения с прямоугольником и центром объекта
    cv2.imshow('Original Image with Bounding Box', frame)

    # Обработка нажатий клавиш
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        selected_color = 'red'    # Выбор красного цвета
    elif key == ord('g'):
        selected_color = 'green'  # Выбор зеленого цвета
    elif key == ord('b'):
        selected_color = 'blue'   # Выбор синего цвета

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
