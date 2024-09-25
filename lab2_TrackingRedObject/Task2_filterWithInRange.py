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

    elif color == 'black':
        # Диапазон черного цвета
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        return cv2.inRange(hsv_frame, lower_black, upper_black)

    elif color == 'white':
        # Диапазон белого цвета (допустим, светлые участки)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])
        return cv2.inRange(hsv_frame, lower_white, upper_white)

    else:
        return np.zeros_like(hsv_frame[:, :, 0])  # Возвращаем пустую маску


# Открытие камеры
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Ошибка при получении кадра")
        break

    # Преобразование изображения в цветовое пространство HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Получаем маски для всех цветов
    red_mask = get_color_mask(hsv_frame, 'red')
    green_mask = get_color_mask(hsv_frame, 'green')
    blue_mask = get_color_mask(hsv_frame, 'blue')
    black_mask = get_color_mask(hsv_frame, 'black')
    white_mask = get_color_mask(hsv_frame, 'white')

    # Применение масок на изображение
    red_output = cv2.bitwise_and(frame, frame, mask=red_mask)
    green_output = cv2.bitwise_and(frame, frame, mask=green_mask)
    blue_output = cv2.bitwise_and(frame, frame, mask=blue_mask)
    black_output = cv2.bitwise_and(frame, frame, mask=black_mask)
    white_output = cv2.bitwise_and(frame, frame, mask=white_mask)

    # Отображение исходного изображения и всех масок параллельно
    cv2.imshow('Original Image', frame)
    cv2.imshow('Red Object Image', red_output)
    cv2.imshow('Green Object Image', green_output)
    cv2.imshow('Blue Object Image', blue_output)
    cv2.imshow('Black Object Image', black_output)
    cv2.imshow('White Object Image', white_output)

    # Ожидание нажатия клавиши для выхода
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
