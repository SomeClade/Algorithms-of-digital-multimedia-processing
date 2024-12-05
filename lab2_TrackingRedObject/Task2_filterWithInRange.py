import cv2
import numpy as np

# Функция для получения маски в зависимости от выбранного цвета
def get_color_mask(hsv_frame, color):
    if color == 'red':
        # Диапазон красного цвета
        lower_red_1 = np.array([0, 120, 70])
        upper_red_1 = np.array([10, 255, 255])
        # находится с двух концов

        lower_red_2 = np.array([170, 120, 70])
        upper_red_2 = np.array([180, 255, 255])

        # Применение фильтра для выделения красного
        mask1 = cv2.inRange(hsv_frame, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv_frame, lower_red_2, upper_red_2)

        # Объединение двух масок
        return cv2.threshold(mask1 | mask2, 1, 255, cv2.THRESH_BINARY)[1]

    elif color == 'blue':
        # Диапазон синего цвета
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        return cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

    elif color == 'green':
        # Диапазон зеленого цвета
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        return cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

    elif color == 'black':
        # Диапазон черного цвета
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask = cv2.inRange(hsv_frame, lower_black, upper_black)
        return cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

    elif color == 'white':
        # Диапазон белого цвета (допустим, светлые участки)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])
        mask = cv2.inRange(hsv_frame, lower_white, upper_white)
        return cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

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

    # Накладываем маски на исходное изображение вручную
    red_output = frame.copy()
    red_output[red_mask == 0] = 0

    green_output = frame.copy()
    green_output[green_mask == 0] = 0

    blue_output = frame.copy()
    blue_output[blue_mask == 0] = 0

    black_output = frame.copy()
    black_output[black_mask == 0] = 0

    white_output = frame.copy()
    white_output[white_mask == 0] = 0

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
