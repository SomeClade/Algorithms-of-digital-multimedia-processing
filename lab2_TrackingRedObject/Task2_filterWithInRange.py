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

    elif color =='black':
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        return cv2.inRange(hsv_frame, lower_black, upper_black)
    elif color == 'white':
        lower_white = np.array([255, 255, 255])
        upper_white = np.array([255, 255, 255])

    else:
        return np.zeros_like(hsv_frame[:, :, 0])  # Возвращаем пустую маску

    '''
     cv2.inRange() выделяет пиксели, попадающие в определённый диапазон цветовых 
     '''


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

    # Наложение маски на изображение
    color_output = cv2.bitwise_and(frame, frame, mask=color_mask)

    cv2.imshow('Original Image', frame)
    cv2.imshow(f'{selected_color.capitalize()} Object Image', color_output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        selected_color = 'red'
    elif key == ord('g'):
        selected_color = 'green'
    elif key == ord('b'):
        selected_color = 'blue'
    elif key == ord('k'):
        selected_color = 'black'

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
