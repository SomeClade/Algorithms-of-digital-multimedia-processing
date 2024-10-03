import cv2
import numpy as np

# Функция для получения маски красного цвета
def get_red_mask(hsv_frame):
    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 120, 70])
    upper_red_2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv_frame, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_frame, lower_red_2, upper_red_2)
    return cv2.bitwise_or(mask1, mask2)

# Функция для получения маски голубого цвета
def get_blue_mask(hsv_frame):
    lower_blue = np.array([102, 58, 89])
    upper_blue = np.array([120, 255, 255])
    lower_blue1 = np.array([112, 68, 89])
    upper_blue1 = np.array([130, 255, 255])
    mask1 = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv_frame, lower_blue1, upper_blue1)
    return cv2.bitwise_or(mask1, mask2)

# Функция для выполнения морфологических операций (открытие и закрытие)
def apply_morphological_transformations(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# Функция для рисования прицела с затемнением за его пределами
def thisIsAwp(image, cx, cy):
    outer_circle_radius = 300
    crosshair_radius = 20
    gap = 10
    line_thickness = 2
    tick_length = 10
    tick_spacing = 20
    color = (0, 0, 0)

    mask = np.zeros_like(image)
    cv2.circle(mask, (cx, cy), outer_circle_radius, (255, 255, 255), -1)
    masked_image = cv2.bitwise_and(image, mask)
    cv2.circle(masked_image, (cx, cy), outer_circle_radius, color, line_thickness)
    cv2.circle(masked_image, (cx, cy), crosshair_radius, color, line_thickness)

    def draw_line_with_ticks(start, end, tick_direction):
        cv2.line(masked_image, start, end, color, line_thickness)
        for i in range(gap + crosshair_radius, outer_circle_radius, tick_spacing):
            tick_start = (cx + i * tick_direction[0], cy + i * tick_direction[1])
            if tick_direction[0] == 0:
                cv2.line(masked_image, (tick_start[0] - tick_length // 2, tick_start[1]),
                         (tick_start[0] + tick_length // 2, tick_start[1]), color, line_thickness)
            else:
                cv2.line(masked_image, (tick_start[0], tick_start[1] - tick_length // 2),
                         (tick_start[0], tick_start[1] + tick_length // 2), color, line_thickness)

    draw_line_with_ticks((cx, cy - crosshair_radius - gap), (cx, cy - outer_circle_radius), (0, -1))
    draw_line_with_ticks((cx, cy + crosshair_radius + gap), (cx, cy + outer_circle_radius), (0, 1))
    draw_line_with_ticks((cx - crosshair_radius - gap, cy), (cx - outer_circle_radius, cy), (-1, 0))
    draw_line_with_ticks((cx + crosshair_radius + gap, cy), (cx + outer_circle_radius, cy), (1, 0))

    return masked_image

# Окна программы и их позиции
window_name_1 = 'Original Camera Feed with Mask and Crosshair'
window_name_2 = 'Mask (Threshold)'
cv2.namedWindow(window_name_1)
cv2.namedWindow(window_name_2)

# Открытие камеры
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

# Выбор цвета: 'red' для красного и 'blue' для голубого
selected_color = 'blue'

# Словарь для эмуляции switch case
color_mask_functions = {
    'red': get_red_mask,
    'blue': get_blue_mask
}

while True:
    # Чтение кадров с камеры
    ret, frame = cap.read()
    if not ret:
        print("Ошибка при получении кадра")
        break

    # Преобразование изображения в цветовое пространство HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Получение маски для выбранного цвета
    mask_function = color_mask_functions.get(selected_color, get_red_mask)
    color_mask = mask_function(hsv_frame)

    # Применяем морфологические преобразования
    mask_morph = apply_morphological_transformations(color_mask)

    # Находим моменты изображения
    moments = cv2.moments(mask_morph)

    # Площадь объекта (момент m00)
    area = moments['m00']

    # Если площадь объекта больше порогового значения (для исключения шумов)
    if area > 1000:
        # Вычисляем координаты центра масс объекта
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # Рисуем прицел AWP в центре обнаруженного объекта
        frame = thisIsAwp(frame, cx, cy)

        # Отображаем центр объекта и его площадь
        cv2.putText(frame, f"Area: {int(area)}", (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Отображение результата
    cv2.imshow(window_name_1, frame)
    cv2.imshow(window_name_2, mask_morph)

    # Ожидание клавиши для выхода
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
