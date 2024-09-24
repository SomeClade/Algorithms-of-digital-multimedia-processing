import cv2
import numpy as np
import pyautogui


# Функция для получения маски красного цвета
def get_red_mask(hsv_frame):
    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 120, 70])
    upper_red_2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv_frame, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_frame, lower_red_2, upper_red_2)
    return cv2.bitwise_or(mask1, mask2)


# Функция для выполнения морфологических операций (открытие и закрытие)
def apply_morphological_transformations(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Открытие (удаляет мелкие шумы)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Закрытие (заполняет маленькие пробелы)
    return mask



def thisIsAwp(image, cx, cy):
    # Параметры прицела
    outer_circle_radius = 300  # Радиус внешнего круга
    crosshair_radius = 20  # Радиус центрального круга
    gap = 10  # Разрыв между кругом и линиями
    line_thickness = 2  # Толщина линий и круга
    tick_length = 10  # Длина отметок на линиях
    tick_spacing = 20  # Расстояние между отметками
    color = (0, 0, 0)  # Цвет прицела (черный)

    # Рисуем внешнюю окружность прицела
    cv2.circle(image, (cx, cy), outer_circle_radius, color, line_thickness)

    # Рисуем круг в центре
    cv2.circle(image, (cx, cy), crosshair_radius, color, line_thickness)

    # Функция для рисования линии с отметками
    def draw_line_with_ticks(start, end, tick_direction):
        # Рисуем основную линию
        cv2.line(image, start, end, color, line_thickness)

        # Рисуем отметки на линии
        for i in range(gap + crosshair_radius, outer_circle_radius, tick_spacing):
            # Вычисляем положение начала и конца отметки
            tick_start = (cx + i * tick_direction[0], cy + i * tick_direction[1])

            # Рисуем отметку
            if tick_direction[0] == 0:  # Вертикальные отметки
                cv2.line(image, (tick_start[0] - tick_length // 2, tick_start[1]),
                         (tick_start[0] + tick_length // 2, tick_start[1]), color, line_thickness)
            else:  # Горизонтальные отметки
                cv2.line(image, (tick_start[0], tick_start[1] - tick_length // 2),
                         (tick_start[0], tick_start[1] + tick_length // 2), color, line_thickness)

    # Рисуем вертикальные линии с отметками
    draw_line_with_ticks((cx, cy - crosshair_radius - gap), (cx, cy - outer_circle_radius), (0, -1))
    draw_line_with_ticks((cx, cy + crosshair_radius + gap), (cx, cy + outer_circle_radius), (0, 1))

    # Рисуем горизонтальные линии с отметками
    draw_line_with_ticks((cx - crosshair_radius - gap, cy), (cx - outer_circle_radius, cy), (-1, 0))
    draw_line_with_ticks((cx + crosshair_radius + gap, cy), (cx + outer_circle_radius, cy), (1, 0))


# Окна программы и их позиции
window_name_1 = 'Original Screen with Red Mask and Crosshair'
window_name_2 = 'Red Mask (Threshold)'
cv2.namedWindow(window_name_1)
cv2.namedWindow(window_name_2)

while True:
    # Перед захватом экрана, временно перемещаем окна за пределы экрана для эффекта мерцания
    cv2.moveWindow(window_name_1, -10000, -10000)
    cv2.moveWindow(window_name_2, -10000, -10000)

    # Захват экрана с помощью pyautogui
    screen = pyautogui.screenshot()
    frame = np.array(screen)  # Преобразование скриншота в массив NumPy
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Преобразование в формат BGR

    # Возвращаем окна в видимые координаты
    cv2.moveWindow(window_name_1, 0, 0)  # Позиция окна 1 (например, верхний левый угол)
    cv2.moveWindow(window_name_2, 500, 0)  # Позиция окна 2 (например, справа от окна 1)

    # Преобразование изображения в цветовое пространство HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Получаем маску для красного цвета
    red_mask = get_red_mask(hsv_frame)

    # Применяем морфологические преобразования
    red_mask_morph = apply_morphological_transformations(red_mask)

    # Находим моменты изображения
    moments = cv2.moments(red_mask_morph)

    # Площадь объекта (момент m00)
    area = moments['m00']

    # Если площадь объекта больше порогового значения (для исключения шумов)
    if area > 1000:
        # Вычисляем координаты центра масс объекта
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # Рисуем прицел AWP в центре обнаруженного объекта
        thisIsAwp(frame, cx, cy)

        # Отображаем центр объекта и его площадь
        cv2.putText(frame, f"Area: {int(area)}", (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Отображение результата
    cv2.imshow(window_name_1, frame)
    cv2.imshow(window_name_2, red_mask_morph)

    # Ожидание времени для предотвращения быстрого цикла
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# Закрытие окон
cv2.destroyAllWindows()
