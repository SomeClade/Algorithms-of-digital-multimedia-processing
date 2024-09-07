import cv2

# Открываем камеру
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка открытия камеры")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Не удалось получить кадр с камеры")
        break

    # Получаем размеры кадра
    height, width = frame.shape[:2]

    # Определяем координаты центра
    center_x, center_y = width // 2, height // 2

    # Получаем центральный пиксель (формат BGR)
    center_pixel = frame[center_y, center_x]

    # Извлекаем значения цветовых каналов BGR
    blue, green, red = center_pixel

    # Определяем цвета
    if red >= green and red >= blue:
        cross_color = (0, 0, 255)
    elif green >= red and green >= blue:
        cross_color = (0, 255, 0)
    else:
        cross_color = (255, 0, 0)

    # Рисуем
    cv2.line(frame, (center_x - 50, center_y), (center_x + 50, center_y), cross_color, 5)  # Горизонтальная линия
    cv2.line(frame, (center_x, center_y - 50), (center_x, center_y + 50), cross_color, 5)  # Вертикальная линия

    # Отображаем кадр с крестом
    cv2.imshow('Webcam with Dynamic Cross', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
