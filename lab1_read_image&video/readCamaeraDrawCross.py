import cv2

# открываем камеру, где 0 камера по умолчанию
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка открытия камеры")
    exit()

while True:
    # захват кадра пока работаем
    ret, frame = cap.read()

    if not ret:
        print("Не удалось получить кадр с камеры")
        break

    # Получаем размеры кадра
    height, width = frame.shape[:2]

    # Считаем центр изображения
    center_x, center_y = width // 2, height // 2

    # горизонтальная линия в 2 пикселя красная
    cv2.line(frame, (center_x - 50, center_y), (center_x + 50, center_y), (0, 0, 255), 2)

    # вертикальная линия в 2 пикселя красная
    cv2.line(frame, (center_x, center_y - 50), (center_x, center_y + 50), (0, 0, 255), 2)

    # Отображаем кадр с красным крестом
    cv2.imshow('Camera with Red Cross', frame)

    # Нажмите 'q' для выхода
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
