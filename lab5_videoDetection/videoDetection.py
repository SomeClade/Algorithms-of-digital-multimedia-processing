import cv2
import numpy as np

# Открываем видеофайл для чтения
cap = cv2.VideoCapture('input_video.mp4')

# Получаем параметры видео для сохранения выходного файла
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Инициализируем объект для записи видео
out = cv2.VideoWriter('output_video.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (frame_width, frame_height))

# Инициализируем фоновый вычитатель
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Применяем фоновое вычитание
    fg_mask = backSub.apply(frame)

    # Удаляем тени (опционально)
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

    # Применяем морфологические операции для удаления шума
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    # Находим контуры
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    movement = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            movement = True
            # Улучшаем контур с помощью приближения
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

    # Если было движение, записываем кадр
    if movement:
        out.write(frame)

    # Отображаем кадр
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fg_mask)

    # Прерываем цикл по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()
