import cv2

cap = cv2.VideoCapture('input_video.mp4')

# Получаем параметры видео для сохранения выходного файла
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Инициализируем объект для записи видео
out = cv2.VideoWriter('output_video.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (frame_width, frame_height))

ret, frame1 = cap.read()
if not ret:
    print("Не удалось прочитать видеофайл.")
    cap.release()
    out.release()
    exit()

# Преобразуем кадр в оттенки серого и применяем размытие Гаусса
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    # Преобразуем кадр в оттенки серого и применяем размытие Гаусса
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    # Вычисляем разницу между кадрами
    frame_diff = cv2.absdiff(gray1, gray2)

    # Применяем пороговое значение для выделения движущихся областей
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]  # todo баловаться с значениями

    # Увеличиваем изображение для заполнения "дырок"
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Находим контуры
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    movement = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Порог площади контура
            movement = True
            #  рисуем прямоугольник вокруг движущегося объекта
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Если было движение, записываем кадр
    if movement:
        out.write(frame2)

    # Отображаем кадр
    cv2.imshow('Video', frame2)

    # Обновляем предыдущий кадр
    gray1 = gray2.copy()

    # Прерываем цикл по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()
