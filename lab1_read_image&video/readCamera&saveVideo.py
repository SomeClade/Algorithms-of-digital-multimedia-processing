import cv2

# открываем камеру, где 0 камера по умолчанию
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка открытия камеры")
    exit()

# Параметры видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 60

# Инициализируем объект для записи видео (кодек XVID, файл .avi)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, fps, (frame_width, frame_height))

print("Запись видео началась. Нажмите 'q' для завершения.")

while True:
    # Читаем кадр с камеры
    ret, frame = cap.read()

    if not ret:
        print("Ошибка при захвате кадра")
        break

    # Рисуем видео в окне
    cv2.imshow('Webcam Live', frame)

    # Записываем кадр в файл
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()

print("Запись завершена. Демонстрация записанного видео.")

# демонстрация записи
cap = cv2.VideoCapture('output_video.avi')

if not cap.isOpened():
    print("Ошибка открытия записанного видео")
    exit()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Видео закончилось")
        break

    # Отображаем записанное видео
    cv2.imshow('Recorded Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
