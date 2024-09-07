import cv2

# открываем видео
cap = cv2.VideoCapture('video/video.mp4')

if not cap.isOpened():
    print("Ошибка открытия видеофайла")
    exit()

while cap.isOpened():
    ret, frame = cap.read()  # Читаем кадр
    if not ret:
        print("Не удалось прочитать кадр или видео закончилось")
        break

    # изменяем размер видео
    resized_frame = cv2.resize(frame, (640, 480))

    # изменяем цветовую гамму
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)  # чб

    # оригинальный кадр
    cv2.imshow('Original Video', frame)

    # изменный по размеру кадр
    cv2.imshow('Resized Video', resized_frame)

    # чб кадр
    cv2.imshow('Grayscale Video', gray_frame)

    # 25мс для отображения кадра
    if cv2.waitKey(25) & 0xFF == ord('q'):  # q = exit
        break

# чистим память  О_О
cap.release()
cv2.destroyAllWindows()
