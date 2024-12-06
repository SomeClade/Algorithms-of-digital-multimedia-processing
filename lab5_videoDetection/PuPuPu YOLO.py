import cv2
import  torch
import numpy as np

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]  # Класс 'person'

# Открываем видеофайл для чтения
cap = cv2.VideoCapture('input_video.mp4')

# Получаем параметры видео для сохранения выходного файла
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Инициализируем объект для записи видео
out = cv2.VideoWriter('output_video.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Применяем модель YOLOv5 к кадру
    results = model(frame)

    # Получаем данные детекции
    detections = results.xyxy[0]  # координаты в формате [x1, y1, x2, y2, confidence, class]

    movement = False
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        # Обводим контур человека
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        movement = True

    # Если был обнаружен человек, записываем кадр
    if movement:
        out.write(frame)

    # Отображаем кадр
    cv2.imshow('Frame', frame)

    # Прерываем цикл по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()
