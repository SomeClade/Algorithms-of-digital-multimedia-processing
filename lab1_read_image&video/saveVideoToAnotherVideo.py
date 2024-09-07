import cv2

# открываем исходное видео
cap = cv2.VideoCapture('video/video.mp4')

if not cap.isOpened():
    print("Ошибка открытия исходного видеофайла")
    exit()

# чтение параметров исходного видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# настройка выходного видео
output_files = {
    'video/output_video.avi': {'codec': 'XVID', 'fps': fps},
    'video/output_video_low_fps.avi': {'codec': 'XVID', 'fps': fps // 2},  # FPS в 2 раза ниже
    'video/output_video.mp4': {'codec': 'mp4v', 'fps': fps},
}

# словарь для хранения объектов VideoWriter для разных форматов
out_writers = {}

# инициализация сохранения
for file, params in output_files.items():
    fourcc = cv2.VideoWriter_fourcc(*params['codec'])  # Устанавливаем кодек
    out_writers[file] = cv2.VideoWriter(file, fourcc, params['fps'], (frame_width, frame_height))

# чтение видео и запись каждого кадра в разные форматы
while True:
    ret, frame = cap.read()
    if not ret:
        print("Видео закончилось или произошла ошибка")
        break

    # запись с разными параметрами
    for writer in out_writers.values():
        writer.write(frame)

    cv2.imshow('Video', frame)

    # ожидание 1 мс между кадрами
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Нажмите 'q' для выхода
        break

# очистка ресурсов
cap.release()
for writer in out_writers.values():
    writer.release()
cv2.destroyAllWindows()

print("Видео успешно записано в несколько файлов:")
for file in output_files.keys():
    print(f"- {file}")
