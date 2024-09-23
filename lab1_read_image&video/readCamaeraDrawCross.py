import cv2
import numpy as np

# Открываем камеру
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка открытия камеры")
    exit()

while True:
    # Захватываем кадр
    ret, frame = cap.read()

    if not ret:
        print("Не удалось получить кадр с камеры")
        break

    # Получаем размеры кадра
    height, width = frame.shape[:2]

    # Считаем центр изображения
    center_x, center_y = width // 2, height // 2
    radius = 50  # Радиус окружности, на которой будут расположены вершины пентаграммы

    # Вычисляем координаты вершин пентаграммы
    points = []
    for i in range(5):
        angle = 2 * np.pi * i / 5 - np.pi / 2  # Угол для каждой вершины
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        points.append((x, y))

    # Соединяем вершины линиями для создания пентаграммы
    for i in range(5):
        cv2.line(frame, points[i], points[(i + 2) % 5], (0, 0, 255), 2)  # Соединяем через одну вершину

    # Отображаем кадр с пентаграммой
    cv2.imshow('Camera with Pentagram', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
