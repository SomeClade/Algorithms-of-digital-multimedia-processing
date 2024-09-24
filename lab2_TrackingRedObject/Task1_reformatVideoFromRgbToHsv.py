import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

while True:
    # Захват кадра с камеры
    ret, frame = cap.read()

    if not ret:
        print("Ошибка при получении кадра")
        break

    # Преобразование изображения в цветовое пространство HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    cv2.imshow('Original Image', frame)
    cv2.imshow('HSV Image', hsv_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
