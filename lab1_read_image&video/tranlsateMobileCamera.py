import cv2

# video_url = 'https://public.ivideon.com/camera/100-F7h5IwKKjv5VrKQzFZwgaF/0/?lang=ru'

video_url = "https://212.192.149.207:8080/video"

cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Ошибка открытия видеопотока с телефона")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Ошибка при получении кадра")
        break
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('Video from Phone Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
