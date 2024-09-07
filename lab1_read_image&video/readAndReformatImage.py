import cv2

# Устанавливаем 3 расширения
extensions = ['jpg', 'png', 'bmp']

# Массив флагов для чтения
read_flags = [cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED]

# Массив флагов для окна
window_flags = [cv2.WINDOW_NORMAL, cv2.WINDOW_AUTOSIZE, cv2.WINDOW_FULLSCREEN]

# Вывод окон с разными флагами
for ext in extensions:
    filename = f'images/img1.{ext}'  #  Кидаем разрешение к файлу
    for read_flag in read_flags:
        img = cv2.imread(filename, read_flag)  # читаем изображение но уже с новым флагом
        if img is None: # ну вдруг
            print(f"Не удалось открыть изображение: {filename} с флагом {read_flag}")
            continue

        for window_flag in window_flags:
            cv2.namedWindow('Image', window_flag)  # Создаем окно с флагом
            cv2.imshow('Image', img)  # Выводим изображение
            cv2.imshow('Image', img)
            print(f"Открыто изображение {filename} с флагом чтения {read_flag} и флагом окна {window_flag}")
            resizedImage= cv2.resize(img, (400,300))
            cv2.imshow('Image', resizedImage) #ресайз изображения
            cv2.waitKey(0)
            cv2.destroyAllWindows()
